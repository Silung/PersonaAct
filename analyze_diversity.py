#!/usr/bin/env python3
"""
分析session数据中一级分类多样性随step的变化
"""
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 全局变量：存储图片分类结果
IMAGE_CLASSIFICATIONS = {}


def load_image_classifications(classification_file: Path) -> Dict[str, str]:
    """
    加载图片分类结果
    返回：{image_path: "一级分类-二级分类"}
    """
    if not classification_file.exists():
        print(f"警告: 分类文件不存在: {classification_file}")
        return {}
    
    try:
        with open(classification_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 加载分类文件失败: {e}")
        return {}


def extract_categories(response: str = None, screenshot: str = None) -> List[Tuple[str, str]]:
    """
    提取分类信息，优先使用重新分类的结果
    参数：
        response: 原始response字段（用于fallback）
        screenshot: 截图路径（用于查找重新分类的结果）
    返回：[(一级分类, 二级分类), ...]的列表
    """
    # 如果有screenshot且在分类结果中，优先使用重新分类的结果
    if screenshot and IMAGE_CLASSIFICATIONS:
        # 标准化路径（转换为Unix风格）
        screenshot_normalized = screenshot.replace('\\', '/')
        
        # 尝试多种路径格式
        possible_paths = [
            screenshot_normalized,
            str(Path(screenshot)),
            str(Path(screenshot).as_posix())
        ]
        
        for path in possible_paths:
            if path in IMAGE_CLASSIFICATIONS:
                category_str = IMAGE_CLASSIFICATIONS[path]
                # 解析 "一级分类-二级分类" 格式
                parts = category_str.split('-', 1)
                if len(parts) == 2:
                    level1 = parts[0].strip()
                    level2 = parts[1].strip()
                    if level1 and level2:
                        return [(level1, level2)]
                break
    
    # Fallback: 从response字段中提取分类信息
    if not response:
        return []
    
    # 查找"分类："后面的内容
    match = re.search(r'分类[：:]\s*([^\n]+)', response)
    if not match:
        return []
    
    categories_str = match.group(1).strip()
    if not categories_str:
        return []
    
    # 按分号分割多个分类
    category_pairs = []
    for cat_str in categories_str.split('；'):
        cat_str = cat_str.strip()
        if not cat_str:
            continue
        
        # 按短横线分割一级和二级分类
        parts = cat_str.split('-', 1)
        if len(parts) == 2:
            level1 = parts[0].strip()
            level2 = parts[1].strip()
            if level1 and level2:
                category_pairs.append((level1, level2))
    
    return category_pairs


def parse_timestamp(timestamp: str) -> int:
    """
    解析时间戳字符串为整数，用于排序
    格式：20260112T133447097
    """
    # 移除T并转换为整数
    return int(timestamp.replace('T', ''))


def timestamp_to_datetime(timestamp: str) -> datetime:
    """
    将时间戳字符串转换为datetime对象
    格式：20260112T133447097 -> 2026-01-12 13:34:47.097
    """
    # 解析格式：YYYYMMDDTHHMMSSMMM
    date_part = timestamp[:8]  # YYYYMMDD
    time_part = timestamp[9:]  # HHMMSSMMM
    
    year = int(date_part[:4])
    month = int(date_part[4:6])
    day = int(date_part[6:8])
    hour = int(time_part[:2])
    minute = int(time_part[2:4])
    second = int(time_part[4:6])
    microsecond = int(time_part[6:]) * 1000 if len(time_part) > 6 else 0
    
    return datetime(year, month, day, hour, minute, second, microsecond)


def calculate_time_span_minutes(timestamps: List[str]) -> float:
    """
    计算时间戳列表的时间跨度（分钟）
    """
    if len(timestamps) < 2:
        return 0.0
    
    try:
        datetimes = [timestamp_to_datetime(ts) for ts in timestamps if ts]
        if len(datetimes) < 2:
            return 0.0
        
        time_span = (max(datetimes) - min(datetimes)).total_seconds() / 60.0  # 转换为分钟
        return max(time_span, 0.01)  # 避免除以0，最小值设为0.01分钟（约0.6秒）
    except Exception as e:
        print(f"警告: 计算时间跨度失败: {e}")
        return 0.0


def load_session_files(data_dir: Path) -> List[Dict]:
    """
    加载指定目录下所有session数据
    支持两种格式：
    1. 新格式：session_*/metadata.json（每个session独立目录）
    2. 旧格式：session*.json（所有session在同一目录）
    """
    # 先尝试新格式：查找 session_* 子目录下的 metadata.json
    session_dirs = sorted([d for d in data_dir.glob('session_*') if d.is_dir()])
    session_files = []
    
    if session_dirs:
        print(f"使用新格式：找到 {len(session_dirs)} 个 session 目录")
        for session_dir in session_dirs:
            metadata_file = session_dir / 'metadata.json'
            if metadata_file.exists():
                session_files.append(metadata_file)
            else:
                print(f"警告: {session_dir.name} 目录下未找到 metadata.json")
    
    # 如果没有找到新格式，尝试旧格式：session*.json
    if not session_files:
        print(f"未找到新格式数据，尝试旧格式...")
        session_files = sorted(data_dir.glob('session*.json'))
    
    if not session_files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到session数据文件（新格式或旧格式）")
    
    all_steps = []
    for session_file in session_files:
        try:
            # 检查文件是否为空
            if session_file.stat().st_size == 0:
                print(f"跳过空文件: {session_file.name}")
                continue
            
            print(f"加载文件: {session_file.name}")
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                session_id = data.get('session_id', session_file.stem)
                
                # 提取所有steps
                reverse_persona = data.get('reverse_persona', False)
                for action in data.get('actions', []):
                    screenshot = action.get('screenshot', '')
                    step_info = {
                        'session_id': session_id,
                        'step': action.get('step'),
                        'timestamp': action.get('timestamp'),
                        'response': action.get('response', ''),
                        'screenshot': screenshot,
                        'viewing_duration': action.get('viewing_duration', 0.0),  # 实际观看时长（秒）
                        'categories': extract_categories(
                            response=action.get('response', ''),
                            screenshot=screenshot
                        ),
                        'reverse_persona': reverse_persona  # 保留reverse_persona信息
                    }
                    all_steps.append(step_info)
        except json.JSONDecodeError as e:
            print(f"警告: 跳过无效的JSON文件 {session_file.name}: {e}")
            continue
        except Exception as e:
            print(f"警告: 处理文件 {session_file.name} 时出错: {e}")
            continue
    
    return all_steps


def calculate_entropy(category_distribution: Dict[str, float]) -> float:
    """
    计算类别熵: H = -sum(p(c) * log(p(c)))
    """
    entropy = 0.0
    for prob in category_distribution.values():
        if prob > 0:
            entropy -= prob * np.log(prob)
    return entropy


def analyze_diversity(all_steps: List[Dict], window_size: int = 50, category_level: int = 1) -> Dict:
    """
    分析多样性随step的变化
    使用滑动窗口，只统计最近window_size个step
    所有session的step统一按时间排序，使用全局连续索引
    
    参数：
        category_level: 1=统计一级分类，2=统计二级分类
    """
    # 按时间戳排序（统一所有session的step）
    sorted_steps = sorted(all_steps, key=lambda x: parse_timestamp(x['timestamp']) if x['timestamp'] else 0)
    
    # 累积统计（用于最终统计）
    all_categories_seen = set()  # 所有分类集合（一级或二级）
    
    diversity_over_steps = []
    
    for i, step_info in enumerate(sorted_steps):
        # 使用全局索引（从1开始），而不是各个session内部的step编号
        global_step_index = i + 1
        original_step = step_info['step']
        categories = step_info['categories']
        
        # 更新全局集合（用于最终统计）
        for level1, level2 in categories:
            if category_level == 1:
                all_categories_seen.add(level1)
            else:  # category_level == 2
                all_categories_seen.add(f"{level1}-{level2}")
        
        # 滑动窗口：统计最近window_size个step中的分类
        window_start = max(0, i - window_size + 1)
        window_steps = sorted_steps[window_start:i+1]
        
        # 统计窗口内的分类
        window_categories_seen = set()
        # 统计加权多样性：一级分类不同算1，一级相同但二级不同算0.2
        level1_to_level2 = defaultdict(set)  # 一级分类 -> 二级分类集合
        for window_step in window_steps:
            for level1, level2 in window_step['categories']:
                if category_level == 1:
                    window_categories_seen.add(level1)
                else:  # category_level == 2
                    window_categories_seen.add(f"{level1}-{level2}")
                level1_to_level2[level1].add(level2)
        
        # 计算加权多样性分数
        if category_level == 1:
            # 规则：一级类目不同时只记1，不再考虑二级类目；一级类目相同时再考虑二级类目
            weighted_diversity = 0.0
            for level1, level2_set in level1_to_level2.items():
                # 一级分类不同：每个一级分类贡献1分（不考虑二级分类）
                weighted_diversity += 1.0
                # 一级分类相同：如果同一个一级分类下有多个不同的二级分类，每个额外的二级分类贡献0.2分
                if len(level2_set) > 1:
                    weighted_diversity += (len(level2_set) - 1) * 0.2
        else:  # category_level == 2
            # 二级类目：每个不同的"一级-二级"组合贡献1分
            weighted_diversity = float(len(window_categories_seen))
        
        # 计算Bubble Breadth: 类别熵（使用窗口内所有step，包括reversed persona）
        # 统计窗口内所有step的分类分布
        category_counts = Counter()
        total_categories = 0
        for window_step in window_steps:
            for level1, level2 in window_step['categories']:
                if category_level == 1:
                    category_counts[level1] += 1
                else:  # category_level == 2
                    category_counts[f"{level1}-{level2}"] += 1
                total_categories += 1
        
        # 计算类别概率分布和熵
        if total_categories > 0:
            category_probs = {cat: count / total_categories for cat, count in category_counts.items()}
            entropy = calculate_entropy(category_probs)
        else:
            entropy = 0.0
        
        # 计算分类暴露率（Category Exposure Rate）：单位时间内接触到的不同分类数量（categories/分钟）
        # 使用窗口内所有step的viewing_duration总和（秒）转换为分钟
        total_viewing_seconds = sum(ws.get('viewing_duration', 0.0) for ws in window_steps)
        time_span_minutes = total_viewing_seconds / 60.0  # 转换为分钟
        if time_span_minutes > 0:
            category_exposure_rate = len(window_categories_seen) / time_span_minutes
        else:
            category_exposure_rate = 0.0
        
        # 记录当前step的多样性（滑动窗口内的）
        diversity_over_steps.append({
            'global_step': global_step_index,  # 全局连续索引
            'original_step': original_step,  # 原始session内的step编号
            'session_id': step_info['session_id'],
            'timestamp': step_info['timestamp'],
            'category_count': len(window_categories_seen),  # 分类数量（一级或二级）
            'level1_count': len(window_categories_seen),  # 为了向后兼容保留这个字段
            'weighted_diversity': weighted_diversity,  # 加权多样性分数
            'entropy': entropy,  # Bubble Breadth: 类别熵
            'category_exposure_rate': category_exposure_rate,  # 分类暴露率（categories/分钟）
            'time_span_minutes': time_span_minutes,  # 窗口时间跨度（分钟）
            'window_size': len(window_steps),  # 实际窗口大小（前20个step可能不足20）
            'categories': categories,
            'reverse_persona': step_info.get('reverse_persona', False)  # 保留reverse_persona信息
        })
    
    return {
        'total_steps': len(sorted_steps),
        'window_size': window_size,
        'category_level': category_level,
        'final_category_count': len(all_categories_seen),
        'final_level1_count': len(all_categories_seen),  # 为了向后兼容保留这个字段
        'diversity_over_steps': diversity_over_steps,
        'all_categories': sorted(all_categories_seen),
        'all_level1_categories': sorted(all_categories_seen)  # 为了向后兼容保留这个字段
    }


def print_results(results: Dict):
    """
    打印分析结果
    """
    category_level = results.get('category_level', 1)
    level_name = "一级分类" if category_level == 1 else "二级分类"
    
    print("\n" + "="*80)
    print(f"{level_name}多样性分析结果")
    print("="*80)
    
    print(f"\n总步数: {results['total_steps']}")
    print(f"\n滑动窗口大小: {results.get('window_size', 20)}")
    print(f"\n分类级别: {level_name}")
    print(f"\n最终统计（所有step）:")
    print(f"  {level_name}数量: {results.get('final_category_count', results['final_level1_count'])}")
    
    all_categories = results.get('all_categories', results['all_level1_categories'])
    print(f"\n{level_name}列表 ({len(all_categories)}):")
    for cat in all_categories:
        print(f"  - {cat}")
    
    print(f"\n多样性随step的变化（滑动窗口内，全局排序）:")
    print(f"{'全局Step':<10} {'原始Step':<10} {'Session ID':<20} {'窗口大小':<10} {level_name+'数':<12} {'加权多样性':<12}")
    print("-" * 90)
    
    for item in results['diversity_over_steps']:
        window_size = item.get('window_size', results.get('window_size', 20))
        global_step = item.get('global_step', item.get('step', ''))
        original_step = item.get('original_step', '')
        weighted_div = item.get('weighted_diversity', 0)
        category_count = item.get('category_count', item['level1_count'])
        print(f"{global_step:<10} {original_step:<10} {item['session_id']:<20} {window_size:<10} {category_count:<12} {weighted_div:<12.2f}")


def save_results(results: Dict, output_file: Path):
    """
    保存结果到JSON文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")


def smooth_data(data: List[float], window_size: int = 5) -> List[float]:
    """
    使用移动平均平滑数据
    """
    if len(data) < window_size:
        return data
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window_data = data[start:end]
        smoothed.append(np.mean(window_data))
    
    return smoothed


def smooth_data_forward_only(data: List[float], window_size: int = 5) -> List[float]:
    """
    只使用向后的窗口进行平滑（不向前延伸）
    第一个点平滑前后重叠（值不变）
    """
    if len(data) == 0:
        return data
    
    if len(data) == 1 or window_size <= 1:
        return data.copy()
    
    smoothed = []
    
    for i in range(len(data)):
        if i == 0:
            # 第一个点：平滑前后重叠
            smoothed.append(data[i])
        else:
            # 只使用向后的窗口（包括当前点）
            start = max(0, i - window_size + 1)
            end = i + 1
            window_data = data[start:end]
            smoothed.append(np.mean(window_data))
    
    return smoothed


def plot_diversity(results: Dict, output_file: Path = None, smooth_window: int = 5):
    """
    绘制分类多样性随step变化的图表
    smooth_window: 平滑窗口大小，0表示不平滑
    只绘制persona变成reverse_persona前后各300个step
    """
    diversity_data = results['diversity_over_steps']
    
    if not diversity_data:
        print("警告: 没有数据可绘制")
        return
    
    # 获取分类级别
    category_level = results.get('category_level', 1)
    level_name = "一级分类" if category_level == 1 else "二级分类"
    
    # 获取窗口大小
    window_size = results.get('window_size', 20)
    skip_steps = 0  # 不跳过任何数据，展示所有step
    
    # 提取数据（使用全局step索引）
    all_steps = [item.get('global_step', item.get('step', i+1)) for i, item in enumerate(diversity_data)]
    all_category_counts = [item.get('category_count', item['level1_count']) for item in diversity_data]
    all_weighted_diversities = [item.get('weighted_diversity', 0) for item in diversity_data]
    all_entropies = [item.get('entropy', 0) for item in diversity_data]
    all_exposure_rates = [item.get('category_exposure_rate', 0) for item in diversity_data]
    all_reverse_persona = [item.get('reverse_persona', False) for item in diversity_data]
    
    # 找到第一个reverse_persona=True的step索引
    reverse_persona_start_idx = None
    for i in range(len(all_reverse_persona)):
        if all_reverse_persona[i]:
            reverse_persona_start_idx = i
            break
    
    if reverse_persona_start_idx is None:
        print("警告: 未找到reverse_persona=True的step，将显示所有数据")
        # 使用所有数据
        steps = all_steps
        category_counts = all_category_counts
        weighted_diversities = all_weighted_diversities
        entropies = all_entropies
        exposure_rates = all_exposure_rates
        print(f"共{len(steps)}个数据点用于展示和平滑")
    else:
        # 只取reverse_persona切换点前后各300个step
        range_size = 300
        
        # 统计切换点前后的 persona 类型分布
        total_before = reverse_persona_start_idx
        total_after = len(all_steps) - reverse_persona_start_idx
        
        normal_before = sum(1 for i in range(reverse_persona_start_idx) if not all_reverse_persona[i])
        reversed_after = sum(1 for i in range(reverse_persona_start_idx, len(all_steps)) if all_reverse_persona[i])
        
        print(f"\n找到reverse_persona切换点在第{reverse_persona_start_idx}个step（全局step {all_steps[reverse_persona_start_idx]}）")
        print(f"切换点之前共 {total_before} 个 steps，其中 normal persona: {normal_before}")
        print(f"切换点之后共 {total_after} 个 steps，其中 reversed persona: {reversed_after}")
        
        # 从切换点往前收集 range_size 个 normal persona 的 steps
        normal_indices = []
        for i in range(reverse_persona_start_idx - 1, -1, -1):
            if not all_reverse_persona[i]:
                normal_indices.append(i)
                if len(normal_indices) >= range_size:
                    break
        normal_indices.reverse()  # 恢复时间顺序
        
        # 从切换点往后收集 range_size 个 reversed persona 的 steps
        reversed_indices = []
        for i in range(reverse_persona_start_idx, len(all_steps)):
            if all_reverse_persona[i]:
                reversed_indices.append(i)
                if len(reversed_indices) >= range_size:
                    break
        
        # 合并索引
        selected_indices = normal_indices + reversed_indices
        
        # 提取对应的数据
        steps = [all_steps[i] for i in selected_indices]
        category_counts = [all_category_counts[i] for i in selected_indices]
        weighted_diversities = [all_weighted_diversities[i] for i in selected_indices]
        entropies = [all_entropies[i] for i in selected_indices]
        exposure_rates = [all_exposure_rates[i] for i in selected_indices]
        
        # 计算相对step位置（从0开始的连续索引）
        reverse_persona_relative_idx = len(normal_indices)  # 切换点在筛选后数据中的位置（从0开始）
        relative_steps = list(range(len(steps)))  # 从0开始的连续索引
        
        print(f"\n实际收集到 {len(normal_indices)} 个 normal persona steps 和 {len(reversed_indices)} 个 reversed persona steps")
        print(f"共{len(steps)}个数据点")
        print(f"在筛选后的数据中，reverse_persona切换点位于索引{reverse_persona_relative_idx}")
        
        if len(reversed_indices) < range_size:
            print(f"警告: reversed persona steps 不足 {range_size} 个，可能数据采集未完成")
    
    # 平滑处理（使用只向后的窗口）
    if smooth_window > 0 and len(category_counts) > 0:
        category_counts_smooth = smooth_data_forward_only(category_counts, smooth_window)
        weighted_diversities_smooth = smooth_data_forward_only(weighted_diversities, smooth_window)
        entropies_smooth = smooth_data_forward_only(entropies, smooth_window)
        exposure_rates_smooth = smooth_data_forward_only(exposure_rates, smooth_window)
    else:
        category_counts_smooth = category_counts
        weighted_diversities_smooth = weighted_diversities
        entropies_smooth = entropies
        exposure_rates_smooth = exposure_rates
    
    # 确定使用相对step还是绝对step
    if reverse_persona_start_idx is not None:
        # 使用从1开始的连续索引
        x_data = relative_steps
        x_label = 'Step'
        title_suffix = f'（reverse_persona切换点前后各300个step）'
        # 记录切换点位置用于绘制垂直线
        switch_point = reverse_persona_relative_idx
    else:
        # 使用绝对step位置
        x_data = steps
        x_label = 'Step'
        title_suffix = f'（滑动窗口：最近{window_size}个step）'
        switch_point = None
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 根据分类级别设置标签
    if category_level == 1:
        count_label = '一级分类数量'
        diversity_label = '加权多样性（一级=1，二级=0.2）'
    else:
        count_label = '二级分类数量'
        diversity_label = '加权多样性（每个二级=1）'
    
    # 绘制原始数据（透明度较低）
    ax.plot(x_data, category_counts, label=f'{count_label}（原始）', 
            linewidth=1, color='#1f77b4', alpha=0.3)
    ax.plot(x_data, weighted_diversities, label=f'{diversity_label}（原始）', 
            linewidth=1, color='#ff7f0e', alpha=0.3)
    ax.plot(x_data, entropies, label='类别熵（原始）', 
            linewidth=1, color='#2ca02c', alpha=0.3)
    ax.plot(x_data, exposure_rates, label='分类暴露率（原始）', 
            linewidth=1, color='#d62728', alpha=0.3)
    
    # 绘制平滑后的曲线
    ax.plot(x_data, category_counts_smooth, label=f'{count_label}（平滑）', 
            linewidth=2.5, color='#1f77b4')
    ax.plot(x_data, weighted_diversities_smooth, label=f'{diversity_label}（平滑）', 
            linewidth=2.5, color='#ff7f0e')
    ax.plot(x_data, entropies_smooth, label='Bubble Breadth: 类别熵（平滑）', 
            linewidth=2.5, color='#2ca02c')
    ax.plot(x_data, exposure_rates_smooth, label='分类暴露率（categories/分钟）（平滑）', 
            linewidth=2.5, color='#d62728')
    
    # 如果找到了reverse_persona切换点，添加垂直线标记
    if switch_point is not None:
        ax.axvline(x=switch_point, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label='reverse_persona切换点')
    
    # 设置图表属性
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{level_name}数量', fontsize=12, fontweight='bold')
    ax.set_title(f'{level_name}多样性随Step的变化{title_suffix}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置x轴刻度
    if len(x_data) > 0:
        x_min, x_max = min(x_data), max(x_data)
        if x_max - x_min > 20:
            # 如果step范围较大，只显示部分刻度
            step_interval = max(1, (x_max - x_min) // 20)
            ax.set_xticks(range(int(x_min), int(x_max) + 1, step_interval))
        else:
            ax.set_xticks(range(int(x_min), int(x_max) + 1))
    
    plt.tight_layout()
    
    # 保存图表
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_file}")
    else:
        plt.savefig('diversity_plot.png', dpi=300, bbox_inches='tight')
        print(f"图表已保存到: diversity_plot.png")
    
    # 显示图表（如果在交互式环境中）
    try:
        plt.show()
    except:
        pass  # 如果无法显示（如无GUI环境），则跳过
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='分析session数据中分类多样性随step的变化')
    parser.add_argument('--name', type=str, required=True, help='数据目录名称（如zsl）')
    parser.add_argument('--output', type=str, default=None, help='输出JSON文件路径（可选）')
    parser.add_argument('--window-size', type=int, default=50, help='滑动窗口大小（默认50）')
    parser.add_argument('--smooth-window', type=int, default=5, help='平滑窗口大小（默认5，设为0禁用平滑）')
    parser.add_argument('--category-level', type=int, default=1, choices=[1, 2], 
                        help='分类级别：1=一级分类，2=二级分类（默认1）')
    parser.add_argument('--use-reclassification', action='store_true', 
                        help='使用重新分类的结果（从image_classifications.json读取）')
    
    args = parser.parse_args()
    
    # 确定数据目录路径
    # 优先尝试deploy/{name}（用户指定的路径）
    data_dir = Path('deploy') / args.name
    if not data_dir.exists():
        # 如果不存在，尝试deploy_log/{name}（实际数据位置）
        data_dir = Path('deploy_log') / args.name
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: deploy/{args.name} 或 deploy_log/{args.name}")
    
    print(f"数据目录: {data_dir}")
    print(f"滑动窗口大小: {args.window_size}")
    print(f"分类级别: {'一级分类' if args.category_level == 1 else '二级分类'}")
    
    # 加载图片分类结果（如果使用重新分类）
    global IMAGE_CLASSIFICATIONS
    if args.use_reclassification:
        classification_file = data_dir / 'image_classifications.json'
        IMAGE_CLASSIFICATIONS = load_image_classifications(classification_file)
        if IMAGE_CLASSIFICATIONS:
            print(f"已加载 {len(IMAGE_CLASSIFICATIONS)} 个图片的重新分类结果")
        else:
            print("警告: 未找到重新分类结果，将使用原始分类")
    
    # 加载数据
    all_steps = load_session_files(data_dir)
    print(f"共加载 {len(all_steps)} 个steps")
    
    # 分析多样性
    results = analyze_diversity(all_steps, window_size=args.window_size, category_level=args.category_level)
    
    # 打印结果
    # print_results(results)
    
    # 保存结果（如果指定了输出文件）
    if args.output:
        output_path = Path(args.output)
        save_results(results, output_path)
    else:
        # 默认保存到数据目录
        output_path = data_dir / 'diversity_analysis.json'
        save_results(results, output_path)
    
    # 绘制图表
    plot_output = data_dir / 'diversity_plot.png'
    plot_diversity(results, plot_output, smooth_window=args.smooth_window)


if __name__ == '__main__':
    main()

