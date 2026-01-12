#!/usr/bin/env python3
"""
分析session数据中一级分类多样性随step的变化
"""
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def extract_categories(response: str) -> List[Tuple[str, str]]:
    """
    从response字段中提取分类信息
    格式：分类：一级分类-二级分类 或 分类：一级分类-二级分类；一级分类-二级分类
    返回：[(一级分类, 二级分类), ...]的列表
    """
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


def load_session_files(data_dir: Path) -> List[Dict]:
    """
    加载指定目录下所有session*.json文件
    """
    session_files = sorted(data_dir.glob('session*.json'))
    if not session_files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到session*.json文件")
    
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
                for action in data.get('actions', []):
                    step_info = {
                        'session_id': session_id,
                        'step': action.get('step'),
                        'timestamp': action.get('timestamp'),
                        'response': action.get('response', ''),
                        'categories': extract_categories(action.get('response', ''))
                    }
                    all_steps.append(step_info)
        except json.JSONDecodeError as e:
            print(f"警告: 跳过无效的JSON文件 {session_file.name}: {e}")
            continue
        except Exception as e:
            print(f"警告: 处理文件 {session_file.name} 时出错: {e}")
            continue
    
    return all_steps


def analyze_diversity(all_steps: List[Dict], window_size: int = 20) -> Dict:
    """
    分析多样性随step的变化（只统计一级分类）
    使用滑动窗口，只统计最近window_size个step
    所有session的step统一按时间排序，使用全局连续索引
    """
    # 按时间戳排序（统一所有session的step）
    sorted_steps = sorted(all_steps, key=lambda x: parse_timestamp(x['timestamp']) if x['timestamp'] else 0)
    
    # 累积统计（用于最终统计）
    all_level1_seen = set()  # 所有一级分类集合
    
    diversity_over_steps = []
    
    for i, step_info in enumerate(sorted_steps):
        # 使用全局索引（从1开始），而不是各个session内部的step编号
        global_step_index = i + 1
        original_step = step_info['step']
        categories = step_info['categories']
        
        # 更新全局集合（用于最终统计）
        for level1, level2 in categories:
            all_level1_seen.add(level1)
        
        # 滑动窗口：统计最近window_size个step中的一级分类
        window_start = max(0, i - window_size + 1)
        window_steps = sorted_steps[window_start:i+1]
        
        # 统计窗口内的一级分类
        window_level1_seen = set()
        # 统计加权多样性：一级分类不同算1，一级相同但二级不同算0.2
        level1_to_level2 = defaultdict(set)  # 一级分类 -> 二级分类集合
        for window_step in window_steps:
            for level1, level2 in window_step['categories']:
                window_level1_seen.add(level1)
                level1_to_level2[level1].add(level2)
        
        # 计算加权多样性分数
        # 每个一级分类算1分，每个二级分类算0.2分（但一级分类已经算1分了，所以二级分类额外算0.2）
        weighted_diversity = 0.0
        for level1, level2_set in level1_to_level2.items():
            weighted_diversity += 1.0  # 一级分类贡献1分
            if len(level2_set) > 1:
                # 如果该一级分类下有多个二级分类，每个额外的二级分类贡献0.2分
                weighted_diversity += (len(level2_set) - 1) * 0.2
        
        # 记录当前step的多样性（滑动窗口内的）
        diversity_over_steps.append({
            'global_step': global_step_index,  # 全局连续索引
            'original_step': original_step,  # 原始session内的step编号
            'session_id': step_info['session_id'],
            'timestamp': step_info['timestamp'],
            'level1_count': len(window_level1_seen),
            'weighted_diversity': weighted_diversity,  # 加权多样性分数
            'window_size': len(window_steps),  # 实际窗口大小（前20个step可能不足20）
            'categories': categories
        })
    
    return {
        'total_steps': len(sorted_steps),
        'window_size': window_size,
        'final_level1_count': len(all_level1_seen),
        'diversity_over_steps': diversity_over_steps,
        'all_level1_categories': sorted(all_level1_seen)
    }


def print_results(results: Dict):
    """
    打印分析结果
    """
    print("\n" + "="*80)
    print("一级分类多样性分析结果")
    print("="*80)
    
    print(f"\n总步数: {results['total_steps']}")
    print(f"\n滑动窗口大小: {results.get('window_size', 20)}")
    print(f"\n最终统计（所有step）:")
    print(f"  一级分类数量: {results['final_level1_count']}")
    
    print(f"\n一级分类列表 ({len(results['all_level1_categories'])}):")
    for cat in results['all_level1_categories']:
        print(f"  - {cat}")
    
    print(f"\n多样性随step的变化（滑动窗口内，全局排序）:")
    print(f"{'全局Step':<10} {'原始Step':<10} {'Session ID':<20} {'窗口大小':<10} {'一级分类数':<12} {'加权多样性':<12}")
    print("-" * 90)
    
    for item in results['diversity_over_steps']:
        window_size = item.get('window_size', results.get('window_size', 20))
        global_step = item.get('global_step', item.get('step', ''))
        original_step = item.get('original_step', '')
        weighted_div = item.get('weighted_diversity', 0)
        print(f"{global_step:<10} {original_step:<10} {item['session_id']:<20} {window_size:<10} {item['level1_count']:<12} {weighted_div:<12.2f}")


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


def plot_diversity(results: Dict, output_file: Path = None, smooth_window: int = 5):
    """
    绘制一级分类多样性随step变化的图表
    smooth_window: 平滑窗口大小，0表示不平滑
    """
    diversity_data = results['diversity_over_steps']
    
    if not diversity_data:
        print("警告: 没有数据可绘制")
        return
    
    # 提取数据（使用全局step索引）
    steps = [item.get('global_step', item.get('step', i+1)) for i, item in enumerate(diversity_data)]
    level1_counts = [item['level1_count'] for item in diversity_data]
    weighted_diversities = [item.get('weighted_diversity', 0) for item in diversity_data]
    
    # 平滑处理
    if smooth_window > 0:
        level1_counts_smooth = smooth_data(level1_counts, smooth_window)
        weighted_diversities_smooth = smooth_data(weighted_diversities, smooth_window)
    else:
        level1_counts_smooth = level1_counts
        weighted_diversities_smooth = weighted_diversities
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制原始数据（透明度较低）
    ax.plot(steps, level1_counts, marker='o', label='一级分类数量（原始）', 
            linewidth=1, markersize=3, color='#1f77b4', alpha=0.3)
    ax.plot(steps, weighted_diversities, marker='s', label='加权多样性（原始）', 
            linewidth=1, markersize=3, color='#ff7f0e', alpha=0.3)
    
    # 绘制平滑后的曲线
    ax.plot(steps, level1_counts_smooth, label='一级分类数量（平滑）', 
            linewidth=2.5, color='#1f77b4')
    ax.plot(steps, weighted_diversities_smooth, label='加权多样性（一级=1，二级=0.2）（平滑）', 
            linewidth=2.5, color='#ff7f0e')
    
    # 设置图表属性
    window_size = results.get('window_size', 20)
    ax.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('一级分类数量', fontsize=12, fontweight='bold')
    ax.set_title(f'一级分类多样性随Step的变化（滑动窗口：最近{window_size}个step）', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置x轴刻度
    if len(steps) > 0:
        step_min, step_max = min(steps), max(steps)
        if step_max - step_min > 20:
            # 如果step范围较大，只显示部分刻度
            ax.set_xticks(range(step_min, step_max + 1, max(1, (step_max - step_min) // 20)))
        else:
            ax.set_xticks(range(step_min, step_max + 1))
    
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
    parser = argparse.ArgumentParser(description='分析session数据中一级分类多样性随step的变化')
    parser.add_argument('--name', type=str, required=True, help='数据目录名称（如zsl）')
    parser.add_argument('--output', type=str, default=None, help='输出JSON文件路径（可选）')
    parser.add_argument('--window-size', type=int, default=20, help='滑动窗口大小（默认20）')
    parser.add_argument('--smooth-window', type=int, default=5, help='平滑窗口大小（默认5，设为0禁用平滑）')
    
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
    
    # 加载数据
    all_steps = load_session_files(data_dir)
    print(f"共加载 {len(all_steps)} 个steps")
    
    # 分析多样性
    results = analyze_diversity(all_steps, window_size=args.window_size)
    
    # 打印结果
    print_results(results)
    
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

