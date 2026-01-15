#!/usr/bin/env python3
"""
统计Bubble Depth指标
- Bubble Depth: 使用COI (Counterfactual Overlap Index) 衡量escape potential
"""
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from scipy.spatial.distance import jensenshannon


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
    
    match = re.search(r'分类[：:]\s*([^\n]+)', response)
    if not match:
        return []
    
    categories_str = match.group(1).strip()
    if not categories_str:
        return []
    
    category_pairs = []
    for cat_str in categories_str.split('；'):
        cat_str = cat_str.strip()
        if not cat_str:
            continue
        
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
    return int(timestamp.replace('T', ''))


def load_session_files(data_dir: Path, range_size: int = 400) -> List[Dict]:
    """
    加载指定目录下所有session数据
    支持两种格式：
    1. 新格式：session_*/metadata.json（每个session独立目录）
    2. 旧格式：session*.json（所有session在同一目录）
    只保留 reverse_persona 切换点前后各 range_size 个 step
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
    
    # 首先收集所有 steps 并按时间排序
    all_steps = []
    for session_file in session_files:
        try:
            if session_file.stat().st_size == 0:
                print(f"跳过空文件: {session_file.name}")
                continue
            
            print(f"加载文件: {session_file.name}")
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                session_id = data.get('session_id', session_file.stem)
                collector = data.get('collector', '')
                reverse_persona = data.get('reverse_persona', False)
                
                for action in data.get('actions', []):
                    screenshot = action.get('screenshot', '')
                    step_info = {
                        'session_id': session_id,
                        'collector': collector,
                        'reverse_persona': reverse_persona,
                        'step': action.get('step'),
                        'timestamp': action.get('timestamp'),
                        'viewing_duration': action.get('viewing_duration', 0),
                        'response': action.get('response', ''),
                        'screenshot': screenshot,
                        'categories': extract_categories(
                            response=action.get('response', ''),
                            screenshot=screenshot
                        )
                    }
                    all_steps.append(step_info)
        except json.JSONDecodeError as e:
            print(f"警告: 跳过无效的JSON文件 {session_file.name}: {e}")
            continue
        except Exception as e:
            print(f"警告: 处理文件 {session_file.name} 时出错: {e}")
            continue
    
    # 按时间戳排序
    sorted_steps = sorted(all_steps, key=lambda x: parse_timestamp(x['timestamp']) if x['timestamp'] else 0)
    
    # 找到第一个 reverse_persona=True 的 step
    reverse_persona_start_idx = None
    for i, step_info in enumerate(sorted_steps):
        if step_info['reverse_persona']:
            reverse_persona_start_idx = i
            break
    
    # 筛选切换点前后各 range_size 个 step
    if reverse_persona_start_idx is not None:
        # 统计切换点前后的 persona 类型分布
        total_before = reverse_persona_start_idx
        total_after = len(sorted_steps) - reverse_persona_start_idx
        
        normal_before = sum(1 for i in range(reverse_persona_start_idx) if not sorted_steps[i]['reverse_persona'])
        reversed_after = sum(1 for i in range(reverse_persona_start_idx, len(sorted_steps)) if sorted_steps[i]['reverse_persona'])
        
        print(f"\n找到 reverse_persona 切换点在第 {reverse_persona_start_idx} 个 step")
        print(f"切换点之前共 {total_before} 个 steps，其中 normal persona: {normal_before}")
        print(f"切换点之后共 {total_after} 个 steps，其中 reversed persona: {reversed_after}")
        
        # 从切换点往前收集 range_size 个 normal persona 的 steps
        normal_steps = []
        for i in range(reverse_persona_start_idx - 1, -1, -1):
            if not sorted_steps[i]['reverse_persona']:
                normal_steps.append(sorted_steps[i])
                if len(normal_steps) >= range_size:
                    break
        normal_steps.reverse()  # 恢复时间顺序
        
        # 从切换点往后收集 range_size 个 reversed persona 的 steps
        reversed_steps = []
        for i in range(reverse_persona_start_idx, len(sorted_steps)):
            if sorted_steps[i]['reverse_persona']:
                reversed_steps.append(sorted_steps[i])
                if len(reversed_steps) >= range_size:
                    break
        
        filtered_steps = normal_steps + reversed_steps
        print(f"\n实际收集到 {len(normal_steps)} 个 normal persona steps 和 {len(reversed_steps)} 个 reversed persona steps")
        print(f"共 {len(filtered_steps)} 个 steps")
        
        if len(reversed_steps) < range_size:
            print(f"警告: reversed persona steps 不足 {range_size} 个，可能数据采集未完成")
    else:
        filtered_steps = sorted_steps
        print(f"\n警告: 未找到 reverse_persona 切换点，将使用所有数据")
    
    # 重新组织成 sessions 格式
    sessions_dict = defaultdict(lambda: {
        'actions': [],
        'collector': '',
        'reverse_persona': False,
        'session_id': ''
    })
    
    for step_info in filtered_steps:
        session_id = step_info['session_id']
        if not sessions_dict[session_id]['session_id']:
            sessions_dict[session_id]['session_id'] = session_id
            sessions_dict[session_id]['collector'] = step_info['collector']
            sessions_dict[session_id]['reverse_persona'] = step_info['reverse_persona']
        
        action_info = {
            'step': step_info['step'],
            'timestamp': step_info['timestamp'],
            'viewing_duration': step_info['viewing_duration'],
            'response': step_info['response'],
            'categories': step_info['categories']
        }
        sessions_dict[session_id]['actions'].append(action_info)
    
    all_sessions = list(sessions_dict.values())
    return all_sessions


def calculate_coi(sessions: List[Dict], duration_threshold: float = None, category_level: int = 1) -> Dict:
    """
    计算多种COI指标 (Counterfactual Overlap Index)
    
    注意：所有计算都基于转换前后各400个step（在load_session_files中筛选）
    
    参数:
        sessions: session数据列表
        duration_threshold: 观看时长阈值
        category_level: 使用的类目级别，1表示一级类目，2表示二级类目（一级-二级）
    
    所有指标均基于指定级别的类目
    
    1. 概率COI: 基于类目概率分布，在交集类别上计算概率乘积之和
       公式: COI = sum(P_cultivation(cat) * P_counterfactual(cat) for cat in intersection)
    
    2. 归一化COI: 将概率COI归一化到交集类别的平均概率
       公式: normalized_COI = probability_COI / avg(P_cultivation(intersection), P_counterfactual(intersection))
    
    3. JS-COI: 基于类目分布的Jensen-Shannon Divergence
       公式: JS_COI = 1 - JS_divergence (using base=2)
       JS_divergence = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)
       使用base=2时，JS_divergence范围0-1
       JS-COI范围0-1，1表示分布完全相同（bubble强），0表示完全不同（成功逃离）
    
    使用这些COI指标衡量escape potential
    """
    # 按collector分组
    sessions_by_persona = defaultdict(list)
    for session in sessions:
        collector = session.get('collector', '')
        if collector:
            sessions_by_persona[collector].append(session)
    
    coi_results = []
    
    for persona, persona_sessions in sessions_by_persona.items():
        # 分离正常persona和reversed persona的session
        normal_sessions = [s for s in persona_sessions if not s.get('reverse_persona', False)]
        reversed_sessions = [s for s in persona_sessions if s.get('reverse_persona', False)]
        
        if not normal_sessions or not reversed_sessions:
            print(f"跳过persona {persona}: 缺少正常或reversed session")
            continue
        
        # 计算每个正常session的观看时长中位数作为阈值
        if duration_threshold is None:
            all_durations = []
            for session in normal_sessions:
                for action in session['actions']:
                    if action.get('viewing_duration', 0) > 0:
                        all_durations.append(action['viewing_duration'])
            if all_durations:
                duration_threshold = np.median(all_durations)
            else:
                duration_threshold = 15.0  # 默认阈值
        
        # 收集cultivation phase的类别频次（正常persona）
        cultivation_category_counts = Counter()
        for session in normal_sessions:
            for action in session['actions']:
                for cat_pair in action['categories']:
                    if category_level == 1:
                        category = cat_pair[0]  # 只取一级类目
                    else:  # category_level == 2
                        category = f"{cat_pair[0]}-{cat_pair[1]}"  # 一级-二级
                    cultivation_category_counts[category] += 1
        
        # 收集counterfactual phase的类别频次（reversed persona）
        counterfactual_category_counts = Counter()
        for session in reversed_sessions:
            for action in session['actions']:
                for cat_pair in action['categories']:
                    if category_level == 1:
                        category = cat_pair[0]  # 只取一级类目
                    else:  # category_level == 2
                        category = f"{cat_pair[0]}-{cat_pair[1]}"  # 一级-二级
                    counterfactual_category_counts[category] += 1
        
        # 计算总频次
        cultivation_total = sum(cultivation_category_counts.values())
        counterfactual_total = sum(counterfactual_category_counts.values())
        
        # 计算概率分布
        if cultivation_total > 0 and counterfactual_total > 0:
            cultivation_probs = {cat: count / cultivation_total 
                                for cat, count in cultivation_category_counts.items()}
            counterfactual_probs = {cat: count / counterfactual_total 
                                   for cat, count in counterfactual_category_counts.items()}
            
            # 计算概率COI：只在交集类别上计算概率乘积之和
            # 这样可以避免样本量差异导致的低估问题
            # 只考虑两个阶段都出现的类别，反映共同类别上的分布相似度
            intersection_categories = set(cultivation_probs.keys()) & set(counterfactual_probs.keys())
            
            if intersection_categories:
                # 方法1: 在交集类别上计算概率乘积之和
                probability_coi = sum(cultivation_probs[cat] * counterfactual_probs[cat] 
                                    for cat in intersection_categories)
                
                # 方法2: 归一化到交集类别的总概率（可选，用于对比）
                # 计算交集类别在两个阶段中的总概率
                cultivation_intersection_prob = sum(cultivation_probs[cat] for cat in intersection_categories)
                counterfactual_intersection_prob = sum(counterfactual_probs[cat] for cat in intersection_categories)
                
                # 归一化COI：除以交集类别的平均概率，使结果更稳定
                avg_intersection_prob = (cultivation_intersection_prob + counterfactual_intersection_prob) / 2
                if avg_intersection_prob > 0:
                    normalized_coi = probability_coi / avg_intersection_prob
                else:
                    normalized_coi = 0.0
            else:
                # 如果没有交集类别，COI为0
                probability_coi = 0.0
                normalized_coi = 0.0
                cultivation_intersection_prob = 0.0
                counterfactual_intersection_prob = 0.0
        else:
            # 如果总频次为0，初始化所有变量
            probability_coi = 0.0
            normalized_coi = 0.0
            cultivation_intersection_prob = 0.0
            counterfactual_intersection_prob = 0.0
            cultivation_probs = {}
            counterfactual_probs = {}
            cultivation_cat_counts = Counter()
            counterfactual_cat_counts = Counter()
            js_divergence = 0.0
            js_coi = 0.0
        
        # 保留集合信息用于输出
        cultivation_categories = set(cultivation_category_counts.keys())
        counterfactual_categories = set(counterfactual_category_counts.keys())
        intersection = cultivation_categories & counterfactual_categories
        union = cultivation_categories | counterfactual_categories
        
        # 类目频次（根据category_level参数，可能是一级或二级类目）
        cultivation_cat_counts = cultivation_category_counts
        counterfactual_cat_counts = counterfactual_category_counts
        
        # 获取所有类目
        all_categories = cultivation_categories | counterfactual_categories
        
        if all_categories and cultivation_total > 0 and counterfactual_total > 0:
            # 构建两个分布的概率向量（按相同的类别顺序）
            sorted_cats = sorted(all_categories)
            cultivation_cat_probs = np.array([
                cultivation_cat_counts.get(cat, 0) / cultivation_total 
                for cat in sorted_cats
            ])
            counterfactual_cat_probs = np.array([
                counterfactual_cat_counts.get(cat, 0) / counterfactual_total 
                for cat in sorted_cats
            ])
            
            # 计算JS Divergence（scipy的jensenshannon返回的是JS距离，即sqrt(JS divergence)）
            # 使用base=2，这样JS divergence的范围是0到1，更直观
            js_distance = jensenshannon(cultivation_cat_probs, counterfactual_cat_probs, base=2)
            js_divergence = js_distance ** 2  # JS divergence = (JS distance)^2
            
            # 转换为COI：值越小表示分布越相似（COI越高表示bubble越强）
            # 使用base=2时，JS divergence范围是0到1
            js_coi = 1 - js_divergence  # 1表示完全相同，0表示完全不同
        else:
            js_divergence = 0.0
            js_coi = 0.0
        
        # 使用概率COI作为主要指标（在交集类别上计算）
        coi = probability_coi
        
        # 统计top10类别（基于指定的类目级别）
        cultivation_top10 = cultivation_cat_counts.most_common(10)
        counterfactual_top10 = counterfactual_cat_counts.most_common(10)
        
        # 转换为包含比例的格式
        cultivation_top10_with_ratio = [
            (cat, count, count / cultivation_total if cultivation_total > 0 else 0) 
            for cat, count in cultivation_top10
        ]
        counterfactual_top10_with_ratio = [
            (cat, count, count / counterfactual_total if counterfactual_total > 0 else 0) 
            for cat, count in counterfactual_top10
        ]
        
        coi_results.append({
            'persona': persona,
            'coi': coi,  # 概率COI（在交集类别上计算）
            'normalized_coi': normalized_coi,  # 归一化COI
            'js_coi': js_coi,  # 基于JS Divergence的COI
            'js_divergence': js_divergence,  # JS Divergence原始值
            'cultivation_categories': len(cultivation_categories),
            'counterfactual_categories': len(counterfactual_categories),
            'cultivation_total': cultivation_total,
            'counterfactual_total': counterfactual_total,
            'intersection': len(intersection),
            'union': len(union),
            'cultivation_intersection_prob': cultivation_intersection_prob,
            'counterfactual_intersection_prob': counterfactual_intersection_prob,
            'duration_threshold': duration_threshold,
            'category_level': category_level,  # 记录使用的类目级别
            'cultivation_top10': cultivation_top10_with_ratio,
            'counterfactual_top10': counterfactual_top10_with_ratio
        })
    
    return {
        'coi_results': coi_results,
        'avg_coi': np.mean([r['coi'] for r in coi_results]) if coi_results else 0.0,
        'avg_js_coi': np.mean([r['js_coi'] for r in coi_results]) if coi_results else 0.0
    }


def print_results(coi_results: Dict):
    """打印分析结果"""
    print("\n" + "="*80)
    print("COI 分析结果（基于转换前后各400个step）")
    print("="*80)
    
    if coi_results and 'coi_results' in coi_results:
        # 获取类目级别（从第一个结果中）
        category_level = coi_results['coi_results'][0].get('category_level', 1) if coi_results['coi_results'] else 1
        category_level_desc = "一级类目" if category_level == 1 else "二级类目（一级-二级）"
        
        print(f"\n使用类目级别: {category_level_desc}")
        print(f"平均概率COI: {coi_results.get('avg_coi', 0):.4f}")
        print(f"平均JS-COI: {coi_results.get('avg_js_coi', 0):.4f}")
        print(f"\n注意: ")
        print(f"  - 概率COI: 基于{category_level_desc}，在交集类别上计算概率乘积之和")
        print(f"  - JS-COI: 基于{category_level_desc}分布的JS Divergence，1表示完全相同，0表示完全不同")
        print(f"  - 所有计算仅基于转换前后各400个step")
        print(f"\n各Persona的COI结果:")
        print(f"{'Persona':<20} {'概率COI':<12} {'JS-COI':<12} {'JS-Div':<12} {'交集':<10} {'Cult_Total':<12} {'Counter_Total':<12}")
        print("-" * 110)
        
        for item in coi_results['coi_results']:
            print(f"{item['persona']:<20} {item['coi']:<12.4f} {item.get('js_coi', 0):<12.4f} "
                  f"{item.get('js_divergence', 0):<12.4f} {item.get('intersection', 0):<10} "
                  f"{item.get('cultivation_total', 0):<12} {item.get('counterfactual_total', 0):<12}")
        
        # 打印每个persona的top10类别
        print("\n" + "="*80)
        print("各Persona的Top10类别统计")
        print("="*80)
        
        for item in coi_results['coi_results']:
            persona = item['persona']
            print(f"\n--- {persona} ---")
            
            print(f"\nCultivation阶段 Top10类别（共{item['cultivation_total']}个样本）：")
            print(f"{'排名':<6} {'类别':<30} {'数量':<10} {'占比':<10}")
            print("-" * 60)
            for i, (cat, count, ratio) in enumerate(item.get('cultivation_top10', []), 1):
                print(f"{i:<6} {cat:<30} {count:<10} {ratio*100:>6.2f}%")
            
            print(f"\nCounterfactual阶段 Top10类别（共{item['counterfactual_total']}个样本）：")
            print(f"{'排名':<6} {'类别':<30} {'数量':<10} {'占比':<10}")
            print("-" * 60)
            for i, (cat, count, ratio) in enumerate(item.get('counterfactual_top10', []), 1):
                print(f"{i:<6} {cat:<30} {count:<10} {ratio*100:>6.2f}%")


def save_results(coi_results: Dict, output_file: Path):
    """保存结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coi_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='统计COI (Counterfactual Overlap Index) 指标')
    parser.add_argument('--name', type=str, required=True, help='数据目录名称（如zsl）')
    parser.add_argument('--output', type=str, default=None, help='输出JSON文件路径（可选）')
    parser.add_argument('--duration-threshold', type=float, default=None, help='观看时长阈值（默认使用中位数）')
    parser.add_argument('--use-reclassification', action='store_true', 
                        help='使用重新分类的结果（从image_classifications.json读取）')
    parser.add_argument('--category-level', type=int, default=1, choices=[1, 2],
                        help='使用的类目级别：1=一级类目，2=二级类目（一级-二级）')
    
    args = parser.parse_args()
    
    # 确定数据目录路径
    data_dir = Path('deploy') / args.name
    if not data_dir.exists():
        data_dir = Path('deploy_log') / args.name
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: deploy/{args.name} 或 deploy_log/{args.name}")
    
    print(f"数据目录: {data_dir}")
    
    # 加载图片分类结果（如果使用重新分类）
    global IMAGE_CLASSIFICATIONS
    if args.use_reclassification:
        classification_file = data_dir / 'image_classifications.json'
        IMAGE_CLASSIFICATIONS = load_image_classifications(classification_file)
        if IMAGE_CLASSIFICATIONS:
            print(f"已加载 {len(IMAGE_CLASSIFICATIONS)} 个图片的重新分类结果")
        else:
            print("警告: 未找到重新分类结果，将使用原始分类")
    
    # ============================================================
    # 重要：加载数据时只保留 reverse_persona 切换点前后各 400 个 step
    # 这意味着COI的计算仅基于这800个steps（转换前400 + 转换后400）
    # ============================================================
    all_sessions = load_session_files(data_dir, range_size=400)
    print(f"共加载 {len(all_sessions)} 个sessions（筛选后，仅包含转换前后各400步）")
    
    # 统计正常和reversed persona的session数量
    normal_count = sum(1 for s in all_sessions if not s.get('reverse_persona', False))
    reversed_count = sum(1 for s in all_sessions if s.get('reverse_persona', False))
    
    # 统计 step 数量
    total_steps = sum(len(s['actions']) for s in all_sessions)
    normal_steps = sum(len(s['actions']) for s in all_sessions if not s.get('reverse_persona', False))
    reversed_steps = sum(len(s['actions']) for s in all_sessions if s.get('reverse_persona', False))
    
    print(f"正常 persona sessions: {normal_count}，steps: {normal_steps}")
    print(f"Reversed persona sessions: {reversed_count}，steps: {reversed_steps}")
    print(f"总 steps: {total_steps}")
    
    # 计算COI
    print(f"\n计算COI（使用{'一级类目' if args.category_level == 1 else '二级类目'}）...")
    coi_results = calculate_coi(
        all_sessions, 
        duration_threshold=args.duration_threshold,
        category_level=args.category_level
    )
    
    # 打印结果
    print_results(coi_results)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / 'coi_results.json'
    save_results(coi_results, output_path)


if __name__ == '__main__':
    main()

