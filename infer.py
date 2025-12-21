import os
import json
import re
import argparse
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from scipy.stats import pearsonr, entropy

# SMAPE 实现
def smape(y_true, y_pred, eps=1e-8):
    """Symmetric Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return np.mean(np.abs(y_pred - y_true) / denominator) * 2

# MAE 实现
def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_pred - y_true))

# KL 散度计算（用于观看时长分布比较）
def compute_kl_divergence(y_true, y_pred, bins=20):
    """
    计算预测分布和真实分布之间的 KL 散度
    KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # 确定分箱范围（使用真实值和预测值的最大值）
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    
    # 创建分箱
    bin_edges = np.linspace(min_val, max_val + 1e-8, bins + 1)
    
    # 计算直方图（归一化为概率分布）
    hist_true, _ = np.histogram(y_true, bins=bin_edges)
    hist_pred, _ = np.histogram(y_pred, bins=bin_edges)
    
    # 归一化为概率分布，添加平滑避免 0 概率
    eps = 1e-10
    p = (hist_true + eps) / (hist_true.sum() + eps * bins)
    q = (hist_pred + eps) / (hist_pred.sum() + eps * bins)
    
    # 计算 KL 散度
    kl_div = entropy(p, q)
    
    return kl_div

# 解析 code block 里的watch(xx)，like/comment/share
def parse_pred_action(pred_str):
    '''返回 (主action, 时长/内容) 目前只取第一个匹配, 不区分多个动作'''
    # 匹配 watch(xx)
    match = re.search(r"watch\(([^)]*)\)", pred_str)
    if match:
        try:
            sec = float(match.group(1))
        except:
            sec = None
        return 'watch', sec
    if re.search(r"like\(\)", pred_str):
        return 'like', None
    if re.search(r"comment\\s*\(", pred_str):
        return 'comment', None
    if re.search(r"share\\s*\(", pred_str):
        return 'share', None
    return 'none', None

def parse_solution_action(solution_str):
    # solution都用code block，摘第一个命令
    return parse_pred_action(solution_str)

def load_persona(persona_file: str) -> dict:
    """加载persona.json文件"""
    if persona_file and os.path.exists(persona_file):
        with open(persona_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# 从 prompts.py 导入统一的 prompt 函数
from prompts import get_system_prompt, get_user_prompt


def evaluate_dataset(dataset_name, test_file, output_dir=None, is_persona=False, persona_file=None, 
                     api_base="http://127.0.0.1:8012/v1", api_key="1234567890", model="qwen"):
    """评估单个数据集"""
    test_file_path = f'data/{dataset_name}/{test_file}'
    
    if not os.path.exists(test_file_path):
        print(f"跳过 {dataset_name}: 测试文件不存在 {test_file_path}")
        return None
    
    # 确定输出目录
    if output_dir is None:
        output_dir = f'result/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化 OpenAI client
    client = OpenAI(base_url=api_base, api_key=api_key)
    
    # 加载persona信息（如果指定了persona_file）
    persona = {}
    if persona_file:
        persona = load_persona(persona_file)
        if persona:
            print(f"已加载persona信息: {persona_file}")
    
    print(f"\n开始评估数据集: {dataset_name}")
    print(f"测试文件: {test_file_path}")
    
    # 读取测试集
    with open(test_file_path, encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    
    y_true = []
    y_pred = []
    type_true = []
    type_pred = []
    
    print(f"共{len(all_data)}条样本，开始推理...")
    
    for item in tqdm(all_data, desc=f"推理 {dataset_name}"):
        messages = item['messages'].copy()  # 复制messages，避免修改原始数据
        
        # 如果最后一条是assistant的消息，移除它（这是ground truth）
        if messages[-1]['role'] == 'assistant':
            messages = messages[:-1]
        
        # 如果提供了persona，使用统一的 system prompt 函数
        if persona:
            # 使用 prompts.py 中的统一函数生成 system prompt
            system_content = get_system_prompt(use_persona=True, persona=persona)
            if messages and messages[0]['role'] == 'system':
                # 如果已有system message，替换为统一的格式
                messages[0]['content'] = system_content
            else:
                # 如果没有system message，创建一个新的
                messages.insert(0, {
                    "role": "system",
                    "content": system_content
                })
        
        gt_type, gt_value = parse_solution_action(item['solution'])
        type_true.append(gt_type)
        y_true.append(gt_value if gt_value is not None else 0.0)
        
        # 调用vllm
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=256,
                temperature=0.0
            )
            pred_out = completion.choices[0].message.content
        except Exception as e:
            pred_out = ""
            print(f"推理异常：{e}")
        pred_type, pred_value = parse_pred_action(pred_out)
        type_pred.append(pred_type)
        # 只eval观看时长（其他action统一算0）
        if pred_type == 'watch' and pred_value is not None:
            y_pred.append(float(pred_value))
        else:
            y_pred.append(0.0)
    
    # 评估 type acc
    c_type_acc = np.mean(np.array(type_true) == np.array(type_pred))
    # 相关系数
    pear_corr = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0
    c_smape = smape(y_true, y_pred)
    c_mae = mae(y_true, y_pred)
    
    # 计算观看时长分布的 KL 散度
    # 只对 watch 类型的样本计算 KL 散度
    watch_indices = [i for i, t in enumerate(type_true) if t == 'watch']
    if len(watch_indices) > 0:
        y_true_watch = np.array([y_true[i] for i in watch_indices])
        y_pred_watch = np.array([y_pred[i] for i in watch_indices])
        c_kl_div = compute_kl_divergence(y_true_watch, y_pred_watch)
    else:
        c_kl_div = 0.0
    
    print(f"\n===== 评估结果 ({dataset_name}) =====")
    print(f"Type ACC ：{c_type_acc:.4f}")
    print(f"Pearson  ：{pear_corr:.4f}")
    print(f"SMAPE    ：{c_smape:.4f}")
    print(f"MAE      ：{c_mae:.4f}")
    print(f"KL Div   ：{c_kl_div:.4f}  (观看时长分布)")
    
    # 明细可选输出
    prefix = 'persona_' if is_persona else ''
    output_detail_file = os.path.join(output_dir, f'{prefix}output_infer_eval_detail_{dataset_name}.json')
    data_out = []
    for i in range(len(y_true)):
        data_out.append({
            'gt_type': type_true[i],
            'pred_type': type_pred[i],
            'gt_watch': y_true[i],
            'pred_watch': y_pred[i]
        })
    with open(output_detail_file, 'w', encoding='utf8') as f:
        json.dump(data_out, f, ensure_ascii=False, indent=2)
    
    print(f"详细结果已保存到: {output_detail_file}")
    
    return {
        'dataset': dataset_name,
        'samples': len(all_data),
        'type_acc': c_type_acc,
        'pearson': pear_corr,
        'smape': c_smape,
        'mae': c_mae,
        'kl_div': c_kl_div
    }

# 解析命令行参数
parser = argparse.ArgumentParser(description='Inference on test dataset')
parser.add_argument('--name', type=str, default=None, 
                    help='User name (yqg, zsl, etc.). Only test data/{name}/')
parser.add_argument('--test_file', type=str, default='video_action_test_item.jsonl',
                    help='Test file name')
parser.add_argument('--persona', action='store_true',
                    help='Test persona version of the dataset (persona_video_action_test_item.jsonl)')
parser.add_argument('--api_base', type=str, default='http://127.0.0.1:8012/v1',
                    help='API base URL (default: http://127.0.0.1:8012/v1)')
parser.add_argument('--api_key', type=str, default='1234567890',
                    help='API key (default: 1234567890)')
parser.add_argument('--model', type=str, default='qwen',
                    help='Model name (default: qwen)')
args = parser.parse_args()

# 如果指定了--persona，修改test_file
if args.persona and not args.test_file.startswith('persona_'):
    args.test_file = 'persona_' + args.test_file

# 确定要测试的数据集
if args.name:
    # 指定了name，只测试该用户
    test_path = os.path.join('data', args.name, args.test_file)
    if not os.path.exists(test_path):
        print(f"错误：测试文件不存在 {test_path}")
        exit(1)
    datasets_to_test = [args.name]
else:
    # 查找所有可用的数据集
    datasets_to_test = []
    if os.path.exists('data'):
        for name in os.listdir('data'):
            dataset_path = os.path.join('data', name)
            if os.path.isdir(dataset_path):
                test_path = os.path.join(dataset_path, args.test_file)
                if os.path.exists(test_path):
                    datasets_to_test.append(name)
    datasets_to_test = sorted(datasets_to_test)

if not datasets_to_test:
    print("未找到任何可用的测试数据集")
    exit(1)

print(f"将测试数据集: {', '.join(datasets_to_test)}")
print(f"测试文件: {args.test_file}")
if args.persona:
    print("使用Persona模式：将加载persona.json文件")

# 评估所有数据集
results = []
is_persona = args.test_file.startswith('persona_')
file_type = "Persona" if is_persona else "Normal"

for dataset in datasets_to_test:
    output_dir = f'result/{dataset}'
    # 如果使用persona模式，加载对应的persona.json文件
    persona_file = None
    if args.persona:
        persona_file = os.path.join('data', dataset, 'persona.json')
        if not os.path.exists(persona_file):
            print(f"警告: persona文件不存在 {persona_file}，将不使用persona信息")
            persona_file = None
    
    result = evaluate_dataset(
        dataset, 
        args.test_file, 
        output_dir=output_dir, 
        is_persona=is_persona, 
        persona_file=persona_file,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model
    )
    if result:
        result['type'] = file_type
        result['test_file'] = args.test_file
        results.append(result)

# 汇总结果
if len(results) > 0:
    print("\n" + "="*80)
    print(f"汇总结果 ({file_type})")
    print("="*80)
    print(f"{'Dataset':<10} {'Samples':<8} {'Type ACC':<10} {'Pearson':<10} {'SMAPE':<10} {'MAE':<10} {'KL Div':<10}")
    print("-" * 80)
    for result in results:
        print(f"{result['dataset']:<10} {result['samples']:<8} {result['type_acc']:<10.4f} "
              f"{result['pearson']:<10.4f} {result['smape']:<10.4f} {result['mae']:<10.4f} "
              f"{result['kl_div']:<10.4f}")
    
    # 计算加权平均值
    total_samples = sum(r['samples'] for r in results)
    if total_samples > 0:
        weighted_type_acc = sum(r['type_acc'] * r['samples'] for r in results) / total_samples
        weighted_pearson = sum(r['pearson'] * r['samples'] for r in results) / total_samples
        weighted_smape = sum(r['smape'] * r['samples'] for r in results) / total_samples
        weighted_mae = sum(r['mae'] * r['samples'] for r in results) / total_samples
        weighted_kl_div = sum(r['kl_div'] * r['samples'] for r in results) / total_samples
        
        print("-" * 80)
        print(f"{'Average':<10} {total_samples:<8} {weighted_type_acc:<10.4f} "
              f"{weighted_pearson:<10.4f} {weighted_smape:<10.4f} {weighted_mae:<10.4f} "
              f"{weighted_kl_div:<10.4f}")

# 保存汇总结果
if results:
    # 确定汇总文件的保存目录
    if args.name:
        # 如果指定了name，保存到对应目录
        summary_dir = f'result/{args.name}'
        os.makedirs(summary_dir, exist_ok=True)
    else:
        # 否则保存到result根目录
        summary_dir = 'result'
        os.makedirs(summary_dir, exist_ok=True)
    
    prefix = 'persona_' if is_persona else ''
    summary_file = os.path.join(summary_dir, f'{prefix}output_infer_eval_summary.json')
    with open(summary_file, 'w', encoding='utf8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n汇总结果已保存到: {summary_file}")
