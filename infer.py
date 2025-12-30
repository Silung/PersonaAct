import os
import json
import re
import argparse
import numpy as np
from datetime import datetime
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


def append_result_to_markdown(markdown_file: str, start_time: datetime, end_time: datetime,
                              model: str, api_base: str, data_dir: str, test_file: str,
                              persona: bool, think: bool, results: list):
    """
    将实验结果追加到 markdown 文件，每个数据集单独一行
    
    Args:
        markdown_file: markdown 文件路径
        start_time: 开始时间
        end_time: 结束时间
        model: 模型名称
        api_base: API base URL
        data_dir: 数据目录
        test_file: 测试文件名
        persona: 是否使用 Persona 模式
        think: 是否使用 Think 模式
        results: 结果列表，每个元素包含一个数据集的评估结果
    """
    # 确保 result 目录存在
    markdown_dir = os.path.dirname(markdown_file)
    if markdown_dir:
        os.makedirs(markdown_dir, exist_ok=True)
    else:
        os.makedirs('result', exist_ok=True)
    
    # 计算运行时长（秒）
    duration = (end_time - start_time).total_seconds()
    
    # 格式化时间
    time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    duration_str = f"{duration:.1f}"
    
    # 准备表格行数据
    persona_str = "是" if persona else "否"
    think_str = "是" if think else "否"
    
    # 转义 markdown 表格中的特殊字符（|）
    def escape_md(text):
        return str(text).replace('|', '\\|')
    
    # 检查文件是否存在，如果不存在则创建表头
    file_exists = os.path.exists(markdown_file)
    
    with open(markdown_file, 'a', encoding='utf-8') as f:
        if not file_exists:
            # 创建表头
            header = "| 运行时间 | 运行时长(秒) | 模型 | API Base | 数据目录 | 测试文件 | Persona | Think | 数据集 | 样本数 | Type ACC | Pearson | SMAPE | MAE | KL Div |\n"
            separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            f.write(header)
            f.write(separator)
        
        # 为每个数据集追加一行
        if results:
            for result in results:
                dataset_name = result.get('dataset', 'unknown')
                samples = result.get('samples', 0)
                type_acc = result.get('type_acc', 0.0)
                pearson = result.get('pearson', 0.0)
                smape = result.get('smape', 0.0)
                mae = result.get('mae', 0.0)
                kl_div = result.get('kl_div', 0.0)
                
                row = f"| {escape_md(time_str)} | {escape_md(duration_str)} | {escape_md(model)} | {escape_md(api_base)} | {escape_md(data_dir)} | {escape_md(test_file)} | {escape_md(persona_str)} | {escape_md(think_str)} | {escape_md(dataset_name)} | {samples} | {type_acc:.4f} | {pearson:.4f} | {smape:.4f} | {mae:.4f} | {kl_div:.4f} |\n"
                f.write(row)


# 从 prompts.py 导入统一的 prompt 函数
from prompts import get_system_prompt, get_user_prompt


def convert_image_tags_to_urls(content: str, images: list) -> list:
    """
    将包含<image>标签的文本转换为多模态content格式
    
    Args:
        content: 包含<image>标签的文本
        images: 图片路径列表
    
    Returns:
        list: 多模态content（图片+文本），如果没有<image>标签则返回原字符串
    """
    if not images or '<image>' not in content:
        # 如果没有图片或没有<image>标签，返回纯文本格式
        return content
    
    # 分割文本，按<image>标签切分
    parts = content.split('<image>')
    
    # 构建多模态content
    result = []
    image_idx = 0
    
    for i, part in enumerate(parts):
        # 如果不是第一部分，说明前面有<image>标签，先添加图片
        if i > 0 and image_idx < len(images):
            img_path = images[image_idx]
            # 转换为绝对路径
            if not os.path.isabs(img_path):
                img_path = os.path.abspath(img_path)
            
            result.append({
                "type": "image_url",
                "image_url": {"url": f"file://{img_path}"}
            })
            image_idx += 1
        
        # 添加文本部分（如果非空）
        if part.strip():
            result.append({
                "type": "text",
                "text": part
            })
    
    return result if result else content


def evaluate_dataset(dataset_name, test_file, data_dir="data", output_dir=None, is_persona=False, persona_file=None, 
                     api_base="http://127.0.0.1:8012/v1", api_key="1234567890", model="qwen"):
    """评估单个数据集"""
    test_file_path = os.path.join(data_dir, dataset_name, test_file)
    
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
    raw_outputs = []  # 保存原始输出
    ground_truths = []  # 保存ground truth原始文本
    input_messages = []  # 保存推理时的输入messages
    input_images = []  # 保存输入图片路径
    
    print(f"共{len(all_data)}条样本，开始推理...")
    
    for item in tqdm(all_data, desc=f"推理 {dataset_name}"):
        messages = item['messages'].copy()  # 复制messages，避免修改原始数据
        
        # 保存图片路径
        images = item.get('images', [])
        input_images.append(images)
        
        # 如果最后一条是assistant的消息，移除它（这是ground truth）
        if messages[-1]['role'] == 'assistant':
            messages = messages[:-1]
        
        # 转换所有消息中的<image>标签为实际的图片URL
        for msg in messages:
            if msg['role'] == 'user' and isinstance(msg.get('content'), str):
                # 将<image>标签转换为图片URL格式
                msg['content'] = convert_image_tags_to_urls(msg['content'], images)
        
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
        ground_truths.append(item['solution'])  # 保存ground truth原始文本
        
        # 保存输入messages（去掉图片路径，只保留文本内容用于调试）
        messages_for_save = []
        for msg in messages:
            msg_copy = {"role": msg["role"]}
            # 如果content是字符串，直接保存；如果是列表（多模态），只保存文本部分的前500字符
            if isinstance(msg.get("content"), str):
                # 保存前1000字符，避免太长
                msg_copy["content"] = msg["content"][:1000] if len(msg["content"]) > 1000 else msg["content"]
            else:
                msg_copy["content"] = str(msg.get("content"))[:1000]
            messages_for_save.append(msg_copy)
        input_messages.append(messages_for_save)
        
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
        
        raw_outputs.append(pred_out)  # 保存原始输出
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
            'pred_watch': y_pred[i],
            'ground_truth': ground_truths[i],  # 原始ground truth
            'model_output': raw_outputs[i],     # 模型原始输出
            'input_messages': input_messages[i],  # 推理时的输入messages
            'input_images': input_images[i]  # 输入图片路径列表
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
parser.add_argument('--data_dir', type=str, default='data',
                    help='Data directory (default: data, can be data, data3, etc.)')
parser.add_argument('--name', type=str, default=None, 
                    help='User name (yqg, zsl, etc.). Only test {data_dir}/{name}/')
parser.add_argument('--test_file', type=str, default=None,
                    help='Test file name (auto-determined if not specified)')
parser.add_argument('--persona', action='store_true',
                    help='Test persona version of the dataset (auto-use persona_video_action_test_item.jsonl)')
parser.add_argument('--think', action='store_true',
                    help='Test thinking version of the dataset (auto-use thinking_*_test_item.jsonl)')
parser.add_argument('--api_base', type=str, default='http://127.0.0.1:8012/v1',
                    help='API base URL (default: http://127.0.0.1:8012/v1)')
parser.add_argument('--api_key', type=str, default='1234567890',
                    help='API key (default: 1234567890)')
parser.add_argument('--model', type=str, default='qwen',
                    help='Model name (default: qwen)')
args = parser.parse_args()

# 自动确定测试文件名
if args.test_file is None:
    # 如果没有指定test_file，根据persona和think选项自动确定
    prefix = ''
    if args.think:
        prefix += 'thinking_'
    if args.persona:
        prefix += 'persona_'
    args.test_file = f'{prefix}video_action_test_item.jsonl'
else:
    # 如果指定了test_file，根据选项自动添加前缀
    if args.think and not args.test_file.startswith('thinking_'):
        args.test_file = 'thinking_' + args.test_file
    if args.persona and not args.test_file.startswith('persona_'):
        # 如果已经有thinking_前缀，在其后添加persona_
        if args.test_file.startswith('thinking_'):
            args.test_file = 'thinking_persona_' + args.test_file[9:]  # 去掉thinking_后添加thinking_persona_
        else:
            args.test_file = 'persona_' + args.test_file

# 确定要测试的数据集
if args.name:
    # 指定了name，只测试该用户
    test_path = os.path.join(args.data_dir, args.name, args.test_file)
    if not os.path.exists(test_path):
        print(f"错误：测试文件不存在 {test_path}")
        exit(1)
    datasets_to_test = [args.name]
else:
    # 查找所有可用的数据集
    datasets_to_test = []
    if os.path.exists(args.data_dir):
        for name in os.listdir(args.data_dir):
            dataset_path = os.path.join(args.data_dir, name)
            if os.path.isdir(dataset_path):
                test_path = os.path.join(dataset_path, args.test_file)
                if os.path.exists(test_path):
                    datasets_to_test.append(name)
    datasets_to_test = sorted(datasets_to_test)

if not datasets_to_test:
    print("未找到任何可用的测试数据集")
    exit(1)

# 记录开始时间
start_time = datetime.now()

print(f"数据目录: {args.data_dir}")
print(f"将测试数据集: {', '.join(datasets_to_test)}")
print(f"测试文件: {args.test_file}")
if args.think:
    print("使用Thinking模式：将使用thinking_开头的数据文件")
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
        persona_file = os.path.join(args.data_dir, dataset, 'persona.json')
        if not os.path.exists(persona_file):
            print(f"警告: persona文件不存在 {persona_file}，将不使用persona信息")
            persona_file = None
    
    result = evaluate_dataset(
        dataset, 
        args.test_file, 
        data_dir=args.data_dir,
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

# 记录结束时间
end_time = datetime.now()

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
    
    # 追加结果到 markdown 文件
    markdown_file = os.path.join('result', 'infer_results_history.md')
    append_result_to_markdown(
        markdown_file=markdown_file,
        start_time=start_time,
        end_time=end_time,
        model=args.model,
        api_base=args.api_base,
        data_dir=args.data_dir,
        test_file=args.test_file,
        persona=args.persona,
        think=args.think,
        results=results
    )
    print(f"结果已追加到: {markdown_file}")
