#!/usr/bin/env python3
"""
使用LLM对视频截图进行分类
"""
import json
import argparse
import base64
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# 二级分类体系
CATEGORY_TAXONOMY = """
## 分类体系（二级分类）
一级类目：["其他", "生活", "亲子", "健康", "美食", "穿搭", "美容", "音乐", "舞蹈", "影视", "休闲", "娱乐", "旅游", "体育", "文化艺术", "知识", "教育", "游戏", "动漫", "数码科技", "直播", "交通", "新闻资讯"], 
二级类目: ["其他", "装修装饰", "绿植园艺", "日常生活", "生活收纳", "好物推荐", "生活窍门", "情感婚恋", "萌宠", "厨卫用品", "清洁保养", "孕产知识", "孕期饮食", "育儿经验", "萌宝", "母婴产品", "减肥瘦身", "个卫计生", "养生保健", "疾病知识", "医疗器械", "烹饪教程", "美食侦探", "日常饮食", "食材", "茶艺", "酒水饮料", "零食点心", "水果", "中老年穿搭", "儿童服饰", "少女穿搭", "熟女穿搭", "情侣装", "亲子装", "家居内衣", "男士休闲", "男士正装", "运动服饰", "潮牌", "泳装", "婚纱礼服", "汉服及周边", "珠宝配饰", "美妆", "护肤", "美发", "美甲", "美体", "医美", "MV", "音乐现场", "音乐演奏", "电音", "中国舞", "宅舞", "舞蹈教程", "街舞", "爵士舞", "现代舞", "芭蕾", "剧集", "电影", "综艺", "少儿", "纪录片", "短片", "预告资讯", "影视杂谈", "桌游", "解压整蛊", "玩偶", "手办模玩", "搞笑", "玩具", "鬼畜", "明星", "街拍", "时尚风潮", "网红", "T台", "乡村户外", "游轮岛屿", "城市探索", "野外生存", "旅途风光", "旅行攻略", "户外运动", "体育赛事", "健身", "水上运动", "球类运动", "瑜伽", "运动教学", "语言文字", "收藏", "访谈", "民俗", "星座命理", "手工艺", "歌舞戏剧", "绘画", "雕塑", "展览", "魔术杂技", "曲苑杂坛", "历史", "科普", "野生动物", "植物", "地理", "数理化", "天文", "社科人文", "信息技术", "演讲讲座", "职场提升", "语言留学", "在校学习", "办公软件", "摄影技巧", "艺术教育", "电竞赛事", "角色扮演", "动作冒险", "策略游戏", "格斗游戏", "射击游戏", "休闲益智游戏", "卡片游戏", "MOBA", "GMV", "音游", "游戏周边", "短片动画", "动漫", "动漫资讯", "二次元周边", "科技资讯", "影音智能", "手机平板", "影像器材", "电脑设备", "家用电器", "数码配件", "直播带货", "真香吃播", "游戏主播", "唱歌主播", "汽车资讯", "用车知识", "玩车", "出行工具", "工农设备", "船舶", "航空", "时政热点", "社会新闻", "天气", "财经资讯"]
"""


def encode_image_to_base64(image_path: Path) -> str:
    """将图片编码为base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def classify_image_with_llm(image_path: Path, llm_url: str = "http://localhost:8012/v1/chat/completions") -> Tuple[str, str]:
    """
    使用LLM对图片进行分类
    返回：(一级分类, 二级分类)
    """
    # 编码图片
    image_base64 = encode_image_to_base64(image_path)
    
    # 构建prompt
    prompt = f"""请仔细观察这张视频截图，根据以下分类体系进行分类：

{CATEGORY_TAXONOMY}

请你：
1. 判断视频的主题和内容类型
2. 从上述分类体系中选择最合适的一个二级分类
3. 只返回分类结果，格式为：一级分类-二级分类

例如：
- 娱乐-搞笑幽默
- 知识教育-科普知识
- 生活方式-美食烹饪

请直接回复分类结果，不要有其他解释。"""

    # 调用LLM API
    payload = {
        "model": "qwen",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(llm_url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        # 提取分类结果
        content = result['choices'][0]['message']['content'].strip()
        
        # 解析分类（格式：一级分类-二级分类）
        if '-' in content:
            parts = content.split('-', 1)
            level1 = parts[0].strip()
            level2 = parts[1].strip()
            return level1, level2
        else:
            # 如果格式不对，返回"其他-未分类"
            print(f"警告: 无法解析分类结果: {content}")
            return "其他", "未分类"
    
    except Exception as e:
        print(f"错误: 调用LLM失败: {e}")
        return "其他", "未分类"


def load_session_files(data_dir: Path) -> List[Dict]:
    """
    加载指定目录下所有session数据
    支持新格式：session_*/metadata.json
    """
    session_dirs = sorted([d for d in data_dir.glob('session_*') if d.is_dir()])
    
    if not session_dirs:
        raise FileNotFoundError(f"在 {data_dir} 下未找到session目录")
    
    print(f"找到 {len(session_dirs)} 个 session 目录")
    
    all_sessions = []
    for session_dir in session_dirs:
        metadata_file = session_dir / 'metadata.json'
        if metadata_file.exists():
            try:
                if metadata_file.stat().st_size == 0:
                    print(f"跳过空文件: {metadata_file}")
                    continue
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_sessions.append(data)
            except Exception as e:
                print(f"警告: 加载文件 {metadata_file} 失败: {e}")
        else:
            print(f"警告: {session_dir.name} 目录下未找到 metadata.json")
    
    return all_sessions


def classify_single_image(img_path: str, llm_url: str) -> Tuple[str, str, str]:
    """
    分类单张图片的工作函数
    返回：(图片路径, 一级分类, 二级分类)
    """
    try:
        level1, level2 = classify_image_with_llm(Path(img_path), llm_url)
        return img_path, level1, level2
    except Exception as e:
        print(f"\n错误: 分类图片 {img_path} 失败: {e}")
        return img_path, "其他", "未分类"


def classify_all_images(data_dir: Path, llm_url: str, output_file: Path, force_reclassify: bool = False):
    """
    对所有session的图片进行分类（使用10线程并发）
    """
    # 加载已有的分类结果（如果存在）
    existing_classifications = {}
    if output_file.exists() and not force_reclassify:
        print(f"加载已有分类结果: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_classifications = json.load(f)
        print(f"已有 {len(existing_classifications)} 个图片的分类结果")
    
    # 加载所有session
    all_sessions = load_session_files(data_dir)
    print(f"共加载 {len(all_sessions)} 个sessions")
    
    # 收集所有图片路径
    all_image_paths = []
    for session in all_sessions:
        for action in session.get('actions', []):
            screenshot = action.get('screenshot')
            if screenshot:
                # 转换路径分隔符（Windows -> Unix）
                screenshot = screenshot.replace('\\', '/')
                image_path = Path(screenshot)
                if image_path.exists():
                    all_image_paths.append(str(image_path))
                else:
                    print(f"警告: 图片不存在: {screenshot}")
    
    print(f"共找到 {len(all_image_paths)} 张图片")
    
    # 过滤出需要分类的图片（跳过已分类的）
    images_to_classify = []
    if not force_reclassify:
        for img_path in all_image_paths:
            if img_path not in existing_classifications:
                images_to_classify.append(img_path)
        print(f"需要分类 {len(images_to_classify)} 张图片（跳过 {len(all_image_paths) - len(images_to_classify)} 张已分类）")
    else:
        images_to_classify = all_image_paths
        print(f"强制重新分类所有 {len(images_to_classify)} 张图片")
    
    # 如果没有需要分类的图片，直接返回
    if not images_to_classify:
        print("没有需要分类的图片")
        return
    
    # 分类结果和线程锁
    classifications = existing_classifications.copy()
    lock = threading.Lock()
    completed_count = 0
    
    # 使用线程池进行并发分类
    print(f"\n开始分类（使用10线程并发）...")
    
    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 提交所有任务
            future_to_img = {
                executor.submit(classify_single_image, img_path, llm_url): img_path 
                for img_path in images_to_classify
            }
            
            # 使用tqdm显示进度
            with tqdm(total=len(images_to_classify), desc="分类进度") as pbar:
                # 处理完成的任务
                for future in as_completed(future_to_img):
                    try:
                        img_path, level1, level2 = future.result()
                        
                        # 线程安全地更新结果
                        with lock:
                            classifications[img_path] = f"{level1}-{level2}"
                            completed_count += 1
                            
                            # 每10张图片保存一次（增量保存）
                            if completed_count % 10 == 0:
                                with open(output_file, 'w', encoding='utf-8') as f:
                                    json.dump(classifications, f, ensure_ascii=False, indent=2)
                        
                        # 更新进度条
                        pbar.update(1)
                        
                    except Exception as e:
                        img_path = future_to_img[future]
                        print(f"\n错误: 处理图片 {img_path} 失败: {e}")
                        with lock:
                            classifications[img_path] = "其他-未分类"
                        pbar.update(1)
    
    except KeyboardInterrupt:
        print("\n用户中断，保存当前结果...")
    
    # 保存最终结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(classifications, f, ensure_ascii=False, indent=2)
    
    print(f"\n分类完成！结果已保存到: {output_file}")
    print(f"共分类 {len(classifications)} 张图片")
    
    # 统计分类结果
    category_counts = {}
    for category in classifications.values():
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print(f"\n分类统计（前20个）:")
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (category, count) in enumerate(sorted_categories[:20], 1):
        print(f"  {i}. {category}: {count}")


def main():
    parser = argparse.ArgumentParser(description='使用LLM对视频截图进行分类')
    parser.add_argument('--name', type=str, required=True, help='数据目录名称（如zsl_div）')
    parser.add_argument('--output', type=str, default=None, help='输出JSON文件路径（可选）')
    parser.add_argument('--llm-url', type=str, default='http://localhost:8012/v1/chat/completions',
                        help='LLM API URL')
    parser.add_argument('--force', action='store_true', help='强制重新分类所有图片（默认跳过已分类）')
    
    args = parser.parse_args()
    
    # 确定数据目录路径
    data_dir = Path('deploy') / args.name
    if not data_dir.exists():
        data_dir = Path('deploy_log') / args.name
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: deploy/{args.name} 或 deploy_log/{args.name}")
    
    print(f"数据目录: {data_dir}")
    print(f"LLM URL: {args.llm_url}")
    
    # 确定输出文件路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / 'image_classifications.json'
    
    # 开始分类
    classify_all_images(data_dir, args.llm_url, output_path, force_reclassify=args.force)


if __name__ == '__main__':
    main()

