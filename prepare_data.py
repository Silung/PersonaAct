import os
import json
import random
import argparse
import cv2
from pathlib import Path
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Tuple
from prompts import get_system_prompt, get_user_prompt
import librosa
import soundfile as sf
import tempfile


client = OpenAI(
    base_url="http://127.0.0.1:8012/v1",
    api_key="1234567890"
)

def clean_json_response(response: str) -> str:
    """清理模型返回的JSON响应，去除markdown标记"""
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    return response.strip()

def get_category(image_paths: List[str], max_images: int = 10) -> dict:
    """
    根据图片路径分类（支持多标签分类，支持多张图片）
    
    Args:
        image_paths: 图片路径列表（可以是单个字符串或列表）
        max_images: 最多使用多少张图片（默认10）
    
    Returns:
        dict: {"categories": [{"main": "主类", "sub": "子类"}, ...]}
    """
    # 支持单个路径或路径列表
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    # 限制图片数量
    image_paths = image_paths[:max_images]
    
    sys_prompt = (
        "你需要对当前的短视频进行分类。请从以下【主类-子类】结构中选择最合适的分类（可以选择多个）：\n\n"
        "{\n"
        '  "娱乐": ["搞笑段子", "剧情短片", "才艺展示", "挑战实验", "综艺片段", "明星动态", "网红日常"],\n'
        '  "知识教育": ["科普知识", "技能教学", "外语学习", "学习方法", "法律科普", "职业技能", "编程技术"],\n'
        '  "生活记录": ["日常vlog", "家庭生活", "生活技巧", "情感故事", "社区互助", "公益记录", "怀旧故事"],\n'
        '  "情感与心理": ["情感咨询", "恋爱关系", "婚姻家庭", "心理健康", "励志治愈", "情绪调节"],\n'
        '  "美食": ["家常菜", "烘焙甜点", "特色小吃", "美食探店", "异国料理", "饮品调制", "美食测评"],\n'
        '  "时尚美妆": ["穿搭分享", "美妆护肤", "发型造型", "配饰分享", "时尚资讯", "品牌测评", "美甲美睫"],\n'
        '  "运动健身": ["健身训练", "球类运动", "户外探险", "瑜伽普拉提", "运动技巧", "极限运动", "康复训练"],\n'
        '  "科技数码": ["手机测评", "电脑硬件", "智能家居", "软件技巧", "科技前沿", "数码配件", "AI应用"],\n'
        '  "汽车": ["汽车评测", "驾驶技巧", "保养维护", "汽车文化", "新能源汽车", "二手车交易", "汽车改装"],\n'
        '  "游戏": ["手游攻略", "端游实况", "电竞赛事", "游戏测评", "游戏教学", "游戏新闻", "虚拟世界/沙盒创作"],\n'
        '  "音乐舞蹈": ["歌曲翻唱", "舞蹈表演", "乐器演奏", "声乐教学", "舞蹈教学", "音乐创作", "舞蹈编排"],\n'
        '  "影视动漫": ["电影解说", "剧情剪辑", "影视评论", "经典影片", "动漫解说", "动画短片", "配音表演"],\n'
        '  "旅行": ["国内旅行", "国外旅行", "旅行攻略", "露营体验", "景区探秘", "背包客路线", "小众景点"],\n'
        '  "摄影与创作": ["摄影技巧", "器材测评", "后期制作", "人像摄影", "风光摄影", "手机摄影", "摄影作品展示"],\n'
        '  "财经商业": ["商业分析", "创业经验", "理财知识", "投资策略", "副业思路", "营销策略", "经济观察"],\n'
        '  "房产家居": ["住宅装修", "家居收纳", "家居好物", "房产知识", "租房买房", "园艺绿植", "智能家居"],\n'
        '  "医疗健康": ["养生保健", "心理健康科普", "疾病预防", "营养饮食", "中医养生", "运动康复", "医药科普"],\n'
        '  "三农乡村": ["农村生活", "农业种植", "乡村美食", "传统手艺", "乡村振兴", "农产品展示", "田园风光"],\n'
        '  "宠物": ["狗狗日常", "猫咪日常", "宠物训练", "宠物医疗", "萌宠搞笑", "宠物用品", "动物救助"],\n'
        '  "亲子育儿": ["育儿经验", "儿童教育", "萌娃日常", "亲子游戏", "孕期知识", "早教启蒙", "亲子旅行"],\n'
        '  "二次元": ["动漫解说", "cosplay", "宅舞", "同人创作", "声优配音", "虚拟偶像", "动漫周边"],\n'
        '  "文化历史": ["历史科普", "民俗文化", "文物考古", "非遗传承", "文学知识", "历史人物故事", "传统文化"],\n'
        '  "军事法律": ["军事知识", "军事装备", "国防教育", "法律常识", "法律案例", "政策解读", "国际局势"],\n'
        '  "社会资讯": ["社会热点", "民生事件", "公益新闻", "社会观察", "热点评论", "现场记录", "公共安全"],\n'
        '  "广告推广": ["商业广告", "产品推广", "品牌宣传", "直播带货", "开箱测评", "促销活动", "软广植入"]\n'
        "}\n\n"
        "分类规则：\n"
        "1. 可以输出1-3个最相关的分类，如果符合多个需要输出多个类别\n"
        "2. 如果视频内容不符合以上任何分类，可以自由描述\n"
        "3. 输出格式必须是纯 JSON，不包含任何额外文字或符号\n"
        "输出格式：\n"
        "{\n"
        '  "categories": [\n'
        '    {"main": "主类名称", "sub": "子类名称"},\n'
        '    {"main": "主类名称2", "sub": "子类名称2"}  // 可选\n'
        '  ]\n'
        "}"
    )
    
    # 构建多张图片的content
    content = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"file://{os.path.abspath(img_path)}"}
            })
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": content}
    ]

    response = client.chat.completions.create(
        model="qwen",
        messages=messages,
        temperature=0.1
    )
    
    text = clean_json_response(response.choices[0].message.content)
    category = json.loads(text)
    if "categories" not in category:
        category["categories"] = []
    return category

def parse_video_stats(image_path: str) -> dict:
    """
    调用 LLM 视觉模型识别图片中的点赞/评论/收藏/转发数量、作者和标题
    返回 dict: {"like": int, "comment": int, "favorite": int, "share": int, "author": str, "title": str}
    """
    sys_prompt = """请你识别图片中的短视频点赞、评论、收藏、转发数量、作者名称和视频标题。请仅输出JSON格式，示例：
{"like": 100, "comment": 22, "favorite": 8, "share": 3, "author": "用户名", "title": "视频标题"}。
"like"为点赞数，"comment"为评论数，"favorite"为收藏数，"share"为转发数，"author"为作者名称，"title"为视频标题。
发现为空或识别不出来时请输出0或空字符串。如果有单位需要换算，如"14.3万"应输出为143000。请严格保证输出格式为纯 JSON，不要多余内容。"""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"file://{os.path.abspath(image_path)}"}}
        ]}
    ]
    response = client.chat.completions.create(
        model='qwen',
        messages=messages,
        temperature=0.0
    )
    
    text = clean_json_response(response.choices[0].message.content)
    stats = json.loads(text)
    # 确保所有字段存在且为正确类型
    result = {"like": 0, "comment": 0, "favorite": 0, "share": 0, "author": "", "title": ""}
    for key in ["like", "comment", "favorite", "share"]:
        if key in stats:
            val = stats[key]
            result[key] = int(val) if isinstance(val, (int, str)) and str(val).isdigit() else 0
    for key in ["author", "title"]:
        if key in stats:
            result[key] = str(stats[key]) if stats[key] else ""
    return result
def load_session_files(raw_data_dir: str, collector_name: str = None) -> List[Tuple[str, Dict]]:
    """
    加载所有session文件（可选仅加载指定标注者）
    
    Args:
        raw_data_dir: raw_data目录路径
        collector_name: 标注者名字（如只加载某一人，默认None为所有）
    Returns:
        List of (collector_name, session_data) tuples
    """
    session_data = []
    raw_data_path = Path(raw_data_dir)

    # 如果指定collector_name, 只遍历其文件夹
    if collector_name:
        collector_dirs = [raw_data_path / collector_name]
    else:
        # 遍历raw_data下的所有子目录（标注者目录）
        collector_dirs = [d for d in raw_data_path.iterdir() if d.is_dir()]

    for collector_dir in collector_dirs:
        if not collector_dir.is_dir():
            continue
        curr_name = collector_dir.name
        # 在标注者目录下查找session_*.json文件
        session_files = list(collector_dir.glob("session_*.json"))
        for session_file in session_files:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                session_data.append((curr_name, data))
    return session_data




def get_video_screenshots(video_path: str, max_images: int = 10) -> List[str]:
    """
    获取视频的多帧截图路径列表（从screenshots/视频名/目录）
    如果截图超过max_images张，则均匀抽帧
    
    Args:
        video_path: 视频文件路径
        max_images: 最多返回多少张截图
        
    Returns:
        截图文件路径列表
    """
    video_file = Path(video_path)
    screenshots_dir = video_file.parent.parent / "screenshots" / video_file.stem
    
    # 获取所有frame_*.jpg文件
    screenshot_files = sorted(screenshots_dir.glob("frame_*.jpg"))
    
    # 如果截图数量不超过max_images，直接返回全部
    screenshot_paths = [str(f) for f in screenshot_files]
    if len(screenshot_paths) <= max_images:
        return screenshot_paths
    
    # 均匀抽帧
    indices = [int(i * len(screenshot_paths) / max_images) for i in range(max_images)]
    return [screenshot_paths[i] for i in indices]


def extract_video_frames(video_path: str, output_dir: str = None, fps: int = 24) -> List[str]:
    """
    从视频中提取帧（每fps帧取一帧，第n*fps-1帧）
    图像保存到磁盘
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录（如果为None则自动生成为screenshots/视频名/）
        
    Returns:
        截图文件路径列表
    """    
    if not os.path.exists(video_path):
        return []
    
    # 自动生成输出目录
    if output_dir is None:
        video_file = Path(video_path)
        # 保存在screenshots/视频名/目录
        output_dir = video_file.parent.parent / "screenshots" / video_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return []
    
    # 计算要提取的帧索引（每24帧取一帧）
    frame_indices = []
    n = 1
    while True:
        frame_idx = n * fps - 1  # 第23, 47, 71, ... 帧
        if frame_idx < total_frames:
            frame_indices.append(frame_idx)
            n += 1
        else:
            # 不足24帧，取最后一帧
            if not frame_indices or frame_indices[-1] != total_frames - 1:
                frame_indices.append(total_frames - 1)
            break
    output_paths = [str(Path(output_dir) / f"frame_{idx}.jpg") for idx in range(len(frame_indices))]
    
    # 提取帧并保存到磁盘
    screenshots = []
    for frame_number, output_path in zip(frame_indices, output_paths):
        # 如果截图已存在，直接使用
        if os.path.exists(output_path):
            screenshots.append(output_path)
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            cv2.imwrite(output_path, frame)
            screenshots.append(output_path)
        else:
            print(f"警告: 无法读取视频帧 {frame_number} - {video_path}")
    
    cap.release()
    return screenshots


# 全局ASR模型缓存
_asr_model = None
_asr_postprocess = None

def get_asr_model():
    """获取或加载ASR模型（单例模式）- SenseVoiceSmall"""
    global _asr_model, _asr_postprocess
    if _asr_model is None:
        try:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
            
            print("正在加载SenseVoiceSmall模型...")
            _asr_model = AutoModel(
                model="iic/SenseVoiceSmall",
                language='auto',
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device="cuda:0",  # 使用GPU
                disable_update=True
            )
            _asr_postprocess = rich_transcription_postprocess
            print("SenseVoiceSmall模型加载完成")
        except ImportError:
            print("警告: 未安装funasr，无法转录音频。请运行: pip install funasr")
            return None, None
        except Exception as e:
            print(f"警告: 加载ASR模型失败: {e}")
            return None, None
    return _asr_model, _asr_postprocess


def extract_audio_first_2s(audio_path: str, output_path: str = None, duration: float = 2.0) -> str:
    """
    提取音频的前N秒（默认2秒），如果音频不足N秒则提取全部
    
    Args:
        audio_path: 输入音频文件路径
        output_path: 输出音频文件路径（如果为None，则使用临时文件）
        duration: 要提取的时长（秒），默认2秒
        
    Returns:
        输出音频文件路径，失败返回None
    """
    if not os.path.exists(audio_path):
        return None
    
    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=None, duration=None)
    
    # 计算实际提取的时长（不超过音频长度）
    actual_duration = min(duration, len(y) / sr)
    
    # 裁剪前N秒
    samples_to_extract = int(actual_duration * sr)
    y_extracted = y[:samples_to_extract]
    
    # 保存到临时文件或指定路径
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
    
    # 保存裁剪后的音频
    sf.write(output_path, y_extracted, sr)
    return output_path


def transcribe_audio(audio_path: str) -> Dict[str, str]:
    """
    将音频转换为文本（使用SenseVoiceSmall）
    同时转录完整音频和前2秒音频
    
    Args:
        audio_path: 音频文件路径
        
    Returns:
        包含 'text'（完整转录）和 'text_2s'（前2秒转录）的字典
    """
    if not os.path.exists(audio_path):
        return {'text': '', 'text_2s': ''}
    
    model, postprocess_func = get_asr_model()
    if model is None or postprocess_func is None:
        return {'text': '', 'text_2s': ''}
    
    result = {'text': '', 'text_2s': ''}
    
    # 转录完整音频
    full_result = model.generate(
        input=audio_path,
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    if isinstance(full_result, list) and len(full_result) > 0:
        result['text'] = postprocess_func(full_result[0]["text"])
    
    # 转录前2秒音频
    audio_2s_path = extract_audio_first_2s(audio_path, duration=2.0)
    if audio_2s_path:
        audio_2s_result = model.generate(
            input=audio_2s_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        if isinstance(audio_2s_result, list) and len(audio_2s_result) > 0:
            result['text_2s'] = postprocess_func(audio_2s_result[0]["text"])
        
        # 清理临时文件
        if audio_2s_path.startswith(tempfile.gettempdir()):
            os.remove(audio_2s_path)
    
    return result


def analyze_audio_module(session_data: List[Tuple[str, Dict]], output_file: str):
    """
    分析模块：音频转文本
    
    Args:
        session_data: session数据列表
        output_file: 输出文件路径
    """
    print("\n" + "="*60)
    print("开始音频转文本分析")
    print("="*60 + "\n")
    
    results = {}
    
    # 收集所有需要转录的音频
    all_actions = []
    for collector, session in session_data:
        for action in session['actions']:
            audio_path = action.get('audio_path', '').replace('\\', '/')
            if audio_path and os.path.exists(audio_path):
                all_actions.append({
                    'audio_path': audio_path,
                    'collector': collector,
                    'session_id': session['session_id'],
                    'timestamp': action.get('timestamp', '')
                })
    
    # 去重
    unique_audios = {}
    for action in all_actions:
        audio_path = action['audio_path']
        if audio_path not in unique_audios:
            unique_audios[audio_path] = action
    
    all_actions = list(unique_audios.values())
    print(f"共找到 {len(all_actions)} 个音频需要转录")
    
    # 使用多线程处理
    def process_single(action_info):
        audio_path = action_info['audio_path']
        result_dict = transcribe_audio(audio_path)
        # 如果至少有一个转录结果不为空，则返回
        if result_dict.get('text') or result_dict.get('text_2s'):
            return audio_path, result_dict
        return audio_path, None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:  # 音频转录较慢，用2线程
        futures = [executor.submit(process_single, action) for action in all_actions]
        
        with tqdm(total=len(futures), desc="转录进度", ncols=80) as pbar:
            for fut in concurrent.futures.as_completed(futures):
                audio_path, result = fut.result()
                if result:
                    results[audio_path] = result
                pbar.update(1)
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n音频转录结果已保存到: {output_file}")
    print(f"成功转录: {len(results)} 个音频")


def analyze_category_module(session_data: List[Tuple[str, Dict]], output_file: str):
    """
    分析模块：视频分类
    
    Args:
        session_data: session数据列表
        output_file: 输出文件路径
    """
    print("\n" + "="*60)
    print("开始视频分类分析")
    print("="*60 + "\n")
    
    results = {}
    
    # 收集所有需要分析的视频
    all_actions = []
    for collector, session in session_data:
        for action in session['actions']:
            video_path = action.get('video_path', '').replace('\\', '/')
            if video_path:
                # 检查视频的多帧截图是否存在
                screenshots = get_video_screenshots(video_path, max_images=10)
                if screenshots:
                    all_actions.append({
                        'video_path': video_path,
                        'collector': collector,
                        'session_id': session['session_id']
                    })
    
    print(f"共找到 {len(all_actions)} 个视频需要分类")
    
    # 使用多线程处理
    def process_single(action_info):
        video_path = action_info['video_path']
        # 获取视频的多帧截图（至多10张）
        screenshots = get_video_screenshots(video_path, max_images=10)
        
        if not screenshots:
            return video_path, None
        
        # 传递多张截图给get_category进行分类
        category = get_category(screenshots)
        return video_path, {
            'category': category,
            'collector': action_info['collector'],
            'session_id': action_info['session_id'],
            'screenshot_count': len(screenshots)
        }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single, action) for action in all_actions]
        
        with tqdm(total=len(futures), desc="分类进度", ncols=80) as pbar:
            for fut in concurrent.futures.as_completed(futures):
                video_path, result = fut.result()
                if result:
                    results[video_path] = result
                pbar.update(1)
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n分类结果已保存到: {output_file}")
    print(f"成功分类: {len(results)} 个视频")


def analyze_stats_module(session_data: List[Tuple[str, Dict]], output_file: str):
    """
    分析模块：提取点赞、评论等统计数据
    
    Args:
        session_data: session数据列表
        output_file: 输出文件路径
    """
    print("\n" + "="*60)
    print("开始提取统计数据")
    print("="*60 + "\n")
    
    results = {}
    
    # 收集所有需要分析的视频
    all_actions = []
    for collector, session in session_data:
        for action in session['actions']:
            video_path = action.get('video_path', '').replace('\\', '/')
            if video_path:
                # 获取视频的第一帧截图用于统计分析
                screenshots = get_video_screenshots(video_path, max_images=1)
                if screenshots:
                    all_actions.append({
                        'video_path': video_path,
                        'screenshot': screenshots[0],
                        'collector': collector,
                        'session_id': session['session_id']
                    })
    
    print(f"共找到 {len(all_actions)} 个视频需要分析")
    
    # 使用多线程处理
    def process_single(action_info):
        screenshot = action_info['screenshot']
        video_path = action_info['video_path']
        stats = parse_video_stats(screenshot)
        return video_path, {
            'stats': stats
        }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single, action) for action in all_actions]
        
        with tqdm(total=len(futures), desc="统计进度", ncols=80) as pbar:
            for fut in concurrent.futures.as_completed(futures):
                video_path, result = fut.result()
                if result:
                    results[video_path] = result
                pbar.update(1)
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n统计结果已保存到: {output_file}")
    print(f"成功分析: {len(results)} 个视频")


def analyze_action_reason(image_paths: List[str], action_info: dict) -> dict:
    """
    分析用户做出某个动作的原因
    
    Args:
        image_paths: 视频截图路径列表
        action_info: 动作信息，包含 viewing_duration 和 actions
        
    Returns:
        dict: 包含 'description'（视频描述）, 'category'（分类）, 'reason'（行为解释）
    """
    # 构建动作描述
    viewing_duration = action_info.get('viewing_duration', 0)
    actions = action_info.get('actions', [])
    
    action_desc = f"观看了{viewing_duration:.1f}秒"
    for act in actions:
        action_type = act.get('type', '')
        if action_type == 'like':
            action_desc += "，点赞"
        elif action_type == 'comment':
            action_desc += f"，评论：{act.get('text', '')}"
        elif action_type == 'share':
            action_desc += f"，分享给：{act.get('text', '')}"
    
    sys_prompt = (
        "你是一个短视频用户行为分析专家。请根据视频截图和用户的实际行为，完成以下三个任务：\n\n"
        "1. **视频描述**：用一句话（不超过30字）描述视频的主要内容\n"
        "2. **视频分类**：从以下【主类-子类】结构中选择最合适的1-2个分类：\n\n"
        "{\n"
        '  "娱乐": ["搞笑段子", "剧情短片", "才艺展示", "挑战实验", "综艺片段", "明星动态", "网红日常"],\n'
        '  "知识教育": ["科普知识", "技能教学", "外语学习", "学习方法", "法律科普", "职业技能", "编程技术"],\n'
        '  "生活记录": ["日常vlog", "家庭生活", "生活技巧", "情感故事", "社区互助", "公益记录", "怀旧故事"],\n'
        '  "情感与心理": ["情感咨询", "恋爱关系", "婚姻家庭", "心理健康", "励志治愈", "情绪调节"],\n'
        '  "美食": ["家常菜", "烘焙甜点", "特色小吃", "美食探店", "异国料理", "饮品调制", "美食测评"],\n'
        '  "时尚美妆": ["穿搭分享", "美妆护肤", "发型造型", "配饰分享", "时尚资讯", "品牌测评", "美甲美睫"],\n'
        '  "运动健身": ["健身训练", "球类运动", "户外探险", "瑜伽普拉提", "运动技巧", "极限运动", "康复训练"],\n'
        '  "科技数码": ["手机测评", "电脑硬件", "智能家居", "软件技巧", "科技前沿", "数码配件", "AI应用"],\n'
        '  "汽车": ["汽车评测", "驾驶技巧", "保养维护", "汽车文化", "新能源汽车", "二手车交易", "汽车改装"],\n'
        '  "游戏": ["手游攻略", "端游实况", "电竞赛事", "游戏测评", "游戏教学", "游戏新闻", "虚拟世界/沙盒创作"],\n'
        '  "音乐舞蹈": ["歌曲翻唱", "舞蹈表演", "乐器演奏", "声乐教学", "舞蹈教学", "音乐创作", "舞蹈编排"],\n'
        '  "影视动漫": ["电影解说", "剧情剪辑", "影视评论", "经典影片", "动漫解说", "动画短片", "配音表演"],\n'
        '  "旅行": ["国内旅行", "国外旅行", "旅行攻略", "露营体验", "景区探秘", "背包客路线", "小众景点"],\n'
        '  "摄影与创作": ["摄影技巧", "器材测评", "后期制作", "人像摄影", "风光摄影", "手机摄影", "摄影作品展示"],\n'
        '  "财经商业": ["商业分析", "创业经验", "理财知识", "投资策略", "副业思路", "营销策略", "经济观察"],\n'
        '  "房产家居": ["住宅装修", "家居收纳", "家居好物", "房产知识", "租房买房", "园艺绿植", "智能家居"],\n'
        '  "医疗健康": ["养生保健", "心理健康科普", "疾病预防", "营养饮食", "中医养生", "运动康复", "医药科普"],\n'
        '  "三农乡村": ["农村生活", "农业种植", "乡村美食", "传统手艺", "乡村振兴", "农产品展示", "田园风光"],\n'
        '  "宠物": ["狗狗日常", "猫咪日常", "宠物训练", "宠物医疗", "萌宠搞笑", "宠物用品", "动物救助"],\n'
        '  "亲子育儿": ["育儿经验", "儿童教育", "萌娃日常", "亲子游戏", "孕期知识", "早教启蒙", "亲子旅行"],\n'
        '  "二次元": ["动漫解说", "cosplay", "宅舞", "同人创作", "声优配音", "虚拟偶像", "动漫周边"],\n'
        '  "文化历史": ["历史科普", "民俗文化", "文物考古", "非遗传承", "文学知识", "历史人物故事", "传统文化"],\n'
        '  "军事法律": ["军事知识", "军事装备", "国防教育", "法律常识", "法律案例", "政策解读", "国际局势"],\n'
        '  "社会资讯": ["社会热点", "民生事件", "公益新闻", "社会观察", "热点评论", "现场记录", "公共安全"],\n'
        '  "广告推广": ["商业广告", "产品推广", "品牌宣传", "直播带货", "开箱测评", "促销活动", "软广植入"]\n'
        "}\n\n"
        "3. **行为解释**：用一句话（不超过30字）结合视频内容、分类和用户行为（观看时长、点赞/评论/分享）解释用户为什么这样做\n"
        "   - 如果用户快速划走（<3秒），说明不感兴趣的原因\n"
        "   - 如果用户观看较久但没有互动，说明内容吸引但不足以互动的原因\n"
        "   - 如果用户点赞/评论/分享，说明内容打动用户的具体点\n\n"
        "输出格式（纯JSON，不要有其他内容）：\n"
        "{\n"
        '  "description": "视频描述",\n'
        '  "category": {"main": "主类", "sub": "子类"},\n'
        '  "reason": "行为解释"\n'
        "}"
    )
    
    # 构建多张图片的content
    content = []
    for img_path in image_paths[:5]:  # 最多5张图
        if os.path.exists(img_path):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"file://{os.path.abspath(img_path)}"}
            })
    
    content.append({
        "type": "text",
        "text": f"用户行为：{action_desc}\n\n请完成上述三个任务："
    })
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": content}
    ]
    
    response = client.chat.completions.create(
        model="qwen",
        messages=messages,
        temperature=0.3,
        max_tokens=300
    )
    
    text = clean_json_response(response.choices[0].message.content)
    try:
        result = json.loads(text)
        # 确保返回所需的字段
        return {
            'description': result.get('description', ''),
            'category': result.get('category', {'main': '', 'sub': ''}),
            'reason': result.get('reason', '')
        }
    except json.JSONDecodeError:
        # 如果解析失败，返回空结果
        return {
            'description': '',
            'category': {'main': '', 'sub': ''},
            'reason': text  # 降级：直接使用原始文本作为reason
        }


def analyze_reason_module(session_data: List[Tuple[str, Dict]], output_file: str):
    """
    分析模块：分析用户动作原因
    
    Args:
        session_data: session数据列表
        output_file: 输出文件路径
    """
    print("\n" + "="*60)
    print("开始分析用户动作原因")
    print("="*60 + "\n")
    
    results = {}
    
    # 收集所有需要分析的动作
    all_actions = []
    for collector, session in session_data:
        for action in session['actions']:
            video_path = action.get('video_path', '').replace('\\', '/')
            if video_path:
                screenshots = get_video_screenshots(video_path, max_images=5)
                if screenshots:
                    all_actions.append({
                        'video_path': video_path,
                        'screenshots': screenshots,
                        'viewing_duration': action.get('viewing_duration', 0),
                        'actions': action.get('actions', []),
                        'collector': collector,
                        'session_id': session['session_id'],
                        'timestamp': action.get('timestamp', '')
                    })
    
    print(f"共找到 {len(all_actions)} 个动作需要分析原因")
    
    # 使用多线程处理
    def process_single(action_info):
        video_path = action_info['video_path']
        screenshots = action_info['screenshots']
        
        reason_result = analyze_action_reason(screenshots, {
            'viewing_duration': action_info['viewing_duration'],
            'actions': action_info['actions']
        })
        
        # 构建动作字符串
        action_str = f"watch({action_info['viewing_duration']})"
        for act in action_info['actions']:
            action_type = act.get('type', '')
            if action_type == 'like':
                action_str += ", like()"
            elif action_type == 'comment':
                action_str += f", comment({act.get('text', '')})"
            elif action_type == 'share':
                action_str += f", share({act.get('text', '')})"
        
        return video_path, {
            'action': action_str,
            'description': reason_result.get('description', ''),
            'category': reason_result.get('category', {'main': '', 'sub': ''}),
            'reason': reason_result.get('reason', ''),
            'collector': action_info['collector'],
            'session_id': action_info['session_id'],
            'timestamp': action_info['timestamp']
        }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single, action) for action in all_actions]
        
        with tqdm(total=len(futures), desc="分析进度", ncols=80) as pbar:
            for fut in concurrent.futures.as_completed(futures):
                video_path, result = fut.result()
                if result and (result.get('reason') or result.get('description')):
                    results[video_path] = result
                pbar.update(1)
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n动作原因分析结果已保存到: {output_file}")
    print(f"成功分析: {len(results)} 个动作")


def load_persona(persona_file: str) -> dict:
    """加载persona.json文件"""
    if persona_file and os.path.exists(persona_file):
        with open(persona_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def prepare_training_data(
    session_data: List[Tuple[str, Dict]], 
    output_file: str,
    window_size: int = 5,
    max_history: int = 3,
    audio_transcript_file: str = None,
    action_reason_file: str = None,
    persona_file: str = None,
    random_seed: int = 42,
    think: bool = False,
    audio_version: str = 'none'
):
    """
    将session数据切分为带历史的训练数据，并按session划分数据集避免信息泄露
    
    Args:
        session_data: session数据列表
        output_file: 输出文件路径
        window_size: 历史窗口大小
        max_history: 最多使用多少条历史记录
        audio_transcript_file: 音频转录结果文件路径
        action_reason_file: 动作原因分析结果文件路径
        persona_file: persona.json文件路径
        random_seed: 随机种子
        think: 是否使用思考模式
        audio_version: 音频版本，可选 'none'（默认）、'full'、'2s'
    """
    print("\n" + "="*60)
    print("开始准备训练数据")
    print("="*60 + "\n")
    
    # 设置随机种子
    random.seed(random_seed)
    print(f"随机种子: {random_seed}")
    
    # 加载音频转录结果
    audio_transcripts = {}
    if audio_transcript_file and os.path.exists(audio_transcript_file):
        with open(audio_transcript_file, 'r', encoding='utf-8') as f:
            audio_transcripts = json.load(f)
        print(f"已加载 {len(audio_transcripts)} 个音频转录结果")
    
    # 加载动作原因分析结果
    action_reasons = {}
    if action_reason_file and os.path.exists(action_reason_file):
        with open(action_reason_file, 'r', encoding='utf-8') as f:
            action_reasons = json.load(f)
        print(f"已加载 {len(action_reasons)} 个动作原因分析结果")
    
    # 加载persona信息
    persona = load_persona(persona_file)
    if persona:
        print(f"已加载persona信息")
    
    # 第一步：按session划分（10% valid1, 10% test1, 80% 用于进一步划分）
    random.shuffle(session_data)
    total_sessions = len(session_data)

    # 定义生成样本的函数
    def generate_samples_from_session(collector, session):
        """从单个session生成所有样本"""
        samples = []
        actions = session['actions']
        
        for i, action in enumerate(actions):
            xml_path = action.get('xml_path', '').replace('\\', '/')
            video_path = action.get('video_path', '').replace('\\', '/')
            audio_path = action.get('audio_path', '').replace('\\', '/')
            
            # 为当前视频只取第一帧（避免多帧暗示观看时长）
            current_screenshots = get_video_screenshots(video_path, max_images=1)
            
            # 构建历史记录（每个历史只用一张图）
            start_idx = max(0, i - window_size + 1)
            history_actions = actions[start_idx:i]
            
            # 提取历史截图（每个历史视频只用一张）
            history_screenshots = []
            str_actions = []
            for hist_action in history_actions:
                hist_video = hist_action.get('video_path', '').replace('\\', '/')
                item = hist_action.get('actions', [])
                hist_screenshots = get_video_screenshots(hist_video, max_images=1)
                if hist_screenshots and actions:
                    history_screenshots.append(hist_screenshots[0])
                    str_action = '```python\n'
                    str_action += f"watch({hist_action['viewing_duration']})\n"
                    for act in item:
                        action_type = act['type']
                        if action_type == 'like':
                            str_action += "like()\n"
                        elif action_type == 'comment':
                            str_action += f"comment({act['text']})\n"
                        elif action_type == 'share':
                            str_action += f"share({act['text']})\n"
                    str_action += '```'
                    str_actions.append(str_action)
            
            # 随机选择历史记录数量
            if history_screenshots:
                num_history = random.randint(1, min(max_history, len(history_screenshots)))
                selected_history = history_screenshots[-num_history:]
                str_actions = str_actions[-num_history:]
            else:
                selected_history = []
                str_actions = []
            
            # 构建训练样本：历史图片 + 当前视频的多张图片
            all_images = selected_history + current_screenshots
            
            # 构建历史提示
            if len(selected_history) > 0:
                history_placeholder = "Your browsing history:\n"
                for str_action in str_actions:
                    history_placeholder += "<image>\n" + str_action + "\n"
            else:
                history_placeholder = ""
            
            # 获取三种版本的音频转录文本
            audio_text_full = ""  # 完整转录
            audio_text_2s = ""    # 2秒转录
            audio_text_none = ""  # 无转录
            
            if audio_path in audio_transcripts:
                audio_text_full = audio_transcripts[audio_path].get('text', '')
                audio_text_2s = audio_transcripts[audio_path].get('text_2s', '')
            
            # 构建动作标签（用于监督学习）
            user_actions = action.get('actions', [])
            answer = '```python\n'
            answer += f"watch({action['viewing_duration']})\n"
            for act in user_actions:
                action_type = act['type']
                if action_type == 'like':
                    answer += "like()\n"
                elif action_type == 'comment':
                    answer += f"comment({act['text']})\n"
                elif action_type == 'share':
                    answer += f"share({act['text']})\n"
            answer += '```'
            
            # 获取动作原因（用于think模式）
            reason_text = ""
            if video_path in action_reasons:
                reason_data = action_reasons[video_path]
                description = reason_data.get('description', '')
                category = reason_data.get('category', {})
                reason = reason_data.get('reason', '')
                
                # 构建完整的分析文本（描述+分类+解释）
                parts = []
                
                # 1. 视频内容描述
                if description:
                    parts.append(f"视频内容：{description}")
                
                # 2. 视频分类
                category_str = f"{category.get('main', '')}-{category.get('sub', '')}" if category.get('main') else ""
                if category_str:
                    parts.append(f"分类：{category_str}")
                
                # 3. 行为解释
                if reason:
                    parts.append(f"行为分析：{reason}")
                
                # 组合成完整文本
                if parts:
                    reason_text = "\n".join(parts)
            
            # 构建带原因的回答（用于think模式）
            answer_with_reason = ""
            if reason_text:
                answer_with_reason = f"{reason_text}\n\n{answer}"
            else:
                answer_with_reason = answer
            
            # 通用的元数据
            metadata = {
                "collector": collector,
                "session_id": session['session_id'],
                "timestamp": action.get('timestamp', ''),
                "viewing_duration": action.get('viewing_duration', 0),
                "xml_path": xml_path,
                "video_path": action.get('video_path', '').replace('\\', '/'),
                "audio_path": action.get('audio_path', '').replace('\\', '/')
            }
            
            # 辅助函数：创建样本
            def create_sample(audio_text_version, use_thinking=False):
                """创建单个样本"""
                # 生成基础 user prompt（普通样本不使用persona）
                base_user_prompt = get_user_prompt(
                    history_screenshots=selected_history,
                    history_actions=str_actions if selected_history else None,
                    current_screenshots=current_screenshots,
                    audio_transcript=audio_text_version,
                    persona=None
                )
                
                # 如果使用thinking模式，修改prompt
                if use_thinking:
                    user_prompt = base_user_prompt.replace(
                        "Respond strictly with a Python code block",
                        "Start by giving a short reason for the action you are taking, and then output a Python code block"
                    ).replace(
                        "starting with ```python",
                        "beginning with ```python"
                    )
                    assistant_content = answer_with_reason
                    solution_content = answer_with_reason
                else:
                    user_prompt = base_user_prompt
                    assistant_content = answer
                    solution_content = answer
                
                return {
                    "images": all_images,
                    "messages": [
                        {
                            "role": "system",
                            "content": get_system_prompt(use_persona=False)
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        },
                        {
                            "role": "assistant",
                            "content": assistant_content
                        }
                    ],
                    "solution": solution_content,
                    "metadata": metadata.copy()
                }
            
            # 辅助函数：创建persona样本
            def create_persona_sample(audio_text_version, use_thinking=False):
                """创建带persona的样本"""
                # persona信息现在放在user message中
                base_user_prompt = get_user_prompt(
                    history_screenshots=selected_history,
                    history_actions=str_actions if selected_history else None,
                    current_screenshots=current_screenshots,
                    audio_transcript=audio_text_version,
                    persona=persona if persona else None
                )
                
                if use_thinking:
                    user_prompt = base_user_prompt.replace(
                        "Respond strictly with a Python code block",
                        "Start by giving a short reason for the action you are taking, and then output a Python code block"
                    ).replace(
                        "starting with ```python",
                        "beginning with ```python"
                    )
                    assistant_content = answer_with_reason
                    solution_content = answer_with_reason
                else:
                    user_prompt = base_user_prompt
                    assistant_content = answer
                    solution_content = answer
                
                return {
                    "images": all_images,
                    "messages": [
                        {
                            "role": "system",
                            "content": get_system_prompt(use_persona=False)
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        },
                        {
                            "role": "assistant",
                            "content": assistant_content
                        }
                    ],
                    "solution": solution_content,
                    "metadata": metadata.copy()
                }
            
            # 生成三种版本的样本：(完整转录, 2s转录, 无转录)
            sample_full = create_sample(audio_text_full, use_thinking=think)
            sample_2s = create_sample(audio_text_2s, use_thinking=think)
            sample_none = create_sample(audio_text_none, use_thinking=think)
            
            # 生成三种版本的persona样本
            persona_sample_full = create_persona_sample(audio_text_full, use_thinking=think)
            persona_sample_2s = create_persona_sample(audio_text_2s, use_thinking=think)
            persona_sample_none = create_persona_sample(audio_text_none, use_thinking=think)
            
            # 返回三种版本的样本：(完整转录, 2s转录, 无转录)，每种包含(普通, persona)
            samples.append((
                (sample_full, sample_2s, sample_none),
                (persona_sample_full, persona_sample_2s, persona_sample_none)
            ))
        
        return samples
    
    # 生成各个数据集 - 三种版本：完整转录、2秒转录、无转录
    # 每个数据集都有三种版本
    datasets_full = {
        'valid': [],
        'test': [],
        'train': [],
        'train_all': [],
        'sft': []
    }
    datasets_2s = {
        'valid': [],
        'test': [],
        'train': [],
        'train_all': [],
        'sft': []
    }
    datasets_none = {
        'valid': [],
        'test': [],
        'train': [],
        'train_all': [],
        'sft': []
    }
    
    # persona数据集 - 三种版本
    persona_datasets_full = {
        'valid': [],
        'test': [],
        'train': [],
        'train_all': [],
        'sft': []
    }
    persona_datasets_2s = {
        'valid': [],
        'test': [],
        'train': [],
        'train_all': [],
        'sft': []
    }
    persona_datasets_none = {
        'valid': [],
        'test': [],
        'train': [],
        'train_all': [],
        'sft': []
    }
    
    for collector, session in session_data:
        samples = generate_samples_from_session(collector, session)
        
        num_sft_data = int(len(samples)*0.3)
        num_vaild = len(samples) // 10
        num_test = len(samples) // 10
        
        # 如果样本数 <= 4，全部放入train
        if num_vaild == 0:
            # 完整转录版本
            datasets_full['train'].extend([s[0][0] for s in samples])
            datasets_full['train_all'].extend([s[0][0] for s in samples])
            persona_datasets_full['train'].extend([s[1][0] for s in samples])
            persona_datasets_full['train_all'].extend([s[1][0] for s in samples])
            # 2秒转录版本
            datasets_2s['train'].extend([s[0][1] for s in samples])
            datasets_2s['train_all'].extend([s[0][1] for s in samples])
            persona_datasets_2s['train'].extend([s[1][1] for s in samples])
            persona_datasets_2s['train_all'].extend([s[1][1] for s in samples])
            # 无转录版本
            datasets_none['train'].extend([s[0][2] for s in samples])
            datasets_none['train_all'].extend([s[0][2] for s in samples])
            persona_datasets_none['train'].extend([s[1][2] for s in samples])
            persona_datasets_none['train_all'].extend([s[1][2] for s in samples])
        else:
            last = samples[-(num_vaild + num_test):]
            first = samples[:num_sft_data]
            random.shuffle(last)
            random.shuffle(first)
            
            # 完整转录版本
            datasets_full['valid'].extend([s[0][0] for s in last[:num_vaild]])
            datasets_full['test'].extend([s[0][0] for s in last[num_vaild:]])
            persona_datasets_full['valid'].extend([s[1][0] for s in last[:num_vaild]])
            persona_datasets_full['test'].extend([s[1][0] for s in last[num_vaild:]])
            if first:
                datasets_full['sft'].extend([s[0][0] for s in first])
                persona_datasets_full['sft'].extend([s[1][0] for s in first])
            datasets_full['train'].extend([s[0][0] for s in samples[num_sft_data:-(num_vaild + num_test)]])
            datasets_full['train_all'].extend([s[0][0] for s in samples[:-(num_vaild + num_test)]])
            persona_datasets_full['train'].extend([s[1][0] for s in samples[num_sft_data:-(num_vaild + num_test)]])
            persona_datasets_full['train_all'].extend([s[1][0] for s in samples[:-(num_vaild + num_test)]])
            
            # 2秒转录版本
            datasets_2s['valid'].extend([s[0][1] for s in last[:num_vaild]])
            datasets_2s['test'].extend([s[0][1] for s in last[num_vaild:]])
            persona_datasets_2s['valid'].extend([s[1][1] for s in last[:num_vaild]])
            persona_datasets_2s['test'].extend([s[1][1] for s in last[num_vaild:]])
            if first:
                datasets_2s['sft'].extend([s[0][1] for s in first])
                persona_datasets_2s['sft'].extend([s[1][1] for s in first])
            datasets_2s['train'].extend([s[0][1] for s in samples[num_sft_data:-(num_vaild + num_test)]])
            datasets_2s['train_all'].extend([s[0][1] for s in samples[:-(num_vaild + num_test)]])
            persona_datasets_2s['train'].extend([s[1][1] for s in samples[num_sft_data:-(num_vaild + num_test)]])
            persona_datasets_2s['train_all'].extend([s[1][1] for s in samples[:-(num_vaild + num_test)]])
            
            # 无转录版本
            datasets_none['valid'].extend([s[0][2] for s in last[:num_vaild]])
            datasets_none['test'].extend([s[0][2] for s in last[num_vaild:]])
            persona_datasets_none['valid'].extend([s[1][2] for s in last[:num_vaild]])
            persona_datasets_none['test'].extend([s[1][2] for s in last[num_vaild:]])
            if first:
                datasets_none['sft'].extend([s[0][2] for s in first])
                persona_datasets_none['sft'].extend([s[1][2] for s in first])
            datasets_none['train'].extend([s[0][2] for s in samples[num_sft_data:-(num_vaild + num_test)]])
            datasets_none['train_all'].extend([s[0][2] for s in samples[:-(num_vaild + num_test)]])
            persona_datasets_none['train'].extend([s[1][2] for s in samples[num_sft_data:-(num_vaild + num_test)]])
            persona_datasets_none['train_all'].extend([s[1][2] for s in samples[:-(num_vaild + num_test)]])
    
    # 保存数据集
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    base_name = output_path.stem
    ext = output_path.suffix
    
    # 定义三种音频版本的前缀（none为默认版本，不加前缀）
    audio_version_prefixes = {
        'none': '',  # 默认版本，无前缀
        'full': 'audio_full',
        '2s': 'audio_2s'
    }
    
    # 定义三种版本的数据集字典
    all_datasets = {
        'full': datasets_full,
        '2s': datasets_2s,
        'none': datasets_none
    }
    
    all_persona_datasets = {
        'full': persona_datasets_full,
        '2s': persona_datasets_2s,
        'none': persona_datasets_none
    }
    
    # 数据集名称映射到文件后缀（键名必须与datasets字典中的键名一致）
    dataset_file_mapping = {
        'train': 'train',
        'train_all': 'train_all',
        'sft': 'train_sft',
        'valid': 'valid_item',
        'test': 'test_item'
    }
    
    # 计算总样本数（使用完整转录版本作为基准）
    total_samples = sum(len(datasets_full[key]) for key in ['train', 'train_all', 'sft', 'valid', 'test'])
    
    print("\n" + "="*60)
    print(f"数据集划分完成（随机种子={random_seed}）")
    print("="*60)
    
    # 根据audio_version参数决定保存哪些版本
    versions_to_save = [audio_version] if audio_version in audio_version_prefixes else ['none']
    
    # 保存指定版本的数据集
    for version_key in versions_to_save:
        version_prefix = audio_version_prefixes[version_key]
        datasets = all_datasets[version_key]
        
        version_display = "默认（无转录）" if version_prefix == '' else f"{version_prefix.upper()}"
        print(f"\n{version_display} 版本数据集:")
        print("-" * 60)
        
        for display_name, file_suffix in dataset_file_mapping.items():
            dataset = datasets[display_name]
            if dataset:  # 只保存非空数据集
                # 如果是默认版本（无前缀），文件名格式不同
                if version_prefix == '':
                    filename_prefix = f"{'thinking_' if think else ''}video_action_{file_suffix}"
                else:
                    filename_prefix = f"{'thinking_' if think else ''}{version_prefix}_video_action_{file_suffix}"
                output_file_path = output_path.parent / f"{filename_prefix}{ext}"
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    for item in dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                percentage = len(dataset) / total_samples * 100 if total_samples > 0 else 0
                version_label = "默认（无转录）" if version_prefix == '' else f"{version_prefix}"
                print(f"  {display_name:15s} [{version_label:15s}]: {output_file_path.name} ({len(dataset):5d} 样本, {percentage:5.1f}%)")
    
    # 保存persona数据集（指定版本）
    if persona:
        print("\n" + "="*60)
        print("PERSONA数据集:")
        print("="*60)
        
        for version_key in versions_to_save:
            version_prefix = audio_version_prefixes[version_key]
            persona_datasets = all_persona_datasets[version_key]
            
            version_display = "默认（无转录）" if version_prefix == '' else f"{version_prefix.upper()}"
            print(f"\n{version_display} 版本 Persona数据集:")
            print("-" * 60)
            
            for display_name, file_suffix in dataset_file_mapping.items():
                dataset = persona_datasets[display_name]
                if dataset:  # 只保存非空数据集
                    # 如果是默认版本（无前缀），文件名格式不同
                    if version_prefix == '':
                        filename_prefix = f"{'thinking_' if think else ''}persona_video_action_{file_suffix}"
                    else:
                        filename_prefix = f"{'thinking_' if think else ''}{version_prefix}_persona_video_action_{file_suffix}"
                    output_file_path = output_path.parent / f"{filename_prefix}{ext}"
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        for item in dataset:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
                    percentage = len(dataset) / total_samples * 100 if total_samples > 0 else 0
                    version_label = "默认（无转录）" if version_prefix == '' else f"{version_prefix}"
                    print(f"  {display_name:15s} [{version_label:15s}]: {output_file_path.name} ({len(dataset):5d} 样本, {percentage:5.1f}%)")
    
    print(f"\n总样本数: {total_samples}")
    
    # 统计信息（使用完整转录版本）
    all_samples = datasets_full['train'] + datasets_full['valid'] + datasets_full['test']
    total_with_history = sum(1 for s in all_samples if len(s['images']) > 1)
    avg_history = sum(len(s['images']) - 1 for s in all_samples) / len(all_samples) if all_samples else 0
    print(f"包含历史记录的样本: {total_with_history}")
    print(f"平均历史长度: {avg_history:.2f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PersonaAct 数据准备工具')
    
    parser.add_argument('--mode', type=str, default='all',
                       choices=['extract', 'analyze', 'prepare', 'all'],
                       help='运行模式: extract(提取截图), analyze(分析), prepare(准备训练数据), all(全部)')
    
    parser.add_argument('--task', type=str, default='all',
                       choices=['category', 'stats', 'audio', 'reason', 'all'],
                       help='分析任务: category(分类), stats(统计), audio(音频转文本), reason(动作原因分析), all(全部)')
    
    parser.add_argument('--input', type=str, default='raw_data',
                       help='输入目录（raw_data目录）')
    
    parser.add_argument('--output-dir', type=str, default='data',
                       help='输出目录')
    
    parser.add_argument('--window-size', type=int, default=5,
                       help='历史窗口大小')
    
    parser.add_argument('--max-history', type=int, default=3,
                       help='最多使用多少条历史记录')
    
    parser.add_argument('--random-seed', type=int, default=42,
                       help='随机种子（默认42）')

    parser.add_argument('--name', type=str, default=None,
                       help='标注者名字（如只加载某一人，默认None为所有）')

    parser.add_argument('--think', action='store_true', default=False,
                       help='是否使用思考模式')
    
    parser.add_argument('--audio', type=str, default='none',
                       choices=['none', 'full', '2s'],
                       help='音频转录版本: none(无转录,默认), full(完整转录), 2s(2秒转录)')

    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PersonaAct 数据准备工具")
    print("="*60 + "\n")
    
    # 加载所有session文件
    print(f"正在从 {args.input} 加载session文件...")
    session_data = load_session_files(args.input, args.name)
    print(f"共加载 {len(session_data)} 个session")
    
    total_actions = sum(len(session['actions']) for _, session in session_data)
    print(f"共 {total_actions} 个视频动作记录\n")
    
    # 按标注者分组
    collectors_data = {}
    for collector, session in session_data:
        if collector not in collectors_data:
            collectors_data[collector] = []
        collectors_data[collector].append((collector, session))
    
    # 根据模式执行不同的任务
    if args.mode in ['extract', 'all']:
        print("\n" + "="*60)
        print("开始批量提取视频多帧截图")
        print("="*60 + "\n")
        
        # 收集所有需要提取截图的视频
        videos_to_extract = set()
        for collector, session in session_data:
            for action in session['actions']:
                video_path = action.get('video_path', '').replace('\\', '/')
                if video_path:
                    videos_to_extract.add(video_path)
        
        print(f"共有 {len(videos_to_extract)} 个视频需要提取多帧截图")
        
        # 使用多线程批量提取
        def extract_single(video_path):
            screenshots = extract_video_frames(video_path)
            return video_path, len(screenshots) if screenshots else 0
        
        success_count = 0
        total_frames = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(extract_single, video) for video in videos_to_extract]
            
            with tqdm(total=len(futures), desc="提取进度", ncols=80) as pbar:
                for fut in concurrent.futures.as_completed(futures):
                    video_path, frame_count = fut.result()
                    if frame_count > 0:
                        success_count += 1
                        total_frames += frame_count
                    pbar.update(1)
        
        print(f"\n多帧截图提取完成！")
        print(f"成功: {success_count}/{len(videos_to_extract)} 个视频")
        print(f"共提取: {total_frames} 帧截图")
        print(f"平均每个视频: {total_frames/success_count:.1f} 帧" if success_count > 0 else "")
    
    if args.mode in ['analyze', 'all']:
        # 为每个标注者分别进行分析
        for collector_name, collector_session_data in collectors_data.items():
            print(f"\n处理标注者: {collector_name}")
            collector_output_dir = os.path.join(args.output_dir, collector_name)
            
            if args.task in ['category', 'all']:
                category_output = os.path.join(collector_output_dir, 'category_analysis.json')
                analyze_category_module(collector_session_data, category_output)
            
            if args.task in ['stats', 'all']:
                stats_output = os.path.join(collector_output_dir, 'stats_analysis.json')
                analyze_stats_module(collector_session_data, stats_output)
            
            if args.task in ['audio', 'all']:
                audio_output = os.path.join(collector_output_dir, 'audio_transcript.json')
                analyze_audio_module(collector_session_data, audio_output)
            
            if args.task in ['reason', 'all']:
                reason_output = os.path.join(collector_output_dir, 'action_reason.json')
                analyze_reason_module(collector_session_data, reason_output)
    
    if args.mode in ['prepare', 'all']:
        # 为每个标注者分别准备训练数据
        for collector_name, collector_session_data in collectors_data.items():
            print(f"\n处理标注者: {collector_name}")
            collector_output_dir = os.path.join(args.output_dir, collector_name)
            train_output = os.path.join(collector_output_dir, 'video_action_train.jsonl')
            audio_transcript_file = os.path.join(collector_output_dir, 'audio_transcript.json')
            action_reason_file = os.path.join(collector_output_dir, 'action_reason.json')
            persona_file = os.path.join(collector_output_dir, 'persona.json')
            prepare_training_data(
                collector_session_data, 
                train_output,
                window_size=args.window_size,
                max_history=args.max_history,
                audio_transcript_file=audio_transcript_file if os.path.exists(audio_transcript_file) else None,
                action_reason_file=action_reason_file if os.path.exists(action_reason_file) else None,
                persona_file=persona_file if os.path.exists(persona_file) else None,
                random_seed=args.random_seed,
                think=args.think,
                audio_version=args.audio
            )
    
    print("\n" + "="*60)
    print("✓ 所有任务完成！")
    print("="*60)


if __name__ == "__main__":
    main()

