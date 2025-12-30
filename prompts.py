"""训练时使用的prompt模板，需要与训练时完全一致"""


# ========== 数据准备阶段的 prompts ==========

def get_category_prompt() -> str:
    """获取视频分类的 prompt"""
    return (
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


def get_video_stats_prompt() -> str:
    """获取视频统计数据提取的 prompt"""
    return (
        "请你识别图片中的短视频点赞、评论、收藏、转发数量、作者名称和视频标题。请仅输出JSON格式，示例：\n"
        '{"like": 100, "comment": 22, "favorite": 8, "share": 3, "author": "用户名", "title": "视频标题"}。\n'
        '"like"为点赞数，"comment"为评论数，"favorite"为收藏数，"share"为转发数，"author"为作者名称，"title"为视频标题。\n'
        '发现为空或识别不出来时请输出0或空字符串。如果有单位需要换算，如"14.3万"应输出为143000。请严格保证输出格式为纯 JSON，不要多余内容。'
    )


def get_action_reason_prompt(intro: str, action_desc: str) -> str:
    """
    获取动作原因分析的 prompt
    
    Args:
        intro: prompt 开头介绍语（有多个变体用于数据增强）
        action_desc: 用户行为描述，如"观看了15.3秒，点赞"
    """
    examples_text = """
示例1：
用户行为：观看了2.1秒
输出：
{
  "description": "搞笑段子，演员夸张表演引发笑点",
  "category": {"main": "娱乐", "sub": "搞笑段子"},
  "reason": "内容平淡无新意，表演夸张过度显得不真实，快速划走"
}

示例2：
用户行为：观看了15.3秒，点赞
输出：
{
  "description": "美食制作教程，详细展示烘焙步骤",
  "category": {"main": "美食", "sub": "烘焙甜点"},
  "reason": "教程清晰实用，步骤详细易懂，想学习制作方法"
}

示例3：
用户行为：观看了8.5秒
输出：
{
  "description": "旅行vlog，展示某地风景和人文",
  "category": {"main": "旅行", "sub": "国内旅行"},
  "reason": "风景优美吸引人，但节奏较慢缺乏亮点，看完主要画面后离开"
}
"""
    
    return (
        f"{intro}\n\n"
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
        f"参考示例：\n{examples_text}\n"
        "**重要要求**：\n"
        "- 必须输出严格的JSON格式\n"
        "- 所有字段值必须是单行文本，不能包含换行符\n"
        "- description和reason字段不超过30字\n"
        "- 不要输出任何JSON之外的内容\n\n"
        "输出格式（纯JSON）：\n"
        "{\n"
        '  "description": "视频描述",\n'
        '  "category": {"main": "主类", "sub": "子类"},\n'
        '  "reason": "行为解释"\n'
        "}"
    )


def get_action_reason_prompt_variants():
    """获取动作原因分析 prompt 的多个变体（用于数据增强）"""
    return [
        "你是一个短视频用户行为分析专家。请根据视频截图和用户的实际行为，完成以下三个任务：",
        "作为短视频分析师，请仔细观察视频内容和用户行为，完成以下分析：",
        "请分析这个短视频的内容和用户的浏览行为，完成以下任务：",
        "你需要作为专业分析师，根据视频画面和用户互动记录，完成以下工作：",
    ]


# ========== 训练和推理阶段的 prompts ==========

def format_persona_for_system_prompt(persona: dict) -> str:
    """将persona信息格式化为system prompt的一部分"""
    if not persona:
        return ""
    
    # 提取persona_text（详细的用户画像描述）
    persona_text = persona.get('persona_text', '')
    
    # 提取关键特征
    key_traits = persona.get('key_traits', {})
    
    # 构建persona描述
    persona_description = f"""以下是你的用户画像信息，请根据这些特征来做出符合你个人偏好的行为决策：

{persona_text}

### 关键特征
- 内容vs创作者偏好: {key_traits.get('content_vs_creator', 'N/A')}
- 情绪调节倾向: {key_traits.get('emotion_regulation', 'N/A')}
- 新奇容忍度: {key_traits.get('novelty_tolerance', 'N/A')}
- 社交敏感度: {key_traits.get('social_sensitivity', 'N/A')}
"""
    return persona_description


def get_system_prompt(use_persona: bool = False, persona: dict = None) -> str:
    """获取system prompt"""
    # persona信息现在放在user message中，system只保留基础提示
    return "You are a helpful assistant."


def get_user_prompt(history_screenshots: list = None, history_actions: list = None, 
                    current_screenshots: list = None, audio_transcript: str = None,
                    persona: dict = None) -> str:
    """
    获取user prompt，与训练时完全一致
    
    Args:
        history_screenshots: 历史截图路径列表（每个历史视频一张）
        history_actions: 历史动作的代码字符串列表
        current_screenshots: 当前视频截图路径列表（支持多张）
        audio_transcript: 音频转录文本（可选）
        persona: 用户画像信息（可选），如果提供会添加到user prompt开头
    """
    # 构建persona提示（放在user message开头）
    persona_placeholder = ""
    if persona:
        persona_description = format_persona_for_system_prompt(persona)
        if persona_description:
            persona_placeholder = f"{persona_description}\n\n"
    
    # 构建历史提示
    history_placeholder = ""
    if history_screenshots and history_actions and len(history_screenshots) == len(history_actions):
        history_placeholder = "Your browsing history:\n"
        for str_action in history_actions:
            history_placeholder += "<image>\n" + str_action + "\n"
    
    # 构建音频文本提示
    audio_placeholder = ""
    if audio_transcript:
        audio_placeholder = f"\nVideo audio transcript:\n{audio_transcript}"
    
    # 构建当前视频提示（支持多张截图）
    if current_screenshots:
        current_video_placeholder = "<image>" * len(current_screenshots)
    else:
        current_video_placeholder = ""
    
    # 构建完整的user prompt（与训练时完全一致）
    user_prompt = (
        f"{persona_placeholder}"
        "You are a user who is happily enjoying short videos. "
        "You need to analyze the current video interface screenshot and make an appropriate action decision based on what you see. "
        "First, briefly describe the video content and its category, then respond with a Python code block calling the following functions:\n"
        "```python\n"
        "watch(second: float = 5.0) # Continue watching the video\n"
        "like() # Give a like to the video\n"
        "comment(text: str = \"\") # Leave a comment on the video\n"
        "share(who: str = \"\") # share/forward the video to someone\n"
        "```\n"
        "You can call multiple functions in a single code block to perform multiple actions.\n\n"
        "Response format (required):\n"
        "视频内容：[brief description]\n"
        "分类：[category]-[subcategory]\n\n"
        "```python\n"
        "[your action code]\n"
        "```\n\n"
        f"{history_placeholder}"
        f"Below is the video you are currently browsing:\n{current_video_placeholder}"
        f"{audio_placeholder}"
    )
    
    return user_prompt

