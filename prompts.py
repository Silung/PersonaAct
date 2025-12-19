"""训练时使用的prompt模板，需要与训练时完全一致"""

def format_persona_for_system_prompt(persona: dict) -> str:
    """将persona信息格式化为system prompt的一部分"""
    if not persona:
        return ""
    
    # 提取persona_text（详细的用户画像描述）
    persona_text = persona.get('persona_text', '')
    
    # 提取关键特征
    key_traits = persona.get('key_traits', {})
    
    # 提取大五人格
    big_five = persona.get('big_five_personality', {})
    
    # 构建persona描述
    persona_description = f"""以下是你的用户画像信息，请根据这些特征来做出符合你个人偏好的行为决策：

{persona_text}

### 关键特征
- 内容vs创作者偏好: {key_traits.get('content_vs_creator', 'N/A')}
- 情绪调节倾向: {key_traits.get('emotion_regulation', 'N/A')}
- 新奇容忍度: {key_traits.get('novelty_tolerance', 'N/A')}
- 社交敏感度: {key_traits.get('social_sensitivity', 'N/A')}

### 大五人格评分
- 开放性: {big_five.get('openness', {}).get('score', 'N/A')}/5
- 尽责性: {big_five.get('conscientiousness', {}).get('score', 'N/A')}/5
- 外向性: {big_five.get('extraversion', {}).get('score', 'N/A')}/5
- 宜人性: {big_five.get('agreeableness', {}).get('score', 'N/A')}/5
- 神经质: {big_five.get('neuroticism', {}).get('score', 'N/A')}/5
"""
    return persona_description


def get_system_prompt(use_persona: bool = False, persona: dict = None) -> str:
    """获取system prompt"""
    base_prompt = "You are a helpful assistant."
    
    if use_persona and persona:
        persona_prompt = format_persona_for_system_prompt(persona)
        if persona_prompt:
            return f"{base_prompt}\n\n{persona_prompt}"
    
    return base_prompt


def get_user_prompt(history_screenshots: list = None, history_actions: list = None, 
                    current_screenshot: str = None, audio_transcript: str = None) -> str:
    """
    获取user prompt，与训练时完全一致
    
    Args:
        history_screenshots: 历史截图路径列表（每个历史视频一张）
        history_actions: 历史动作的代码字符串列表
        current_screenshot: 当前视频截图路径
        audio_transcript: 音频转录文本（可选）
    """
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
    
    # 构建当前视频提示
    current_video_placeholder = "<image>" if current_screenshot else ""
    
    # 构建完整的user prompt（与训练时完全一致）
    user_prompt = (
        "You are a user who is happily enjoying short videos. "
        "You need to analyze the current video interface screenshot and make an appropriate action decision based on what you see. "
        "Respond strictly with a Python code block (starting with ```python) calling the following functions:\n"
        "```python\n"
        "watch(second: float = 2.0) # Continue watching the video\n"
        "like() # Give a like to the video\n"
        "comment(text: str = \"\") # Leave a comment on the video\n"
        "share(who: str = \"\") # share/forward the video to someone\n"
        "```\n"
        "You can call multiple functions in a single code block to perform multiple actions.\n"
        f"{history_placeholder}"
        f"Below is the video you are currently browsing:\n{current_video_placeholder}"
        f"{audio_placeholder}"
    )
    
    return user_prompt

