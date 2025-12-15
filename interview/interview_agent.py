"""
Interview Agent - 基于 LangChain 的用户访谈 Agent
"""
import json
import re
from typing import Dict, List, Any
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from behavior_analyzer import BehaviorAnalyzer


class InterviewAgent:
    """基于 LangChain 的访谈 Agent"""
    
    SYSTEM_PROMPT = """你是一个专业的用户研究员，正在进行深度访谈，目标是建模用户在短视频平台上的**偏好函数**和**决策机制**。

核心目标：不是给用户分类，而是理解用户的**心理驱动因素**和**行为决策逻辑**。

访谈原则：
1. **基于数据提问（Data-Driven）**：
   - 必须基于用户的真实行为数据提问
   - 例如：看到用户频繁点赞某类内容 → 问"是什么让你决定给这类视频点赞？"
   - 例如：看到用户快速跳过某些视频 → 问"通常什么情况下你会在几秒内划走？"

2. **开放式深度提问**：
   - 使用"为什么"、"什么情况下"、"你觉得"等引导词
   - 不要问选择题（❌"你是不是喜欢XX？"）
   - 要问决策过程（✅"当你看到XX时，你通常是什么反应？为什么？"）

3. **探索心理机制，不只是表面偏好**：
   - 不只问"喜欢什么"，要问"为什么喜欢"、"什么驱动你"
   - 挖掘情境依赖性："在什么情况下会..."
   - 理解权衡机制："熟悉 vs 新奇"、"娱乐 vs 学习"

4. **自然对话**：
   - 像朋友聊天，不是审问
   - 如果用户问你问题，先回答再继续
   - 每次回复简短（不超过80字）

访谈维度（按优先级探索）：

🎯 1. 动机探索（Motivation）
   - 什么驱动用户打开 App、继续观看？

🎯 2. 偏好机制（Preference Criteria）
   - 用户如何评价内容？关注什么维度？

🎯 3. 创作者依附（Creator Affinity）
   - 是否对特定作者有情感连接？

🎯 4. 互动决策（Interaction Strategy）
   - 点赞/评论/分享背后的心理动因

🎯 5. 探索 vs 利用（Exploration-Exploitation）
   - 熟悉内容 vs 新奇内容的权衡

🎯 6. 跳过策略（Disengagement）
   - 什么触发"划走"决策？

🎯 7. 情境依赖（Contextual Behavior）
   - 不同场景下的行为差异

🎯 8. 现实关联（Meaningful Engagement）
   - 内容与现实生活的联系

当收集到足够信息后，生成用户画像。"""

    def __init__(
        self,
        data_dir: str = "data/yqg",
        raw_data_dir: str = "raw_data/yqg",
        api_base: str = "http://127.0.0.1:8012/v1",
        api_key: str = "1234567890",
        model: str = "qwen"
    ):
        self.data_dir = data_dir
        self.raw_data_dir = raw_data_dir
        self.analyzer = BehaviorAnalyzer(data_dir, raw_data_dir)
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            base_url=api_base,
            api_key=api_key,
            model=model,
            temperature=0.7,
        )
        
        # 状态
        self.turn = 0
        self.current_persona = ""
        self.chat_history: List[Dict[str, str]] = []  # 当前环节的历史
        self.full_history: List[Dict[str, str]] = []  # 完整历史（所有环节）
        self.interview_plan: List[Dict[str, Any]] = []
        self.current_section = 0
        self.section_turn = 0
    
    def _format_image_for_vllm(self, image_path: str) -> Dict[str, str]:
        """格式化图片用于 vLLM - 使用本地文件路径"""
        absolute_path = Path(image_path).absolute()
        return {
            "type": "image_url",
            "image_url": {"url": f"file://{absolute_path}"}
        }
        
    def initialize(self) -> Dict[str, Any]:
        """初始化访谈"""
        self.turn = 0
        self.current_persona = ""
        self.chat_history = []
        self.full_history = []
        self.current_section = 0
        self.section_turn = 0
        
        behavior_summary = self.analyzer.get_behavior_summary()
        analysis = self.analyzer.analyze_all()
        self.interview_plan = self._generate_plan_from_outline()
        
        return {
            "behavior_summary": behavior_summary,
            "analysis": analysis,
            "interview_plan": self.interview_plan,
        }
    
    def _load_outline(self) -> str:
        """加载访谈大纲"""
        outline_path = Path(__file__).parent / "outline.md"
        if outline_path.exists():
            with open(outline_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def _generate_plan_from_outline(self) -> List[Dict[str, Any]]:
        """从 outline.md 生成访谈计划"""
        outline_content = self._load_outline()
        
        if not outline_content:
            # 降级到默认计划
            return self._generate_default_plan()
        
        prompt = f"""请从以下访谈大纲中提取访谈阶段信息。

访谈大纲:
{outline_content}

请提取访谈的主要阶段，生成一个 JSON 数组，每个阶段包含：
- title: 阶段标题（简短，不超过10个字）
- goal: 阶段目标（一句话）
- max_turns: 建议对话轮数（默认10）
- key_questions: 2-3个关键问题方向

只输出 JSON 数组，格式如下：
[
  {{"title": "使用场景与动机", "goal": "明确刷视频的功能性角色", "max_turns": 10, "key_questions": ["打开App的触发条件", "刷视频的目的"]}},
  {{"title": "内容偏好", "goal": "理解内容偏好原因", "max_turns": 10, "key_questions": ["喜欢的内容类型", "吸引点"]}}
]"""
        
        messages = [
            SystemMessage(content="你是一个专业的访谈设计专家，擅长从文档中提取结构化信息。"),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        content = response.content.strip()
        
        # 提取 JSON
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group())
            return plan
        
        # 解析失败，使用默认计划
        return self._generate_default_plan()
    
    def _generate_default_plan(self) -> List[Dict[str, Any]]:
        """生成默认访谈计划（降级方案）"""
        return [
            {"title": "使用场景与动机", "goal": "明确刷视频的功能性角色", "max_turns": 3, "key_questions": ["打开App的触发条件", "刷视频的目的"]},
            {"title": "内容偏好", "goal": "理解内容偏好原因", "max_turns": 3, "key_questions": ["喜欢的内容类型", "吸引点"]},
            {"title": "创作者依附", "goal": "判断content-driven还是creator-driven", "max_turns": 2, "key_questions": ["关注的创作者", "作者影响"]},
            {"title": "互动决策", "goal": "理解点赞/评论/分享动机", "max_turns": 2, "key_questions": ["互动行为", "动机"]},
            {"title": "探索与利用", "goal": "判断信息茧房倾向", "max_turns": 2, "key_questions": ["重复内容态度", "新内容接受度"]},
        ]
    
    def get_current_progress(self) -> Dict[str, Any]:
        """获取当前访谈进度"""
        if not self.interview_plan:
            return {"current": 0, "total": 0, "section": "初始化中"}
        
        current_section = self.interview_plan[self.current_section] if self.current_section < len(self.interview_plan) else None
        
        return {
            "current": self.current_section + 1,
            "total": len(self.interview_plan),
            "section": current_section['title'] if current_section else "完成",
            "section_turn": self.section_turn,
            "section_max_turns": current_section['max_turns'] if current_section else 0,
            "goal": current_section['goal'] if current_section else "",
        }
    
    def chat(self, user_input: str, image_paths: List[str] = None) -> Dict[str, Any]:
        """执行一轮对话
        
        Returns:
            Dict with keys:
            - response: AI 回复内容
            - section_changed: 是否切换到下一环节
            - section_info: 当前环节信息
        """
        self.turn += 1
        self.section_turn += 1
        
        # 记录用户输入（支持多模态）
        user_msg = {"role": "user", "content": user_input}
        if image_paths:
            content = [{"type": "text", "text": user_input}]
            for img_path in image_paths:
                content.append(self._format_image_for_vllm(img_path))
            user_msg["content"] = content
        
        self.chat_history.append(user_msg)
        self.full_history.append(user_msg)
        
        # 构建系统消息
        behavior_summary = self.analyzer.get_behavior_summary()
        representative_videos = self.analyzer.get_representative_videos(max_samples=3)
        
        video_examples = []
        for i, video in enumerate(representative_videos, 1):
            cats = video.get('categories', [])
            categories = ', '.join([c.get('main', '') for c in cats[:2]]) or '未分类'
            duration = video.get('viewing_duration', 0)
            transcript = video.get('transcript', '')[:80]
            actions = ', '.join([a.get('type', '') for a in video.get('actions', [])])
            video_desc = f"视频{i}: {categories}, 观看{duration:.1f}秒, {actions if actions else '无互动'}"
            if transcript:
                video_desc += f", 内容:{transcript}"
            video_examples.append(video_desc)
        
        video_context = "\n".join(video_examples) if video_examples else ""
        
        current_section_info = ""
        if self.interview_plan and self.current_section < len(self.interview_plan):
            section = self.interview_plan[self.current_section]
            current_section_info = f"""
当前访谈环节: {section['title']} (第{self.current_section + 1}/{len(self.interview_plan)}环节)
环节目标: {section['goal']}
关键问题方向: {', '.join(section['key_questions'])}
当前环节轮数: {self.section_turn}/{section['max_turns']}

⚠️ 如果你认为已经收集到足够信息，请在回复最后一行输出 "NEXT_SECTION" 来结束当前环节。
"""
        
        system_content = f"""{self.SYSTEM_PROMPT}

用户短视频行为数据摘要:
{behavior_summary}

具体视频样本:
{video_context}
{current_section_info}
当前是第 {self.turn} 轮对话。

回复要自然、简短（不超过80字）。"""
        
        messages = [SystemMessage(content=system_content)]
        
        # 只使用当前环节的历史对话
        for msg in self.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # 调用 LLM
        response = self.llm.invoke(messages)
        result = response.content.strip()
        
        # 检查是否包含 [NEXT_SECTION] 标记或达到最大轮数
        section_changed = False
        auto_ended = False
        
        if "NEXT_SECTION" in result:
            # 移除标记
            result = result.replace("NEXT_SECTION", "").strip()
            section_changed = True
        elif self.interview_plan and self.current_section < len(self.interview_plan):
            # 检查是否达到最大轮数
            section = self.interview_plan[self.current_section]
            if self.section_turn >= section['max_turns']:
                section_changed = True
                auto_ended = True
        
        # 记录回复
        assistant_msg = {"role": "assistant", "content": result}
        self.chat_history.append(assistant_msg)
        self.full_history.append(assistant_msg)
        
        # 判断是否完成所有环节
        interview_completed = False
        
        # 如果需要切换环节
        if section_changed:
            if self.current_section < len(self.interview_plan) - 1:
                # 还有下一个环节
                self.current_section += 1
                self.section_turn = 0
                # 清空当前环节历史，但保留完整历史
                self.chat_history = []
            else:
                # 已经是最后一个环节，访谈完成
                interview_completed = True
        
        return {
            "response": result,
            "section_changed": section_changed,
            "auto_ended": auto_ended,
            "interview_completed": interview_completed,
            "section_info": self.get_current_progress()
        }
    
    def get_first_question(self) -> str:
        """获取第一个问题"""
        behavior_summary = self.analyzer.get_behavior_summary()
        representative_videos = self.analyzer.get_representative_videos(max_samples=3)
        
        video_examples = []
        for i, video in enumerate(representative_videos, 1):
            cats = video.get('categories', [])
            categories = ', '.join([c.get('main', '') for c in cats[:2]]) or '未分类'
            duration = video.get('viewing_duration', 0)
            transcript = video.get('transcript', '')[:100]
            video_desc = f"视频{i}: {categories}, 观看{duration:.1f}秒, 内容:{transcript if transcript else '无转录'}"
            video_examples.append(video_desc)
        
        video_context = "\n".join(video_examples) if video_examples else "暂无具体视频样本"
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"""这是访谈的开始。

用户短视频行为数据摘要:
{behavior_summary}

具体视频样本:
{video_context}

请基于行为数据生成第一个访谈问题。要求：开放式深度提问，自然友好，不超过80字。

只输出问题。""")
        ]
        
        response = self.llm.invoke(messages)
        question = response.content.strip()
        self.chat_history.append({"role": "assistant", "content": question})
        self.full_history.append({"role": "assistant", "content": question})
        return question
    
    def generate_section_question(self) -> str:
        """生成新环节的第一个问题"""
        if not self.interview_plan or self.current_section >= len(self.interview_plan):
            return "让我们继续聊聊吧。"
        
        behavior_summary = self.analyzer.get_behavior_summary()
        current_section = self.interview_plan[self.current_section]
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"""现在进入新的访谈环节。

当前环节: {current_section['title']}
环节目标: {current_section['goal']}
关键问题方向: {', '.join(current_section['key_questions'])}

用户短视频行为数据摘要:
{behavior_summary}

请基于当前环节的目标，生成一个开场问题。要求：
1. 开放式深度提问
2. 自然友好，像朋友聊天
3. 不超过80字

只输出问题。""")
        ]
        
        response = self.llm.invoke(messages)
        question = response.content.strip()
        self.chat_history.append({"role": "assistant", "content": question})
        self.full_history.append({"role": "assistant", "content": question})
        return question
    
    def generate_final_persona(self) -> str:
        """生成最终人设 - 使用完整历史"""
        # 使用完整历史而不是当前环节历史
        conversation_summary = "\n".join([
            f"{'用户' if msg['role'] == 'user' else 'AI'}: {msg['content'] if isinstance(msg['content'], str) else msg['content'][0]['text']}"
            for msg in self.full_history
        ])
        
        behavior_summary = self.analyzer.get_behavior_summary()
        
        prompt = f"""基于以下信息，生成一份详细的用户画像（Persona）。

用户短视频行为数据:
{behavior_summary}

完整访谈对话（所有环节）:
{conversation_summary}

请生成一份结构化的用户画像，包含以下维度（每个维度2-3句话）：

## 📱 使用动机与场景
- 为什么刷短视频？在什么场景下使用？

## 🎯 内容偏好机制
- 喜欢什么类型的内容？评价内容的标准是什么？

## 👤 创作者关系
- 对创作者的依附程度？content-driven 还是 creator-driven？

## 💬 互动决策逻辑
- 什么情况下点赞/评论/分享？互动背后的心理动因？

## 🔄 探索与利用策略
- 对熟悉内容 vs 新奇内容的态度？信息茧房倾向？

## ⏭️ 跳过与停留决策
- 什么触发"划走"？什么让你继续看？

## 🎭 情境依赖性
- 不同时间/心情下的行为差异？

## 💭 情绪与现实关联
- 内容对情绪的影响？与现实生活的联系？

要求：
1. 使用第一人称"我"，自然流畅
2. 每个维度具体、有细节，避免空泛
3. 基于访谈内容和行为数据，不要编造
4. 总字数400-600字

直接输出画像，使用 Markdown 格式。"""
        
        messages = [
            SystemMessage(content="你是一个专业的用户研究员，擅长生成用户画像。"),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        persona = response.content.strip()
        self.current_persona = persona
        return persona
    
    def get_current_persona(self) -> str:
        """获取当前 Persona"""
        return self.current_persona if self.current_persona else "收集中..."
    
    def get_structured_persona(self) -> Dict[str, Any]:
        """获取结构化 Persona"""
        analysis = self.analyzer.analyze_all()
        interaction = analysis['interaction_patterns']
        creators = analysis['creator_preferences']
        engagement = analysis['engagement_metrics']
        
        traits = {}
        
        # 内容 vs 创作者驱动
        if creators['most_liked_creators'] and len(creators['most_liked_creators']) > 2:
            traits['content_vs_creator'] = 'creator-driven'
        else:
            traits['content_vs_creator'] = 'content-driven'
        
        # 情绪调节
        traits['emotion_regulation'] = 'high' if interaction['avg_watch_duration'] > 10 else 'low'
        
        # 新内容接受度
        if interaction['quick_skip_rate'] < 0.2:
            traits['novelty_tolerance'] = 'high'
        elif interaction['quick_skip_rate'] > 0.4:
            traits['novelty_tolerance'] = 'low'
        else:
            traits['novelty_tolerance'] = 'medium'
        
        # 社交敏感度
        traits['social_sensitivity'] = 'high' if engagement.get('prefers_popular') else 'moderate'
        
        return {
            "persona_text": self.current_persona,
            "key_traits": traits,
            "interview_turns": self.turn,
        }
