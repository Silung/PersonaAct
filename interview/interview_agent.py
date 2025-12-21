"""
Interview Agent - 基于 LangChain 的用户访谈 Agent
"""
import json
import re
from typing import Dict, List, Any, Tuple
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from behavior_analyzer import BehaviorAnalyzer

try:
    from duckduckgo_search import DDGS
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False
    print("Warning: duckduckgo_search not installed. Search functionality disabled.")


class InterviewAgent:
    """基于 LangChain 的访谈 Agent"""
    
    SYSTEM_PROMPT = """你是一个专业的用户研究员，正在进行深度访谈，目标是建模用户在短视频平台上的**偏好函数**和**决策机制**。

核心目标：不是给用户分类，而是理解用户的**心理驱动因素**和**行为决策逻辑**。

**⚠️ 重要：访谈是分环节进行的，每个环节有明确的目标和问题方向。你必须严格按照当前环节的目标提问，不要提前问其他环节的问题。**

访谈原则：
1. **严格按照环节提问并主动结束，避免重复提问**：
   - 当前环节会明确告诉你环节目标和关键问题方向
   - **系统会提供"已问过的问题列表"，你必须仔细检查这个列表**
   - **在提问前，必须检查已问过的问题列表和完整对话历史，如果某个问题已经在之前的环节问过并得到回答，不要重复提问**
   - **必须提出与已问过的问题明显不同的新问题**，如果无法提出新问题，请立即结束当前环节
   - 只能问与当前环节相关的问题，不要问其他环节的问题
   - **个人信息环节特别说明**：
     * ✅ 只能问：年龄范围、职业类型、生活状态（如学生、上班族等）
     * ❌ 禁止问：地址、城市、姓名、联系方式等敏感信息（匿名化要求）
     * ❌ 禁止问：内容喜好、使用场景、观看习惯等其他环节的问题
     * **一旦收集到年龄、职业、生活状态这三个基本信息，立即结束环节**
   - **如果已经收集到当前环节的主要信息（覆盖了关键问题方向），或者无法提出与已问过的问题明显不同的新问题，必须立即主动结束当前环节，不要继续问重复或无关的问题**

2. **基于数据提问（Data-Driven）**：
   - 必须基于用户的真实行为数据提问
   - 例如：看到用户频繁点赞某类内容 → 问"是什么让你决定给这类视频点赞？"
   - 例如：看到用户快速跳过某些视频 → 问"通常什么情况下你会在几秒内划走？"

3. **开放式深度提问**：
   - 使用"为什么"、"什么情况下"、"你觉得"等引导词
   - 不要问选择题（❌"你是不是喜欢XX？"）
   - 要问决策过程（✅"当你看到XX时，你通常是什么反应？为什么？"）

4. **探索心理机制，不只是表面偏好**：
   - 不只问"喜欢什么"，要问"为什么喜欢"、"什么驱动你"
   - 挖掘情境依赖性："在什么情况下会..."
   - 理解权衡机制："熟悉 vs 新奇"、"娱乐 vs 学习"

5. **自然对话**：
   - 像朋友聊天，不是审问
   - 如果用户问你问题，先回答再继续
   - 每次回复简短（不超过80字）

访谈维度（这些是后续环节可能探索的内容，当前环节会明确告诉你应该问什么）：

🎯 **基础信息收集（会在相应环节进行）**：
   - **个人信息**：只问年龄范围、职业类型、生活状态（如学生、上班族等）。**禁止问地址、城市、姓名、联系方式等敏感信息（匿名化要求）**。一旦收集到这三个基本信息，立即结束环节。
   - **内容喜好**：具体喜欢什么类型的内容？喜欢哪些作者（可以提问为什么喜欢某作者）？为什么喜欢？

🎯 其他可能探索的维度（在相应环节进行）：
   - 动机探索：什么驱动用户打开 App、继续观看？
   - 偏好机制：用户如何评价内容？关注什么维度？
   - 创作者依附：是否对特定作者有情感连接？
   - 互动决策：点赞/评论/分享背后的心理动因
   - 探索 vs 利用：熟悉内容 vs 新奇内容的权衡
   - 跳过策略：什么触发"划走"决策？
   - 情境依赖：不同场景下的行为差异
   - 现实关联：内容与现实生活的联系

**重要**：
1. 严格按照当前环节的目标和关键问题方向提问，不要提前问其他环节的问题
2. 每个环节都有明确的主题，必须在该环节内完成相应的信息收集
3. **个人信息环节特别要求**：
   - 只能问年龄范围、职业类型、生活状态（如学生、上班族等）
   - **禁止问地址、城市、姓名、联系方式等敏感信息（匿名化要求）**
   - **一旦收集到年龄、职业、生活状态这三个基本信息，立即结束环节**
4. **如果已经收集到当前环节的主要信息（覆盖了关键问题方向），必须立即主动结束当前环节，不要继续问重复或无关的问题**
5. 不要等到达到最大轮数才结束，效率很重要

当收集到足够信息后，生成用户画像。生成的人设应该是一段高信息密度的自述，必须包含个人信息和明确的偏好（如内容类型、作者等）。"""

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
        self.asked_questions: List[str] = []  # 已问过的问题列表
        
        # 搜索工具
        self.ddgs = DDGS() if SEARCH_AVAILABLE else None
        self.section_turn = 0
    
    def _format_image_for_vllm(self, image_path: str) -> Dict[str, str]:
        """格式化图片用于 vLLM - 使用本地文件路径"""
        absolute_path = Path(image_path).absolute()
        return {
            "type": "image_url",
            "image_url": {"url": f"file://{absolute_path}"}
        }
    
    def _search_web(self, query: str, max_results: int = 3) -> str:
        """使用 DuckDuckGo 搜索"""
        if not self.ddgs:
            return "搜索功能不可用（未安装 duckduckgo_search）"
        
        results = self.ddgs.text(query, max_results=max_results)
        if not results:
            return "没有找到相关信息"
        
        summary = []
        for i, r in enumerate(results, 1):
            title = r.get('title', '')
            body = r.get('body', '')[:200]
            summary.append(f"{i}. {title}\n{body}")
        
        return "\n\n".join(summary)
    
    def _parse_and_execute_search(self, text: str) -> Tuple[str, bool]:
        """解析文本中的搜索请求并执行
        
        格式: [SEARCH: 关键词]
        
        Returns:
            (处理后的文本, 是否执行了搜索)
        """
        # 匹配 [SEARCH: xxx] 格式
        pattern = r'\[SEARCH:\s*(.+?)\]'
        matches = re.findall(pattern, text)
        
        if not matches:
            return text, False
        
        # 执行所有搜索
        search_results = []
        for query in matches:
            query = query.strip()
            print(f"🔍 执行搜索: {query}")
            result = self._search_web(query, max_results=2)
            search_results.append(f"[搜索 '{query}' 的结果]\n{result}")
        
        # 移除搜索标记
        cleaned_text = re.sub(pattern, '', text).strip()
        
        # 拼接搜索结果
        all_results = "\n\n".join(search_results)
        
        return cleaned_text, True, all_results
        
    def initialize(self) -> Dict[str, Any]:
        """初始化访谈"""
        self.turn = 0
        self.current_persona = ""
        self.chat_history = []
        self.full_history = []
        self.current_section = 0
        self.section_turn = 0
        self.asked_questions = []  # 重置已问过的问题列表
        
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

**重要要求**：
1. 无论大纲内容如何，必须确保包含以下两个基础环节（如果大纲中没有，需要添加）：
   - **个人信息环节**：收集用户的基本背景信息（年龄范围、职业类型、生活状态等），但需要匿名化处理，不涉及具体姓名、地址等敏感信息
   - **喜好内容环节**：深入了解用户的具体喜好内容类型、偏好原因、价值取向等

2. 请提取访谈的主要阶段，生成一个 JSON 数组，每个阶段包含：
   - title: 阶段标题（简短，不超过10个字）
   - goal: 阶段目标（一句话）
   - max_turns: 建议对话轮数（默认10）
   - key_questions: 2-3个关键问题方向

只输出 JSON 数组，格式如下：
[
  {{"title": "个人信息", "goal": "了解用户基本背景（匿名化）", "max_turns": 8, "key_questions": ["年龄范围", "职业类型", "生活状态"]}},
  {{"title": "内容喜好", "goal": "深入了解具体喜好内容和偏好原因", "max_turns": 10, "key_questions": ["喜欢的内容类型", "偏好原因", "价值取向"]}},
  {{"title": "使用场景与动机", "goal": "明确刷视频的功能性角色", "max_turns": 10, "key_questions": ["打开App的触发条件", "刷视频的目的"]}}
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
            try:
                plan = json.loads(json_match.group())
                return plan
            except json.JSONDecodeError:
                # JSON 解析失败，使用默认计划
                pass
        
        # 解析失败，使用默认计划
        return self._generate_default_plan()
    
    def _generate_default_plan(self) -> List[Dict[str, Any]]:
        """生成默认访谈计划（降级方案）"""
        return [
            {"title": "个人信息", "goal": "了解用户基本背景（匿名化）", "max_turns": 8, "key_questions": ["年龄范围", "职业类型", "生活状态"]},
            {"title": "内容喜好", "goal": "深入了解具体喜好内容和偏好原因", "max_turns": 10, "key_questions": ["喜欢的内容类型", "偏好原因", "价值取向"]},
            {"title": "使用场景与动机", "goal": "明确刷视频的功能性角色", "max_turns": 3, "key_questions": ["打开App的触发条件", "刷视频的目的"]},
            {"title": "创作者依附", "goal": "判断content-driven还是creator-driven", "max_turns": 2, "key_questions": ["关注的创作者", "作者影响"]},
            {"title": "互动决策", "goal": "理解点赞/评论/分享动机", "max_turns": 2, "key_questions": ["互动行为", "动机"]},
            {"title": "探索与利用", "goal": "判断信息茧房倾向", "max_turns": 2, "key_questions": ["重复内容态度", "新内容接受度"]},
        ]
    
    def _generate_previous_sections_summary(self) -> str:
        """生成之前环节的信息摘要，用于避免重复提问"""
        if not self.interview_plan or self.current_section == 0:
            return ""
        
        # 提取之前环节的信息
        summary_parts = []
        for i in range(self.current_section):
            if i < len(self.interview_plan):
                section = self.interview_plan[i]
                summary_parts.append(f"- {section['title']}: {section['goal']}")
        
        if not summary_parts:
            return ""
        
        return "\n".join(summary_parts) + "\n\n⚠️ **重要**：以上是之前环节的主题。在提问前，请检查完整对话历史，如果某个问题已经在之前的环节问过并得到回答，不要重复提问。"
    
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
        
        # 生成已问过的问题列表（用于避免重复）
        asked_questions_text = ""
        if self.asked_questions:
            asked_questions_text = f"""
📋 **已问过的问题列表（请避免重复或相似的问题）**：
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(self.asked_questions)])}

⚠️ **重要**：请仔细检查上述问题列表。如果当前环节的关键问题方向已经通过这些问题得到了充分了解，或者你无法提出与上述问题**明显不同**的新问题，请在回复最后一行输出 "NEXT_SECTION" 来结束当前环节。
"""
        
        current_section_info = ""
        if self.interview_plan and self.current_section < len(self.interview_plan):
            section = self.interview_plan[self.current_section]
            current_section_info = f"""
当前访谈环节: {section['title']} (第{self.current_section + 1}/{len(self.interview_plan)}环节)
环节目标: {section['goal']}
关键问题方向: {', '.join(section['key_questions'])}

{asked_questions_text}

⚠️ **重要约束**：
1. **必须严格按照当前环节的目标和关键问题方向提问**，不要问其他环节的问题
2. **在提问前，必须检查已问过的问题列表和完整对话历史**，如果某个问题已经在之前的环节问过并得到回答，不要重复提问
3. **必须提出与已问过的问题明显不同的新问题**，如果无法提出新问题，请结束当前环节
4. 当前环节是"{section['title']}"，只能问与"{section['goal']}"相关的问题
5. **如果当前环节是"个人信息"**：
   - ✅ 只能问：年龄范围、职业类型、生活状态（如学生、上班族、自由职业等）
   - ❌ 禁止问：地址、具体城市、姓名、联系方式等敏感信息
   - ❌ 禁止问：内容喜好、使用场景、观看习惯等其他环节的问题
   - **一旦收集到年龄、职业、生活状态这三个基本信息，立即结束环节**
6. **如果当前环节是"内容喜好"**，才能问喜欢什么类型的内容、为什么喜欢等
7. 当前环节轮数: {self.section_turn}/{section['max_turns']}

🎯 **主动结束环节（非常重要）**：
- **如果已经收集到当前环节的主要信息（覆盖了关键问题方向：{', '.join(section['key_questions'])}），必须立即主动结束当前环节**
- **如果无法提出与已问过的问题明显不同的新问题，必须立即结束当前环节**
- 特别是"个人信息"环节：一旦收集到年龄、职业、生活状态，立即结束，不要问其他问题
- 不要等到达到最大轮数才结束，也不要问重复或无关的问题
- 如果已经收集到足够信息或无法提出新问题，请在回复最后一行输出 "NEXT_SECTION" 来结束当前环节
- 判断标准：
  1. 是否已经了解了关键问题方向中的主要信息？
  2. 是否还能提出与已问过的问题明显不同的新问题？
  如果两个问题的答案都是"否"，就立即结束
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
        
        # 使用完整历史对话，让agent能看到之前环节的内容，避免重复提问
        # 但为了控制token数量，只使用最近的相关对话
        history_to_use = self.full_history[-20:] if len(self.full_history) > 20 else self.full_history
        
        for msg in history_to_use:
            content = self._extract_text_content(msg["content"])
            if msg["role"] == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        
        # 调用 LLM
        response = self.llm.invoke(messages)
        result = response.content.strip()
        
        # 提取问题（如果回复中包含问题）
        # 尝试从回复中提取问题：通常是第一句话或问号结尾的句子
        question = None
        if "?" in result or "？" in result:
            # 提取第一个问句
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                # 跳过空行和标记行
                if not line or "NEXT_SECTION" in line:
                    continue
                if ("?" in line or "？" in line) and len(line) > 5:
                    # 提取问号前的部分作为问题
                    question = line.rstrip('?？。！').strip()
                    break
            # 如果没有找到，尝试提取第一行（可能是问题但没有问号）
            if not question and lines:
                first_line = lines[0].strip()
                # 跳过标记行
                if "NEXT_SECTION" not in first_line and len(first_line) > 5:
                    question = first_line.rstrip('。！').strip()
        else:
            # 即使没有问号，如果回复很短且像问题，也尝试提取
            lines = result.split('\n')
            if lines:
                first_line = lines[0].strip()
                # 跳过标记行
                if "NEXT_SECTION" not in first_line and len(first_line) > 5 and len(first_line) < 100:
                    # 如果第一行很短，可能是问题
                    question = first_line.rstrip('。！').strip()
        
        # 如果提取到问题，保存到已问过的问题列表
        if question and question not in self.asked_questions:
            self.asked_questions.append(question)
        
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
                # 清空当前环节历史，但保留完整历史和已问过的问题列表
                # 注意：不重置asked_questions，因为跨环节也可能有相似问题
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
        """获取第一个问题 - 基于第一个环节（个人信息）"""
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
        
        # 获取第一个环节信息
        first_section = None
        if self.interview_plan and len(self.interview_plan) > 0:
            first_section = self.interview_plan[0]
        
        # 构建环节信息
        section_info = ""
        goal_text = "了解用户基本信息"
        if first_section:
            section_info = f"""
当前环节: {first_section['title']}
环节目标: {first_section['goal']}
关键问题方向: {', '.join(first_section['key_questions'])}
"""
            goal_text = first_section['goal']
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"""这是访谈的开始。

{section_info}
用户短视频行为数据摘要:
{behavior_summary}

具体视频样本:
{video_context}

请基于当前环节的目标和关键问题方向，生成第一个访谈问题。要求：
1. 必须符合当前环节的目标：{goal_text}
2. 开放式深度提问，自然友好
3. 不超过80字

只输出问题。""")
        ]
        
        response = self.llm.invoke(messages)
        question = response.content.strip()
        # 保存第一个问题到已问过的问题列表
        if question and question not in self.asked_questions:
            self.asked_questions.append(question)
        self.chat_history.append({"role": "assistant", "content": question})
        self.full_history.append({"role": "assistant", "content": question})
        # 初始化 section_turn
        self.section_turn = 1
        return question
    
    def generate_section_question(self) -> str:
        """生成新环节的第一个问题"""
        if not self.interview_plan or self.current_section >= len(self.interview_plan):
            return "让我们继续聊聊吧。"
        
        behavior_summary = self.analyzer.get_behavior_summary()
        current_section = self.interview_plan[self.current_section]
        
        # 生成已问过的问题列表（用于避免重复）
        asked_questions_text = ""
        if self.asked_questions:
            asked_questions_text = f"""
📋 **已问过的问题列表（请避免重复或相似的问题）**：
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(self.asked_questions)])}

⚠️ **重要**：请仔细检查上述问题列表，确保新问题与已问过的问题明显不同。
"""
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"""现在进入新的访谈环节。

当前环节: {current_section['title']}
环节目标: {current_section['goal']}
关键问题方向: {', '.join(current_section['key_questions'])}

{asked_questions_text}
用户短视频行为数据摘要:
{behavior_summary}

请基于当前环节的目标，生成一个开场问题。要求：
1. 开放式深度提问
2. 自然友好，像朋友聊天
3. 不超过80字
4. 确保与已问过的问题明显不同

只输出问题。""")
        ]
        
        response = self.llm.invoke(messages)
        question = response.content.strip()
        # 保存问题到已问过的问题列表
        if question and question not in self.asked_questions:
            self.asked_questions.append(question)
        self.chat_history.append({"role": "assistant", "content": question})
        self.full_history.append({"role": "assistant", "content": question})
        # 更新 section_turn，因为这是新环节的第一个问题
        self.section_turn = 1
        return question
    
    def _extract_text_content(self, msg_content):
        """提取消息的文本内容"""
        if isinstance(msg_content, str):
            return msg_content
        if isinstance(msg_content, list) and msg_content:
            return msg_content[0].get("text", str(msg_content))
        return str(msg_content)
    
    def generate_final_persona(self) -> str:
        """生成最终人设 - 一段高信息密度的话"""
        # 使用完整历史而不是当前环节历史
        conversation_summary = "\n".join([
            f"{'用户' if msg['role'] == 'user' else 'AI'}: {self._extract_text_content(msg['content'])}"
            for msg in self.full_history
        ])
        
        behavior_summary = self.analyzer.get_behavior_summary()
        
        prompt = f"""基于以下信息，生成一段高信息密度的用户自述（第一人称），要求：

用户短视频行为数据:
{behavior_summary}

完整访谈对话（所有环节）:
{conversation_summary}

**必须包含的内容（缺一不可）**：
1. **个人信息**：年龄范围、职业类型、生活状态等（匿名化，不涉及具体姓名、地址）
2. **明确偏好**：具体喜欢的内容类型、喜欢的作者/创作者（如果有），以及为什么喜欢
3. **使用场景**：在什么情况下刷视频，刷视频的目的
4. **行为特征**：结合行为数据，描述具体的观看习惯、互动方式等

**写作要求**：
1. **高信息密度**：只写有用的、具体的信息，没用的、空泛的内容不要写
2. **第一人称自述**：使用"我"来叙述，自然流畅，像用户在自述
3. **具体细节**：必须包含具体的行为细节、真实场景、明确偏好（如具体的内容类型、作者名称等）
4. **有依据**：结合访谈对话和行为数据，给出有依据的描述
5. **避免空泛**：不要写"一般"、"通常"、"可能"等模糊词汇，不要写"我喜欢看视频"、"我会点赞"等无信息量的套话
6. **一段话**：写成一段连贯的话，不要分点、不要用Markdown格式，直接输出文本
7. **字数**：300-500字，确保信息密度高
8. **⚠️ 重要：不要使用具体数据**：
   - ❌ 禁止使用：具体数字（如"点赞率30%"、"观看5次"、"15-20分钟"、"10秒"等）
   - ❌ 禁止使用：百分比、次数、时长等精确数值
   - ✅ 应该使用：描述性语言（如"点赞率较高/较低"、"经常/偶尔观看"、"观看时间较长/较短"、"很快划走"等）
   - ✅ 应该使用：程度词（如"特别"、"比较"、"偶尔"、"经常"、"很少"等）

**示例（好的写法）**：
"我是一名25-30岁的互联网产品经理，平时工作压力较大，主要在通勤路上和睡前刷短视频。我特别喜欢知识类内容，尤其是产品分析、商业案例和编程教程，经常关注'产品经理老王'和'商业洞察'这两个创作者，因为他们的内容既有深度又实用，能帮助我提升工作能力。我每次观看时间都比较长，当看到有价值的知识类视频时会完整看完并点赞，点赞频率比较高，而纯娱乐类视频我通常只看几秒就划走。我很少评论，但会收藏一些特别有用的内容。相比热门内容，我更关注内容质量，即使作者粉丝不多，只要内容好我也会看完。"

**示例（错误的写法 - 包含具体数据）**：
"我点赞率约30%，平均每次观看15-20分钟，喜欢某类内容5次..." ❌

直接输出自述文本，不要其他格式。"""
        
        messages = [
            SystemMessage(content="你是一个专业的用户研究员，擅长生成高信息密度的用户画像。"),
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
        content_prefs = analysis['content_preferences']
        
        # 基于行为数据推断的特征
        traits = {}
        
        # 内容 vs 创作者驱动 - 使用观看时长判断更准确
        most_watched_by_duration = creators.get('most_watched_by_duration', [])
        if most_watched_by_duration and len(most_watched_by_duration) > 2:
            # 如果观看时长最多的前3个创作者的总时长占比很高，说明是creator-driven
            top3_duration = sum(duration for _, duration in most_watched_by_duration[:3])
            total_duration = creators.get('total_watch_duration', 0)
            if total_duration > 0 and top3_duration / total_duration > 0.3:  # 前3个创作者占比超过30%
                traits['content_vs_creator'] = 'creator-driven'
            else:
                traits['content_vs_creator'] = 'content-driven'
        elif creators.get('most_liked_creators') and len(creators['most_liked_creators']) > 2:
            # 降级到使用点赞数据
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
            "behavior_stats": {
                "like_rate": interaction['like_rate'],
                "avg_watch_duration": interaction['avg_watch_duration'],
                "quick_skip_rate": interaction['quick_skip_rate'],
                "top_categories": content_prefs['top_main_categories'][:3],
            }
        }
