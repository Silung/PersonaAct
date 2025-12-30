"""
奖励函数插件（拆分为三个可配置的 reward class）
用于 GRPO 训练的自定义奖励函数：VideoQualityReward, PersonaReward, DiversityReward
"""

import re
import json
import os
from typing import List, Dict, Any

from swift.plugin import ORM, orms, rm_plugins

# ---------------------------
# 模块级辅助：读取 analysis_data、动作解析、 video quality 计算
# ---------------------------

DATA_PATH = "data/analysis_data.json"
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        analysis_data = json.load(f)
else:
    analysis_data = {}

_ACTION_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)


def parse_actions_from_completion(completion: str) -> List[str]:
    """从 completion 的代码块中安全解析并“执行”特定动作函数（watch/skip/like/comment/share）以返回动作列表"""
    recorded_actions: List[str] = []
    if not completion:
        return recorded_actions
    match = _ACTION_RE.search(completion)
    if not match:
        return recorded_actions
    code = match.group(1).strip()
    if not code:
        return recorded_actions

    def watch(second: float = 5.0):
        recorded_actions.append({'type': 'watch', 'second': second})

    def skip():
        recorded_actions.append({'type': 'skip'})

    def like():
        recorded_actions.append({'type': 'like'})

    def comment(text: str = ""):
        recorded_actions.append({'type': 'comment', 'text': text})

    def share(who: str = ""):
        recorded_actions.append({'type': 'share', 'text': who})

    safe_globals = {
        "__builtins__": {},
        "watch": watch,
        "skip": skip,
        "like": like,
        "comment": comment,
        "share": share,
    }

    try:
        exec(code, safe_globals, {})
    except Exception:
        # 忽略执行错误，返回已记录的动作（可能为空）
        pass

    return recorded_actions


def calc_video_quality_score(like: int = 0, comment: int = 0, favorite: int = 0, share: int = 0) -> float:
    """
    归一化的视频质量评分（0..1）
    与原逻辑相似：like*1 + comment*2 + favorite*1.5 + share*1.2，除以常数并裁剪
    """
    raw = like * 1.0 + comment * 2.0 + favorite * 1.5 + share * 1.2
    # 假定 10000 为“高分标尺”，并裁剪到 1.0
    score = min(raw / 10000.0, 1.0)
    return score

# ---------------------------
# VideoQualityReward
# ---------------------------


class VideoQualityReward(ORM):
    """
    视频质量奖励
    参数：
      - invert (bool): 如果 True，则反转质量偏好（喜欢低质量或少人看的视频）
    """
    def __init__(self, invert: bool = False):
        super().__init__()
        self.invert = invert

    def _score_for_vid(self, vid: str) -> float:
        entry = analysis_data.get(vid, {}) if vid else {}
        like = int(entry.get("like", 0) or 0)
        comment = int(entry.get("comment", 0) or 0)
        favorite = int(entry.get("favorite", 0) or 0)
        share = int(entry.get("share", 0) or 0)
        quality = calc_video_quality_score(like, comment, favorite, share)
        # invert 表示偏好低质量/小众时把 combined 反转
        if self.invert:
            quality = 1.0 - quality
        # 裁剪
        quality = max(0.0, min(1.0, quality))
        return quality

    def __call__(self, completions: List[str], metadata: List[Dict] = None, **kwargs) -> List[float]:
        rewards: List[float] = []
        for i, completion in enumerate(completions):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            viewed = meta.get("viewed", [])
            if not viewed:
                # 没有历史，返回 0
                rewards.append(0.0)
                continue
            curr_vid = viewed[-1]
            reward = self._score_for_vid(curr_vid)
            rewards.append(float(reward))
        return rewards


# ---------------------------
# PersonaReward
# ---------------------------


class PersonaReward(ORM):
    """
    基于“人设/兴趣”的奖励
    参数：
      - likes (List[str]): 明确喜欢的主类别列表
      - dislikes (List[str]): 明确不喜欢的主类别列表
      - action_weights (dict): 各动作的基准 reward（会乘以 quality 或其他因子）
    """
    def __init__(
        self,
        likes: List[str] = None,
        dislikes: List[str] = None,
        action_weights: Dict[str, float] = None,
        allow_partial_watch: bool = True
    ):
        super().__init__()
        self.likes = likes or ["科技数码", "人文艺术", "二次元", "游戏", "生活记录"]
        self.dislikes = dislikes or ["时尚", "宠物", "体育运动"]
        # 默认动作权重（可覆盖）
        self.action_weights = action_weights or {
            "watch": 0.2,
            "like": 0.5,
            "comment": 0.7,
            "share": 0.3,
            "skip": -0.2
        }

    def __call__(self, completions: List[str], metadata: List[Dict] = None, **kwargs) -> List[float]:
        rewards: List[float] = []
        for i, completion in enumerate(completions):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            viewed = meta.get("viewed", [])
            if not viewed:
                rewards.append(0.0)
                continue
            curr_vid = viewed[-1]
            entry = analysis_data.get(curr_vid, {})
            curr_main_category = entry["category"]["main_category"]

            actions = parse_actions_from_completion(completion)
            # 计算一个基础动作分（sum of weights for actions happened）
            action_score = 0.0
            for a in actions:
                action_score += float(self.action_weights.get(a, 0.0))

            # 若没有动作（比如空或无法解析），给予小负激励以鼓励明确动作
            if not actions:
                action_score -= 0.5

            reward = 0.0

            # 视频质量因子（用于放大/缩小对合理行为的奖励）
            quality = calc_video_quality_score(
                int(entry.get("like", 0) or 0),
                int(entry.get("comment", 0) or 0),
                int(entry.get("favorite", 0) or 0),
                int(entry.get("share", 0) or 0),
            )

            # 根据当前类别判定合理行为（保留原有规则思路，但使用更可配置的 action_weights）
            if curr_main_category in self.likes:
                # 喜欢类别：watch 是合理的主要动作
                if actions == ["watch"]:
                    reward += 0.5 + 0.2 * quality
                elif set(actions).issubset({"watch", "like", "share", "comment"}) and ("watch" in actions):
                    # 高质量视频允许更多互动
                    if quality > 0.5:
                        reward += 0.5 + 0.7 * quality
                    else:
                        reward -= 0.5
                elif "skip" in actions:
                    reward -= 1.0
                elif not ("watch" in actions):
                    # 只有 like/comment/share 没有 watch，视为不合理
                    reward -= 0.8
                else:
                    reward -= 0.5
            elif curr_main_category in self.dislikes:
                # 不喜欢类别，skip 最合理
                if actions == ["skip"]:
                    reward += 1.0
                else:
                    reward -= 1.0
            else:
                # 中性或未知类别：watch 为最好，skip 次之，其他小正向
                if actions == ["watch"]:
                    reward += 1.0
                elif "skip" in actions:
                    reward += 0.5
                else:
                    reward += 0.1

            # 把动作得分融合进来（动作权重可以放大小）
            reward += 0.5 * action_score

            # 最后裁剪并返回
            rewards.append(float(reward))
        return rewards


# ---------------------------
# DiversityReward
# ---------------------------


class DiversityReward(ORM):
    """
    多样性奖励（统计历史窗口内主类别数量）
    参数：
      - diversity_weight (float): 多样性分数乘数（正数表示鼓励多样性）
      - niche_pref (float in [0,1]): 专一偏好度，1.0 表示强偏好专一（会惩罚多样性），0.0 表示强偏好广泛（奖励多样性）
        实际奖励计算为 (n_unique_categories * 0.2) * (1 - 2*niche_pref) 。（你可以调整公式）
      - cap (float): 多样性分的上限
    """
    def __init__(self, diversity_weight: float = 1.0, niche_pref: float = 0.0, cap: float = 1.0):
        super().__init__()
        self.diversity_weight = float(diversity_weight)
        # 限定 niche_pref 在 [0,1]
        self.niche_pref = min(1.0, max(0.0, float(niche_pref)))
        self.cap = float(cap)

    def __call__(self, completions: List[str], metadata: List[Dict] = None, **kwargs) -> List[float]:
        rewards: List[float] = []
        for i, completion in enumerate(completions):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            viewed = meta.get("viewed", [])
            unique_cats = set()
            for vid in viewed:
                entry = analysis_data.get(vid, {})
                main_cat = None
                try:
                    if isinstance(entry.get("category"), dict):
                        main_cat = entry["category"].get("main_category")
                    else:
                        main_cat = entry.get("category")
                except Exception:
                    main_cat = None
                if main_cat:
                    unique_cats.add(main_cat)

            n_unique = len(unique_cats)
            # 基础多样性分：线性增长，但 capped
            base_div = min(n_unique * 0.2, self.cap)  # 与之前逻辑相似
            # niche_pref: 0 -> 偏好广泛（正向），1 -> 偏好专一（负向）
            # 设计：multiplier = (1 - 2*niche_pref)  => niche_pref=0 => +1, niche_pref=0.5 => 0, niche_pref=1 => -1
            multiplier = 1.0 - 2.0 * self.niche_pref
            diversity_score = base_div * multiplier * self.diversity_weight
            rewards.append(float(diversity_score))
        return rewards


# ---------------------------
# FormatReward
# ---------------------------
class FormatReward(ORM):
    """
    格式奖励：检查输出是否包含正确的 Python 代码块格式并且可以正常解析
    奖励规则：
      - 包含 ```python ... ``` 或 ``` ... ``` 代码块且能成功解析出动作：+1.0
      - 有代码块但无法解析出有效动作：0.0
      - 无代码块：0.0
    """
    def __init__(self):
        super().__init__()

    def __call__(self, completions: List[str], solution: List[str] = None, **kwargs) -> List[float]:
        """
        检查每个 completion 是否包含代码块格式并能成功解析
        
        Args:
            completions: 模型生成的输出列表
            solution: 真值列表（可选，此函数不使用）
            
        Returns:
            奖励分数列表
        """
        rewards = []
        for completion in completions:
            reward = 0.0
            if completion:
                # 使用 parse_actions_from_completion 检查是否能成功解析
                actions = parse_actions_from_completion(completion)
                # 如果能解析出至少一个有效动作，则给予奖励
                if actions:
                    reward = 1.0
            rewards.append(reward)
        return rewards


class ThinkingFormatReward(ORM):
    """
    思考格式奖励（严格版）：要求输出为【先解释、后代码块】结构且动作解析成功
    奖励规则：
      - 必须严格为『解释（非空且不在代码块内）+ 一个 python 代码块（或 ``` … ）』，且能解析出至少一个动作，才给1.0。
      - 其他任何情况 0.0。
    """
    def __init__(self):
        super().__init__()

    def __call__(self, completions: List[str], solution: List[str] = None, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            reward = 0.0
            if completion:
                # 只允许第一个代码块前的文本为解释，且只能有一个代码块
                code_blocks = list(_ACTION_RE.finditer(completion))
                if len(code_blocks) == 1:
                    block = code_blocks[0]
                    exp = completion[:block.start()].strip()
                    code = block.group(0)
                    # 解释必须非空且非代码
                    has_exp = bool(exp) and not exp.startswith("```")
                    # 后面不能还有多余代码块
                    after = completion[block.end():].strip()
                    no_extra = ("```" not in after)
                    # 能否正常解析动作
                    actions = parse_actions_from_completion(completion)
                    if has_exp and no_extra and actions:
                        reward = 1.0
            rewards.append(reward)
        return rewards


# ---------------------------
# SolutionReward
# ---------------------------


class SolutionReward(ORM):
    """
    解决方案准确性奖励：比较生成的代码与真值代码
    奖励规则：
      - 使用部分匹配策略，检查生成的动作调用是否与真值匹配
      - 完全匹配：+1.0
      - 部分匹配：按匹配比例给分
      - 无匹配：0.0
    """
    def __init__(self, exact_match: bool = False, partial_weight: float = 0.5):
        """
        Args:
            exact_match: 是否要求完全匹配（包括顺序）
            partial_weight: 部分匹配的权重系数
        """
        super().__init__()

    def __call__(self, completions: List[str], solution: List[str] = None, **kwargs) -> List[float]:
        """
        比较生成的代码与真值代码
        
        Args:
            completions: 模型生成的输出列表
            solution: 真值列表
            
        Returns:
            奖励分数列表
        """
        rewards = []
        
        if not solution:
            # 如果没有提供 solution，返回 0 分
            return [0.0] * len(completions)
        
        for completion, sol in zip(completions, solution):
            reward = 0.0
            
            try:
                # 使用 parse_actions_from_completion 解析动作
                generated_actions = parse_actions_from_completion(completion)
                solution_actions = parse_actions_from_completion(sol)
                
                seen_actions = set()
                for action in generated_actions:
                    if action['type'] in seen_actions:
                        reward = 0.0
                        raise Exception("重复动作")
                    seen_actions.add(action['type'])
                
                # 将生成的动作转换为字典以便快速查找
                gen_dict = {action['type']: action for action in generated_actions}
                
                # 计算匹配的动作数量
                matched = 0
                for gt_action in solution_actions:
                    action_type = gt_action['type']
                    
                    if action_type not in gen_dict:
                        continue
                    
                    gen_action = gen_dict[action_type]
                    
                    # 根据动作类型进行匹配
                    if action_type == 'watch':
                        # watch 需要匹配 second 参数，使用相对误差和衰减奖励
                        gt_second = gt_action.get('second', 0)
                        gen_second = gen_action.get('second', 0)
                        
                        # 长观看更宽容
                        score = []
                        relative_error = abs(gt_second - gen_second) / gt_second
                        score.append(1 - min(1, relative_error))
                        if (gt_second >= 7) and (gen_second >= 7):
                            score.append(0.5+0.5*(1 - min(1, relative_error)))
                        matched += max(score)
                    elif action_type in ['like', 'skip', 'comment', 'share']:
                        # like 和 skip 只需要类型匹配
                        matched += 1
                
                # 计算奖励分数（使用 F1-score 思想，同时考虑 precision 和 recall）
                total_sol = len(solution_actions)
                total_gen = len(generated_actions)
                
                if total_sol > 0:
                    # Recall: 真值中有多少被匹配
                    recall = matched / total_sol
                    # Precision: 生成的动作中有多少是正确的
                    precision = matched / total_gen if total_gen > 0 else 0.0
                    
                    # 完全匹配：recall=1, precision=1
                    if recall == 1.0 and precision == 1.0:
                        reward = 1.0
                    # 部分匹配：使用 F1-score
                    elif recall > 0 and precision > 0:
                        # F1 = 2 * (precision * recall) / (precision + recall)
                        f1_score = 2 * (precision * recall) / (precision + recall)
                        reward = f1_score * 0.5  # 部分匹配权重
                    else:
                        reward = 0.0
                    
                    # 对多余动作的额外惩罚（可选）
                    if total_gen > total_sol:
                        extra_penalty = (total_gen - total_sol) * 0.1
                        reward = max(0.0, reward - extra_penalty)
                else:
                    reward = 0.0
                            
            except Exception as e:
                # 解析失败，返回 0 分
                reward = 0.0
            
            rewards.append(reward)
        
        return rewards


# ---------------------------
# 注册 reward func（按你的要求映射到或替换为以下键）
# ---------------------------

orms['video_quality'] = VideoQualityReward
orms['persona'] = PersonaReward
orms['diversity'] = DiversityReward
orms['format'] = FormatReward
orms['thinking_format'] = ThinkingFormatReward
orms['solution'] = SolutionReward
