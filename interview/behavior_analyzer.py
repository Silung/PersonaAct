"""
行为分析模块 - 从用户刷视频数据中提取行为特征
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import numpy as np


class BehaviorAnalyzer:
    """分析用户刷视频行为数据，提取偏好特征"""
    
    def __init__(self, data_dir: str = "data/yqg", raw_data_dir: str = "raw_data/yqg"):
        self.data_dir = Path(data_dir)
        self.raw_data_dir = Path(raw_data_dir)
        
        # 加载预处理数据
        self.audio_transcripts = self._load_json("audio_transcript.json")
        self.category_analysis = self._load_json("category_analysis.json")
        self.stats_analysis = self._load_json("stats_analysis.json")
        
        # 加载 session 数据
        self.sessions = self._load_sessions()
        
    def _load_json(self, filename: str) -> Dict:
        """加载 JSON 文件"""
        filepath = self.data_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_sessions(self) -> List[Dict]:
        """加载所有 session 文件"""
        sessions = []
        for session_file in self.raw_data_dir.glob("session_*.json"):
            with open(session_file, 'r', encoding='utf-8') as f:
                sessions.append(json.load(f))
        return sessions
    
    def analyze_all(self) -> Dict[str, Any]:
        """执行完整的行为分析"""
        return {
            "content_preferences": self._analyze_content_preferences(),
            "interaction_patterns": self._analyze_interaction_patterns(),
            "temporal_habits": self._analyze_temporal_habits(),
            "creator_preferences": self._analyze_creator_preferences(),
            "engagement_metrics": self._analyze_engagement_metrics(),
            "emotional_patterns": self._analyze_emotional_patterns(),
        }
    
    def _analyze_content_preferences(self) -> Dict[str, Any]:
        """分析内容偏好"""
        main_categories = Counter()
        sub_categories = Counter()
        liked_categories = Counter()
        
        for session in self.sessions:
            for action in session.get('actions', []):
                video_path = action.get('video_path', '').replace('\\', '/')
                category_info = self.category_analysis.get(video_path, {})
                categories = category_info.get('category', {}).get('categories', [])
                
                has_like = any(a.get('type') == 'like' for a in action.get('actions', []))
                
                for cat in categories:
                    main = cat.get('main', '未知')
                    sub = cat.get('sub', '未知')
                    main_categories[main] += 1
                    sub_categories[f"{main}/{sub}"] += 1
                    if has_like:
                        liked_categories[main] += 1
        
        return {
            "top_main_categories": main_categories.most_common(5),
            "top_sub_categories": sub_categories.most_common(10),
            "liked_categories": liked_categories.most_common(5),
            "total_videos": sum(main_categories.values()),
        }
    
    def _analyze_interaction_patterns(self) -> Dict[str, Any]:
        """分析互动模式"""
        total_actions = 0
        likes = 0
        comments = 0
        shares = 0
        watch_durations = []
        
        for session in self.sessions:
            for action in session.get('actions', []):
                total_actions += 1
                watch_durations.append(action.get('viewing_duration', 0))
                
                for act in action.get('actions', []):
                    if act.get('type') == 'like':
                        likes += 1
                    elif act.get('type') == 'comment':
                        comments += 1
                    elif act.get('type') == 'share':
                        shares += 1
        
        return {
            "total_videos_watched": total_actions,
            "like_rate": likes / total_actions if total_actions > 0 else 0,
            "comment_rate": comments / total_actions if total_actions > 0 else 0,
            "share_rate": shares / total_actions if total_actions > 0 else 0,
            "avg_watch_duration": np.mean(watch_durations) if watch_durations else 0,
            "max_watch_duration": max(watch_durations) if watch_durations else 0,
            "quick_skip_rate": sum(1 for d in watch_durations if d < 3) / len(watch_durations) if watch_durations else 0,
        }
    
    def _analyze_temporal_habits(self) -> Dict[str, Any]:
        """分析时间习惯"""
        session_durations = []
        videos_per_session = []
        
        for session in self.sessions:
            actions = session.get('actions', [])
            videos_per_session.append(len(actions))
            total_duration = sum(a.get('viewing_duration', 0) for a in actions)
            session_durations.append(total_duration)
        
        return {
            "total_sessions": len(self.sessions),
            "avg_session_duration": np.mean(session_durations) if session_durations else 0,
            "avg_videos_per_session": np.mean(videos_per_session) if videos_per_session else 0,
        }
    
    def _analyze_creator_preferences(self) -> Dict[str, Any]:
        """分析创作者偏好"""
        creator_watch = Counter()
        creator_like = Counter()
        
        for session in self.sessions:
            for action in session.get('actions', []):
                video_path = action.get('video_path', '').replace('\\', '/')
                stats = self.stats_analysis.get(video_path, {}).get('stats', {})
                author = stats.get('author', '').strip()
                
                if author:
                    creator_watch[author] += 1
                    has_like = any(a.get('type') == 'like' for a in action.get('actions', []))
                    if has_like:
                        creator_like[author] += 1
        
        return {
            "most_watched_creators": creator_watch.most_common(5),
            "most_liked_creators": creator_like.most_common(5),
            "unique_creators": len(creator_watch),
        }
    
    def _analyze_engagement_metrics(self) -> Dict[str, Any]:
        """分析用户对热门内容的反应"""
        high_engagement_watched = 0
        high_engagement_liked = 0
        low_engagement_watched = 0
        low_engagement_liked = 0
        
        for session in self.sessions:
            for action in session.get('actions', []):
                video_path = action.get('video_path', '').replace('\\', '/')
                stats = self.stats_analysis.get(video_path, {}).get('stats', {})
                total_engagement = stats.get('like', 0) + stats.get('comment', 0)
                
                has_like = any(a.get('type') == 'like' for a in action.get('actions', []))
                
                if total_engagement > 10000:
                    high_engagement_watched += 1
                    if has_like:
                        high_engagement_liked += 1
                else:
                    low_engagement_watched += 1
                    if has_like:
                        low_engagement_liked += 1
        
        return {
            "high_engagement_like_rate": high_engagement_liked / high_engagement_watched if high_engagement_watched > 0 else 0,
            "low_engagement_like_rate": low_engagement_liked / low_engagement_watched if low_engagement_watched > 0 else 0,
            "prefers_popular": high_engagement_liked / high_engagement_watched > low_engagement_liked / low_engagement_watched if high_engagement_watched > 0 and low_engagement_watched > 0 else None,
        }
    
    def _analyze_emotional_patterns(self) -> Dict[str, Any]:
        """分析情绪模式（基于音频转录）"""
        emotional_keywords = {
            "positive": ["开心", "快乐", "喜欢", "爱", "好", "棒", "赞", "哈哈", "笑"],
            "negative": ["难过", "伤心", "痛", "哭", "烦", "累", "讨厌", "恨"],
            "relaxing": ["放松", "治愈", "舒服", "安静", "平静"],
            "exciting": ["刺激", "激动", "紧张", "精彩", "厉害"],
        }
        
        emotion_counts = defaultdict(int)
        total_with_transcript = 0
        
        for session in self.sessions:
            for action in session.get('actions', []):
                audio_path = action.get('audio_path', '').replace('\\', '/')
                transcript = self.audio_transcripts.get(audio_path, {}).get('text', '')
                
                if transcript:
                    total_with_transcript += 1
                    for emotion, keywords in emotional_keywords.items():
                        if any(kw in transcript for kw in keywords):
                            emotion_counts[emotion] += 1
        
        return {
            "emotion_distribution": dict(emotion_counts),
            "total_analyzed": total_with_transcript,
        }
    
    def get_behavior_summary(self) -> str:
        """生成行为摘要文本"""
        analysis = self.analyze_all()
        
        summary_parts = []
        
        # 内容偏好
        content = analysis['content_preferences']
        if content['top_main_categories']:
            top_cats = [f"{cat}({count}次)" for cat, count in content['top_main_categories'][:3]]
            summary_parts.append(f"📺 常看内容类型: {', '.join(top_cats)}")
        
        # 互动模式
        interaction = analysis['interaction_patterns']
        summary_parts.append(f"👍 点赞率: {interaction['like_rate']:.1%}")
        summary_parts.append(f"⏱️ 平均观看时长: {interaction['avg_watch_duration']:.1f}秒")
        summary_parts.append(f"⏭️ 快速划走率: {interaction['quick_skip_rate']:.1%}")
        
        # 创作者偏好
        creators = analysis['creator_preferences']
        if creators['most_liked_creators']:
            top_creators = [name for name, _ in creators['most_liked_creators'][:3]]
            summary_parts.append(f"❤️ 喜欢的创作者: {', '.join(top_creators)}")
        
        # 热门内容偏好
        engagement = analysis['engagement_metrics']
        if engagement['prefers_popular'] is not None:
            pref = "偏好热门内容" if engagement['prefers_popular'] else "不盲从热门"
            summary_parts.append(f"🔥 {pref}")
        
        return "\n".join(summary_parts)
    
    def get_detailed_behaviors(self) -> List[Dict]:
        """获取详细的行为记录（用于访谈参考）"""
        behaviors = []
        
        for session in self.sessions:
            for action in session.get('actions', []):
                video_path = action.get('video_path', '').replace('\\', '/')
                audio_path = action.get('audio_path', '').replace('\\', '/')
                
                category_info = self.category_analysis.get(video_path, {})
                stats = self.stats_analysis.get(video_path, {}).get('stats', {})
                transcript = self.audio_transcripts.get(audio_path, {}).get('text', '')
                
                behaviors.append({
                    "video_path": video_path,
                    "viewing_duration": action.get('viewing_duration', 0),
                    "actions": action.get('actions', []),
                    "categories": category_info.get('category', {}).get('categories', []),
                    "stats": stats,
                    "transcript": transcript[:200] if transcript else "",
                })
        
        return behaviors
    
    def get_representative_videos(self, max_samples: int = 5) -> List[Dict]:
        """获取有代表性的视频样本用于访谈"""
        behaviors = self.get_detailed_behaviors()
        if not behaviors:
            return []
        
        samples = []
        seen_video_paths = set()
        
        # 1. 观看时间最长的视频
        long_watch = sorted(behaviors, key=lambda x: x['viewing_duration'], reverse=True)[:2]
        for video in long_watch:
            if video['video_path'] not in seen_video_paths:
                video['sample_reason'] = 'long_watch'
                samples.append(video)
                seen_video_paths.add(video['video_path'])
        
        # 2. 快速跳过的视频
        quick_skip = [v for v in behaviors if v['viewing_duration'] < 3]
        if quick_skip and quick_skip[0]['video_path'] not in seen_video_paths:
            quick_skip[0]['sample_reason'] = 'quick_skip'
            samples.append(quick_skip[0])
            seen_video_paths.add(quick_skip[0]['video_path'])
        
        # 3. 点赞的视频
        liked = [v for v in behaviors if any(a.get('type') == 'like' for a in v['actions'])]
        if liked and liked[0]['video_path'] not in seen_video_paths:
            liked[0]['sample_reason'] = 'liked'
            samples.append(liked[0])
            seen_video_paths.add(liked[0]['video_path'])
        
        # 4. 不同类别的视频
        seen_categories = set()
        for video in samples:
            for cat in video.get('categories', []):
                seen_categories.add(cat.get('main', ''))
        
        for video in behaviors:
            if len(samples) >= max_samples:
                break
            if video['video_path'] in seen_video_paths:
                continue
            video_cats = {cat.get('main', '') for cat in video.get('categories', [])}
            if video_cats and not video_cats.intersection(seen_categories):
                video['sample_reason'] = 'diverse_category'
                samples.append(video)
                seen_video_paths.add(video['video_path'])
                seen_categories.update(video_cats)
        
        # 添加截图路径
        for video in samples:
            video_name = Path(video['video_path']).stem
            screenshot_dir = self.raw_data_dir / "screenshots" / video_name
            if screenshot_dir.exists():
                screenshots = sorted(screenshot_dir.glob("*.jpg"))
                if screenshots:
                    video['screenshot_path'] = str(screenshots[0])
        
        return samples[:max_samples]
