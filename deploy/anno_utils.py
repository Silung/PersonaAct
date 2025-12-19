#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
操作标注工具类
专用于scrcpy GUI应用的操作信息标注功能
"""

import os
import re
import subprocess
import json
import math
import logging
from pathlib import Path
import uiautomator2 as u2

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class OperationAnnotator:
    """操作标注工具类"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.font_path = os.path.join(os.path.dirname(__file__), "assets", "SourceHanSansSC-Regular.ttf")
        
    def annotate_operation(self, image_path, json_path, output_path=None):
        """
        在图片上标注操作信息
        
        Args:
            image_path (str): 原始图片路径
            json_path (str): JSON操作数据路径
            output_path (str): 输出图片路径，如果为None则自动生成
            
        Returns:
            str: 标注后的图片路径，失败返回None
        """
        if not PIL_AVAILABLE:
            self.logger.warning("PIL/Pillow 不可用，跳过操作标注")
            return None
            
        try:
            # 读取JSON数据
            with open(json_path, 'r', encoding='utf-8') as f:
                operation_data = json.load(f)
            
            # 生成输出路径
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_dir = os.path.dirname(image_path)
                output_path = os.path.join(output_dir, f"{base_name}_annotated.png")
            
            # 打开图片
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # 根据操作类型进行标注
            action_type = operation_data.get('action_type', '')
            
            if action_type in ['click', 'long_press', 'tap']:
                self._draw_point_operation(draw, operation_data, img.size)
            elif action_type == 'swipe':
                self._draw_swipe_operation(draw, operation_data, img.size)
            elif action_type in ['type', 'keyboard_input']:
                self._draw_text_operation(draw, operation_data, img.size)
            elif action_type in ['wait', 'home', 'back', 'open', 'finish']:
                self._draw_action_operation(draw, operation_data, img.size)
                
            # 保存标注后的图片
            img.save(output_path)
            self.logger.info(f"操作标注完成: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"操作标注失败: {e}")
            return None
    
    def _draw_point_operation(self, draw, operation_data, image_size):
        """绘制点击/长按操作"""
        width, height = image_size
        
        # 获取坐标并应用缩放
        x = operation_data["parameters"]["x"]
        y = operation_data["parameters"]["y"]
        
        # 确保坐标在图片范围内
        x = max(0, min(x, width))
        y = max(0, min(y, height))
        
        # 确定颜色
        action_type = operation_data.get('action_type', '')
        color = '#FF0000' if action_type in ['click', 'tap'] else '#FF8800'
        
        # 绘制圆圈
        radius = int(0.02 * min(width, height))
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=color,
            width=4
        )
        
        # 绘制中心点
        draw.ellipse(
            [x - 8, y - 8, x + 8, y + 8],
            fill=color
        )
        
        # 绘制标签
        label = "long_press" if action_type == 'long_press' else "click"
        self._draw_label(draw, x, y + radius + 15, label, color, image_size)
    
    def _draw_swipe_operation(self, draw, operation_data, image_size):
        """绘制滑动操作"""
        width, height = image_size
        
        start_x = operation_data["parameters"]["start_x"]
        start_y = operation_data["parameters"]["start_y"]
        end_x = operation_data["parameters"]["end_x"]
        end_y = operation_data["parameters"]["end_y"]
        
        # 确保坐标在图片范围内
        start_x = max(0, min(start_x, width))
        start_y = max(0, min(start_y, height))
        end_x = max(0, min(end_x, width))
        end_y = max(0, min(end_y, height))
        
        color = '#0088FF'
        
        # 绘制轨迹线
        draw.line([start_x, start_y, end_x, end_y], fill=color, width=6)
        
        # 绘制箭头
        self._draw_arrow_head(draw, start_x, start_y, end_x, end_y, color)
        
        # 绘制起点和终点
        draw.ellipse(
            [start_x - 10, start_y - 10, start_x + 10, start_y + 10],
            fill='#00FF00',
            outline='#FFFFFF',
            width=2
        )
        draw.ellipse(
            [end_x - 10, end_y - 10, end_x + 10, end_y + 10],
            fill=color,
            outline='#FFFFFF',
            width=2
        )
        
        # 绘制方向标签
        direction = operation_data.get('swipe_direction', '')
        mid_x = (start_x + end_x) // 2
        mid_y = (start_y + end_y) // 2
        self._draw_label(draw, mid_x, mid_y - 20, f"swipe {direction}", color, image_size)
    
    def _draw_text_operation(self, draw, operation_data, image_size):
        """绘制文本输入操作"""
        width, height = image_size
        
        # 文本输入操作在屏幕中央偏上显示，避免遮挡重要内容
        x = width // 2
        y = height // 3  # 改为1/3位置，而不是中央
        
        color = '#FF0000'
        text_content = operation_data["parameters"]["text"]       
        # 使用自适应字体大小
        font_size = self._get_adaptive_font_size(image_size, 36)  # 文本输入使用较大字体
        
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # 限制文本长度，避免过长
        display_text = text_content
        if len(display_text) > 20:
            display_text = display_text[:17] + "..."
        
        full_text = f"输入: {display_text}"
        
        # 计算文本尺寸
        bbox = draw.textbbox((0, 0), full_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 自适应边距
        padding = max(15, font_size // 2)
        
        # 确保不超出图片边界
        if x - text_width//2 - padding < 0:
            x = text_width//2 + padding
        elif x + text_width//2 + padding > width:
            x = width - text_width//2 - padding
        
        if y - text_height//2 - padding < 0:
            y = text_height//2 + padding
        elif y + text_height//2 + padding > height:
            y = height - text_height//2 - padding
        
        # 计算背景框位置
        bg_left = x - text_width//2 - padding
        bg_top = y - text_height//2 - padding
        bg_right = x + text_width//2 + padding
        bg_bottom = y + text_height//2 + padding
        
        # 绘制阴影效果
        shadow_offset = max(3, font_size // 10)
        draw.rectangle(
            [bg_left + shadow_offset, bg_top + shadow_offset, 
             bg_right + shadow_offset, bg_bottom + shadow_offset],
            fill='#00000050'  # 半透明阴影
        )
        
        # 绘制背景
        draw.rectangle(
            [bg_left, bg_top, bg_right, bg_bottom],
            fill='#FFFFFF',
            outline=color,
            width=max(3, font_size // 10)
        )
        
        # 绘制文本
        draw.text(
            (x - text_width//2, y - text_height//2),
            full_text,
            fill=color,
            font=font
        )
    
    def _draw_action_operation(self, draw, operation_data, image_size):
        """绘制动作操作（等待、返回、首页等）"""
        width, height = image_size
        
        # 动作操作在屏幕上方显示，避免遮挡主要内容
        x = width // 2
        y = height // 2
        
        action_type = operation_data.get('action_type', '')
        
        # 动作类型映射
        action_labels = {
            'wait': 'wait 等待',
            'home': 'home 首页',
            'back': 'back 返回', 
            'open': 'open 打开应用',
            'finish': 'finish 结束采集'
        }
        
        colors = {}
        
        label = action_labels.get(action_type, action_type)
        color = colors.get(action_type, '#FF0000')
        
        # 添加额外信息
        if action_type == 'wait':
            wait_time = operation_data["parameters"].get('wait_time', 0)
            label = f"wait {wait_time} s"
        elif action_type == 'open':
            app_name = operation_data["parameters"].get('app_name', '')
            label = f"open {app_name}"
        
        # 绘制标签
        self._draw_action_label(draw, x, y, label, color, image_size)
    
    def _draw_arrow_head(self, draw, start_x, start_y, end_x, end_y, color):
        """绘制箭头头部"""
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.sqrt(dx * dx + dy * dy)
        
        if length < 10:
            return
        
        # 箭头参数
        arrow_length = min(30, length * 0.3)
        angle = math.atan2(dy, dx)
        
        # 左侧箭头线
        arrow_x1 = end_x - arrow_length * math.cos(angle - math.pi / 6)
        arrow_y1 = end_y - arrow_length * math.sin(angle - math.pi / 6)
        draw.line([end_x, end_y, arrow_x1, arrow_y1], fill=color, width=4)
        
        # 右侧箭头线
        arrow_x2 = end_x - arrow_length * math.cos(angle + math.pi / 6)
        arrow_y2 = end_y - arrow_length * math.sin(angle + math.pi / 6)
        draw.line([end_x, end_y, arrow_x2, arrow_y2], fill=color, width=4)
    
    def _get_adaptive_font_size(self, image_size, base_size=24):
        """根据图片尺寸自适应字体大小"""
        width, height = image_size
        min_dimension = min(width, height)
        
        # 根据屏幕最小边长调整字体大小
        if min_dimension <= 720:  # 小屏幕
            return max(16, int(base_size * 0.7))
        elif min_dimension <= 1080:  # 中等屏幕
            return base_size
        elif min_dimension <= 1440:  # 大屏幕
            return int(base_size * 1.2)
        else:  # 超大屏幕
            return int(base_size * 1.5)
    
    def _draw_label(self, draw, x, y, text, color, image_size=None):
        """绘制文本标签"""
        if image_size is None:
            image_size = (1080, 1920)  # 默认尺寸
            
        font_size = self._get_adaptive_font_size(image_size, 24)
        
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # 计算文本尺寸
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 自适应边距
        padding = max(8, font_size // 3)
        
        # 计算背景框位置，确保不超出图片边界
        width, height = image_size
        bg_left = max(padding, x - text_width//2 - padding)
        bg_top = max(padding, y - text_height//2 - padding)
        bg_right = min(width - padding, x + text_width//2 + padding)
        bg_bottom = min(height - padding, y + text_height//2 + padding)
        
        # 如果文本框超出边界，调整文本位置
        if bg_right - bg_left < text_width + padding * 2:
            # 水平方向调整
            if x + text_width//2 + padding > width:
                x = width - text_width//2 - padding * 2
            elif x - text_width//2 - padding < 0:
                x = text_width//2 + padding * 2
        
        if bg_bottom - bg_top < text_height + padding * 2:
            # 垂直方向调整
            if y + text_height//2 + padding > height:
                y = height - text_height//2 - padding * 2
            elif y - text_height//2 - padding < 0:
                y = text_height//2 + padding * 2
        
        # 重新计算背景框位置
        bg_left = x - text_width//2 - padding
        bg_top = y - text_height//2 - padding
        bg_right = x + text_width//2 + padding
        bg_bottom = y + text_height//2 + padding
        
        # 绘制阴影效果
        shadow_offset = max(2, font_size // 12)
        draw.rectangle(
            [bg_left + shadow_offset, bg_top + shadow_offset, 
             bg_right + shadow_offset, bg_bottom + shadow_offset],
            fill='#00000040'  # 半透明黑色阴影
        )
        
        # 绘制背景
        draw.rectangle(
            [bg_left, bg_top, bg_right, bg_bottom],
            fill='#FFFFFF',
            outline=color,
            width=max(2, font_size // 12)
        )
        
        # 绘制文本
        draw.text(
            (x - text_width//2, y - text_height//2),
            text,
            fill=color,
            font=font
        )
    
    def _draw_action_label(self, draw, x, y, text, color, image_size=None):
        """绘制动作标签（较大）"""
        if image_size is None:
            image_size = (1080, 1920)  # 默认尺寸
            
        font_size = self._get_adaptive_font_size(image_size, 42)  # 动作标签使用更大的基础字体
        
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # 计算文本尺寸
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 自适应边距（动作标签使用更大边距）
        padding = max(12, font_size // 2)
        width, height = image_size
        
        # 计算位置，确保不超出图片边界
        bg_left = max(padding, x - text_width//2 - padding)
        bg_top = max(padding, y - text_height//2 - padding)
        bg_right = min(width - padding, x + text_width//2 + padding)
        bg_bottom = min(height - padding, y + text_height//2 + padding)
        
        # 如果文本框超出边界，调整文本位置
        if bg_right - bg_left < text_width + padding * 2:
            if x + text_width//2 + padding > width:
                x = width - text_width//2 - padding * 2
            elif x - text_width//2 - padding < 0:
                x = text_width//2 + padding * 2
        
        if bg_bottom - bg_top < text_height + padding * 2:
            if y + text_height//2 + padding > height:
                y = height - text_height//2 - padding * 2
            elif y - text_height//2 - padding < 0:
                y = text_height//2 + padding * 2
        
        # 重新计算背景框位置
        bg_left = x - text_width//2 - padding
        bg_top = y - text_height//2 - padding
        bg_right = x + text_width//2 + padding
        bg_bottom = y + text_height//2 + padding
        
        # 绘制阴影效果
        shadow_offset = max(3, font_size // 10)
        draw.rectangle(
            [bg_left + shadow_offset, bg_top + shadow_offset, 
             bg_right + shadow_offset, bg_bottom + shadow_offset],
            fill='#00000060'  # 更深的阴影
        )
        
        # 绘制背景
        draw.rectangle(
            [bg_left, bg_top, bg_right, bg_bottom],
            fill='#FFFFFF',
            outline=color,
            width=max(3, font_size // 10)
        )
        
        # 绘制文本
        draw.text(
            (x - text_width//2, y - text_height//2),
            text,
            fill=color,
            font=font
        )


class CoordinateConverter:
    """坐标转换工具类"""

    def __init__(
        self, screen_width=None, screen_height=None, touch_width=None, touch_height=None
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.touch_width = touch_width
        self.touch_height = touch_height

    def convert_touch_to_screen_coordinates(self, touch_x, touch_y):
        """将触摸设备坐标转换为屏幕像素坐标"""
        max_x = (
            self.touch_width
            if self.touch_width and self.touch_width > 0
            else self.screen_width
        )
        max_y = (
            self.touch_height
            if self.touch_height and self.touch_height > 0
            else self.screen_height
        )

        if max_x and max_x > 0 and max_y and max_y > 0:
            relative_x = max(0.0, min(1.0, touch_x / max_x))
            relative_y = max(0.0, min(1.0, touch_y / max_y))

            screen_x = (
                int(relative_x * self.screen_width)
                if self.screen_width and self.screen_width > 0
                else touch_x
            )
            screen_y = (
                int(relative_y * self.screen_height)
                if self.screen_height and self.screen_height > 0
                else touch_y
            )

            return screen_x, screen_y
        else:
            raise ValueError("无法获取触摸设备分辨率")

    def convert_to_relative_coordinates(self, x, y):
        """将屏幕坐标转换为相对坐标（0-1范围）"""
        if self.screen_width and self.screen_height:
            rel_x = x / self.screen_width
            rel_y = y / self.screen_height
            return round(rel_x, 4), round(rel_y, 4)
        else:
            raise ValueError("无法获取屏幕尺寸")


class GestureAnalyzer:
    """手势分析工具类"""

    def __init__(self, coordinate_converter, logger=None):
        self.coordinate_converter = coordinate_converter
        self.logger = logger or logging.getLogger(__name__)

    def analyze_gesture(self, touch_session, duration):
        """分析手势类型"""
        if (
            touch_session["start_x"] is None
            or touch_session["start_y"] is None
            or touch_session["current_x"] is None
            or touch_session["current_y"] is None
        ):
            raise ValueError("无法获取触摸会话")

        start_x = touch_session["start_x"]
        start_y = touch_session["start_y"]
        end_x = touch_session["current_x"]
        end_y = touch_session["current_y"]
        path_points = touch_session["path_points"]

        # 计算总距离和直线距离
        total_distance = self._calculate_total_distance(path_points)
        straight_distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

        # 判断手势类型的阈值
        MIN_SWIPE_DISTANCE = 50
        LONG_PRESS_THRESHOLD = 1
        TAP_MOVEMENT_THRESHOLD = 20

        if straight_distance < TAP_MOVEMENT_THRESHOLD:
            # 静止或微小移动
            rel_x, rel_y = self.coordinate_converter.convert_to_relative_coordinates(
                start_x, start_y
            )
            if duration >= LONG_PRESS_THRESHOLD:
                return {
                    "type": "long_press",
                    "params": {
                        "x": start_x,
                        "y": start_y,
                        "relative_x": rel_x,
                        "relative_y": rel_y,
                        "duration": round(duration, 2),
                    },
                }
            else:
                return {
                    "type": "click",
                    "params": {
                        "x": start_x,
                        "y": start_y,
                        "relative_x": rel_x,
                        "relative_y": rel_y,
                    },
                }

        elif straight_distance >= MIN_SWIPE_DISTANCE:
            # 滑动操作
            rel_start_x, rel_start_y = (
                self.coordinate_converter.convert_to_relative_coordinates(
                    start_x, start_y
                )
            )
            rel_end_x, rel_end_y = (
                self.coordinate_converter.convert_to_relative_coordinates(end_x, end_y)
            )

            swipe_params = {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "relative_start_x": rel_start_x,
                "relative_start_y": rel_start_y,
                "relative_end_x": rel_end_x,
                "relative_end_y": rel_end_y,
                "distance": round(straight_distance, 2),
                "duration": round(duration, 2),
                "velocity": (
                    round(straight_distance / duration, 2) if duration > 0 else 0
                ),
                "path_length": round(total_distance, 2),
            }

            # 计算滑动方向
            direction = self._calculate_swipe_direction(start_x, start_y, end_x, end_y)
            swipe_params["direction"] = direction

            # 判断滑动类型
            swipe_type = self._classify_swipe_type(
                path_points, straight_distance, total_distance
            )
            swipe_params["swipe_type"] = swipe_type

            return {"type": "swipe", "params": swipe_params}

        return None

    def _calculate_total_distance(self, path_points):
        """计算路径总长度"""
        if len(path_points) < 2:
            return 0

        total_distance = 0
        for i in range(1, len(path_points)):
            prev_point = path_points[i - 1]
            curr_point = path_points[i]

            dx = curr_point["x"] - prev_point["x"]
            dy = curr_point["y"] - prev_point["y"]
            distance = (dx**2 + dy**2) ** 0.5
            total_distance += distance

        return total_distance

    def _calculate_swipe_direction(self, start_x, start_y, end_x, end_y):
        """计算滑动方向"""
        dx = end_x - start_x
        dy = end_y - start_y

        angle = math.atan2(dy, dx) * 180 / math.pi
        if angle < 0:
            angle += 360

        if 315 <= angle or angle < 45:
            return "right"
        elif 45 <= angle < 135:
            return "down"
        elif 135 <= angle < 225:
            return "left"
        elif 225 <= angle < 315:
            return "up"

        return "unknown"

    def _classify_swipe_type(self, path_points, straight_distance, total_distance):
        """分类滑动类型"""
        if len(path_points) < 3:
            return "simple"

        curvature_ratio = (
            total_distance / straight_distance if straight_distance > 0 else 1
        )

        if curvature_ratio <= 1.2:
            return "linear"
        elif curvature_ratio <= 2.0:
            return "curved"
        else:
            return "complex"


def parse_getevent_line(line):
    """解析getevent输出行"""
    try:
        parts = line.strip().split()
        if len(parts) >= 4:
            device = parts[0].rstrip(":")
            event_type = int(parts[1], 16)
            code = int(parts[2], 16)
            value = int(parts[3], 16)
            return device, event_type, code, value
    except:
        pass
    return None, None, None, None


class ADBUtils:
    """ADB工具类"""

    def __init__(self, device_id=None, logger=None, use_u2=False):
        self.logger = logger or logging.getLogger(__name__)
        self.device_id = device_id
        self.use_u2 = use_u2
        if self.use_u2:
            self.device = u2.connect(self.device_id)
        else:
            self.device = None
        self.logger.info(f"连接设备: {self.device_id}")

    def execute_adb_command(self, command, timeout=30, binary=False):
        """执行ADB命令

        Args:
            command: ADB命令列表
            timeout: 超时时间
            binary: 是否为二进制输出（如截图命令）
        """
        if self.device_id:
            full_cmd = ["adb", "-s", self.device_id] + command
        else:
            full_cmd = ["adb"] + command

        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=not binary,
                timeout=timeout,
                shell=False,
            )
            return result
        except subprocess.TimeoutExpired:
            raise Exception(f"ADB命令超时: {' '.join(full_cmd)}")
        except Exception as e:
            raise Exception(f"ADB命令执行失败: {str(e)}")

    def get_screen_size(self):
        """获取屏幕尺寸"""
        result = self.execute_adb_command(["shell", "wm", "size"])
        if result.returncode == 0:
            match = re.search(r"(\d+)x(\d+)", result.stdout)
            if match:
                width, height = int(match.group(1)), int(match.group(2))
                return width, height
        raise ValueError("无法获取屏幕尺寸")

    def get_touch_resolution(self):
        """获取触摸设备坐标范围"""
        try:
            result = self.execute_adb_command(["shell", "getevent", "-p"])
            if result.returncode != 0:
                self.logger.error("getevent -p 命令执行失败")
                return None, None

            lines = result.stdout.split("\n")
            touch_device_found = False
            max_x = 0
            max_y = 0

            for line in lines:
                if "touch" in line.lower() or "Touch" in line:
                    touch_device_found = True
                    continue

                if touch_device_found and line.strip() == "":
                    break

                if touch_device_found:
                    if "0035" in line:  # ABS_MT_POSITION_X
                        match = re.search(r"max (\d+)", line)
                        if match:
                            max_x = int(match.group(1))
                    elif "0036" in line:  # ABS_MT_POSITION_Y
                        match = re.search(r"max (\d+)", line)
                        if match:
                            max_y = int(match.group(1))

            if max_x > 0 and max_y > 0:
                self.logger.info(f"触摸坐标范围: X(0-{max_x}), Y(0-{max_y})")
                return max_x, max_y
            else:
                self.logger.warning("无法获取触摸坐标范围")
                return None, None

        except Exception as e:
            self.logger.error(f"获取触摸设备坐标范围失败: {str(e)}")
            return None, None

    def screencap(self):
        """执行截图命令"""
        if self.use_u2:
            screenshot = self.device.screenshot()
            return screenshot
        cmd = ["exec-out", "screencap", "-p"]
        result = self.execute_adb_command(cmd, binary=True)
        return result

    def dump_ui_hierarchy(self):
        """获取UI层次结构XML"""
        try:
            if self.use_u2:
                xml_content = self.device.dump_hierarchy()
                return xml_content
            cmd = ["exec-out", "uiautomator", "dump", "/dev/stdout"]
            result = self.execute_adb_command(cmd)
            return result
        except Exception as e:
            self.logger.error(f"获取UI层次结构失败: {str(e)}")
            return None


def get_xml_max_depth(xml_content):
    """
    计算XML内容中最大连续缩进的个数
    
    Args:
        xml_content (str): XML内容字符串
        N (int): 最大允许的连续缩进个数
        
    Returns:
        bool: 如果最大连续缩进个数 >= N 返回False，否则返回True
    """
    lines = xml_content.strip().split('\n')
    
    max_consecutive_indent = 0
    current_consecutive_indent = 0
    prev_indent_level = -1
    
    for line in lines:
        # 跳过空行
        if not line.strip():
            continue
            
        # 检查是否是<node开头的行
        stripped_line = line.strip()
        if not stripped_line.startswith('<node'):
            # 如果不是node行，重置连续计数
            current_consecutive_indent = 0
            prev_indent_level = -1
            continue
            
        # 计算当前行的缩进级别（每2个空格为一级）
        leading_spaces = len(line) - len(line.lstrip())
        current_indent_level = leading_spaces // 2
        
        # 检查是否是连续递增的缩进
        if prev_indent_level != -1 and current_indent_level == prev_indent_level + 1:
            # 连续递增缩进
            current_consecutive_indent += 1
            max_consecutive_indent = max(max_consecutive_indent, current_consecutive_indent)
        else:
            # 重置连续计数
            current_consecutive_indent = 0
            
        prev_indent_level = current_indent_level
    
    return max_consecutive_indent


def get_xml_adb(device_serial):
    # adb shell uiautomator dump /sdcard/window_dump.xml
    # adb pull /sdcard/window_dump.xml
    try:
        cmd = ["adb", "-s", device_serial, "shell", "uiautomator", "dump", "/sdcard/window_dump.xml"]
        subprocess.run(cmd, shell=False)
        cmd = ["adb", "-s", device_serial, "pull", "/sdcard/window_dump.xml"]
        subprocess.run(cmd, shell=False)
        with open("window_dump.xml", "r", encoding="utf-8") as f:
            xml_content = f.read()
        return xml_content
        
    except Exception as e:
        raise Exception(f"ADB命令执行失败: {str(e)}")


def get_xml_content(u2_device, max_depth=50, max_u2_try=2):
    device_serial = u2_device.serial
    for _ in range(max_u2_try):
        try:
            xml_content = u2_device.dump_hierarchy(compressed=True, max_depth=max_depth)
            _depth = get_xml_max_depth(xml_content)
            if _depth < max_depth:
                return xml_content
        except Exception as e:
            print(f"get_xml_content error: {e}")

    xml_content = get_xml_adb(device_serial)
    return xml_content


