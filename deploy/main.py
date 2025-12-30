from argparse import ArgumentParser
from typing import Optional
import time
import os
import sys
import json
import datetime
import threading
import random
import queue
import uiautomator2 as u2
import cv2
import numpy as np
import wave
import re

import pyaudio
import logging
from adbutils import adb
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent, QPixmap, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QLabel, QSpinBox, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QComboBox
from PySide6.QtCore import QTimer, QSize
from ui_main import Ui_MainWindow
from pathlib import Path
from adb_utils import ADBUtils

import scrcpy

sys.path.append(str(Path(__file__).parent.parent))
from prompts import get_system_prompt, get_user_prompt
from openai import OpenAI

# 图片压缩配置：最大像素数（与训练时保持一致）
MAX_PIXELS = 288000  # 训练时设置的 MAX_PIXELS

if not QApplication.instance():
    app = QApplication([])
else:
    app = QApplication.instance()


class MainWindow(QMainWindow):
    def __init__(
        self,
        max_width: Optional[int],
        serial: Optional[str] = None,
        encoder_name: Optional[str] = None,
    ):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.max_width = max_width

        self.devices = self.list_devices()
        if serial:
            self.choose_device(serial)
        self.device = adb.device(serial=self.ui.combo_device.currentText())
        self.alive = True
        
        self.u2_device = u2.connect(self.device.serial)
        self.log_info(f"已连接到uiautomator2设备: {self.device.serial}")
        
        # ADB工具
        self.adb_utils = ADBUtils(device_serial=self.device.serial, logger=logging.getLogger(__name__))
        
        # AI交互相关
        self.ai_client = None
        self.ai_persona = None
        self.ai_history_screenshots = []
        self.ai_history_actions = []
        self.ai_max_history = 3
        self.ai_step_count = 0
        self.is_interacting = False
        self.session_data = None  # 存储会话数据，用于生成最终结果
        
        # LLM延迟跟踪（用于watch函数补偿）
        self.llm_delay_history = []  # 最近10次延迟记录
        self.llm_delay_max_history = 10  # 最多保存10次延迟
        self.llm_avg_delay = 1.0  # 初始延迟1秒
        self.llm_start_time_after_swipe = None  # 下滑后1秒的时间点（作为延迟计算起点）
        
        # 输出目录
        self.output_dir = None
        self.session_dir = None
        
        # （已移除无用的截图缓存功能）
        
        # 当前 XML 文本（异步获取）
        self.current_xml_text = None
        
        # 休息功能相关
        self.rest_enabled = False
        self.work_duration = 30  # 工作时长（分钟）
        self.rest_duration = 5  # 休息时长（分钟）
        self.rest_random_offset = 2  # 随机偏移（分钟）
        self.work_start_time = None  # 工作开始时间
        self.is_resting = False  # 是否正在休息
        self.rest_check_timer = QTimer()  # 检查休息时间的定时器
        self.rest_check_timer.timeout.connect(self._check_rest_time)
        self.current_platform = "bilibili"  # 当前平台
        
        # 状态更新定时器
        self.status_update_timer = QTimer()
        self.status_update_timer.timeout.connect(self._update_status_info)
        self.status_update_timer.start(1000)  # 每秒更新一次
        self.interaction_start_time = None  # 交互开始时间
        self.platform_packages = {
            "bilibili": "tv.danmaku.bili",
            "抖音": "com.ss.android.ugc.aweme",
            "快手": "com.smile.gifmake",
            "小红书": "com.xingin.xhs"
        }
        
        # 视频录制
        self.is_recording = False
        self.current_video_path = None
        self.current_audio_path = None
        self.video_writer = None
        self.video_frame_count = 0
        self.video_fps = 30
        self.video_writer_lock = threading.Lock()  # 保护 video_writer 的线程锁
        self.current_recording_start_time = None  # 记录录制开始时间
        
        # 异步视频写入队列
        self.video_frame_queue = queue.Queue(maxsize=120)  # 最多缓存4秒视频（30fps）
        self.video_write_thread = None
        self.video_write_thread_running = False
        
        # 音频录制
        self.is_audio_recording = False
        self.audio_buffer = []
        self.audio_sample_rate = 48000
        self.audio_channels = 1
        
        # 音频播放
        self.audio_stream = None
        self.pyaudio_instance = None
        
        self.window_size_initialized = False
        self.last_display_size = None
        
        # 初始化音频播放
        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = self.pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.audio_sample_rate,
            output=True,
            frames_per_buffer=2048
        )
        print(f"✅ 音频播放初始化成功")

        # Setup client
        self.client = scrcpy.Client(
            device=self.device,
            flip=self.ui.flip.isChecked(),
            bitrate=1000000000,
            encoder_name=encoder_name,
            max_fps=60,
            audio=True,
            audio_codec="opus",
            audio_bit_rate=128000,
        )
        self.client.add_listener(scrcpy.EVENT_INIT, self.on_init)
        self.client.add_listener(scrcpy.EVENT_FRAME, self.on_frame)
        self.client.add_listener(scrcpy.EVENT_AUDIO, self.on_audio)

        # Bind事件
        self.ui.button_test_model.clicked.connect(self.test_model_connection)
        self.ui.button_start.clicked.connect(self.start_interaction)
        self.ui.button_stop.clicked.connect(self.stop_interaction)
        self.ui.combo_device.currentTextChanged.connect(self.choose_device)
        self.ui.flip.stateChanged.connect(self.on_flip)
        self.ui.persona_select.currentTextChanged.connect(self.on_persona_changed)
        self.ui.button_toggle_pointer.clicked.connect(self.toggle_pointer_location)
        self.ui.image_quality_slider.valueChanged.connect(self.on_image_quality_changed)
        
        # Bind mouse event for controlling device
        self.ui.label.mousePressEvent = self.on_mouse_event(scrcpy.ACTION_DOWN)
        self.ui.label.mouseMoveEvent = self.on_mouse_event(scrcpy.ACTION_MOVE)
        self.ui.label.mouseReleaseEvent = self.on_mouse_event(scrcpy.ACTION_UP)
        
        # 初始化休息功能UI
        self._init_rest_ui()
        
        # 初始化图片质量显示
        self.ui.image_quality_value_label.setText(str(self.ui.image_quality_slider.value()))

    def _init_rest_ui(self):
        """初始化休息功能UI"""
        try:
            # 创建休息功能分组框
            rest_group = QGroupBox("休息设置")
            rest_layout = QVBoxLayout()
            
            # 启用休息功能复选框
            self.rest_enable_checkbox = QCheckBox("启用休息功能")
            self.rest_enable_checkbox.stateChanged.connect(self._on_rest_enable_changed)
            rest_layout.addWidget(self.rest_enable_checkbox)
            
            # 平台选择
            platform_layout = QHBoxLayout()
            platform_layout.addWidget(QLabel("平台:"))
            self.platform_combo = QComboBox()
            self.platform_combo.addItems(["bilibili", "抖音", "快手", "小红书"])
            self.platform_combo.currentTextChanged.connect(self._on_platform_changed)
            platform_layout.addWidget(self.platform_combo)
            rest_layout.addLayout(platform_layout)
            
            # 工作时长
            work_layout = QHBoxLayout()
            work_layout.addWidget(QLabel("工作时长(分钟):"))
            self.work_duration_spinbox = QSpinBox()
            self.work_duration_spinbox.setRange(1, 180)
            self.work_duration_spinbox.setValue(30)
            self.work_duration_spinbox.valueChanged.connect(self._on_work_duration_changed)
            work_layout.addWidget(self.work_duration_spinbox)
            rest_layout.addLayout(work_layout)
            
            # 休息时长
            rest_time_layout = QHBoxLayout()
            rest_time_layout.addWidget(QLabel("休息时长(分钟):"))
            self.rest_duration_spinbox = QSpinBox()
            self.rest_duration_spinbox.setRange(1, 60)
            self.rest_duration_spinbox.setValue(5)
            self.rest_duration_spinbox.valueChanged.connect(self._on_rest_duration_changed)
            rest_time_layout.addWidget(self.rest_duration_spinbox)
            rest_layout.addLayout(rest_time_layout)
            
            # 随机偏移
            offset_layout = QHBoxLayout()
            offset_layout.addWidget(QLabel("随机偏移(分钟):"))
            self.rest_offset_spinbox = QSpinBox()
            self.rest_offset_spinbox.setRange(0, 30)
            self.rest_offset_spinbox.setValue(2)
            self.rest_offset_spinbox.valueChanged.connect(self._on_rest_offset_changed)
            offset_layout.addWidget(self.rest_offset_spinbox)
            rest_layout.addLayout(offset_layout)
            
            rest_group.setLayout(rest_layout)
            
            # 将分组框添加到左侧配置布局
            if hasattr(self.ui, 'config_layout'):
                self.ui.config_layout.addWidget(rest_group)
                self.log_info("✅ 休息功能UI已添加")
            else:
                self.log_info("⚠️ 无法添加休息功能UI：找不到config_layout")
        except Exception as e:
            self.log_info(f"⚠️ 初始化休息功能UI失败: {e}")
    
    def _on_rest_enable_changed(self, state):
        """休息功能启用状态改变"""
        self.rest_enabled = (state == 2)  # Qt.Checked == 2
        if self.rest_enabled:
            self.log_info("✅ 休息功能已启用")
        else:
            self.log_info("❌ 休息功能已禁用")
    
    def _on_platform_changed(self, platform):
        """平台改变"""
        self.current_platform = platform
        self.log_info(f"平台切换到: {platform}")
    
    def _on_work_duration_changed(self, value):
        """工作时长改变"""
        self.work_duration = value
    
    def _on_rest_duration_changed(self, value):
        """休息时长改变"""
        self.rest_duration = value
    
    def _on_rest_offset_changed(self, value):
        """随机偏移改变"""
        self.rest_random_offset = value
    
    def _set_rest_controls_enabled(self, enabled: bool):
        """启用或禁用休息设置控件
        
        Args:
            enabled: True表示启用，False表示禁用
        """
        if hasattr(self, 'rest_enable_checkbox'):
            self.rest_enable_checkbox.setEnabled(enabled)
        if hasattr(self, 'platform_combo'):
            self.platform_combo.setEnabled(enabled)
        if hasattr(self, 'work_duration_spinbox'):
            self.work_duration_spinbox.setEnabled(enabled)
        if hasattr(self, 'rest_duration_spinbox'):
            self.rest_duration_spinbox.setEnabled(enabled)
        if hasattr(self, 'rest_offset_spinbox'):
            self.rest_offset_spinbox.setEnabled(enabled)
    
    def _set_all_config_enabled(self, enabled: bool):
        """启用或禁用所有配置控件（在交互状态下锁定配置）
        
        Args:
            enabled: True表示启用，False表示禁用
        """
        # 设备配置
        self.ui.combo_device.setEnabled(enabled)
        self.ui.flip.setEnabled(enabled)
        
        # AI配置
        self.ui.model_url_input.setEnabled(enabled)
        self.ui.persona_input.setEnabled(enabled)
        self.ui.persona_select.setEnabled(enabled)
        self.ui.use_persona_checkbox.setEnabled(enabled)
        self.ui.history_spinbox.setEnabled(enabled)
        self.ui.button_test_model.setEnabled(enabled)
        
        # 输出配置
        self.ui.save_video_checkbox.setEnabled(enabled)
        self.ui.save_audio_checkbox.setEnabled(enabled)
        self.ui.save_xml_checkbox.setEnabled(enabled)
        
        # 休息设置
        self._set_rest_controls_enabled(enabled)
    
    def _check_rest_time(self):
        """检查是否需要休息"""
        if not self.rest_enabled or not self.is_interacting or self.is_resting:
            return
        
        if self.work_start_time is None:
            return
        
        # 计算已工作时间
        elapsed_minutes = (time.time() - self.work_start_time) / 60
        
        # 计算随机工作时长
        random_offset = random.uniform(-self.rest_random_offset, self.rest_random_offset)
        target_work_duration = self.work_duration + random_offset
        
        if elapsed_minutes >= target_work_duration:
            self.log_info(f"⏰ 已工作 {elapsed_minutes:.1f} 分钟，开始休息...")
            threading.Thread(target=self._perform_rest, daemon=True).start()
    
    def _perform_rest(self):
        """执行休息流程"""
        self.is_resting = True
        
        # 更新状态显示为休息中
        self.ui.status_label.setText("状态: 休息中")
        self.ui.status_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #FF9800; padding: 8px; background-color: #FFF3E0; border-radius: 5px; }")
        
        try:
            # 1. 关闭app
            self.log_info(f"📱 关闭应用: {self.current_platform}")
            package_name = self.platform_packages.get(self.current_platform, "tv.danmaku.bili")
            result = self.adb_utils.execute_command(["shell", "am", "force-stop", package_name])
            time.sleep(1)
            
            # 2. 返回桌面
            self.log_info("🏠 返回桌面")
            self.adb_utils.execute_command(["shell", "input", "keyevent", "KEYCODE_HOME"])
            time.sleep(1)
            
            # 3. 计算随机休息时长
            random_offset = random.uniform(-self.rest_random_offset, self.rest_random_offset)
            actual_rest_duration = max(1, self.rest_duration + random_offset)
            
            self.log_info(f"😴 休息 {actual_rest_duration:.1f} 分钟...")
            time.sleep(actual_rest_duration * 60)
            
            # 4. 打开app
            self.log_info(f"📱 打开应用: {self.current_platform}")
            self.adb_utils.execute_command([
                "shell", "monkey", "-p", package_name, 
                "-c", "android.intent.category.LAUNCHER", "1"
            ])
            time.sleep(3)
            
            # 5. 如果是bilibili，点击特定位置
            if self.current_platform == "bilibili":
                self.log_info("👆 点击 bilibili 特定位置(500, 600)")
                self.u2_device.click(500, 600)
                time.sleep(1)
            
            # 重置工作开始时间
            self.work_start_time = time.time()
            self.log_info("✅ 休息结束，继续工作")
            
            # 恢复状态显示为正在交互
            self.ui.status_label.setText("状态: 正在交互")
            self.ui.status_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #4CAF50; padding: 8px; background-color: #E8F5E9; border-radius: 5px; }")
            
        except Exception as e:
            self.log_info(f"⚠️ 休息流程出错: {e}")
        finally:
            self.is_resting = False
    
    def _update_status_info(self):
        """更新状态详细信息（每秒调用一次）"""
        if not self.is_interacting:
            self.ui.info_label.setText("会话: 未开始\n平台: -\n运行时长: 0分钟")
            return
        
        # 计算运行时长
        if self.interaction_start_time:
            elapsed = time.time() - self.interaction_start_time
            minutes = int(elapsed / 60)
            seconds = int(elapsed % 60)
            duration_str = f"{minutes}分{seconds}秒"
        else:
            duration_str = "0分钟"
        
        # 获取会话ID
        session_id = self.session_data.get('session_id', 'unknown') if self.session_data else 'unknown'
        
        # 更新显示
        info_text = f"会话: {session_id}\n平台: {self.current_platform}\n运行时长: {duration_str}"
        self.ui.info_label.setText(info_text)
    
    def log_info(self, message):
        msg = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}"
        print(msg)
        self.ui.log_text.append(msg)
        scrollbar = self.ui.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_persona_changed(self):
        self.ai_persona = None
    
    def on_image_quality_changed(self, value):
        """图片质量滑块值改变时的回调"""
        self.ui.image_quality_value_label.setText(str(value))

    def test_model_connection(self):
        model_url = self.ui.model_url_input.text().strip()
        if not model_url:
            QMessageBox.warning(self, "提示", "请输入模型URL")
            return
        
        # 检查设备是否可用
        if not hasattr(self, 'u2_device') or self.u2_device is None:
            QMessageBox.warning(self, "提示", "设备未连接，无法截图")
            return
        
        self.log_info(f"测试模型连接: {model_url}")
        self.log_info(f"正在截图...")
        
        try:
            # 1. 截图
            screenshot = self.u2_device.screenshot()
            
            # 2. 保存到临时文件
            import tempfile
            temp_dir = Path(tempfile.gettempdir())
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]
            test_screenshot_path = temp_dir / f"test_screenshot_{timestamp}.png"
            screenshot.save(str(test_screenshot_path))
            self.log_info(f"✅ 截图已保存: {test_screenshot_path.name}")
            
            # 3. 创建测试客户端
            test_client = OpenAI(base_url=model_url, api_key="1234567890")
            
            # 4. 加载persona（如果选择了且启用了使用Persona）
            persona_template = self.ui.persona_select.currentText()
            use_persona = self.ui.use_persona_checkbox.isChecked() and persona_template != "None"
            test_persona = None
            if use_persona:
                test_persona = self._load_persona(persona_template)
            
            # 5. 构建prompt（测试模式，不使用历史记录）
            system_prompt = get_system_prompt(use_persona, test_persona)
            user_prompt = get_user_prompt(
                history_screenshots=None,
                history_actions=None,
                current_screenshots=[str(test_screenshot_path)],
                audio_transcript=None
            )
            
            # 6. 编码图片
            import base64
            from PIL import Image
            import io
            
            # 获取图片质量滑块的值
            image_quality = self.ui.image_quality_slider.value()
            
            def encode_image(image_path, max_pixels=MAX_PIXELS, quality=85):
                img = Image.open(image_path)
                width, height = img.size
                total_pixels = width * height
                
                # 如果总像素数超过限制，按比例缩放
                if total_pixels > max_pixels:
                    import math
                    scale = math.sqrt(max_pixels / total_pixels)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                buffer.seek(0)
                encoded = base64.b64encode(buffer.read()).decode('utf-8')
                return encoded
            
            # 7. 构建消息
            base64_image = encode_image(str(test_screenshot_path), quality=image_quality)
            user_content = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            # 8. 调用LLM（记录延迟）
            self.log_info(f"🤖 正在调用LLM...")
            llm_start_time = time.time()
            
            response = test_client.chat.completions.create(
                model="qwen",
                messages=messages,
                max_tokens=256,
                temperature=0.0
            )
            
            llm_end_time = time.time()
            llm_delay = llm_end_time - llm_start_time
            
            response_text = response.choices[0].message.content
            self.log_info(f"✅ 模型连接成功")
            self.log_info(f"⏱️ LLM推理延迟: {llm_delay:.2f}秒")
            self.log_info(f"📝 AI回复:\n{response_text}")
            
            # 9. 显示结果对话框（包含延迟信息）
            msg = QMessageBox(self)
            msg.setWindowTitle("测试结果")
            msg.setText("模型连接成功！")
            delay_info = f"推理延迟: {llm_delay:.2f}秒\n\n"
            msg.setInformativeText(f"{delay_info}AI回复:\n\n{response_text[:450]}{'...' if len(response_text) > 450 else ''}")
            msg.setDetailedText(f"推理延迟: {llm_delay:.2f}秒\n\n完整回复:\n{response_text}")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.exec()
            
            # 10. 清理临时文件
            try:
                if test_screenshot_path.exists():
                    test_screenshot_path.unlink()
            except Exception as e:
                self.log_info(f"⚠️ 清理临时文件失败: {e}")
                
        except Exception as e:
            error_msg = str(e)
            self.log_info(f"❌ 测试失败: {error_msg}")
            QMessageBox.critical(self, "测试失败", f"测试过程中出现错误:\n\n{error_msg}")

    def _load_persona(self, persona_name):
        persona_path = Path(__file__).parent.parent / "data" / persona_name / "persona.json"
        if not persona_path.exists():
            self.log_info(f"⚠️ Persona文件不存在: {persona_path}")
            return None
        
        with open(persona_path, encoding='utf-8') as f:
            persona = json.load(f)
        
        self.log_info(f"✅ 加载Persona: {persona_name}")
        return persona

    def start_interaction(self):
        if self.is_interacting:
            self.log_info("交互已在进行中")
            return
        
        model_url = self.ui.model_url_input.text().strip()
        if not model_url:
            QMessageBox.warning(self, "提示", "请输入模型URL")
            return
        
        # 获取persona（必填）
        persona_name = self.ui.persona_input.text().strip()
        if not persona_name:
            QMessageBox.warning(self, "提示", "请输入Persona名称（必填）")
            return
        
        self.ai_client = OpenAI(base_url=model_url, api_key="1234567890")
        self.ai_max_history = self.ui.history_spinbox.value()
        
        # 加载persona模板（如果选择了且启用了使用Persona）
        persona_template = self.ui.persona_select.currentText()
        use_persona = self.ui.use_persona_checkbox.isChecked() and persona_template != "None"
        if use_persona:
            self.ai_persona = self._load_persona(persona_template)
        else:
            self.ai_persona = None
        
        deploy_log_dir = Path(__file__).parent.parent / "deploy_log"
        deploy_log_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir = deploy_log_dir
        
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 使用persona作为目录名，同一个人设下的不同session放在一起
        self.session_dir = deploy_log_dir / persona_name
        self.session_dir.mkdir(exist_ok=True, parents=True)
        
        (self.session_dir / "screenshots").mkdir(exist_ok=True)
        (self.session_dir / "videos").mkdir(exist_ok=True)
        (self.session_dir / "audios").mkdir(exist_ok=True)
        (self.session_dir / "xml").mkdir(exist_ok=True)
        
        self.ai_history_screenshots = []
        self.ai_history_actions = []
        self.ai_step_count = 0
        
        # 初始化会话数据，将persona保存在collector字段中
        platform_name = self.current_platform  # 使用当前平台
        
        self.session_data = {
            "session_id": session_id,
            "collector": persona_name,  # persona保存在collector中
            "platform": platform_name,
            "start_time": datetime.datetime.now().isoformat(),
            "actions": []
        }
        
        self.is_interacting = True
        self.interaction_start_time = time.time()  # 记录开始时间
        self.ui.button_start.setEnabled(False)
        self.ui.button_stop.setEnabled(True)
        
        # 禁用所有配置控件（锁定配置）
        self._set_all_config_enabled(False)
        
        # 更新状态显示
        self.ui.status_label.setText("状态: 正在交互")
        self.ui.status_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #4CAF50; padding: 8px; background-color: #E8F5E9; border-radius: 5px; }")
        
        self.log_info(f"✅ 开始交互")
        self.log_info(f"Persona: {persona_name}")
        self.log_info(f"输出目录: {self.session_dir}")
        
        # 启动休息功能定时器
        if self.rest_enabled:
            self.work_start_time = time.time()
            self.rest_check_timer.start(30000)  # 每30秒检查一次
            self.log_info(f"⏰ 休息功能已启动，工作 {self.work_duration} 分钟后休息 {self.rest_duration} 分钟")
        
        threading.Thread(target=self._interaction_loop, daemon=True).start()

    def stop_interaction(self):
        if not self.is_interacting:
            return
        
        self.is_interacting = False
        self.interaction_start_time = None  # 重置开始时间
        self.ui.button_start.setEnabled(True)
        self.ui.button_stop.setEnabled(False)
        
        # 重新启用所有配置控件
        self._set_all_config_enabled(True)
        
        # 更新状态显示
        self.ui.status_label.setText("状态: 未交互")
        self.ui.status_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #9E9E9E; padding: 8px; background-color: #F5F5F5; border-radius: 5px; }")
        
        # 停止休息检查定时器
        if self.rest_check_timer.isActive():
            self.rest_check_timer.stop()
            self.log_info("⏰ 休息检查定时器已停止")
        
        if self.is_recording:
            self.stop_recording(save_files=False)
        
        # 生成最终结果JSON
        if self.session_data:
            self.session_data["end_time"] = datetime.datetime.now().isoformat()
            self.session_data["total_steps"] = self.ai_step_count
            
            # 保存最终会话文件（直接保存在persona目录下）
            session_file = self.session_dir / f"session_{self.session_data['session_id']}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, ensure_ascii=False, indent=2)
            
            self.log_info(f"✅ 最终结果已保存: {session_file.name}")
            self.log_info(f"共 {self.ai_step_count} 步")
        
        self.log_info(f"✅ 交互结束")

    def _interaction_loop(self):
        while self.is_interacting:
            self._perform_ai_step()
            time.sleep(0.5)

    def _perform_ai_step(self):
        self.ai_step_count += 1
        self.ui.step_label.setText(f"步数: {self.ai_step_count}")
        
        self.log_info(f"\n{'='*50}")
        self.log_info(f"Step {self.ai_step_count}")
        self.log_info(f"{'='*50}")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]
        step_start_time = time.time()
        
        # 1. 实时截图
        screenshot_filename = f"step_{self.ai_step_count}_{timestamp}.png"
        screenshot_path = self.session_dir / "screenshots" / screenshot_filename
        screenshot = self.u2_device.screenshot()
        screenshot.save(str(screenshot_path))
        self.log_info(f"📸 截图: {screenshot_filename}")
        
        # 2. 使用异步获取的 XML
        xml_path = None
        if self.ui.save_xml_checkbox.isChecked():
            xml_filename = f"{timestamp}.xml"
            xml_path = self.session_dir / "xml" / xml_filename
            
            # 优先使用异步获取的 XML
            if self.current_xml_text is not None:
                xml_path.write_text(self.current_xml_text, encoding='utf-8')
                self.log_info(f"📝 XML: {xml_filename} (使用异步获取)")
                self.current_xml_text = None  # 清空已使用的 XML
            # 否则实时获取
            else:
                xml_text = self.u2_device.dump_hierarchy()
                xml_path.write_text(xml_text, encoding='utf-8')
                self.log_info(f"📝 XML: {xml_filename} (实时获取)")
        
        # 3. 开始录制（如果需要）
        video_path = None
        audio_path = None
        if self.ui.save_video_checkbox.isChecked() or self.ui.save_audio_checkbox.isChecked():
            video_path = self.session_dir / "videos" / f"{timestamp}.mp4" if self.ui.save_video_checkbox.isChecked() else None
            audio_path = self.session_dir / "audios" / f"{timestamp}.wav" if self.ui.save_audio_checkbox.isChecked() else None
            
            if video_path or audio_path:
                self.start_recording(video_path, audio_path)
        
        # 4. 调用LLM
        self.log_info(f"🤖 调用LLM...")
        response = self._get_ai_response(str(screenshot_path))
        self.log_info(f"响应:\n{response}")
        
        # 5. 解析代码
        code = self._parse_code(response)
        if not code:
            self.log_info(f"⚠️ 未找到代码块")
            return
        
        self.log_info(f"执行代码:")
        # 解析代码中的动作
        actions = self._parse_actions_from_code(code)
        
        # 6. 执行代码
        self._execute_ai_code(code)
        
        # 7. 停止录制
        if self.is_recording:
            stopped_video_path, stopped_audio_path = self.stop_recording(save_files=True)
            # 使用实际保存的路径
            if stopped_video_path:
                video_path = stopped_video_path
            if stopped_audio_path:
                audio_path = stopped_audio_path
        
        # 计算观看时长（从步骤开始到结束）
        viewing_duration = time.time() - step_start_time
        
        # 8. 保存历史数据
        self.ai_history_screenshots.append(str(screenshot_path))
        self.ai_history_actions.append(f"```python\n{code}\n```")
        
        # 9. 保存会话数据到JSON
        if self.session_data:
            # 计算相对于项目根目录的路径
            project_root = Path(__file__).parent.parent
            
            action_entry = {
                "step": self.ai_step_count,
                "timestamp": timestamp,
                "actions": actions,
                "viewing_duration": round(viewing_duration, 2),
                "screenshot": str(screenshot_path.relative_to(project_root)),
                "response": response,
                "code": code
            }
            
            # 添加文件路径（相对于项目根目录）
            if xml_path and xml_path.exists():
                action_entry["xml_path"] = str(xml_path.relative_to(project_root))
            if video_path and video_path.exists():
                action_entry["video_path"] = str(video_path.relative_to(project_root))
            if audio_path and audio_path.exists():
                action_entry["audio_path"] = str(audio_path.relative_to(project_root))
            
            self.session_data["actions"].append(action_entry)
            
            # 实时保存会话文件（增量保存）
            session_file = self.session_dir / f"session_{self.session_data['session_id']}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, ensure_ascii=False, indent=2)
        
        # 10. 滑动到下一个视频
        self.u2_device.swipe(0.5, 0.8, 0.5, 0.2, duration=0.1)
        self.log_info(f"→ swipe up (下一个视频)")
        time.sleep(1)
        
        # 记录下滑后1秒的时间点，作为下一次延迟计算的起点
        self.llm_start_time_after_swipe = time.time()
        
        # 在 swipe 后异步获取下一个视频的 XML
        def _fetch_next_xml():
            time.sleep(1)  # 等待 1 秒让 UI 稳定
            if self.u2_device:
                try:
                    self.current_xml_text = self.u2_device.dump_hierarchy()
                    if self.current_xml_text:
                        self.log_info(f"📝 XML已异步获取，大小: {len(self.current_xml_text)} chars")
                except Exception as e:
                    self.log_info(f"⚠️ XML异步获取失败: {e}")
                    self.current_xml_text = None
        
        threading.Thread(target=_fetch_next_xml, daemon=True).start()

    def _get_ai_response(self, screenshot_path):
        # 根据复选框和模板选择决定是否使用persona
        persona_template = self.ui.persona_select.currentText()
        use_persona = self.ui.use_persona_checkbox.isChecked() and persona_template != "None"
        
        # 每次调用时都从UI读取历史窗口大小（确保使用最新值）
        current_max_history = self.ui.history_spinbox.value()
        
        system_prompt = get_system_prompt(use_persona, self.ai_persona)
        user_prompt = get_user_prompt(
            history_screenshots=self.ai_history_screenshots[-current_max_history:] if self.ai_history_screenshots else None,
            history_actions=self.ai_history_actions[-current_max_history:] if self.ai_history_actions else None,
            current_screenshots=[screenshot_path],
            audio_transcript=None
        )
        
        # 收集所有图像
        all_images = []
        if self.ai_history_screenshots:
            all_images.extend(self.ai_history_screenshots[-current_max_history:])
        all_images.append(screenshot_path)
        
        # 将图像编码为base64并构建消息内容
        import base64
        from PIL import Image
        import io
        
        # 获取图片质量滑块的值
        image_quality = self.ui.image_quality_slider.value()
        
        def encode_image(image_path, max_pixels=MAX_PIXELS, quality=85):
            """压缩并编码图片
            
            Args:
                image_path: 图片路径
                max_pixels: 最大像素数（宽*高），默认288000（与训练时保持一致）
                quality: JPEG质量（1-100）
            """
            import math
            
            # 打开图片
            img = Image.open(image_path)
            
            # 如果图片总像素数超过限制，进行压缩
            width, height = img.size
            total_pixels = width * height
            
            if total_pixels > max_pixels:
                # 计算缩放比例（保持宽高比）
                scale = math.sqrt(max_pixels / total_pixels)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # 缩放图片
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 转换为RGB（如果是RGBA）
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # 保存到内存缓冲区
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            
            # 编码为base64
            encoded = base64.b64encode(buffer.read()).decode('utf-8')
            
            return encoded
        
        # 构建用户消息内容，包含文本和图像
        user_content = []
        
        # 添加文本部分
        user_content.append({
            "type": "text",
            "text": user_prompt
        })
        
        # 添加所有图像
        for img_path in all_images:
            base64_image = encode_image(img_path, quality=image_quality)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # 如果已经记录了下滑后1秒的时间点，使用它作为延迟计算的起点
        # 否则使用当前时间（第一次调用时）
        if self.llm_start_time_after_swipe is not None:
            llm_start_time = self.llm_start_time_after_swipe
            self.llm_start_time_after_swipe = None  # 清空，等待下一次下滑后设置
        else:
            llm_start_time = time.time()
        
        response = self.ai_client.chat.completions.create(
            model="qwen",
            messages=messages,
            max_tokens=256,
            temperature=0.0
        )
        
        # 记录LLM返回后的时间，计算延迟（从下滑后1秒到LLM返回的时间）
        llm_end_time = time.time()
        llm_delay = llm_end_time - llm_start_time
        
        # 更新延迟历史记录（平滑算法：保留最近10次）
        self.llm_delay_history.append(llm_delay)
        if len(self.llm_delay_history) > self.llm_delay_max_history:
            self.llm_delay_history.pop(0)  # 移除最旧的记录
        
        # 计算最近10次延迟的均值
        self.llm_avg_delay = sum(self.llm_delay_history) / len(self.llm_delay_history)
        
        return response.choices[0].message.content

    def _parse_code(self, response_text):
        code_pattern = r"```python\s*(.*?)\s*```"
        match = re.search(code_pattern, response_text, re.DOTALL)
        
        if match:
            code = match.group(1).strip()
            return code
        
        return None
    
    def _parse_actions_from_code(self, code):
        """从代码中解析动作（like, comment, share）"""
        actions = []
        
        # 检查是否调用了like()
        if re.search(r'\blike\s*\(', code):
            actions.append({"type": "like"})
        
        # 检查是否调用了comment()
        comment_match = re.search(r'\bcomment\s*\(\s*["\']?([^"\']*)["\']?\s*\)', code)
        if comment_match:
            comment_text = comment_match.group(1) if comment_match.group(1) else ""
            actions.append({"type": "comment", "text": comment_text})
        
        # 检查是否调用了share()
        share_match = re.search(r'\bshare\s*\(\s*["\']?([^"\']*)["\']?\s*\)', code)
        if share_match:
            share_who = share_match.group(1) if share_match.group(1) else ""
            actions.append({"type": "share", "who": share_who})
        
        return actions

    def _execute_ai_code(self, code):
        screen_width, screen_height = self.u2_device.window_size()
        
        def tap(x, y):
            self.u2_device.click(x, y)
            self.log_info(f"  → tap({x}, {y})")
        
        def swipe(x1, y1, x2, y2, duration=0.1):
            self.u2_device.swipe(x1, y1, x2, y2, duration)
            self.log_info(f"  → swipe({x1}, {y1}, {x2}, {y2})")
        
        def watch(second=5.0):
            # 减去LLM延迟（从query到回复的时间）
            adjusted_second = max(0.0, second - self.llm_avg_delay)
            self.log_info(f"  → watch({second}s)")
            if adjusted_second > 0:
                time.sleep(adjusted_second)
        
        def like():
            # like_x = int(screen_width * 0.95)
            # like_y = int(screen_height * 0.65)
            # tap(like_x, like_y)
            # time.sleep(0.3)
            self.log_info(f"  → like()")
        
        def comment(text=""):
            # comment_x = int(screen_width * 0.95)
            # comment_y = int(screen_height * 0.75)
            # tap(comment_x, comment_y)
            # time.sleep(0.5)
            # if text:
            #     self.u2_device.send_keys(text)
            #     time.sleep(0.3)
            #     self.u2_device.press("enter")
            #     time.sleep(0.5)
            # self.u2_device.press("back")
            # time.sleep(0.3)
            self.log_info(f"  → comment({text})")

        
        def share(who=""):
            # share_x = int(screen_width * 0.95)
            # share_y = int(screen_height * 0.85)
            # tap(share_x, share_y)
            # time.sleep(0.5)
            # self.u2_device.press("back")
            # time.sleep(0.3)
            self.log_info(f"  → share({who})")
        
        local_vars = {
            'tap': tap,
            'swipe': swipe,
            'watch': watch,
            'like': like,
            'comment': comment,
            'share': share
        }
        
        exec(code, {}, local_vars)

    def _video_write_worker(self):
        """后台线程：异步写入视频帧"""
        while self.video_write_thread_running:
            try:
                # 从队列获取帧，超时1秒
                frame_data = self.video_frame_queue.get(timeout=1.0)
                
                if frame_data is None:
                    # None 是停止信号
                    break
                
                frame, video_writer = frame_data
                
                # 写入视频帧
                if video_writer and video_writer.isOpened():
                    try:
                        video_writer.write(frame)
                        self.video_frame_count += 1
                    except Exception as e:
                        self.log_info(f"⚠️ 后台写入视频帧失败: {e}")
                
                self.video_frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.log_info(f"⚠️ 视频写入线程异常: {e}")
    
    def start_recording(self, video_path, audio_path):
        if self.is_recording:
            self.stop_recording()
        
        self.current_video_path = video_path
        self.current_audio_path = audio_path
        with self.video_writer_lock:
            self.video_writer = None
        self.video_frame_count = 0
        self.audio_buffer = []
        self.is_recording = True
        self.is_audio_recording = True
        self.current_recording_start_time = time.time()  # 记录录制开始时间
        
        # 清空队列
        while not self.video_frame_queue.empty():
            try:
                self.video_frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # 启动后台写入线程
        if not self.video_write_thread_running:
            self.video_write_thread_running = True
            self.video_write_thread = threading.Thread(target=self._video_write_worker, daemon=True)
            self.video_write_thread.start()
            self.log_info("✅ 视频异步写入线程已启动")
        
        return True

    def stop_recording(self, save_files=True):
        """停止录制并异步写入音频
        
        Args:
            save_files: 是否保存文件。False时直接丢弃，不保存到磁盘
        """
        if not self.is_recording:
            return None, None
        
        # 先设置标志，阻止新的写入
        self.is_recording = False
        self.is_audio_recording = False
        
        # 等待视频队列清空（最多等待3秒）
        queue_wait_start = time.time()
        while not self.video_frame_queue.empty():
            if time.time() - queue_wait_start > 3.0:
                self.log_info(f"⚠️ 视频队列等待超时，队列剩余: {self.video_frame_queue.qsize()} 帧")
                break
            time.sleep(0.1)
        
        if self.video_frame_queue.empty():
            self.log_info(f"✅ 视频队列已清空")
        
        video_path = self.current_video_path
        audio_path = self.current_audio_path
        frame_count = self.video_frame_count
        audio_data = self.audio_buffer.copy() if save_files else []
        
        # 使用锁保护 video_writer 的访问
        with self.video_writer_lock:
            video_writer = self.video_writer
            # 清空引用，防止on_frame继续写入
            self.video_writer = None
        
        self.video_frame_count = 0
        self.audio_buffer = []
        
        def _write_files():
            # 关闭视频writer
            if video_writer:
                try:
                    video_writer.release()
                    if save_files and video_path and video_path.exists():
                        video_size = video_path.stat().st_size / 1024
                        self.log_info(f"✅ 视频: {video_path.name} ({frame_count}帧, {self.video_fps}fps, {video_size:.1f}KB)")
                    elif not save_files and video_path and video_path.exists():
                        # 不保存时删除文件
                        video_path.unlink()
                except Exception as e:
                    self.log_info(f"⚠️ 视频保存失败: {e}")
            
            # 写入音频
            if save_files and audio_data and audio_path:
                try:
                    with wave.open(str(audio_path), 'wb') as wav_file:
                        wav_file.setnchannels(self.audio_channels)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(self.audio_sample_rate)
                        for data in audio_data:
                            wav_file.writeframes(data)
                    
                    audio_size = audio_path.stat().st_size / 1024
                    self.log_info(f"✅ 音频: {audio_path.name} ({audio_size:.1f}KB)")
                except Exception as e:
                    self.log_info(f"⚠️ 音频保存失败: {e}")
        
        threading.Thread(target=_write_files, daemon=True).start()
        
        return video_path, audio_path

    def on_audio(self, audio_data):
        if audio_data is None or not isinstance(audio_data, np.ndarray):
            return
        
        if audio_data.ndim == 2:
            audio_mono = audio_data.mean(axis=0).astype(np.float32)
        else:
            audio_mono = audio_data.astype(np.float32)
        
        if audio_mono.dtype == np.int16:
            audio_mono = audio_mono / 32768.0
        
        if self.audio_stream:
            self.audio_stream.write(audio_mono.tobytes(), exception_on_underflow=False)
        
        if self.is_audio_recording:
            audio_int16 = (audio_mono * 32767).astype(np.int16)
            self.audio_buffer.append(audio_int16.tobytes())
    
    def choose_device(self, device):
        if device not in self.devices:
            msgBox = QMessageBox()
            msgBox.setText(f"Device serial [{device}] not found!")
            msgBox.exec()
            return

        self.ui.combo_device.setCurrentText(device)
        if getattr(self, "client", None):
            self.client.stop()
            self.client.device = adb.device(serial=device)
            
        self.device = adb.device(serial=device)
        self.u2_device = u2.connect(device)
        # 更新ADB工具的设备序列号
        if hasattr(self, 'adb_utils'):
            self.adb_utils = ADBUtils(device_serial=device, logger=logging.getLogger(__name__))
        self.log_info(f"已连接到设备: {device}")

    def list_devices(self):
        self.ui.combo_device.clear()
        items = [i.serial for i in adb.device_list()]
        self.ui.combo_device.addItems(items)
        return items

    def on_flip(self, _):
        self.client.flip = self.ui.flip.isChecked()

    def toggle_pointer_location(self):
        """切换指针位置显示"""
        is_enabled = self.ui.button_toggle_pointer.isChecked()
        success = self.adb_utils.toggle_pointer_location(is_enabled)
        
        if success:
            status = "开启" if is_enabled else "关闭"
            self.log_info(f"指针位置显示已{status}")
            self.ui.button_toggle_pointer.setText(f"{'关闭' if is_enabled else '显示'}指针位置")
        else:
            self.log_info("指针位置设置失败")
            self.ui.button_toggle_pointer.setChecked(not is_enabled)

    def on_mouse_event(self, action=scrcpy.ACTION_DOWN):
        """处理鼠标事件并转换为设备触摸事件"""
        def handler(evt: QMouseEvent):
            # 清除输入框焦点，避免干扰
            focused_widget = QApplication.focusWidget()
            if focused_widget is not None:
                focused_widget.clearFocus()
            
            # 获取当前显示的图片尺寸和设备真实尺寸
            device_width, device_height = self.client.resolution
            pixmap = self.ui.label.pixmap()
            
            if pixmap:
                # 获取label的实际尺寸
                label_width = self.ui.label.width()
                label_height = self.ui.label.height()
                
                # 获取图片的显示尺寸
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                
                # 计算图片在label中的偏移量（居中显示）
                offset_x = max(0, (label_width - pixmap_width) / 2)
                offset_y = max(0, (label_height - pixmap_height) / 2)
                
                # 获取鼠标在label中的位置
                mouse_x = evt.position().x()
                mouse_y = evt.position().y()
                
                # 减去偏移量，得到鼠标在图片上的实际位置
                image_x = mouse_x - offset_x
                image_y = mouse_y - offset_y
                
                # 确保坐标在图片范围内
                if 0 <= image_x <= pixmap_width and 0 <= image_y <= pixmap_height:
                    # 计算缩放比例并转换坐标
                    ratio_x = device_width / pixmap_width
                    ratio_y = device_height / pixmap_height
                    touch_x = image_x * ratio_x
                    touch_y = image_y * ratio_y
                else:
                    # 点击在图片外，不处理
                    return
            else:
                # 回退到原来的计算方式
                ratio = self.max_width / max(self.client.resolution)
                touch_x = evt.position().x() / ratio
                touch_y = evt.position().y() / ratio
            
            # 通过scrcpy控制发送触摸事件到设备
            self.client.control.touch(touch_x, touch_y, action)

        return handler

    def on_init(self):
        self.setWindowTitle(f"Persona视频交互系统 - {self.client.device_name}")
        if self.audio_stream:
            self.log_info("✅ 音频播放已启用")
        
    def on_frame(self, frame):
        app.processEvents()
        
        if frame is not None:
            device_width, device_height = self.client.resolution
            ratio = min(self.max_width / device_width, 800 / device_height)
            
            display_width = int(device_width * ratio)
            display_height = int(device_height * ratio)
            
            image = QImage(
                frame,
                frame.shape[1],
                frame.shape[0],
                frame.shape[1] * 3,
                QImage.Format_BGR888,
            )
            pix = QPixmap(image)
            
            scaled_pix = pix.scaled(display_width, display_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.label.setPixmap(scaled_pix)
            
            current_size = (display_width, display_height)
            if not self.window_size_initialized or self.last_display_size != current_size:
                self.update_window_size(display_width, display_height)
                self.last_display_size = current_size
                self.window_size_initialized = True
            
            # 如果正在录制，将帧放入异步队列
            if self.is_recording:
                with self.video_writer_lock:
                    if self.video_writer is None:
                        # 第一帧到来时初始化VideoWriter
                        try:
                            height, width = frame.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            self.video_writer = cv2.VideoWriter(
                                str(self.current_video_path), 
                                fourcc, 
                                self.video_fps, 
                                (width, height)
                            )
                            if self.video_writer.isOpened():
                                self.log_info(f"✅ VideoWriter初始化成功 ({width}x{height})")
                            else:
                                self.log_info(f"⚠️ VideoWriter初始化失败")
                                self.video_writer = None
                                return
                        except Exception as e:
                            self.log_info(f"⚠️ VideoWriter初始化异常: {e}")
                            self.video_writer = None
                            return
                    
                    # 将帧放入异步队列（非阻塞）
                    if self.video_writer and self.video_writer.isOpened():
                        try:
                            # 复制帧数据，避免引用问题
                            frame_copy = frame.copy()
                            # 非阻塞放入队列
                            self.video_frame_queue.put_nowait((frame_copy, self.video_writer))
                        except queue.Full:
                            # 队列满了，丢弃这一帧
                            pass
            
    def update_window_size(self, display_width, display_height):
        left_panel_width = 400
        margin = 30
        window_width = left_panel_width + display_width + margin
        
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            max_width = screen_geometry.width() - 50
            max_height = screen_geometry.height() - 50
            window_width = min(window_width, max_width)
            window_height = max(display_height + 80, 700)
            window_height = min(window_height, max_height)
        else:
            window_height = max(display_height + 80, 700)
        
        self.ui.label.setFixedSize(display_width, display_height)
        self.resize(window_width, window_height)

    def closeEvent(self, _):
        # 停止视频写入线程
        if self.video_write_thread_running:
            self.video_write_thread_running = False
            # 发送停止信号
            try:
                self.video_frame_queue.put_nowait(None)
            except queue.Full:
                pass
            # 等待线程结束
            if self.video_write_thread:
                self.video_write_thread.join(timeout=2.0)
            self.log_info("✅ 视频写入线程已停止")
        
        if self.is_recording:
            video_path = self.current_video_path
            self.is_recording = False
            with self.video_writer_lock:
                if self.video_writer:
                    try:
                        self.video_writer.release()
                    except Exception as e:
                        self.log_info(f"⚠️ 关闭VideoWriter失败: {e}")
                    self.video_writer = None
            if video_path and video_path.exists():
                video_path.unlink()
            self.video_frame_count = 0
        
        self.is_audio_recording = False
        self.audio_buffer = []
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        
        self.client.stop()
        self.alive = False


def main():
    parser = ArgumentParser(description="Persona视频交互系统")
    parser.add_argument(
        "-m",
        "--max_width",
        type=int,
        default=800,
        help="窗口最大宽度",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        help="设备序列号",
    )
    parser.add_argument("--encoder_name", type=str, help="编码器名称")
    args = parser.parse_args()

    m = MainWindow(args.max_width, args.device, args.encoder_name)
    m.show()

    m.client.start()
    while m.alive:
        m.client.start()


if __name__ == "__main__":
    main()
