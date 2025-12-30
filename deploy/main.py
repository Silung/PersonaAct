from argparse import ArgumentParser
from typing import Optional
import time
import os
import sys
import json
import datetime
import threading
import random
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
        
        # Bind mouse event for controlling device
        self.ui.label.mousePressEvent = self.on_mouse_event(scrcpy.ACTION_DOWN)
        self.ui.label.mouseMoveEvent = self.on_mouse_event(scrcpy.ACTION_MOVE)
        self.ui.label.mouseReleaseEvent = self.on_mouse_event(scrcpy.ACTION_UP)
        
        # 初始化休息功能UI
        self._init_rest_ui()

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
            
        except Exception as e:
            self.log_info(f"⚠️ 休息流程出错: {e}")
        finally:
            self.is_resting = False
    
    def log_info(self, message):
        msg = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}"
        print(msg)
        self.ui.log_text.append(msg)
        scrollbar = self.ui.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_persona_changed(self):
        self.ai_persona = None

    def test_model_connection(self):
        model_url = self.ui.model_url_input.text().strip()
        if not model_url:
            QMessageBox.warning(self, "提示", "请输入模型URL")
            return
        
        self.log_info(f"测试模型连接: {model_url}")
        
        test_client = OpenAI(base_url=model_url, api_key="1234567890")
        response = test_client.chat.completions.create(
            model="qwen",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10,
            temperature=0.0
        )
        self.log_info(f"✅ 模型连接成功")
        QMessageBox.information(self, "成功", "模型连接成功！")

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
        
        # 加载persona模板（如果选择了）
        persona_template = self.ui.persona_select.currentText()
        if persona_template != "None":
            self.ai_persona = self._load_persona(persona_template)
        
        deploy_log_dir = Path(__file__).parent.parent / "deploy_log"
        deploy_log_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir = deploy_log_dir
        
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 使用persona作为目录名的一部分
        self.session_dir = deploy_log_dir / persona_name / f"session_{session_id}"
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
        self.ui.button_start.setEnabled(False)
        self.ui.button_stop.setEnabled(True)
        
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
        self.ui.button_start.setEnabled(True)
        self.ui.button_stop.setEnabled(False)
        
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
            
            # 保存最终会话文件
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
        
        code = self._parse_code(response)
        if not code:
            self.log_info(f"⚠️ 未找到代码块")
            return
        
        self.log_info(f"执行代码:")
        # 解析代码中的动作
        actions = self._parse_actions_from_code(code)
        self._execute_ai_code(code)
        
        if self.is_recording:
            stopped_video_path, stopped_audio_path = self.stop_recording(save_files=True)
            # 使用实际保存的路径
            if stopped_video_path:
                video_path = stopped_video_path
            if stopped_audio_path:
                audio_path = stopped_audio_path
        
        # 计算观看时长（从步骤开始到结束）
        viewing_duration = time.time() - step_start_time
        
        self.ai_history_screenshots.append(str(screenshot_path))
        self.ai_history_actions.append(f"```python\n{code}\n```")
        
        # 将step信息直接添加到会话数据的actions中
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
        
        self.u2_device.swipe(0.5, 0.8, 0.5, 0.2, duration=0.1)
        self.log_info(f"→ swipe up (下一个视频)")
        time.sleep(1)
        
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
        use_persona = self.ui.persona_select.currentText() != "None"
        
        system_prompt = get_system_prompt(use_persona, self.ai_persona)
        user_prompt = get_user_prompt(
            history_screenshots=self.ai_history_screenshots[-self.ai_max_history:] if self.ai_history_screenshots else None,
            history_actions=self.ai_history_actions[-self.ai_max_history:] if self.ai_history_actions else None,
            current_screenshots=[screenshot_path],
            audio_transcript=None
        )
        
        # 收集所有图像
        all_images = []
        if self.ai_history_screenshots:
            all_images.extend(self.ai_history_screenshots[-self.ai_max_history:])
        all_images.append(screenshot_path)
        
        # 将图像编码为base64并构建消息内容
        import base64
        
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        # 构建用户消息内容，包含文本和图像
        user_content = []
        
        # 添加文本部分
        user_content.append({
            "type": "text",
            "text": user_prompt
        })
        
        # 添加所有图像
        for img_path in all_images:
            base64_image = encode_image(img_path)
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
        
        # 记录LLM调用前的时间
        llm_start_time = time.time()
        
        response = self.ai_client.chat.completions.create(
            model="qwen",
            messages=messages,
            max_tokens=256,
            temperature=0.0
        )
        
        # 记录LLM返回后的时间，计算延迟
        llm_end_time = time.time()
        llm_delay = llm_end_time - llm_start_time
        
        # 更新延迟历史记录（平滑算法：保留最近10次）
        self.llm_delay_history.append(llm_delay)
        if len(self.llm_delay_history) > self.llm_delay_max_history:
            self.llm_delay_history.pop(0)  # 移除最旧的记录
        
        # 计算最近10次延迟的均值
        self.llm_avg_delay = sum(self.llm_delay_history) / len(self.llm_delay_history)
        
        self.log_info(f"⏱️ LLM延迟: {llm_delay:.3f}s, 平均延迟: {self.llm_avg_delay:.3f}s")
        
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
            self.log_info(f"  → watch({second}s, 减去延迟{self.llm_avg_delay:.3f}s, 实际等待{adjusted_second:.3f}s)")
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
            
            # 如果正在录制，直接写入视频文件
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
                    
                    if self.video_writer and self.video_writer.isOpened():
                        try:
                            self.video_writer.write(frame)
                            self.video_frame_count += 1
                        except Exception as e:
                            # 捕获 OpenCV 的 C++ 异常
                            self.log_info(f"⚠️ 写入视频帧失败: {e}")
                            # 如果写入失败，关闭 writer 避免后续错误
                            try:
                                self.video_writer.release()
                            except:
                                pass
                            self.video_writer = None
            
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
