from argparse import ArgumentParser
from typing import Optional
import time
import os
import sys
import json
import datetime
import threading
import uiautomator2 as u2
import cv2
import numpy as np
import wave
import re

import pyaudio
from adbutils import adb
from PySide6.QtGui import QImage, QPixmap, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PySide6.QtCore import QTimer, QSize
from ui_main import Ui_MainWindow
from pathlib import Path

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
        
        # AI交互相关
        self.ai_client = None
        self.ai_persona = None
        self.ai_history_screenshots = []
        self.ai_history_actions = []
        self.ai_max_history = 3
        self.ai_step_count = 0
        self.is_interacting = False
        
        # 输出目录
        self.output_dir = None
        self.session_dir = None
        
        # 视频录制
        self.is_recording = False
        self.current_video_path = None
        self.current_audio_path = None
        self.video_writer = None
        self.video_frame_count = 0
        self.video_fps = 30
        
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
        
        self.ai_client = OpenAI(base_url=model_url, api_key="1234567890")
        self.ai_max_history = self.ui.history_spinbox.value()
        
        persona_name = self.ui.persona_select.currentText()
        if persona_name != "None":
            self.ai_persona = self._load_persona(persona_name)
        
        deploy_log_dir = Path(__file__).parent.parent / "deploy_log"
        deploy_log_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir = deploy_log_dir
        
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        persona_suffix = f"_{persona_name}" if persona_name != "None" else "_no_persona"
        self.session_dir = deploy_log_dir / f"{session_id}{persona_suffix}"
        self.session_dir.mkdir(exist_ok=True)
        
        (self.session_dir / "screenshots").mkdir(exist_ok=True)
        (self.session_dir / "videos").mkdir(exist_ok=True)
        (self.session_dir / "audios").mkdir(exist_ok=True)
        (self.session_dir / "xml").mkdir(exist_ok=True)
        
        self.ai_history_screenshots = []
        self.ai_history_actions = []
        self.ai_step_count = 0
        
        self.is_interacting = True
        self.ui.button_start.setEnabled(False)
        self.ui.button_stop.setEnabled(True)
        
        self.log_info(f"✅ 开始交互")
        self.log_info(f"输出目录: {self.session_dir}")
        
        threading.Thread(target=self._interaction_loop, daemon=True).start()

    def stop_interaction(self):
        if not self.is_interacting:
            return
        
        self.is_interacting = False
        self.ui.button_start.setEnabled(True)
        self.ui.button_stop.setEnabled(False)
        
        if self.is_recording:
            self.stop_recording(save_files=False)
        
        self.log_info(f"✅ 交互结束，共 {self.ai_step_count} 步")

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
        
        screenshot_filename = f"step_{self.ai_step_count}_{timestamp}.png"
        screenshot_path = self.session_dir / "screenshots" / screenshot_filename
        screenshot = self.u2_device.screenshot()
        screenshot.save(str(screenshot_path))
        self.log_info(f"📸 截图: {screenshot_filename}")
        
        if self.ui.save_xml_checkbox.isChecked():
            xml_filename = f"step_{self.ai_step_count}_{timestamp}.xml"
            xml_path = self.session_dir / "xml" / xml_filename
            xml_text = self.u2_device.dump_hierarchy()
            xml_path.write_text(xml_text, encoding='utf-8')
            self.log_info(f"📝 XML: {xml_filename}")
        
        if self.ui.save_video_checkbox.isChecked() or self.ui.save_audio_checkbox.isChecked():
            video_path = self.session_dir / "videos" / f"step_{self.ai_step_count}_{timestamp}.mp4" if self.ui.save_video_checkbox.isChecked() else None
            audio_path = self.session_dir / "audios" / f"step_{self.ai_step_count}_{timestamp}.wav" if self.ui.save_audio_checkbox.isChecked() else None
            
            if video_path or audio_path:
                self.start_recording(video_path, audio_path)
        
        self.log_info(f"🤖 调用LLM...")
        response = self._get_ai_response(str(screenshot_path))
        self.log_info(f"响应:\n{response}")
        
        code = self._parse_code(response)
        if not code:
            self.log_info(f"⚠️ 未找到代码块")
            return
        
        self.log_info(f"执行代码:")
        self._execute_ai_code(code)
        
        if self.is_recording:
            self.stop_recording(save_files=True)
        
        self.ai_history_screenshots.append(str(screenshot_path))
        self.ai_history_actions.append(f"```python\n{code}\n```")
        
        step_data = {
            "step": self.ai_step_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "screenshot": str(screenshot_path),
            "response": response,
            "code": code
        }
        
        step_file = self.session_dir / f"step_{self.ai_step_count}.json"
        with open(step_file, 'w', encoding='utf-8') as f:
            json.dump(step_data, f, ensure_ascii=False, indent=2)
        
        self.u2_device.swipe(0.5, 0.8, 0.5, 0.2, duration=0.1)
        self.log_info(f"→ swipe up (下一个视频)")
        time.sleep(1)

    def _get_ai_response(self, screenshot_path):
        use_persona = self.ui.persona_select.currentText() != "None"
        
        system_prompt = get_system_prompt(use_persona, self.ai_persona)
        user_prompt = get_user_prompt(
            history_screenshots=self.ai_history_screenshots[-self.ai_max_history:] if self.ai_history_screenshots else None,
            history_actions=self.ai_history_actions[-self.ai_max_history:] if self.ai_history_actions else None,
            current_screenshot=screenshot_path,
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
        
        response = self.ai_client.chat.completions.create(
            model="qwen",
            messages=messages,
            max_tokens=256,
            temperature=0.0
        )
        
        return response.choices[0].message.content

    def _parse_code(self, response_text):
        code_pattern = r"```python\s*(.*?)\s*```"
        match = re.search(code_pattern, response_text, re.DOTALL)
        
        if match:
            code = match.group(1).strip()
            return code
        
        return None

    def _execute_ai_code(self, code):
        screen_width, screen_height = self.u2_device.window_size()
        
        def tap(x, y):
            self.u2_device.click(x, y)
            self.log_info(f"  → tap({x}, {y})")
        
        def swipe(x1, y1, x2, y2, duration=0.1):
            self.u2_device.swipe(x1, y1, x2, y2, duration)
            self.log_info(f"  → swipe({x1}, {y1}, {x2}, {y2})")
        
        def watch(second=2.0):
            self.log_info(f"  → watch({second}s)")
            time.sleep(second)
        
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
        self.video_writer = None
        self.video_frame_count = 0
        self.audio_buffer = []
        self.is_recording = True
        self.is_audio_recording = True
        
        return True

    def stop_recording(self, save_files=True):
        if not self.is_recording:
            return None, None
        
        self.is_recording = False
        self.is_audio_recording = False
        
        video_path = self.current_video_path
        audio_path = self.current_audio_path
        frame_count = self.video_frame_count
        audio_data = self.audio_buffer.copy() if save_files else []
        video_writer = self.video_writer
        
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
                        self.log_info(f"✅ 视频: {video_path.name} ({frame_count}帧, {video_size:.1f}KB)")
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
        self.log_info(f"已连接到设备: {device}")

    def list_devices(self):
        self.ui.combo_device.clear()
        items = [i.serial for i in adb.device_list()]
        self.ui.combo_device.addItems(items)
        return items

    def on_flip(self, _):
        self.client.flip = self.ui.flip.isChecked()

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
                if self.video_writer is None:
                    # 第一帧到来时初始化VideoWriter
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
                
                if self.video_writer:
                    self.video_writer.write(frame)
                    self.video_frame_count += 1
            
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
            if self.video_writer:
                self.video_writer.release()
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
