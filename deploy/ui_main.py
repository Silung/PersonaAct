# -*- coding: utf-8 -*-

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 800)
        
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.main_horizontal_layout = QHBoxLayout(self.centralwidget)
        self.main_horizontal_layout.setObjectName("main_horizontal_layout")
        
        # ================ 左侧配置面板 ================
        self.config_panel = QWidget()
        self.config_panel.setObjectName("config_panel")
        self.config_panel.setMinimumWidth(350)
        self.config_panel.setMaximumWidth(400)
        
        self.config_layout = QVBoxLayout(self.config_panel)
        self.config_layout.setObjectName("config_layout")
        
        # 设备配置组
        self.device_group = QGroupBox("设备配置")
        self.device_group.setObjectName("device_group")
        self.device_layout = QVBoxLayout(self.device_group)
        
        self.device_select_layout = QHBoxLayout()
        self.label_device = QLabel("设备:")
        self.combo_device = QComboBox()
        self.combo_device.setMinimumSize(QSize(150, 0))
        self.device_select_layout.addWidget(self.label_device)
        self.device_select_layout.addWidget(self.combo_device)
        self.device_layout.addLayout(self.device_select_layout)
        
        self.flip = QCheckBox("翻转屏幕")
        self.flip.setObjectName("flip")
        self.device_layout.addWidget(self.flip)
        
        self.config_layout.addWidget(self.device_group)
        
        # AI配置组
        self.ai_group = QGroupBox("AI配置")
        self.ai_group.setObjectName("ai_group")
        self.ai_layout = QVBoxLayout(self.ai_group)
        
        self.model_url_layout = QHBoxLayout()
        self.label_model_url = QLabel("模型URL:")
        self.model_url_input = QLineEdit()
        self.model_url_input.setText("http://127.0.0.1:8012/v1")
        self.model_url_layout.addWidget(self.label_model_url)
        self.model_url_layout.addWidget(self.model_url_input)
        self.ai_layout.addLayout(self.model_url_layout)
        
        self.persona_input_layout = QHBoxLayout()
        self.label_persona = QLabel("Persona*:")
        self.persona_input = QLineEdit()
        self.persona_input.setObjectName("persona_input")
        self.persona_input.setPlaceholderText("请输入persona名称（必填）")
        self.persona_input_layout.addWidget(self.label_persona)
        self.persona_input_layout.addWidget(self.persona_input)
        self.ai_layout.addLayout(self.persona_input_layout)
        
        self.persona_select_layout = QHBoxLayout()
        self.label_persona_template = QLabel("模板:")
        self.persona_select = QComboBox()
        self.persona_select.setObjectName("persona_select")
        self.persona_select.addItems(["None", "zsl", "yqg"])
        self.persona_select_layout.addWidget(self.label_persona_template)
        self.persona_select_layout.addWidget(self.persona_select)
        self.ai_layout.addLayout(self.persona_select_layout)
        
        # 使用Persona复选框
        self.use_persona_checkbox = QCheckBox("使用Persona")
        self.use_persona_checkbox.setObjectName("use_persona_checkbox")
        self.use_persona_checkbox.setChecked(True)  # 默认启用
        self.ai_layout.addWidget(self.use_persona_checkbox)
        
        self.history_layout = QHBoxLayout()
        self.label_history = QLabel("历史窗口:")
        self.history_spinbox = QSpinBox()
        self.history_spinbox.setMinimum(1)
        self.history_spinbox.setMaximum(10)
        self.history_spinbox.setValue(3)
        self.history_layout.addWidget(self.label_history)
        self.history_layout.addWidget(self.history_spinbox)
        self.ai_layout.addLayout(self.history_layout)
        
        # 图片质量滑块
        self.image_quality_layout = QVBoxLayout()
        self.image_quality_label_layout = QHBoxLayout()
        self.label_image_quality = QLabel("图片质量:")
        self.image_quality_value_label = QLabel("85")
        self.image_quality_value_label.setMinimumWidth(30)
        self.image_quality_value_label.setAlignment(Qt.AlignRight)
        self.image_quality_label_layout.addWidget(self.label_image_quality)
        self.image_quality_label_layout.addStretch()
        self.image_quality_label_layout.addWidget(self.image_quality_value_label)
        self.image_quality_layout.addLayout(self.image_quality_label_layout)
        
        self.image_quality_slider = QSlider(Qt.Horizontal)
        self.image_quality_slider.setObjectName("image_quality_slider")
        self.image_quality_slider.setMinimum(1)
        self.image_quality_slider.setMaximum(100)
        self.image_quality_slider.setValue(85)
        self.image_quality_slider.setTickPosition(QSlider.TicksBelow)
        self.image_quality_slider.setTickInterval(10)
        self.image_quality_layout.addWidget(self.image_quality_slider)
        self.ai_layout.addLayout(self.image_quality_layout)
        
        self.button_test_model = QPushButton("测试模型连接")
        self.button_test_model.setObjectName("button_test_model")
        self.ai_layout.addWidget(self.button_test_model)
        
        self.config_layout.addWidget(self.ai_group)
        
        # 交互控制组
        self.control_group = QGroupBox("交互控制")
        self.control_group.setObjectName("control_group")
        self.control_layout = QVBoxLayout(self.control_group)
        
        self.button_start = QPushButton("开始交互")
        self.button_start.setObjectName("button_start")
        self.button_start.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px; padding: 10px; }")
        self.button_start.setEnabled(True)
        
        self.button_stop = QPushButton("停止交互")
        self.button_stop.setObjectName("button_stop")
        self.button_stop.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; font-size: 14px; padding: 10px; }")
        self.button_stop.setEnabled(False)
        
        self.control_layout.addWidget(self.button_start)
        self.control_layout.addWidget(self.button_stop)
        
        # 状态显示
        self.status_label = QLabel("状态: 未交互")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #9E9E9E; padding: 8px; background-color: #F5F5F5; border-radius: 5px; }")
        self.control_layout.addWidget(self.status_label)
        
        # 详细信息显示
        self.info_label = QLabel("会话: 未开始\n平台: -\n运行时长: 0分钟")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("QLabel { font-size: 12px; color: #666; padding: 5px; }")
        self.control_layout.addWidget(self.info_label)
        
        self.step_label = QLabel("步数: 0")
        self.step_label.setAlignment(Qt.AlignCenter)
        self.step_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; color: #2196F3; padding: 10px; }")
        self.control_layout.addWidget(self.step_label)
        
        self.config_layout.addWidget(self.control_group)
        
        # 输出配置组
        self.output_group = QGroupBox("输出配置")
        self.output_group.setObjectName("output_group")
        self.output_layout = QVBoxLayout(self.output_group)
        
        self.output_info_label = QLabel("数据保存在: ../deploy_log/")
        self.output_info_label.setStyleSheet("QLabel { color: #666; font-size: 12px; }")
        self.output_layout.addWidget(self.output_info_label)
        
        self.save_video_checkbox = QCheckBox("保存视频")
        self.save_video_checkbox.setChecked(True)
        self.output_layout.addWidget(self.save_video_checkbox)
        
        self.save_audio_checkbox = QCheckBox("保存音频")
        self.save_audio_checkbox.setChecked(True)
        self.output_layout.addWidget(self.save_audio_checkbox)
        
        self.save_xml_checkbox = QCheckBox("保存XML")
        self.save_xml_checkbox.setChecked(True)
        self.output_layout.addWidget(self.save_xml_checkbox)
        
        self.config_layout.addWidget(self.output_group)
        
        # 调试组
        self.debug_group = QGroupBox("调试选项")
        self.debug_group.setObjectName("debug_group")
        self.debug_layout = QVBoxLayout(self.debug_group)
        
        # 指针位置显示按钮
        self.button_toggle_pointer = QPushButton("显示指针位置")
        self.button_toggle_pointer.setObjectName("button_toggle_pointer")
        self.button_toggle_pointer.setCheckable(True)
        self.button_toggle_pointer.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
            }
            QPushButton:checked {
                background-color: #FF5722;
            }
        """)
        self.debug_layout.addWidget(self.button_toggle_pointer)
        
        self.config_layout.addWidget(self.debug_group)
        
        # 日志显示
        self.log_group = QGroupBox("运行日志")
        self.log_group.setObjectName("log_group")
        self.log_layout = QVBoxLayout(self.log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setObjectName("log_text")
        self.log_text.setReadOnly(True)
        self.log_layout.addWidget(self.log_text)
        
        self.config_layout.addWidget(self.log_group)
        
        self.config_layout.addStretch()
        
        self.main_horizontal_layout.addWidget(self.config_panel)
        
        # ================ 右侧屏幕展示 ================
        self.screen_panel = QWidget()
        self.screen_panel.setObjectName("screen_panel")
        self.screen_layout = QVBoxLayout(self.screen_panel)
        
        self.screen_title = QLabel("实时屏幕")
        self.screen_title.setAlignment(Qt.AlignCenter)
        self.screen_title.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        self.screen_title.setFixedHeight(30)
        self.screen_layout.addWidget(self.screen_title)
        
        self.label = QLabel()
        self.label.setObjectName("label")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("QLabel { border: 1px solid gray; }")
        self.label.setText("等待设备连接...")
        self.screen_layout.addWidget(self.label)
        
        self.screen_layout.addStretch()
        
        self.main_horizontal_layout.addWidget(self.screen_panel)
        
        self.main_horizontal_layout.setStretch(0, 0)
        self.main_horizontal_layout.setStretch(1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QCoreApplication.translate("MainWindow", "Persona视频交互系统", None)
        )
