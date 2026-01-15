# -*- coding: utf-8 -*-

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 900)
        
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.main_horizontal_layout = QHBoxLayout(self.centralwidget)
        self.main_horizontal_layout.setObjectName("main_horizontal_layout")
        self.main_horizontal_layout.setSpacing(10)
        
        # ================ 第一列：设备配置、AI配置、休息设置 ================
        self.column1_panel = QWidget()
        self.column1_panel.setObjectName("column1_panel")
        self.column1_panel.setMinimumWidth(280)
        self.column1_panel.setMaximumWidth(320)
        
        self.column1_layout = QVBoxLayout(self.column1_panel)
        self.column1_layout.setObjectName("column1_layout")
        self.column1_layout.setSpacing(10)
        self.column1_layout.setContentsMargins(5, 5, 5, 5)
        
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
        
        self.column1_layout.addWidget(self.device_group)
        
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
        
        self.api_key_layout = QHBoxLayout()
        self.label_api_key = QLabel("API Key:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setText("1234567890")
        self.api_key_layout.addWidget(self.label_api_key)
        self.api_key_layout.addWidget(self.api_key_input)
        self.ai_layout.addLayout(self.api_key_layout)
        
        self.model_layout = QHBoxLayout()
        self.label_model = QLabel("Model:")
        self.model_input = QLineEdit()
        self.model_input.setText("qwen")
        self.model_layout.addWidget(self.label_model)
        self.model_layout.addWidget(self.model_input)
        self.ai_layout.addLayout(self.model_layout)
        
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
        
        # Reverse Persona复选框
        self.reverse_persona_checkbox = QCheckBox("Reverse Persona")
        self.reverse_persona_checkbox.setObjectName("reverse_persona_checkbox")
        self.reverse_persona_checkbox.setChecked(False)  # 默认关闭
        self.reverse_persona_checkbox.setToolTip("开启后，watch时间反转为max(0,15-t)，like/comment/share无效")
        self.ai_layout.addWidget(self.reverse_persona_checkbox)
        
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
        
        self.column1_layout.addWidget(self.ai_group)
        
        # 休息设置组
        self.rest_group = QGroupBox("休息设置")
        self.rest_group.setObjectName("rest_group")
        self.rest_layout = QVBoxLayout(self.rest_group)
        
        # 启用休息功能
        self.rest_enable_checkbox = QCheckBox("启用休息功能")
        self.rest_enable_checkbox.setObjectName("rest_enable_checkbox")
        self.rest_enable_checkbox.setChecked(False)
        self.rest_layout.addWidget(self.rest_enable_checkbox)
        
        # 平台选择
        self.platform_layout = QHBoxLayout()
        self.platform_label = QLabel("平台:")
        self.platform_combo = QComboBox()
        self.platform_combo.setObjectName("platform_combo")
        self.platform_combo.addItems(["bilibili", "douyin", "kuaishou"])
        self.platform_layout.addWidget(self.platform_label)
        self.platform_layout.addWidget(self.platform_combo)
        self.rest_layout.addLayout(self.platform_layout)
        
        # 工作时长
        self.work_duration_layout = QHBoxLayout()
        self.work_duration_label = QLabel("工作时长(分钟):")
        self.work_duration_spinbox = QSpinBox()
        self.work_duration_spinbox.setObjectName("work_duration_spinbox")
        self.work_duration_spinbox.setRange(1, 180)
        self.work_duration_spinbox.setValue(30)
        self.work_duration_layout.addWidget(self.work_duration_label)
        self.work_duration_layout.addWidget(self.work_duration_spinbox)
        self.rest_layout.addLayout(self.work_duration_layout)
        
        # 休息时长
        self.rest_duration_layout = QHBoxLayout()
        self.rest_duration_label = QLabel("休息时长(分钟):")
        self.rest_duration_spinbox = QSpinBox()
        self.rest_duration_spinbox.setObjectName("rest_duration_spinbox")
        self.rest_duration_spinbox.setRange(1, 60)
        self.rest_duration_spinbox.setValue(5)
        self.rest_duration_layout.addWidget(self.rest_duration_label)
        self.rest_duration_layout.addWidget(self.rest_duration_spinbox)
        self.rest_layout.addLayout(self.rest_duration_layout)
        
        # 随机偏移
        self.rest_offset_layout = QHBoxLayout()
        self.rest_offset_label = QLabel("随机偏移(分钟):")
        self.rest_offset_spinbox = QSpinBox()
        self.rest_offset_spinbox.setObjectName("rest_offset_spinbox")
        self.rest_offset_spinbox.setRange(0, 30)
        self.rest_offset_spinbox.setValue(2)
        self.rest_offset_layout.addWidget(self.rest_offset_label)
        self.rest_offset_layout.addWidget(self.rest_offset_spinbox)
        self.rest_layout.addLayout(self.rest_offset_layout)
        
        self.column1_layout.addWidget(self.rest_group)
        
        # 添加弹性空间
        self.column1_layout.addStretch()
        
        self.main_horizontal_layout.addWidget(self.column1_panel)
        
        # ================ 第二列：输出配置、调试选项、交互控制 ================
        self.column2_panel = QWidget()
        self.column2_panel.setObjectName("column2_panel")
        self.column2_panel.setMinimumWidth(220)
        self.column2_panel.setMaximumWidth(260)
        
        self.column2_layout = QVBoxLayout(self.column2_panel)
        self.column2_layout.setObjectName("column2_layout")
        self.column2_layout.setSpacing(10)
        self.column2_layout.setContentsMargins(5, 5, 5, 5)
        
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
        
        self.column2_layout.addWidget(self.output_group)
        
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
        
        self.column2_layout.addWidget(self.debug_group)
        
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
        
        self.column2_layout.addWidget(self.control_group)
        
        # 添加弹性空间
        self.column2_layout.addStretch()
        
        self.main_horizontal_layout.addWidget(self.column2_panel)
        
        # ================ 中间：实时屏幕 ================
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
        
        # ================ 右侧：运行日志 ================
        self.log_panel = QWidget()
        self.log_panel.setObjectName("log_panel")
        self.log_panel.setMinimumWidth(300)
        self.log_panel.setMaximumWidth(400)
        
        self.log_panel_layout = QVBoxLayout(self.log_panel)
        self.log_panel_layout.setContentsMargins(5, 5, 5, 5)
        
        self.log_group = QGroupBox("运行日志")
        self.log_group.setObjectName("log_group")
        self.log_layout = QVBoxLayout(self.log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setObjectName("log_text")
        self.log_text.setReadOnly(True)
        self.log_layout.addWidget(self.log_text)
        
        self.log_panel_layout.addWidget(self.log_group)
        
        self.main_horizontal_layout.addWidget(self.log_panel)
        
        # 设置各列的拉伸比例
        self.main_horizontal_layout.setStretch(0, 0)  # 第一列：固定宽度
        self.main_horizontal_layout.setStretch(1, 0)  # 第二列：固定宽度
        self.main_horizontal_layout.setStretch(2, 1)  # 实时屏幕：可拉伸
        self.main_horizontal_layout.setStretch(3, 0)  # 日志：固定宽度

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QCoreApplication.translate("MainWindow", "Persona视频交互系统", None)
        )
