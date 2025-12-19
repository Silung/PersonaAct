"""
Mobile Monitor - 通过ADB监控手机屏幕操作并记录轨迹
"""

import subprocess
import json
import time
from PIL import Image
import os
import threading
import logging
import shutil
from datetime import datetime
from pathlib import Path
from token import OP
from adb_input import ADBInput
# 注意：移除了wx的导入，因为这是Qt版本

# 导入自定义工具类
from anno_utils import (
    CoordinateConverter,
    GestureAnalyzer,
    OperationAnnotator,
    ADBUtils,
    parse_getevent_line,
    get_xml_content
)


class MobileMonitor:
    """手机屏幕操作监控类"""

    def __init__(
        self,
        output_dir="./init_captures",
        device_id=None,
        debug=True,
        screenshot_delay=1.0,
        draw_operations=False,
        save_xml=False,
        cache_interval=1,
        logger=None,
        control_panel=None,
        draw_path_points=True,
        num_image_cache=5,
        use_u2=True,
    ):
        """
        初始化监控器

        Args:
            output_dir (str): 输出目录路径
            device_id (str): 设备ID，如果为None则使用第一个连接的设备
            debug (bool): 是否启用调试模式
            screenshot_delay (float): 动作结束后截图延迟时间（秒）
            draw_operations (bool): 是否在图像上绘制操作轨迹
        """
        self.output_dir = Path(output_dir)
        self.device_id = device_id
        self.use_u2 = use_u2
        self.save_xml = save_xml
        self.step_count = 0
        self.is_monitoring = False
        self.monitor_thread = None
        self.debug = debug
        self.screenshot_delay = screenshot_delay
        self.draw_operations = draw_operations
        self.draw_path_points = draw_path_points
        self.screen_width = None
        self.screen_height = None
        self.touch_width = None
        self.touch_height = None

        # 截图缓存相关属性
        self.cache_interval = cache_interval  # 缓存截图间隔（秒）
        self.cache_screenshots = (
            []
        )  # 缓存的截图信息 [{'timestamp': float, 'path': str}, ...]
        self.cache_index = 0
        self.cache_thread = None
        self.num_image_cache = num_image_cache
        self.is_caching = False
        self.operation_start_time = None

        # 内存缓存相关属性
        self.preloaded_image = None  # 预加载的图片数据
        self.preloaded_timestamp = None  # 预加载图片的时间戳
        self.preloaded_filename = None  # 预加载图片的文件名
        self.preloaded_xml = None  # 预加载的XML数据
        self.preloaded_xml_filename = None  # 预加载XML的文件名
        self.closest_time_diff = 0.1

        # 创建输出目录和cache子目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        if logger is not None:
            self.logger = logger
        else:
            self._setup_logging()

        # 初始化ADB连接
        self._init_adb_connection()

        # 初始化ADB输入工具
        self.adb_input = ADBInput(device_id=device_id, logger=self.logger)

        # 保存control_panel引用，用于状态更新
        self.control_panel = control_panel

        # 初始化工具类
        self.adb_utils = ADBUtils(
            device_id=self.device_id, logger=self.logger, use_u2=self.use_u2
        )
        self.coordinate_converter = None  # 将在获取屏幕信息后初始化
        self.gesture_analyzer = None
        self.image_annotator = OperationAnnotator(logger=self.logger)

    def _setup_logging(self):
        """设置日志记录"""
        try:
            # 创建logger
            self.logger = logging.getLogger("MobileMonitor")
            self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

            # 避免重复添加handler
            if not self.logger.handlers:
                # 创建控制台handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.DEBUG if self.debug else logging.INFO)

                # 创建formatter
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                console_handler.setFormatter(formatter)

                # 添加控制台handler
                self.logger.addHandler(console_handler)

                # 尝试创建文件handler
                try:
                    log_file = self.output_dir / "monitor.log"
                    file_handler = logging.FileHandler(log_file, encoding="utf-8")
                    file_handler.setLevel(logging.DEBUG)
                    file_handler.setFormatter(formatter)
                    self.logger.addHandler(file_handler)
                    self.logger.info("日志系统初始化完成，文件日志已启用")
                except Exception as e:
                    self.logger.warning(f"无法创建日志文件: {str(e)}，仅使用控制台日志")
            else:
                self.logger.info("日志系统初始化完成")

        except Exception as e:
            # 如果日志设置失败，至少要有基本的错误输出
            print(f"日志系统初始化失败: {str(e)}")
            # 创建一个最基本的logger
            self.logger = logging.getLogger("MobileMonitor")
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
            self.logger.addHandler(handler)

    def _init_adb_connection(self):
        """初始化ADB连接"""
        try:
            self.logger.info("开始初始化ADB连接...")

            # 检查ADB是否可用
            self.logger.debug("检查ADB版本...")
            result = subprocess.run(
                ["adb", "version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                self.logger.error(f"ADB版本检查失败: {result.stderr}")
                raise Exception("ADB未正确安装或不在PATH中")

            self.logger.info(f"ADB版本: {result.stdout.strip()}")

            # 获取连接的设备列表
            self.logger.debug("获取连接的设备列表...")
            devices = self._get_connected_devices()
            self.logger.info(f"发现设备: {devices}")

            if not devices:
                self.logger.error("没有发现连接的Android设备")
                raise Exception("没有连接的Android设备")

            if self.device_id is None:
                self.device_id = devices[0]
                self.logger.info(f"自动选择设备: {self.device_id}")
            elif self.device_id not in devices:
                self.logger.error(
                    f"指定的设备 {self.device_id} 未在连接列表中: {devices}"
                )
                raise Exception(f"指定的设备 {self.device_id} 未连接")

            # 检查设备状态和权限
            self._check_device_permissions()

            self.logger.info(f"ADB连接成功，设备ID: {self.device_id}")

        except Exception as e:
            self.logger.error(f"ADB连接初始化失败: {str(e)}")
            raise Exception(f"ADB连接初始化失败: {str(e)}")

    def _check_device_permissions(self):
        """检查设备权限和状态"""
        try:
            self.logger.debug("检查设备权限...")

            # 检查设备是否授权
            result = self._adb_command(["get-state"], timeout=10)
            if result.returncode != 0:
                self.logger.error(f"无法获取设备状态: {result.stderr}")
                raise Exception("设备未正确连接或未授权")

            device_state = result.stdout.strip()
            self.logger.info(f"设备状态: {device_state}")

            if device_state != "device":
                self.logger.error(f"设备状态异常: {device_state}")
                raise Exception(f"设备状态异常: {device_state}, 请确保已授权ADB调试")

            # 检查Android版本
            result = self._adb_command(["shell", "getprop", "ro.build.version.release"])
            if result.returncode == 0:
                android_version = result.stdout.strip()
                self.logger.info(f"Android版本: {android_version}")

            # 检查API级别
            result = self._adb_command(["shell", "getprop", "ro.build.version.sdk"])
            if result.returncode == 0:
                api_level = result.stdout.strip()
                self.logger.info(f"API级别: {api_level}")

                # 对于Android 15 (API 35)的特殊处理
                if int(api_level) >= 35:
                    self.logger.warning("检测到Android 15或更高版本，可能需要额外权限")
                    self.logger.warning("Android 15的安全增强可能导致getevent访问受限")
                    self.logger.warning("建议尝试以下解决方案：")
                    self.logger.warning(
                        "1. 检查开发者选项中是否启用了'USB调试（安全设置）'"
                    )
                    self.logger.warning(
                        "2. 尝试执行: adb shell su -c 'getevent'（需要root权限）"
                    )
                    self.logger.warning("3. 检查是否需要授予额外的系统权限")

            # 测试getevent权限
            self.logger.debug("测试getevent权限...")
            result = self._adb_command(["shell", "ls", "/dev/input/"], timeout=5)
            if result.returncode != 0:
                self.logger.error("无法访问/dev/input/目录")
                raise Exception("无法访问输入设备，可能需要root权限或特殊配置")

            input_devices = result.stdout.strip().split("\n")
            self.logger.info(f"找到输入设备: {input_devices}")

            # 测试getevent命令
            self.logger.debug("测试getevent命令...")
            test_process = subprocess.Popen(
                ["adb", "-s", self.device_id, "shell", "timeout", "1", "getevent"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = test_process.communicate(timeout=3)
                if test_process.returncode not in [0, 124]:  # 124是timeout的正常退出码
                    self.logger.warning(
                        f"getevent测试返回码: {test_process.returncode}"
                    )
                    self.logger.warning(f"getevent错误输出: {stderr}")
                else:
                    self.logger.info("getevent命令测试成功")
            except subprocess.TimeoutExpired:
                test_process.kill()
                self.logger.warning("getevent测试超时，可能权限不足")

            # 针对Android 15进行额外的兼容性检查
            if int(api_level) >= 35:
                self._check_android15_compatibility()

        except Exception as e:
            self.logger.error(f"设备权限检查失败: {str(e)}")
            raise

    def _check_android15_compatibility(self):
        """检查Android 15的兼容性"""
        self.logger.info("正在进行Android 15兼容性检查...")

        try:
            # 检查SELinux状态
            result = self._adb_command(["shell", "getenforce"], timeout=5)
            if result.returncode == 0:
                selinux_status = result.stdout.strip()
                self.logger.info(f"SELinux状态: {selinux_status}")
                if selinux_status == "Enforcing":
                    self.logger.warning("SELinux处于强制模式，可能影响getevent访问")

            # 检查是否能访问具体的input设备
            result = self._adb_command(["shell", "ls", "-l", "/dev/input/"], timeout=5)
            if result.returncode == 0:
                input_files = result.stdout.strip()
                self.logger.debug(f"输入设备详情:\n{input_files}")

                # 检查是否有可读权限
                if "crw-" in input_files:
                    self.logger.warning("某些输入设备权限受限，这在Android 15中很常见")

            # 尝试更安全的替代方案检查
            self.logger.info("尝试检查替代的事件获取方案...")
            result = self._adb_command(["shell", "which", "sendevent"], timeout=5)
            if result.returncode == 0:
                self.logger.info("sendevent命令可用，这是一个好兆头")

            # 检查是否启用了开发者选项的高级调试
            result = self._adb_command(
                ["shell", "settings", "get", "global", "development_settings_enabled"],
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip() == "1":
                self.logger.info("开发者选项已启用")

                # 检查USB调试安全设置
                result = self._adb_command(
                    ["shell", "settings", "get", "global", "adb_enabled"], timeout=5
                )
                if result.returncode == 0:
                    adb_enabled = result.stdout.strip()
                    self.logger.info(f"ADB启用状态: {adb_enabled}")

        except Exception as e:
            self.logger.warning(f"Android 15兼容性检查时出现问题: {str(e)}")

    def _get_connected_devices(self):
        """获取已连接的设备列表"""
        try:
            result = subprocess.run(
                ["adb", "devices"], capture_output=True, text=True, timeout=10
            )
            devices = []
            for line in result.stdout.split("\n")[1:]:  # 跳过标题行
                if line.strip() and "device" in line:
                    device_id = line.split()[0]
                    devices.append(device_id)
            return devices
        except Exception:
            return []

    def _adb_command(self, command, timeout=30, shell=False):
        """执行ADB命令"""
        if self.device_id:
            full_cmd = ["adb", "-s", self.device_id] + command
        else:
            full_cmd = ["adb"] + command
        # self.logger.info(f"执行ADB命令: {' '.join(full_cmd)}")

        try:
            result = subprocess.run(
                full_cmd, capture_output=True, text=True, timeout=timeout, shell=False
            )
            return result
        except subprocess.TimeoutExpired:
            raise Exception(f"ADB命令超时: {' '.join(full_cmd)}")
        except Exception as e:
            raise Exception(f"ADB命令执行失败: {str(e)}")

    def _go_to_home(self):
        """回到桌面首页"""
        self.logger.info("正在回到桌面首页...")
        result = self._adb_command(["shell", "input", "keyevent", "KEYCODE_HOME"])
        if result.returncode != 0:
            raise Exception("回到桌面失败")
        time.sleep(2)  # 等待界面稳定

    def _take_screenshot(self, filename):
        """截图并保存"""
        # 在设备上截图
        st_time = time.time()
        result = self._adb_command(
            ["shell", "screencap", "/sdcard/temp_screenshot.png"]
        )
        if result.returncode != 0:
            raise Exception("设备截图失败")

        # 拉取截图到本地
        local_path = self.output_dir / filename
        result = self._adb_command(
            ["pull", "/sdcard/temp_screenshot.png", str(local_path)]
        )
        if result.returncode != 0:
            raise Exception("截图文件拉取失败")

        # 删除设备上的临时文件
        self._adb_command(["shell", "rm", "/sdcard/temp_screenshot.png"])

        self.logger.debug(
            f"截图已保存: {local_path}, t-cost: {time.time() - st_time:.3f}秒"
        )
        return str(local_path)

    def _convert_touch_to_screen_coordinates(self, touch_x, touch_y):
        """将触摸设备坐标转换为屏幕像素坐标"""
        if self.coordinate_converter:
            return self.coordinate_converter.convert_touch_to_screen_coordinates(
                touch_x, touch_y
            )
        else:
            raise ValueError("坐标转换器未初始化")

    def _update_screen_coordinates(self, touch_session):
        """更新屏幕坐标"""
        if (
            touch_session.get("raw_x") is not None
            and touch_session.get("raw_y") is not None
        ):

            # 进行坐标转换
            screen_x, screen_y = self._convert_touch_to_screen_coordinates(
                touch_session["raw_x"], touch_session["raw_y"]
            )
            touch_session["current_x"] = screen_x
            touch_session["current_y"] = screen_y

    def _convert_to_relative_coordinates(self, x, y):
        """将屏幕坐标转换为相对坐标（0-1范围）"""
        if self.coordinate_converter:
            return self.coordinate_converter.convert_to_relative_coordinates(x, y)
        else:
            raise ValueError("坐标转换器未初始化")

    def _get_screen_size(self):
        """获取屏幕尺寸"""
        return self.adb_utils.get_screen_size()

    def _get_touch_resolution(self):
        """获取触摸设备坐标范围"""
        return self.adb_utils.get_touch_resolution()

    def _parse_getevent_line(self, line):
        """解析getevent输出行"""
        return parse_getevent_line(line)

    def _monitor_touch_events(self):
        """监控触摸事件"""
        self.logger.info("开始监控触摸事件...")

        # 启动getevent监控
        if self.device_id:
            cmd = ["adb", "-s", self.device_id, "shell", "getevent"]
        else:
            cmd = ["adb", "shell", "getevent"]

        self.logger.debug(f"执行命令: {' '.join(cmd)}")

        process = None

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            self.logger.info("getevent进程已启动，等待触摸事件...")

            # 触摸状态跟踪
            touch_session = {
                "is_touching": False,
                "start_time": None,
                "start_x": None,
                "start_y": None,
                "current_x": None,
                "current_y": None,
                "raw_x": None,
                "raw_y": None,
                "path_points": [],  # 记录触摸路径
                "last_update_time": None,
                "coordinates_set": False,
            }

            while self.is_monitoring:
                line = process.stdout.readline()
                if not line:
                    self.logger.warning("getevent输出流结束")
                    break

                current_time = time.time()

                # 解析事件行
                device, event_type, code, value = self._parse_getevent_line(line)

                if event_type is None:
                    continue

                # 触摸事件类型
                EV_ABS = 0x03  # 绝对坐标事件
                EV_KEY = 0x01  # 按键事件
                EV_SYN = 0x00  # 同步事件

                # 触摸代码
                ABS_MT_POSITION_X = 0x35  # X坐标
                ABS_MT_POSITION_Y = 0x36  # Y坐标
                ABS_MT_TRACKING_ID = 0x39  # 触摸点ID
                BTN_TOUCH = 0x14A  # 触摸按钮
                SYN_REPORT = 0x00  # 报告同步

                # 处理触摸事件
                if event_type == EV_ABS:
                    if code == ABS_MT_POSITION_X:
                        # 存储原始触摸坐标
                        touch_session["raw_x"] = value
                        self._update_screen_coordinates(touch_session)
                    elif code == ABS_MT_POSITION_Y:
                        # 存储原始触摸坐标
                        touch_session["raw_y"] = value
                        self._update_screen_coordinates(touch_session)
                    elif code == ABS_MT_TRACKING_ID:
                        if (
                            value == -1 or value == 4294967295
                        ):  # 触摸结束 (处理32位无符号整数的-1)
                            self.logger.info(
                                f"检测到触摸结束事件，tracking_id: {value}"
                            )
                            self._handle_touch_end(touch_session, current_time)
                        else:  # 触摸开始或继续
                            if not touch_session["is_touching"]:
                                self.logger.info(f"检测到触摸开始事件，ID: {value}")
                                self._handle_touch_start(touch_session, current_time)

                elif event_type == EV_SYN and code == SYN_REPORT:
                    # 同步报告 - 一个完整的触摸事件数据包
                    if (
                        touch_session["is_touching"]
                        and touch_session["current_x"] is not None
                        and touch_session["current_y"] is not None
                    ):
                        self._handle_touch_move(touch_session, current_time)

        except Exception as e:
            self.logger.error(f"监控触摸事件时出错: {str(e)}")
        finally:
            if process:
                self.logger.debug("终止getevent进程...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                    self.logger.debug("getevent进程已终止")
                except subprocess.TimeoutExpired:
                    self.logger.warning("强制终止getevent进程...")
                    process.kill()

            self.logger.info("触摸事件监控结束")

    def _handle_touch_start(self, touch_session, current_time):
        """处理触摸开始"""
        touch_session["is_touching"] = True
        touch_session["start_time"] = current_time
        # 初始化为None，等待后续坐标事件
        touch_session["start_x"] = None
        touch_session["start_y"] = None
        touch_session["raw_x"] = None
        touch_session["raw_y"] = None
        touch_session["path_points"] = []
        touch_session["last_update_time"] = current_time
        touch_session["coordinates_set"] = False  # 标记是否已设置坐标

        # 预加载最接近当前时间的缓存截图到内存
        self._preload_closest_screenshot_to_memory(current_time)

        self.logger.info("触摸开始事件，等待坐标数据...")

    def _handle_touch_move(self, touch_session, current_time):
        """处理触摸移动"""
        if (
            touch_session["current_x"] is not None
            and touch_session["current_y"] is not None
        ):

            # 如果是第一次收到坐标，设置起始坐标
            if not touch_session.get("coordinates_set", False):
                touch_session["start_x"] = touch_session["current_x"]
                touch_session["start_y"] = touch_session["current_y"]
                touch_session["coordinates_set"] = True

                # 添加起始点到路径
                touch_session["path_points"].append(
                    {
                        "x": touch_session["start_x"],
                        "y": touch_session["start_y"],
                        "timestamp": current_time,
                    }
                )

            # 添加路径点（避免过于频繁的记录）
            if (
                touch_session["last_update_time"] is None
                or current_time - touch_session["last_update_time"] > 0.05
            ):  # 10ms间隔

                touch_session["path_points"].append(
                    {
                        "x": touch_session["current_x"],
                        "y": touch_session["current_y"],
                        "timestamp": current_time,
                    }
                )
                touch_session["last_update_time"] = current_time

    def _handle_touch_end(self, touch_session, current_time):
        """处理触摸结束"""
        if not touch_session["is_touching"]:
            self.logger.debug("收到触摸结束事件，但当前不在触摸状态")
            return

        touch_session["is_touching"] = False

        # 计算触摸持续时间
        duration = current_time - touch_session["start_time"]
        self.logger.info(f"触摸结束: 持续时间 {duration:.3f}秒")

        # 分析手势类型
        gesture = (
            self.gesture_analyzer.analyze_gesture(touch_session, duration)
            if self.gesture_analyzer
            else None
        )

        if gesture:
            self.logger.info(f"识别手势: {gesture['type']}, 参数: {gesture['params']}")

            # 通知前端显示动作类型
            self._notify_action_to_frontend(gesture["type"], success=True)

            # 添加延迟后再截图和记录
            if self.screenshot_delay > 0:
                self.logger.debug(f"等待 {self.screenshot_delay}秒 后截图...")
                time.sleep(self.screenshot_delay)
            self._record_action(gesture["type"], gesture["params"], touch_session)
        else:
            self.logger.warning("无法识别手势类型")
            self._notify_action_to_frontend("unknown", success=False)

        # 重置触摸会话
        self._reset_touch_session(touch_session)

    # 手势分析方法已移动到GestureAnalyzer类

    def _reset_touch_session(self, touch_session):
        """重置触摸会话"""
        touch_session.update(
            {
                "is_touching": False,
                "start_time": None,
                "start_x": None,
                "start_y": None,
                "current_x": None,
                "current_y": None,
                "raw_x": None,
                "raw_y": None,
                "path_points": [],
                "last_update_time": None,
                "coordinates_set": False,
            }
        )

    def _record_action(self, action_type, params, touch_session=None):
        """记录操作动作"""
        try:
            self.step_count += 1
            operation_time = time.time()
            self.logger.info(f"开始记录第 {self.step_count} 步操作: {action_type}")

            step_screenshot_filename = f"step_{self.step_count}.png"
            step_screenshot_path = self.output_dir / step_screenshot_filename
            step_xml_filename = f"step_{self.step_count}.xml"
            step_xml_path = self.output_dir / step_xml_filename

            time.sleep(self.screenshot_delay)
            self.logger.info(f"sleep {self.screenshot_delay} seconds, take screenshot")
            tmp_img_path = str(step_screenshot_path).replace(".png", "_temp.png")
            # tmp_xml_path = str(step_xml_path).replace(".xml", "_temp.xml")
            if self.use_u2:
                result = self._screencap()
                result.save(tmp_img_path)
            else:
                self._take_screenshot(tmp_img_path)
            # if self.save_xml:
            #     xml_result = self._dump_ui_hierarchy()

            # if self.save_xml and self.use_u2:
            #     with open(tmp_xml_path, "w", encoding="utf-8") as f:
            #         f.write(xml_result)
            #     self.logger.debug(f"已保存实时XML: {step_xml_filename}")
            # elif (
            #     self.save_xml
            #     and xml_result
            #     and xml_result.returncode == 0
            #     and xml_result.stdout
            # ):
            #     with open(
            #         tmp_xml_path,
            #         "w",
            #         encoding="utf-8",
            #     ) as f:
            #         f.write(xml_result.stdout)
            #     self.logger.debug(f"已保存实时XML: {step_xml_filename}")

            # 优先使用内存中的预加载图片和XML
            if self.preloaded_image is not None:
                # 使用预加载的图片数据
                with open(step_screenshot_path, "wb") as f:
                    f.write(self.preloaded_image)

                time_diff = (
                    abs(self.preloaded_timestamp - operation_time)
                    if self.preloaded_timestamp
                    else 0
                )
                self.logger.info(
                    f"使用预加载图片: {self.preloaded_filename} -> {step_screenshot_filename}, 时间差: {time_diff:.3f}秒"
                )

                # 处理XML文件
                if self.preloaded_xml is not None:
                    # 使用预加载的XML数据
                    with open(step_xml_path, "w", encoding="utf-8") as f:
                        f.write(self.preloaded_xml)
                    self.logger.info(
                        f"使用预加载XML: {self.preloaded_xml_filename} -> {step_xml_filename}"
                    )

                # 清空预加载缓存
                self.preloaded_image = None
                self.preloaded_timestamp = None
                self.preloaded_filename = None
                self.preloaded_xml = None
                self.preloaded_xml_filename = None
            else:
                raise ValueError("没有可用的缓存截图")

            # 创建操作记录
            action_data = {
                "step": self.step_count,
                "timestamp": datetime.now().isoformat(),
                "operation_time": operation_time,
                "action_type": action_type,
                "parameters": params,
                "annotation_type": "local",  # 设置annotation_mode为local
            }

            # 保存JSON文件
            json_filename = f"step_{self.step_count}.json"
            json_path = self.output_dir / json_filename

            with open(step_screenshot_path, 'rb') as f:
                image_size = Image.open(f).size
                action_data["image_size"] = {"width": image_size[0], "height": image_size[1]}

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(action_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"✅ 成功记录操作: {action_type}, 参数: {params}")

            # 在截图上绘制操作标注
            annotated_filename = f"step_{self.step_count}_annotated.png"
            if self.draw_operations:
                success = self.image_annotator.annotate_operation(
                    str(step_screenshot_path),
                    str(json_path)
                )
                if not success:
                    # 绘制失败，直接复制原图
                    annotated_path = self.output_dir / annotated_filename
                    shutil.copy2(step_screenshot_path, annotated_path)
                    self.logger.warning(
                        f"绘制操作标注失败，直接复制原图: {annotated_filename}"
                    )

            xml_status = "已保存" if step_xml_path.exists() else "未保存"
            self.logger.info(
                f"✅ 已保存文件: {json_filename}, {step_screenshot_filename}, XML: {xml_status}"
            )

        except Exception as e:
            self.logger.error(f"❌ 记录操作失败: {str(e)}")

    # 图像绘制方法已移动到ImageAnnotator类

    def _screencap(self):
        """截图"""
        return self.adb_utils.screencap()

    def _dump_ui_hierarchy(self):
        """获取UI层次结构XML"""
        # return self.adb_utils.dump_ui_hierarchy()
        return get_xml_content(self.adb_utils.device)

    def _cache_screenshots_periodically(self):
        """定期截图并缓存"""
        self.logger.info(f"开始定期截图缓存，间隔: {self.cache_interval}秒")

        while self.is_caching:
            try:
                current_time = time.time()

                # 确定缓存文件名，存放到cache子目录
                cache_index = self.cache_index % self.num_image_cache
                self.cache_index += 1
                cache_filename = f"cache_{cache_index}.png"
                cache_path = self.cache_dir / cache_filename
                xml_filename = f"cache_{cache_index}.xml"
                xml_path = self.cache_dir / xml_filename

                # lastest_cache_path = self.cache_dir / "latest.png"

                # 截图
                # result = self._adb_command(['exec-out', 'screencap', '-p', '>', str(cache_path)], shell=True)
                result = self._screencap()
                xml_result = None
                if self.save_xml:
                    xml_result = self._dump_ui_hierarchy()

                if self.use_u2:
                    result.save(str(cache_path))

                elif result.returncode == 0 and len(result.stdout) > 0:
                    # 保存截图
                    with open(str(cache_path), "wb") as f:
                        f.write(result.stdout)

                # 保存XML文件
                xml_saved = False
                if self.save_xml and self.use_u2:
                    try:
                        with open(str(xml_path), "w", encoding="utf-8") as f:
                            f.write(xml_result)
                        xml_saved = True
                    except Exception as e:
                        self.logger.error(f"保存XML文件失败: {str(e)}")
                elif (
                    self.save_xml
                    and xml_result
                    and xml_result.returncode == 0
                    and xml_result.stdout
                ):
                    try:
                        with open(str(xml_path), "w", encoding="utf-8") as f:
                            f.write(xml_result.stdout)
                        xml_saved = True
                    except Exception as e:
                        self.logger.error(f"保存XML文件失败: {str(e)}")

                # 更新缓存信息
                cache_info = {
                    "timestamp": current_time,
                    "path": str(cache_path),
                    "filename": cache_filename,
                    "xml_path": str(xml_path) if xml_saved else None,
                    "xml_filename": xml_filename if xml_saved else None,
                }

                # copy cache_path to latest.png
                # shutil.copy2(str(cache_path), str(lastest_cache_path))

                # 保持最多num_image_cache个缓存
                if len(self.cache_screenshots) >= self.num_image_cache:
                    self.cache_screenshots.pop(0)
                self.cache_screenshots.append(cache_info)

                t_cost = time.time() - current_time
                xml_status = "已保存" if xml_saved else "失败"
                self.logger.debug(
                    f"缓存已保存: {cache_filename} (时间戳: {current_time}), XML: {xml_status}, t-cost: {t_cost:.3f}秒"
                )

                time.sleep(self.cache_interval)

            except Exception as e:
                self.logger.error(f"缓存截图时出错: {str(e)}")
                time.sleep(self.cache_interval)

        self.logger.info("定期截图缓存结束")

    def _get_closest_cached_screenshot(self, target_time):
        """获取与目标时间最接近的缓存截图"""
        if not self.cache_screenshots:
            self.logger.warning("没有可用的缓存截图")
            return None

        cache_screenshots = sorted(
            self.cache_screenshots, key=lambda x: x["timestamp"], reverse=True
        )
        min_time_diff = self.closest_time_diff
        for screenshot in cache_screenshots:
            if target_time - screenshot["timestamp"] > min_time_diff:
                break
        time_diff = target_time - screenshot["timestamp"]
        self.logger.info(
            f"选择缓存截图: {screenshot['filename']}, 时间差: {time_diff:.3f}秒"
        )
        return screenshot

    def _preload_closest_screenshot_to_memory(self, target_time):
        """预加载最接近目标时间的缓存截图和XML到内存"""
        try:
            closest_cache = self._get_closest_cached_screenshot(target_time)
            if closest_cache:
                # 读取图片到内存
                with open(closest_cache["path"], "rb") as f:
                    image_data = f.read()

                self.preloaded_image = image_data
                self.preloaded_timestamp = closest_cache["timestamp"]
                self.preloaded_filename = closest_cache["filename"]

                # 读取对应的XML到内存
                self.preloaded_xml = None
                self.preloaded_xml_filename = None

                if (
                    self.save_xml
                    and closest_cache.get("xml_path")
                    and Path(closest_cache["xml_path"]).exists()
                ):
                    try:
                        with open(
                            closest_cache["xml_path"], "r", encoding="utf-8"
                        ) as f:
                            xml_data = f.read()
                        self.preloaded_xml = xml_data
                        self.preloaded_xml_filename = closest_cache["xml_filename"]
                        xml_status = "已预加载"
                    except Exception as e:
                        self.logger.error(f"预加载XML文件失败: {str(e)}")
                        xml_status = "预加载失败"
                else:
                    xml_status = "无缓存XML"

                time_diff = abs(closest_cache["timestamp"] - target_time)
                self.logger.info(
                    f"预加载缓存到内存: {closest_cache['filename']}, XML: {xml_status}, 时间差: {time_diff:.3f}秒"
                )
                return True
            else:
                self.logger.warning("没有可用的缓存截图进行预加载")
                return False

        except Exception as e:
            self.logger.error(f"预加载缓存截图失败: {str(e)}")
            return False

    def wait(self, wait_time=1.0):
        """
        等待指定时间后记录图片

        Args:
            wait_time (float): 等待时间（秒）
        """
        self.operation_start_time = time.time()

        # 预加载最接近当前时间的缓存截图到内存
        self._preload_closest_screenshot_to_memory(self.operation_start_time)

        self.logger.info(f"开始等待 {wait_time} 秒...")
        time.sleep(wait_time)

        # 调用_record_action保存图片
        params = {
            "wait_time": wait_time,
        }

        # 通知前端
        self._notify_action_to_frontend("wait", success=True)

        self._record_action("wait", params)

    def input(self, text, sleep_time=1.0):
        """
        使用ADB输入文本并记录图片

        Args:
            text (str): 要输入的文本内容
        """
        self.operation_start_time = time.time()

        # 预加载最接近当前时间的缓存截图到内存
        self._preload_closest_screenshot_to_memory(self.operation_start_time)

        self.logger.info(f"开始输入文本: {text}")

        # 使用ADBInput输入文本
        success = self.adb_input.input_text_safe(text)

        time.sleep(sleep_time)

        if success:
            self.logger.info("文本输入成功")
        else:
            self.logger.error("文本输入失败")

        # 调用_record_action保存图片
        params = {
            "text": text,
            "success": success,
        }

        # 通知前端
        self._notify_action_to_frontend("type", success)

        self._record_action("type", params)

    def finished(self):
        """
        结束监控并记录完成状态

        Args:
            text (str): 完成描述文本
        """
        self.operation_start_time = time.time()

        # 预加载最接近当前时间的缓存截图到内存
        self._preload_closest_screenshot_to_memory(self.operation_start_time)

        self.logger.info(f"监控完成")

        params = {}

        # 通知前端
        self._notify_action_to_frontend("finish", success=True)

        self._record_action("finish", params)

        self.logger.info(f"✅ 完成状态已保存")

        # 结束监控
        self.stop_monitoring()

    def back(self):
        """
        执行返回操作并记录

        Returns:
            bool: 成功返回True，失败返回False
        """
        self.operation_start_time = time.time()

        # 预加载最接近当前时间的缓存截图到内存
        self._preload_closest_screenshot_to_memory(self.operation_start_time)

        self.logger.info("执行返回操作...")

        try:
            # 发送返回键
            result = self._adb_command(["shell", "input", "keyevent", "KEYCODE_BACK"])

            success = result.returncode == 0
            if success:
                self.logger.info("返回操作成功")
            else:
                self.logger.error(f"返回操作失败: {result.stderr}")

            # 记录操作
            params = {
                "success": success,
            }

            # 通知前端
            self._notify_action_to_frontend("back", success)

            self._record_action("back", params)

            return success

        except Exception as e:
            self.logger.error(f"返回操作出错: {str(e)}")
            return False

    def home(self):
        """
        执行返回操作并记录
        """
        self.operation_start_time = time.time()
        self._preload_closest_screenshot_to_memory(self.operation_start_time)
        try:
            result = self._adb_command(["shell", "input", "keyevent", "KEYCODE_HOME"])
            success = result.returncode == 0
            if success:
                self.logger.info("回到桌面操作成功")
            else:
                self.logger.error(f"回到桌面操作失败: {result.stderr}")

            # 记录操作
            params = {
                "success": success,
            }

            # 通知前端
            self._notify_action_to_frontend("home", success)

            self._record_action("home", params)

            return success
        except Exception as e:
            self.logger.error(f"回到桌面操作出错: {str(e)}")
            return False

    def open_app(self, app_name, app_package_name):
        """
        打开指定应用并记录

        Args:
            app_package_name (str): 应用包名，如 'com.tencent.mm'

        Returns:
            bool: 成功返回True，失败返回False
        """
        self.operation_start_time = time.time()

        # 预加载最接近当前时间的缓存截图到内存
        self._preload_closest_screenshot_to_memory(self.operation_start_time)

        self.logger.info(f"打开应用: {app_package_name}")

        try:
            # 方法1：使用monkey命令启动应用
            result = self._adb_command(
                [
                    "shell",
                    "monkey",
                    "-p",
                    app_package_name,
                    "-c",
                    "android.intent.category.LAUNCHER",
                    "1",
                ]
            )

            success = result.returncode == 0

            if success:
                self.logger.info(f"应用启动成功: {app_package_name}")
                # 等待应用启动
                time.sleep(2.0)
            else:
                self.logger.error(f"应用启动失败: {result.stderr}")
                # 尝试备用方法：使用am start命令
                self.logger.info("尝试备用启动方法...")
                success = self._try_alternative_launch(app_package_name)

            # 记录操作
            params = {
                "package_name": app_package_name,
                "app_name": app_name,
                "success": success,
            }

            # 通知前端
            self._notify_action_to_frontend("open", success)

            self._record_action("open", params)

            return success

        except Exception as e:
            self.logger.error(f"打开应用出错: {str(e)}")
            return False

    def _try_alternative_launch(self, app_package_name):
        """
        尝试备用的应用启动方法

        Args:
            app_package_name (str): 应用包名

        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            # 方法2：获取主Activity并启动
            result = self._adb_command(
                [
                    "shell",
                    "cmd",
                    "package",
                    "resolve-activity",
                    "--brief",
                    app_package_name,
                ]
            )

            if result.returncode == 0 and result.stdout.strip():
                # 解析主Activity
                lines = result.stdout.strip().split("\n")
                main_activity = None

                for line in lines:
                    if "/" in line and app_package_name in line:
                        main_activity = line.strip()
                        break

                if main_activity:
                    # 启动主Activity
                    result = self._adb_command(
                        ["shell", "am", "start", "-n", main_activity]
                    )

                    if result.returncode == 0:
                        self.logger.info(f"使用备用方法成功启动: {main_activity}")
                        time.sleep(2.0)
                        return True

            # 方法3：简单的intent启动
            result = self._adb_command(
                [
                    "shell",
                    "am",
                    "start",
                    "-a",
                    "android.intent.action.MAIN",
                    "-c",
                    "android.intent.category.LAUNCHER",
                    app_package_name,
                ]
            )

            if result.returncode == 0:
                self.logger.info(f"使用Intent方法启动成功: {app_package_name}")
                time.sleep(2.0)
                return True

            return False

        except Exception as e:
            self.logger.error(f"备用启动方法失败: {str(e)}")
            return False

    def start_monitoring(self, return_home=True):
        """开始监控"""
        if self.is_monitoring:
            print("监控已在运行中")
            return

        try:
            # 回到桌面首页
            if return_home:
                self._go_to_home()

            # 记录初始状态
            initial_data = {
                "step": 0,
                "timestamp": datetime.now().isoformat(),
                "action_type": "initial",
                "parameters": {"description": "初始桌面状态"},
            }

            with open(self.output_dir / "step_0.json", "w", encoding="utf-8") as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=2)

            # 获取屏幕尺寸
            width, height = self._get_screen_size()
            if width and height:
                self.screen_width = width
                self.screen_height = height
                self.logger.info(f"屏幕尺寸: {width}x{height}")
            else:
                self.logger.warning("无法获取屏幕尺寸，将使用原始坐标")

            # 获取触摸设备分辨率
            touch_width, touch_height = self._get_touch_resolution()
            if touch_width and touch_height:
                self.touch_width = touch_width
                self.touch_height = touch_height
                self.logger.info(f"触摸设备分辨率: {touch_width}x{touch_height}")
            else:
                raise ValueError("无法获取触摸设备分辨率")

            # 初始化坐标转换器和手势分析器
            self.coordinate_converter = CoordinateConverter(
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                touch_width=self.touch_width,
                touch_height=self.touch_height,
            )
            self.gesture_analyzer = GestureAnalyzer(
                coordinate_converter=self.coordinate_converter, logger=self.logger
            )

            # 开始监控
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_touch_events)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

            # 启动截图缓存线程
            self.is_caching = True
            self.cache_process = threading.Thread(
                target=self._cache_screenshots_periodically
            )
            self.cache_process.daemon = True
            self.cache_process.start()

            self.logger.info("开始监控屏幕操作...")
            self.logger.info(f"截图缓存已启动，间隔: {self.cache_interval}秒")
            self.logger.info("按Ctrl+C停止监控")

        except Exception as e:
            self.is_monitoring = False
            raise Exception(f"开始监控失败: {str(e)}")

    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            self.logger.info("监控未在运行")
            return

        self.logger.info("正在停止监控...")
        self.is_monitoring = False
        self.is_caching = False

        # 停止监控线程
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        # 停止缓存线程
        if self.cache_thread and self.cache_thread.is_alive():
            self.cache_thread.join(timeout=5)

        self.logger.info("监控已停止")
        self.logger.info(f"总共记录了 {self.step_count} 个操作步骤")

    def _notify_action_to_frontend(self, action_type, success=True):
        """
        通知前端显示动作类型

        Args:
            action_type (str): 动作类型
            success (bool): 是否成功
        """
        # Qt版本暂时不需要特殊的前端通知
        # 可以在这里添加信号发射或其他Qt特定的通知机制
        pass

    def get_summary(self):
        """获取监控摘要"""
        return {
            "device_id": self.device_id,
            "output_dir": str(self.output_dir),
            "step_count": self.step_count,
            "is_monitoring": self.is_monitoring,
        }
