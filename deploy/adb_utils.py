#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADB工具类
统一管理所有ADB相关操作
"""

import os
import subprocess
import logging


class ADBUtils:
    """ADB工具类，封装常用的ADB操作"""
    
    def __init__(self, device_serial=None, logger=None):
        """
        初始化ADB工具
        
        Args:
            device_serial (str): 设备序列号，None表示使用默认设备
            logger: 日志对象
        """
        self.device_serial = device_serial
        self.logger = logger or logging.getLogger(__name__)
    
    def _build_adb_command(self, adb_args):
        """
        构建ADB命令
        
        Args:
            adb_args (list): ADB参数列表
            
        Returns:
            str: 完整的ADB命令字符串
        """
        if self.device_serial:
            return f"adb -s {self.device_serial} {' '.join(adb_args)}"
        else:
            return f"adb {' '.join(adb_args)}"
    
    def execute_command(self, adb_args, use_popen=False):
        """
        执行ADB命令
        
        Args:
            adb_args (list): ADB参数列表
            use_popen (bool): 是否使用popen获取输出
            
        Returns:
            int or str: 如果use_popen为False返回退出码，否则返回输出内容
        """
        command = self._build_adb_command(adb_args)
        
        try:
            if use_popen:
                result = os.popen(command).read().strip()
                self.logger.debug(f"ADB命令执行: {command} -> {result}")
                return result
            else:
                result = os.system(command)
                self.logger.debug(f"ADB命令执行: {command} -> 退出码: {result}")
                return result
        except Exception as e:
            self.logger.error(f"ADB命令执行失败: {command}, 错误: {e}")
            return -1 if not use_popen else ""
    
    def toggle_pointer_location(self, enable):
        """
        切换指针位置显示
        
        Args:
            enable (bool): True为开启，False为关闭
            
        Returns:
            bool: 操作是否成功
        """
        value = "1" if enable else "0"
        result = self.execute_command(["shell", "settings", "put", "system", "pointer_location", value])
        
        if result == 0:
            status = "开启" if enable else "关闭"
            self.logger.info(f"指针位置显示已{status}")
            return True
        else:
            self.logger.error("指针位置设置失败")
            return False
    
    def get_pointer_location_status(self):
        """
        获取指针位置显示状态
        
        Returns:
            bool: True表示已开启，False表示已关闭
        """
        result = self.execute_command(["shell", "settings", "get", "system", "pointer_location"], use_popen=True)
        return result == "1"
    
    def open_app(self, package_name):
        """
        打开应用
        
        Args:
            package_name (str): 应用包名
            
        Returns:
            bool: 操作是否成功
        """
        result = self.execute_command([
            "shell", "monkey", "-p", package_name, 
            "-c", "android.intent.category.LAUNCHER", "1"
        ])
        
        if result == 0:
            self.logger.info(f"应用 {package_name} 启动成功")
            return True
        else:
            self.logger.error(f"应用 {package_name} 启动失败")
            return False
    
    def send_keycode(self, keycode):
        """
        发送按键事件
        
        Args:
            keycode (int): 按键码
            
        Returns:
            bool: 操作是否成功
        """
        result = self.execute_command(["shell", "input", "keyevent", str(keycode)])
        
        if result == 0:
            self.logger.debug(f"按键 {keycode} 发送成功")
            return True
        else:
            self.logger.error(f"按键 {keycode} 发送失败")
            return False
    
    def send_home_key(self):
        """发送HOME键"""
        return self.send_keycode(3)  # KEYCODE_HOME = 3
    
    def send_back_key(self):
        """发送BACK键"""
        return self.send_keycode(4)  # KEYCODE_BACK = 4
    
    def get_device_info(self):
        """
        获取设备基本信息
        
        Returns:
            dict: 设备信息字典
        """
        info = {}
        
        # 获取设备型号
        model = self.execute_command(["shell", "getprop", "ro.product.model"], use_popen=True)
        if model:
            info["model"] = model
        
        # 获取Android版本
        version = self.execute_command(["shell", "getprop", "ro.build.version.release"], use_popen=True)
        if version:
            info["android_version"] = version
        
        # 获取API级别
        api_level = self.execute_command(["shell", "getprop", "ro.build.version.sdk"], use_popen=True)
        if api_level:
            info["api_level"] = api_level
        
        # 获取屏幕分辨率
        screen_size = self.execute_command(["shell", "wm", "size"], use_popen=True)
        if screen_size and "Physical size:" in screen_size:
            try:
                size_part = screen_size.split("Physical size:")[1].strip()
                width, height = map(int, size_part.split("x"))
                info["screen_resolution"] = {"width": width, "height": height}
            except:
                pass
        
        return info
    
    def check_device_connection(self):
        """
        检查设备连接状态
        
        Returns:
            bool: 设备是否连接
        """
        result = self.execute_command(["devices"], use_popen=True)
        if self.device_serial:
            return self.device_serial in result and "device" in result
        else:
            # 检查是否有任何设备连接
            lines = result.split('\n')
            for line in lines[1:]:  # 跳过标题行
                if line.strip() and "device" in line:
                    return True
            return False
    
    def install_apk(self, apk_path):
        """
        安装APK文件
        
        Args:
            apk_path (str): APK文件路径
            
        Returns:
            bool: 安装是否成功
        """
        result = self.execute_command(["install", "-r", apk_path])
        
        if result == 0:
            self.logger.info(f"APK安装成功: {apk_path}")
            return True
        else:
            self.logger.error(f"APK安装失败: {apk_path}")
            return False
    
    def capture_screenshot(self, output_path):
        """
        截取屏幕截图
        
        Args:
            output_path (str): 输出文件路径
            
        Returns:
            bool: 截图是否成功
        """
        result = self.execute_command(["exec-out", "screencap", "-p", ">", output_path])
        
        if result == 0 and os.path.exists(output_path):
            self.logger.debug(f"截图保存成功: {output_path}")
            return True
        else:
            self.logger.error(f"截图保存失败: {output_path}")
            return False


def get_connected_devices():
    """
    获取已连接的设备列表
    
    Returns:
        list: 设备序列号列表
    """
    try:
        result = os.popen("adb devices").read()
        devices = []
        
        lines = result.split('\n')
        for line in lines[1:]:  # 跳过标题行
            if line.strip() and '\t' in line:
                device_id = line.split('\t')[0]
                if device_id:
                    devices.append(device_id)
        
        return devices
    except Exception as e:
        logging.getLogger(__name__).error(f"获取设备列表失败: {e}")
        return []


def check_adb_available():
    """
    检查ADB是否可用
    
    Returns:
        bool: ADB是否可用
    """
    try:
        result = os.system("adb version > /dev/null 2>&1")
        return result == 0
    except:
        return False
