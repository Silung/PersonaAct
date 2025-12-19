#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADB文本输入工具
通过ADB命令实现文本输入功能，支持中文文本输入
"""

import subprocess
import time
import logging
import re


class ADBInput:
    def __init__(self, device_id=None, logger=None, operation_delay=0.15):
        """初始化ADB输入工具"""
        self.operation_delay = operation_delay
        self.device_id = device_id
        self.adb_keyboard_ime = "com.android.adbkeyboard/.AdbIME"
        self.logger = logger or logging.getLogger(__name__)
        self.original_ime = self.get_current_ime()
        
    def _contains_chinese(self, text):
        """
        检查文本是否包含中文字符
        
        Args:
            text (str): 要检查的文本
            
        Returns:
            bool: 包含中文返回True，否则返回False
        """
        # 匹配中文字符的正则表达式
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        return bool(chinese_pattern.search(text))

    def _run_adb_command(self, command):
        """
        执行ADB命令

        Args:
            command (str): ADB命令字符串

        Returns:
            tuple: (returncode, stdout, stderr)
        """
        try:
            self.logger.debug(f"执行ADB命令: {command}")
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, encoding="utf-8"
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            self.logger.error(f"执行ADB命令失败: {e}")
            return -1, "", str(e)

    def get_current_ime(self):
        """
        获取当前默认输入法

        Returns:
            str: 当前输入法包名，失败返回None
        """
        if self.device_id:
            command = f"adb -s {self.device_id} shell settings get secure default_input_method"
        else:
            command = "adb shell settings get secure default_input_method"
        returncode, stdout, stderr = self._run_adb_command(command)

        if returncode != 0:
            self.logger.error(f"获取当前输入法失败: {stderr}")
            return None

        # 获取输出的输入法包名，去除换行符
        ime_name = stdout.strip()
        if ime_name:
            self.logger.info(f"当前输入法: {ime_name}")
            return ime_name
        else:
            self.logger.warning("未找到当前输入法信息")
            return None

    def enable_adb_keyboard(self):
        """
        激活ADB键盘输入法

        Returns:
            bool: 成功返回True，失败返回False
        """
        # 启用ADB键盘
        if self.device_id:
            enable_command = (
                f"adb -s {self.device_id} shell ime enable {self.adb_keyboard_ime}"
            )
        else:
            enable_command = f"adb shell ime enable {self.adb_keyboard_ime}"
        returncode, stdout, stderr = self._run_adb_command(enable_command)

        if returncode != 0:
            self.logger.error(f"启用ADB键盘失败: {stderr}")
            return False

        # 设置ADB键盘为当前输入法
        if self.device_id:
            set_command = (
                f"adb -s {self.device_id} shell ime set {self.adb_keyboard_ime}"
            )
        else:
            set_command = f"adb shell ime set {self.adb_keyboard_ime}"
        returncode, stdout, stderr = self._run_adb_command(set_command)
        time.sleep(self.operation_delay)

        if returncode != 0:
            self.logger.error(f"设置ADB键盘失败: {stderr}")
            return False

        self.logger.info("ADB键盘激活成功")
        return True
    
    def input_text_simple(self, text):
        """
        使用默认ADB input text方法输入文本（适用于不含中文的文本）
        
        Args:
            text (str): 要输入的文本内容
            
        Returns:
            bool: 成功返回True，失败返回False
        """
        if not text:
            self.logger.warning("输入文本为空")
            return False
            
        # 转义特殊字符，适用于shell input text命令
        # 需要转义的字符：空格、引号、反斜杠等
        escaped_text = text.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
        # 如果包含空格，需要用引号包围
        if ' ' in escaped_text:
            escaped_text = f'"{escaped_text}"'
            
        if self.device_id:
            command = f'adb -s {self.device_id} shell input text {escaped_text}'
        else:
            command = f'adb shell input text {escaped_text}'
            
        print(command)
        returncode, stdout, stderr = self._run_adb_command(command)
        time.sleep(self.operation_delay)
        
        if returncode != 0:
            self.logger.error(f"简单文本输入失败: {stderr}")
            return False
            
        self.logger.info(f"简单文本输入成功: {text}")
        return True

    def input_text(self, text):
        """
        输入文本

        Args:
            text (str): 要输入的文本内容

        Returns:
            bool: 成功返回True，失败返回False
        """
        if not text:
            self.logger.warning("输入文本为空")
            return False

        # 转义特殊字符
        escaped_text = text.replace('"', '\\"').replace("'", "\\'")
        if self.device_id:
            command = f'adb -s {self.device_id} shell am broadcast -a ADB_INPUT_TEXT --es msg "{escaped_text}"'
        else:
            command = (
                f'adb shell am broadcast -a ADB_INPUT_TEXT --es msg "{escaped_text}"'
            )

        returncode, stdout, stderr = self._run_adb_command(command)
        time.sleep(self.operation_delay)

        if returncode != 0:
            self.logger.error(f"文本输入失败: {stderr}")
            return False

        self.logger.info(f"文本输入成功: {text}")
        return True

    def restore_original_ime(self, ime_name):
        """
        恢复原始输入法

        Args:
            ime_name (str): 要恢复的输入法包名

        Returns:
            bool: 成功返回True，失败返回False
        """
        if not ime_name:
            self.logger.warning("输入法名称为空，无法恢复")
            return False

        # 启用原始输入法
        if self.device_id:
            enable_command = f"adb -s {self.device_id} shell ime enable {ime_name}"
        else:
            enable_command = f"adb shell ime enable {ime_name}"
        returncode, stdout, stderr = self._run_adb_command(enable_command)

        if returncode != 0:
            self.logger.error(f"启用原始输入法失败: {stderr}")
            return False

        # 设置原始输入法为当前输入法
        if self.device_id:
            set_command = f"adb -s {self.device_id} shell ime set {ime_name}"
        else:
            set_command = f"adb shell ime set {ime_name}"
        returncode, stdout, stderr = self._run_adb_command(set_command)

        if returncode != 0:
            self.logger.error(f"设置原始输入法失败: {stderr}")
            return False

        self.logger.info(f"原始输入法恢复成功: {ime_name}")
        return True

    def input_text_safe(self, text):
        """
        安全的文本输入方法，自动检测文本类型并选择合适的输入方式

        Args:
            text (str): 要输入的文本内容

        Returns:
            bool: 成功返回True，失败返回False
        """
        if not text:
            self.logger.warning("输入文本为空")
            return False
            
        # 检查文本是否包含中文
        contains_chinese = self._contains_chinese(text)
        
        if contains_chinese:
            # 包含中文，使用ADB键盘方法
            self.logger.info(f"检测到中文字符，使用ADB键盘输入: {text}")
            
            # 1. 获取当前输入法
            if not self.original_ime:
                self.logger.error("无法获取当前输入法")
                return False

            try:
                # 2. 激活ADB键盘
                if not self.enable_adb_keyboard():
                    return False

                # 3. 输入文本
                if not self.input_text(text):
                    return False

                return True

            finally:
                # 4. 恢复原始输入法
                if self.original_ime:
                    self.restore_original_ime(self.original_ime)
        else:
            # 不包含中文，使用简单的input text方法
            self.logger.info(f"检测到纯英文/数字文本，使用默认输入方法: {text}")
            return self.input_text_simple(text)


def main():
    """主函数示例"""
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 创建ADB输入实例
    adb_input = ADBInput()

    # 测试不同类型的文本
    test_cases = [
        "Hello World",  # 纯英文
        "123456",       # 纯数字
        "test@email.com",  # 包含符号的英文
        "中文内容测试",    # 中文
        "Hello 你好 World",  # 中英混合
        "Price: $99.99"     # 包含特殊字符
    ]
    
    for text in test_cases:
        print(f"\n正在输入文本: {text}")
        if adb_input.input_text_safe(text):
            print("✓ 文本输入成功!")
        else:
            print("✗ 文本输入失败!")
        
        # 等待一下再输入下一个
        time.sleep(1)


if __name__ == "__main__":
    main()
