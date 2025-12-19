#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动构建脚本
自动找到所需的依赖文件并更新main.spec，然后执行PyInstaller构建
"""

import os
import sys
import site
import glob
import re
from pathlib import Path


def find_package_path(package_name):
    """查找指定包的安装路径"""
    try:
        import importlib.util
        spec = importlib.util.find_spec(package_name)
        if spec and spec.origin:
            return os.path.dirname(spec.origin)
    except ImportError:
        pass
    
    # 备用方法：在site-packages中查找
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        package_path = os.path.join(site_dir, package_name)
        if os.path.exists(package_path):
            return package_path
    
    return None


def find_jar_file(package_path, jar_filename):
    """在包路径中查找指定的jar文件"""
    if not package_path or not os.path.exists(package_path):
        return None
    
    # 递归查找jar文件
    for root, dirs, files in os.walk(package_path):
        if jar_filename in files:
            return os.path.join(root, jar_filename)
    
    return None


def find_data_files():
    """自动查找所需的数据文件"""
    data_files = []
    
    # 查找scrcpy-server.jar
    print("正在查找 scrcpy-server.jar...")
    scrcpy_path = find_package_path('scrcpy')
    if scrcpy_path:
        scrcpy_jar = find_jar_file(scrcpy_path, 'scrcpy-server.jar')
        if scrcpy_jar:
            data_files.append((scrcpy_jar, 'scrcpy'))
            print(f"找到 scrcpy-server.jar: {scrcpy_jar}")
        else:
            print("警告: 未找到 scrcpy-server.jar")
    else:
        print("警告: 未找到 scrcpy 包")
    
    # 查找u2.jar
    print("正在查找 u2.jar...")
    uiautomator2_path = find_package_path('uiautomator2')
    if uiautomator2_path:
        u2_jar = find_jar_file(uiautomator2_path, 'u2.jar')
        if u2_jar:
            data_files.append((u2_jar, 'uiautomator2/assets'))
            print(f"找到 u2.jar: {u2_jar}")
        else:
            print("警告: 未找到 u2.jar")
    else:
        print("警告: 未找到 uiautomator2 包")
    
    # 添加assets目录（如果存在）
    current_dir = os.getcwd()
    assets_dir = os.path.join(current_dir, 'assets')
    if os.path.exists(assets_dir):
        data_files.append(('assets', 'assets'))
        print(f"找到 assets 目录: {assets_dir}")
    else:
        print("警告: 未找到 assets 目录")
    
    return data_files


def update_spec_file(spec_file_path, data_files):
    """更新main.spec文件中的datas部分"""
    if not os.path.exists(spec_file_path):
        print(f"错误: 未找到 {spec_file_path}")
        return None
    
    # 读取现有内容
    with open(spec_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 构建新的datas部分
    datas_lines = ['datas = [']
    for src, dst in data_files:
        datas_lines.append(f"    ('{src}', '{dst}'),")
    datas_lines.append(']')
    new_datas = '\n'.join(datas_lines)
    
    # 使用正则表达式替换datas部分
    pattern = r'datas=\[.*?\]'
    new_content = re.sub(pattern, new_datas, content, flags=re.DOTALL)
    
    # 写回文件
    tmp_spec_file_path = spec_file_path.replace('.spec', '_temp.spec')
    with open(tmp_spec_file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"已更新 {spec_file_path}")
    return tmp_spec_file_path


def run_pyinstaller(spec_file_path):
    """执行PyInstaller构建"""
    if not os.path.exists(spec_file_path):
        print(f"错误: 未找到 {spec_file_path}")
        return False
    
    print(f"正在执行 PyInstaller 构建...")
    cmd = f"pyinstaller {spec_file_path}"
    print(f"执行命令: {cmd}")
    
    result = os.system(cmd)
    if result == 0:
        print("构建成功!")
        return True
    else:
        print(f"构建失败，退出码: {result}")
        return False


def main():
    """主函数"""
    print("=== 自动构建脚本 ===")
    
    # 获取当前目录
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    
    # 查找main.spec文件
    spec_files = glob.glob(os.path.join(current_dir, "**/main.spec"), recursive=True)
    
    if not spec_files:
        print("错误: 未找到 main.spec 文件")
        return 1
    
    # 如果找到多个spec文件，让用户选择
    if len(spec_files) > 1:
        print("找到多个 main.spec 文件:")
        for i, spec_file in enumerate(spec_files):
            print(f"  {i+1}. {spec_file}")
        
        try:
            choice = int(input("请选择要构建的spec文件 (输入数字): ")) - 1
            if 0 <= choice < len(spec_files):
                spec_file_path = spec_files[choice]
            else:
                print("无效选择")
                return 1
        except ValueError:
            print("无效输入")
            return 1
    else:
        spec_file_path = spec_files[0]
    
    print(f"使用 spec 文件: {spec_file_path}")
    
    # 查找数据文件
    print("\n=== 查找依赖文件 ===")
    data_files = find_data_files()
    
    if not data_files:
        print("错误: 未找到任何依赖文件")
        return 1
    
    print(f"\n找到 {len(data_files)} 个数据文件:")
    for src, dst in data_files:
        print(f"  {src} -> {dst}")
    
    # 更新spec文件
    print("\n=== 更新 spec 文件 ===")
    tmp_spec_file_path = update_spec_file(spec_file_path, data_files)
    if not tmp_spec_file_path:
        return 1
    
    # 执行构建
    print("\n=== 执行构建 ===")
    if not run_pyinstaller(tmp_spec_file_path):
        return 1
    
    print("\n构建完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
