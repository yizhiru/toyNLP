import os
import shutil


def remake_dir(path):
    """删除已有目录后重新创建"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def read_lines(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return [line.strip() for line in f if line.strip()]
