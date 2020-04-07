import os
import shutil
import codecs


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def read_lines(path):
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fr:
        lines = []
        for line in fr.readlines():
            line = line.strip()
            if not line:
                continue
            else:
                lines.append(line)
        return lines
