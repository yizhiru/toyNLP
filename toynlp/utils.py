import os
import shutil


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        os.mkdir(path)
