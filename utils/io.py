# -*- coding:utf-8 -*-

import os


def rebuild_dir(path):
    """mkdir path if file or None"""
    if os.path.isfile(path):
        os.remove(path)

    if not os.path.exists(path):
        os.makedirs(path)
