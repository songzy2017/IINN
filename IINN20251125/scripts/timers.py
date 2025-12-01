import math
import time


def timeprinter(ts):
    s = math.floor(ts % 60)
    m = math.floor((ts // 60) %60)
    h = math.floor(ts // 3600)
    return f'{h:0>3d}:{m:0>2d}:{s:0>2d}'


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
