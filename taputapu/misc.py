#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from enum import Enum


class Color(Enum):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREY = (128, 128, 128)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    CYAN = (0, 255, 255)
    INDIGO = (128, 0, 255)
    ORANGE = (255, 128, 0)
    MINT = (128, 255, 128)