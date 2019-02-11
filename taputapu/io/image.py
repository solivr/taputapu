#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from PIL import Image
from typing import Tuple


def get_image_shape_without_loading(filename: str) -> Tuple[int, int]:
    """
    Gives the shape (h,w) of the image without loading its content into memory
    :param filename: filename of the image
    :return: (H, W)
    """
    image = Image.open(filename)
    shape = image.size
    image.close()
    return shape
