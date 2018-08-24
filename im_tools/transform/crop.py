#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"


import numpy as np
from typing import Tuple


def crop_image(image: np.ndarray, x_y_w_h: Tuple[int, int, int, int], margin_wh: Tuple[int, int]=(0, 0)) -> np.ndarray:
    """
    Crops image
    :param image: original full image
    :param x_y_w_h: coordinates of the segment
    :param margin_wh: width, height margin
    :return: cropped image
    """

    def check_dimensions(xmin: int, ymin: int, xmax: int, ymax: int):
        ymin = np.minimum(np.maximum(0, int(ymin)), image.shape[0])
        ymax = np.maximum(0, np.minimum(int(ymax), image.shape[0]))
        xmin = np.maximum(0, np.minimum(int(xmin), image.shape[1]))
        xmax = np.maximum(0, np.minimum(int(xmax), image.shape[1]))
        if xmax - xmin < 1 or ymax - ymin < 1:
            print("(h,w) = ({},{}). Cannot be equal or smaller than zero".format(ymax - ymin, xmax - xmin))
            return None, None, None, None
        return xmin, ymin, xmax, ymax

    x, y, w, h = x_y_w_h
    xmin = x - margin_wh[0]
    ymin = y - margin_wh[1]
    xmax = x + w + margin_wh[0]
    ymax = y + h + margin_wh[1]

    xmin, ymin, xmax, ymax = check_dimensions(xmin, ymin, xmax, ymax)

    crop = np.ascontiguousarray(image[ymin:ymax, xmin:xmax], np.uint8)

    return crop
