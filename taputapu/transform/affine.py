#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import numpy as np
import cv2
from typing import Tuple


def _find_background_color(image: np.ndarray) -> int:
    """
    Given a grayscale image, finds the background color value
    :param image: grayscale image
    :return: background color value (int)
    """

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    thresh_value, thresholded_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find which is the background (0 or 255). Supposing that the background color occurrence is higher
    # than the writing color
    counts, bin_edges = np.histogram(thresholded_image, bins=2)
    background_color = int(np.median(image[thresholded_image == 255*np.argmax(counts)]))

    return background_color


def apply_slant(image: np.ndarray, alpha: float) -> np.ndarray:
    """
    This function applies a slant to the writing image
    :param image: np.ndarray in grayscale
    :param alpha: angle of slant (radian, positive or negative)
    :return: slanted image
    """

    shape_image = image.shape
    shift = max(-alpha * shape_image[0], 0)
    output_size = (int(shape_image[1] + np.ceil(abs(alpha * shape_image[0]))), int(shape_image[0]))

    warpM = np.array([[1, alpha, shift], [0, 1, 0]])

    # Find color of background in order to replicate it in the borders
    border_value = _find_background_color(image)

    img_warp = cv2.warpAffine(image, np.array(warpM), output_size, borderValue=border_value)

    return img_warp


def resize_image_coordinates(input_coordinates: np.ndarray, input_shape: Tuple[int, int],
                       resized_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resizes the cordinates to fit the resized image

    :param input_coordinates: (x,y) coordinates with shape (N,2)
    :param input_shape: shape of the input image (H,W)
    :param resized_shape: shape of the resized image (H, W)
    :return: the resized coordinates (N, 2)
    """

    rx = input_shape[0] / resized_shape[0]
    ry = input_shape[1] / resized_shape[1]

    return np.stack((np.round(input_coordinates[:, 0] / ry),
                      np.round(input_coordinates[:, 1] / rx)), axis=1).astype(np.int32)