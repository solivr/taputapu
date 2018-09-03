#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import numpy as np
import cv2


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
