#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

"""
Compute the likeliness of an image region to contain vessels or other image ridges ,
according to the method described by Frangi et al. :`
``
Frangi A.F., Niessen W.J., Vincken K.L., Viergever M.A. (1998)
Multiscale vessel enhancement filtering. In: Wells W.M., Colchester A., Delp S.
(eds) Medical Image Computing and Computer-Assisted Intervention — MICCAI’98. MICCAI 1998.
Lecture Notes in Computer Science, vol 1496. Springer, Berlin, Heidelberg
```
Code adapted from Matlab to python from :
https://www.mathworks.com/matlabcentral/fileexchange/24409-hessian-based-frangi-vesselness-filter

"""

import numpy as np
from typing import Tuple
from scipy.ndimage.filters import convolve


def hessian2d(image: np.ndarray, sigma: float=1) -> Tuple[float, float, float]:
    """
    This function Hessian2 filters the image with 2nd derivatives of a
    Gaussian with parameter Sigma.
    :param image: image, in flotaing point precision (float64)
    :param sigma: sigma of the gaussian kernel used
    :return: the 2nd derivatives
    """
    # Make kernel coordinates
    x, y = np.meshgrid(np.arange(-np.round(3 * sigma), np.round(3 * sigma) + 1),
                       np.arange(-np.round(3 * sigma), np.round(3 * sigma) + 1), indexing='ij')

    # Build the gaussian 2nd derivatives filters
    d_gaussxx = 1/(2 * np.pi * sigma ** 4) * (x ** 2 / sigma ** 2 - 1) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    d_gaussxy = (1 / (2 * np.pi * sigma ** 6)) * (x * y) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    d_gaussyy = d_gaussxx.conj().T

    d_xx = convolve(image, d_gaussxx, mode='constant', cval=0.0)
    d_xy = convolve(image, d_gaussxy, mode='constant', cval=0.0)
    d_yy = convolve(image, d_gaussyy, mode='constant', cval=0.0)

    return d_xx, d_xy, d_yy


def eig2image(d_xx: float, d_xy: float, d_yy: float) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    This function eig2image calculates the eigenvalues from the
    hessian matrix, sorted by abs value. And gives the direction
    of the ridge (eigenvector smallest eigenvalue) .
    | Dxx  Dxy |
    | Dxy  Dyy |
    """
    # Compute the eigenvectors of J, v1 and v2
    tmp = np.sqrt((d_xx - d_yy) ** 2 + 4 * d_xy ** 2)
    v2x = 2 * d_xy
    v2y = d_yy - d_xx + tmp

    # Normalize
    mag = np.sqrt(v2x**2 + v2y**2)
    i = np.invert(np.isclose(mag, np.zeros(mag.shape)))
    v2x[i] = v2x[i]/mag[i]
    v2y[i] = v2y[i]/mag[i]

    # The eigenvectors are orthogonal
    v1x = -v2y.copy()
    v1y = v2x.copy()

    # Compute the eigenvalues
    mu1 = 0.5*(d_xx + d_yy + tmp)
    mu2 = 0.5*(d_xx + d_yy - tmp)

    # Sort eigenvalues by absolute value abs(lambda1)<abs(lambda2)
    check = np.absolute(mu1) > np.absolute(mu2)

    lambda1 = mu1.copy()
    lambda1[check] = mu2[check]
    lambda2 = mu2.copy()
    lambda2[check] = mu1[check]

    Ix = v1x.copy()
    Ix[check] = v2x[check]
    Iy = v1y.copy()
    Iy[check] = v2y[check]

    return lambda1, lambda2, Ix, Iy


def frangi_filter2d(image: np.ndarray, scale_range: np.array=np.array([1, 10]), scale_ratio: float=2,
                    beta_one: float=0.5, beta_two: float=15, verbose: bool=False, black_white: bool=True):
    """
    This function FRANGIFILTER2D uses the eigenvectors of the Hessian to
    compute the likeliness of an image region to vessels, according
    to the method described by Frangi:2001 (Chapter 2). Adapted from MATLAB code
    :param image: imput image (grayscale)
    :param scale_range: The range of sigmas used, default [1 10]
    :param scale_ratio: Step size between sigmas, default 2
    :param beta_one: Frangi correction constant, default 0.5
    :param beta_two: Frangi correction constant, default 15
    :param verbose: Show debug information, default false
    :param black_white: Detect black ridges (default) set to true, for white ridges set to false.
    :return: The vessel enhanced image (pixel is the maximum found in all scales)
    """

    if len(scale_range) > 1:
        sigmas = np.arange(scale_range[0], scale_range[1] + 1, scale_ratio)
        sigmas = sorted(sigmas)
    else:
        sigmas = [scale_range[0]]
    beta = 2 * beta_one ** 2
    c = 2 * beta_two ** 2

    # Make matrices to store all filterd images
    all_filtered = np.zeros([image.shape[0], image.shape[1], len(sigmas)])
    all_angles = np.zeros([image.shape[0], image.shape[1], len(sigmas)])

    # Frangi filter for all sigmas
    for i in range(len(sigmas)):
        # Show progress
        if verbose:
            print('Current Frangi Filter Sigma: ', str(sigmas[i]))

        # Make 2D hessian
        Dxx, Dxy, Dyy = hessian2d(image, sigmas[i])

        # Correct for scale
        Dxx *= (sigmas[i]**2)
        Dxy *= (sigmas[i]**2)
        Dyy *= (sigmas[i]**2)

        # Calculate (abs sorted) eigenvalues and vectors
        lambda2, lambda1, Ix, Iy = eig2image(Dxx, Dxy, Dyy)

        # Compute the direction of the minor eigenvector
        angles = np.arctan2(Ix, Iy)

        # Compute some similarity measures
        near_zeros = np.isclose(lambda1, np.zeros(lambda1.shape))
        lambda1[near_zeros] = 2**(-52)
        Rb = (lambda2/lambda1)**2
        S2 = lambda1**2 + lambda2**2

        # Compute the output image
        image_filtered = np.exp(-Rb/beta)*(np.ones(image.shape) - np.exp(-S2 / c))

        # see pp. 45
        if black_white:
            image_filtered[lambda1 < 0] = 0
        else:
            image_filtered[lambda1 > 0] = 0

        # store the results in 3D matrices
        all_filtered[:, :, i] = image_filtered.copy()
        all_angles[:, :, i] = angles.copy()

    # Return for every pixel the value of the scale(sigma) with the maximum
    # output pixel value
    if len(sigmas) > 1:
        out_image = np.amax(all_filtered, axis=2)
        out_image = out_image.reshape(image.shape[0], image.shape[1], order='F')
        which_scale = np.argmax(all_filtered, axis=2)
        which_scale = np.reshape(which_scale, image.shape, order='F')

        indices = range(image.size) + (which_scale.flatten(order='F') - 1) * image.size
        values = np.take(all_angles.flatten(order='F'), indices)
        direction = np.reshape(values, image.shape, order='F')
    else:
        out_image = all_filtered.reshape(image.shape[0], image.shape[1], order='F')
        which_scale = np.ones(image.shape)
        direction = np.reshape(all_angles, image.shape, order='F')

    return out_image, which_scale, direction
