import cv2
import numpy as np


def create_1d_gaussian_kernel(sigma, radius=0):
    """
    Creates a Nx1 gaussian filter kernel.

    :param sigma: The sigma (standard deviation) parameter for the gaussian.
    :param radius: The filter radius, so that N = 2*radius + 1. If set to 0, the radius will be computed so that radius = 3.5 * sigma.
    :return: Nx1 gaussian filter kernel.
    """
    if radius <= 0:
        radius = int(np.ceil(3.5 * sigma))

    length = 2 * radius + 1
    x = np.arange(0, length) - radius
    kernel = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-x * x / (2 * sigma * sigma))

    return kernel, x


def create_1d_derivated_gaussian_kernel(sigma, radius=0):
    """
    Creates a Nx1 derivated gaussian filter kernel.

    :param sigma: The sigma (standard deviation) parameter for the gaussian.
    :param radius: The filter radius, so that N = 2*radius + 1. If set to 0, the radius will be computed so that radius = 3.5 * sigma.
    :return:
    """
    if radius <= 0:
        radius = int(np.ceil(3.5 * sigma))

    kernel, x = create_1d_gaussian_kernel(sigma, radius)

    kernel = kernel * x * (-1. / (sigma*sigma))

    return kernel, x
