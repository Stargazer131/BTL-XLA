import numpy as np
from scipy import ndimage


class Matrix:
    x1d = np.array([[1],
                    [-1]], dtype=np.float64)
    y1d = np.array([[1, -1]], dtype=np.float64)
    x_robert = np.array([[-1, 0],
                        [0, 1]], dtype=np.float64)
    y_robert = np.array([[0, 1],
                        [-1, 0]], dtype=np.float64)
    x_prewitt = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]], dtype=np.float64)
    y_prewitt = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]], dtype=np.float64)
    x_sobel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=np.float64)
    y_sobel = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=np.float64)
    laplacian_filter = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float64)
    laplacian_variant_filter = np.array([
        [0, 1, 0],
        [1, -8, 1],
        [0, 1, 0]
    ], dtype=np.float64)
    laplacian_enhancer = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float64)
    laplacian_variant_enhancer = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ], dtype=np.float64)


def create_custom_weighted_kernel(k: int):
    limit = (k - 1) // 2
    matrix = np.zeros((k, k))
    for x in range(limit + 1):
        for y in range(limit + 1):
            matrix[x, y] = 2 ** (x + y)

    flip_right_matrix = np.fliplr(matrix[:limit + 1, :limit + 1])
    flip_down_matrix = np.flipud(matrix[:limit + 1, :limit + 1])

    matrix[:limit + 1, limit:] = flip_right_matrix
    matrix[limit:, :limit + 1] = flip_down_matrix
    matrix[limit:, limit:] = np.flipud(flip_right_matrix)
    matrix /= np.sum(matrix)

    return matrix


def create_custom_morphing_kernel(k: int):
    limit = (k - 1) // 2
    matrix = np.ones((k, k))
    for x in range(limit + 1):
        for y in range(limit + 1):
            if y < limit-x:
                matrix[x, y] = 0

    flip_right_matrix = np.fliplr(matrix[:limit + 1, :limit + 1])
    flip_down_matrix = np.flipud(matrix[:limit + 1, :limit + 1])

    matrix[:limit + 1, limit:] = flip_right_matrix
    matrix[limit:, :limit + 1] = flip_down_matrix
    matrix[limit:, limit:] = np.flipud(flip_right_matrix)

    return matrix


def line2d_coefficient(x1, y1, x2, y2):
    a = (y1 - y2) / (x1 - x2)
    b = y1 - x1 * a
    return a, b


def distance(x, y, a, b):
    ts = np.abs(a * x - y + b)
    ms = np.sqrt(a ** 2 + b ** 2)
    return ts / ms


def mask1d(pixel_values: np.ndarray):
    x_gradient = ndimage.convolve(pixel_values, Matrix.x1d, mode='constant', cval=0)
    y_gradient = ndimage.convolve(pixel_values, Matrix.y1d, mode='constant', cval=0)
    gradient = np.round(np.sqrt(x_gradient ** 2 + y_gradient ** 2))
    return gradient


def robert(pixel_values: np.ndarray):
    x_gradient = ndimage.convolve(pixel_values, Matrix.x_robert, mode='constant', cval=0)
    y_gradient = ndimage.convolve(pixel_values, Matrix.y_robert, mode='constant', cval=0)
    gradient = np.round(np.sqrt(x_gradient ** 2 + y_gradient ** 2))
    return gradient


def prewitt(pixel_values: np.ndarray):
    x_gradient = ndimage.convolve(pixel_values, Matrix.x_prewitt, mode='constant', cval=0)
    y_gradient = ndimage.convolve(pixel_values, Matrix.y_prewitt, mode='constant', cval=0)
    gradient = np.round(np.sqrt(x_gradient ** 2 + y_gradient ** 2))
    return gradient


def sobel(pixel_values: np.ndarray):
    x_gradient = ndimage.convolve(pixel_values, Matrix.x_sobel, mode='constant', cval=0)
    y_gradient = ndimage.convolve(pixel_values, Matrix.y_sobel, mode='constant', cval=0)
    gradient = np.round(np.sqrt(x_gradient ** 2 + y_gradient ** 2))
    return gradient


def gaussian_filter(pixel_values: np.ndarray, kernel_size: int, sigma: float):
    # Calculate the center of the kernel
    center = (kernel_size - 1) / 2

    # Create the Gaussian kernel
    gaussian_kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2)),
        (kernel_size, kernel_size), dtype=np.float64
    )
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel.flatten())

    result = ndimage.convolve(pixel_values, gaussian_kernel, mode='constant', cval=0)
    return np.round(result)
