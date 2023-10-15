import numpy as np
from scipy import ndimage


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


def otsu(pixel_values: np.ndarray):
    grayscale_frequency = np.bincount(pixel_values) / len(pixel_values)
    pi_cumsum = np.cumsum(grayscale_frequency)
    mk_cumsum = np.cumsum(grayscale_frequency * np.array(range(len(grayscale_frequency))))
    mg = mk_cumsum[-1]  # total cumulative sum
    a = (mg * pi_cumsum - mk_cumsum) ** 2
    b = pi_cumsum * (1 - pi_cumsum)
    variance = a / (b + np.finfo(np.float64).eps)  # add epsilon to prevent divide by 0
    otsu_threshold = np.mean(np.argwhere(variance == np.max(variance)))
    return otsu_threshold


def mask1d(pixel_values: np.ndarray):
    x1d = np.array([[1],
                    [-1]])
    y1d = np.array([[1, -1]])
    x_gradient = ndimage.convolve(pixel_values, x1d, mode='constant', cval=0)
    y_gradient = ndimage.convolve(pixel_values, y1d, mode='constant', cval=0)
    gradient = np.round(np.sqrt(x_gradient ** 2 + y_gradient ** 2))
    return gradient


def robert(pixel_values: np.ndarray):
    x_robert = np.array([[-1, 0],
                        [0, 1]])
    y_robert = np.array([[0, 1],
                        [-1, 0]])
    x_gradient = ndimage.convolve(pixel_values, x_robert, mode='constant', cval=0)
    y_gradient = ndimage.convolve(pixel_values, y_robert, mode='constant', cval=0)
    gradient = np.round(np.sqrt(x_gradient ** 2 + y_gradient ** 2))
    return gradient


def prewitt(pixel_values: np.ndarray):
    x_prewitt = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]])
    y_prewitt = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]])
    x_gradient = ndimage.convolve(pixel_values, x_prewitt, mode='constant', cval=0)
    y_gradient = ndimage.convolve(pixel_values, y_prewitt, mode='constant', cval=0)
    gradient = np.round(np.sqrt(x_gradient ** 2 + y_gradient ** 2))
    return gradient


def sobel(pixel_values: np.ndarray):
    x_sobel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
    y_sobel = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])
    x_gradient = ndimage.convolve(pixel_values, x_sobel, mode='constant', cval=0)
    y_gradient = ndimage.convolve(pixel_values, y_sobel, mode='constant', cval=0)
    gradient = np.round(np.sqrt(x_gradient ** 2 + y_gradient ** 2))
    return gradient

