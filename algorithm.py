import numpy as np


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


def otsu_threshold(pixel_values: np.ndarray):
    grayscale_frequency = np.bincount(pixel_values) / len(pixel_values)
    pi_cumsum = np.cumsum(grayscale_frequency)
    mk_cumsum = np.cumsum(grayscale_frequency * np.array(range(len(grayscale_frequency))))
    mg = mk_cumsum[-1]  # total cumulative sum
    a = (mg * pi_cumsum - mk_cumsum) ** 2
    b = pi_cumsum * (1 - pi_cumsum)
    variance = a / (b + np.finfo(np.float64).eps)  # add epsilon to prevent divide by 0
    otsu_threshold_ = np.mean(np.argwhere(variance == np.max(variance)))
    return np.where(pixel_values > otsu_threshold_, 255, 0)


