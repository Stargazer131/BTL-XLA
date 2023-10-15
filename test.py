import numpy as np
from scipy import ndimage


def test_symmetric_matrix():
    k = 7
    limit = (k - 1) // 2
    matrix = np.zeros((k, k))
    for x in range(limit + 1):
        for y in range(limit + 1):
            matrix[x, y] = x + y + 1

    flip_right_matrix = np.fliplr(matrix[:limit + 1, :limit + 1])
    flip_down_matrix = np.flipud(matrix[:limit + 1, :limit + 1])

    matrix[:limit + 1, limit:] = flip_right_matrix
    matrix[limit:, :limit + 1] = flip_down_matrix
    matrix[limit:, limit:] = np.flipud(flip_right_matrix)

    print(matrix)


def test_min_max_filter():
    data = np.array(
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 9, 8, 7],
         [6, 5, 4, 3]]
    )

    pixel_values = data.copy()
    k = 3

    # min filter algorithm
    for i in range(4):
        for j in range(4):
            x_limit = [i - k // 2, i + k // 2]
            y_limit = [j - k // 2, j + k // 2]
            if x_limit[0] < 0 or y_limit[0] < 0 or x_limit[1] >= 4 or y_limit[1] >= 4:
                print(i, j, 0)
                pixel_values[i, j] = 0
            else:
                print(i, j, pixel_values[x_limit[0]: x_limit[1]+1, y_limit[0]: y_limit[1]+1])
                pixel_values[i, j] = np.min(data[x_limit[0]: x_limit[1]+1, y_limit[0]: y_limit[1]+1])

    print(pixel_values)


def test_box_filter():
    data = np.array(
        [[1, 2, 3, 2],
         [4, 16, 2, 1],
         [4, 2, 1, 1],
         [2, 1, 2, 1]], dtype=np.float64
    )

    k = 3
    kernel = np.full((k, k), 1 / (k * k))
    pixel_values = ndimage.convolve(data, kernel, mode='constant', cval=0)
    pixel_values = np.round(pixel_values)

    print(data.dtype)
    print(pixel_values)


def test_k_nearest_mean_filter():
    parameter = [3, 3, 2]

    pixel_values = np.array([
        [3, 1, 0, 7, 8, 3],
        [9, 4, 0, 5, 9, 2],
        [6, 4, 2, 7, 1, 2],
        [5, 6, 9, 6, 4, 3],
        [0, 4, 8, 6, 5, 3],
        [6, 0, 0, 7, 1, 8]
    ])

    k, kernel_size, thresh_hold = parameter[0], parameter[1], parameter[2]
    for i in range(pixel_values.shape[0]):
        for j in range(pixel_values.shape[1]):
            neighbour = []
            center_value = pixel_values[i, j]
            for x in range(i - kernel_size // 2, i + kernel_size // 2 + 1):
                for y in range(j - kernel_size // 2, j + kernel_size // 2 + 1):
                    if 0 <= x < pixel_values.shape[0] and 0 <= y < pixel_values.shape[1]:
                        neighbour.append(pixel_values[x, y])
                    else:
                        neighbour.append(0)
            neighbour.sort(key=lambda val: abs(val - center_value))
            neighbour = np.array(neighbour)[:k]
            print(f'{pixel_values[i, j]} ({i},{j})')
            print(neighbour)
            k_nearest_mean = np.round(np.mean(neighbour))
            print(f'{center_value} - {k_nearest_mean} = {abs(center_value - k_nearest_mean)}')
            if abs(center_value - k_nearest_mean) > thresh_hold:
                pixel_values[i, j] = k_nearest_mean

    print(pixel_values)

def test_bincount():
    pixel_values = np.array([
        [7, 1, 3, 8, 8, 8],
        [2, 0, 6, 5, 7, 0],
        [6, 6, 4, 2, 7, 6],
        [3, 8, 7, 1, 0, 1],
        [2, 4, 9, 6, 2, 8],
        [1, 4, 8, 7, 5, 3]
    ])

    grayscale_frequency = np.bincount(pixel_values.flatten())
    p_sum = 9 * np.cumsum(grayscale_frequency/36)
    print(p_sum)

test_bincount()