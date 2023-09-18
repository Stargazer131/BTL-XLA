import numpy as np


def test_k_nearest_mean_filter():
    parameter = [3, 3, 2]
    n = 4

    data = np.array(
        [[1, 2, 3, 2],
         [4, 16, 2, 1],
         [4, 2, 1, 1],
         [2, 1, 2, 1]]
    )

    pixel_values = np.array(data).reshape(data.shape[0], data.shape[1])  # Get the grayscale pixel values
    k, kernel_size, thresh_hold = parameter[0], parameter[1], parameter[2]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            neighbour = []
            center_value = pixel_values[i, j]
            for x in range(i - kernel_size // 2, i + kernel_size // 2 + 1):
                for y in range(j - kernel_size // 2, j + kernel_size // 2 + 1):
                    if 0 <= x < data.shape[0] and 0 <= y < data.shape[1]:
                        neighbour.append(pixel_values[x, y])
                    else:
                        neighbour.append(0)
            neighbour.sort(key=lambda val: abs(val - center_value))
            neighbour = np.array(neighbour)[:k]
            k_nearest_mean = int(np.mean(neighbour))
            if abs(center_value - k_nearest_mean) > thresh_hold:
                pixel_values[i, j] = k_nearest_mean

    print(pixel_values)


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

