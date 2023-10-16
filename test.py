import numpy as np
import scipy.ndimage
from scipy import ndimage
import algorithm


def test_symmetric_matrix():
    print(algorithm.create_custom_weighted_kernel(5))


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


def test_gaussian():
    pixel_values = np.array([
        [7, 1, 3, 8, 8, 8],
        [2, 0, 6, 5, 7, 0],
        [6, 6, 4, 2, 7, 6],
        [3, 8, 7, 1, 0, 1],
        [2, 4, 9, 6, 2, 8],
        [1, 4, 8, 7, 5, 3]
    ], dtype=np.float64)

    print(algorithm.gaussian_filter(pixel_values))


def test_theta():
    def convert(angle):
        if 22.5 <= angle < 67.5:
            return 45
        elif 67.5 <= angle < 112.5:
            return 90
        elif 112.5 <= angle <= 157.5:
            return 135
        else:
            return 0


    pixel_values = np.array([
        [7, 1, 3, 8, 8, 8],
        [2, 0, 6, 5, 7, 0],
        [6, 6, 4, 2, 7, 6],
        [3, 8, 7, 1, 0, 1],
        [2, 4, 9, 6, 2, 8],
        [1, 4, 8, 7, 5, 3]
    ], dtype=np.float64)
    x_gradient = ndimage.convolve(pixel_values, algorithm.Matrix.x_sobel, mode='constant', cval=0)
    y_gradient = ndimage.convolve(pixel_values, algorithm.Matrix.y_sobel, mode='constant', cval=0)
    gradient = np.sqrt(x_gradient**2 + y_gradient**2)
    theta = np.degrees(np.arctan(y_gradient / (x_gradient + np.finfo(np.float64).eps)))
    theta = np.where(theta >= 0, theta, theta+180)
    theta = np.vectorize(convert)(theta)
    print(theta)


def test_roll():
    # Example input array
    pixel_values = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]
    )

    theta = np.array([
        [0, 135, 45],
        [0, 90, 0],
        [90, 0, 45]]
    )

    # Shift every element one element to the right
    # shifted_array = np.roll(array, shift=(0, -1), axis=(0, 1))
    # shifted_array = np.roll(array, shift=(0, 1), axis=(0, 1))
    max_0degree = np.maximum(
        np.roll(
            np.pad(pixel_values, pad_width=1, mode='constant', constant_values=0),
            shift=(0, 1), axis=(0, 1)
        ),
        np.roll(
            np.pad(pixel_values, pad_width=1, mode='constant', constant_values=0),
            shift=(0, -1), axis=(0, 1)
        )
    )[1:-1, 1:-1]
    max_0degree = np.where(theta == 0, max_0degree, 0)

    a = np.where(theta == 0, pixel_values, 0)
    b = np.maximum(a, max_0degree)
    print(b)

test_roll()

