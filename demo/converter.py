from PIL import Image
import numpy as np
from scipy import ndimage, signal
from demo import utility


def negative(image: Image.Image, parameter):
    pixel_values = np.array(image.getdata(), dtype=np.uint8)

    # negative algorithm
    pixel_values = np.max(pixel_values) - pixel_values

    image.putdata(pixel_values)


def threshold(image: Image.Image, parameter):
    pixel_values = np.array(image.getdata())

    # threshold algorithm with otsu threshold
    grayscale_frequency = np.bincount(pixel_values) / len(pixel_values)
    pi_cumsum = np.cumsum(grayscale_frequency)
    mk_cumsum = np.cumsum(grayscale_frequency * np.array(range(len(grayscale_frequency))))
    mg = mk_cumsum[-1]  # total cumulative sum
    a = (mg * pi_cumsum - mk_cumsum) ** 2
    b = pi_cumsum * (1 - pi_cumsum)
    variance = a / (b + np.finfo(np.float64).eps)  # add epsilon to prevent divide by 0
    otsu_threshold = np.mean(np.argwhere(variance == np.max(variance)))
    pixel_values = np.where(pixel_values > otsu_threshold, 255, 0)

    image.putdata(pixel_values.astype(np.uint8))


def power_law(image: Image.Image, parameter):
    pixel_values = np.array(image.getdata())
    c, y = parameter[0], parameter[1]

    # power law algorithm
    pixel_values = c * (pixel_values ** y)

    pixel_values = np.clip(pixel_values, 0, 255).astype(np.uint8)
    image.putdata(pixel_values)


def max_filter(image: Image.Image, parameter):
    pixel_values = np.array(image.getdata(), dtype=np.uint8)
    pixel_values = pixel_values.reshape(image.height, image.width)
    k = parameter[0]

    # max filter algorithm
    pixel_values = ndimage.maximum_filter(pixel_values, size=(k, k), mode='constant', cval=0)

    image.putdata(pixel_values.flatten())


def min_filter(image: Image.Image, parameter):
    pixel_values = np.array(image.getdata(), dtype=np.uint8)
    pixel_values = pixel_values.reshape(image.height, image.width)
    k = parameter[0]

    # min filter algorithm
    pixel_values = ndimage.minimum_filter(pixel_values, size=(k, k), mode='constant', cval=0)

    image.putdata(pixel_values.flatten())


def simple_average_filter(image: Image.Image, parameter):
    pixel_values = np.array(image.getdata(), dtype=np.float64)
    pixel_values = pixel_values.reshape(image.height, image.width)
    k = parameter[0]

    # simple average filter or box filter algorithm
    kernel = np.full((k, k), 1 / (k * k))
    pixel_values = ndimage.convolve(pixel_values, kernel, mode='constant', cval=0)

    pixel_values = np.round(pixel_values).astype(np.uint8)
    image.putdata(pixel_values.flatten())


def weighted_average_filter(image: Image.Image, parameter):
    pixel_values = np.array(image.getdata(), dtype=np.float64)
    pixel_values = pixel_values.reshape(image.height, image.width)
    k = parameter[0]

    # weighted average filter algorithm
    kernel = utility.create_custom_weighted_kernel(k)  # using the so-so same structure as gaussian kernel
    pixel_values = ndimage.convolve(pixel_values, kernel, mode='constant', cval=0)

    pixel_values = np.round(pixel_values).astype(np.uint8)
    image.putdata(pixel_values.flatten())


def k_nearest_mean_filter(image: Image.Image, parameter):
    pixel_values = np.array(image.getdata())
    pixel_values = pixel_values.reshape(image.height, image.width)
    k, kernel_size, thresh_hold = parameter[0], parameter[1], parameter[2]
    original_data = pixel_values.copy()

    # k nearest neighbour mean filter algorithm
    for i in range(image.height):
        for j in range(image.width):
            neighbour = []
            center_value = original_data[i, j]
            for x in range(i - kernel_size // 2, i + kernel_size // 2 + 1):
                for y in range(j - kernel_size // 2, j + kernel_size // 2 + 1):
                    if 0 <= x < image.height and 0 <= y < image.width:
                        neighbour.append(original_data[x, y])
                    else:
                        neighbour.append(0)
            neighbour.sort(key=lambda val: abs(val - center_value))
            neighbour = np.array(neighbour)[:k]
            k_nearest_mean = np.round(np.mean(neighbour))
            if abs(center_value - k_nearest_mean) > thresh_hold:
                pixel_values[i, j] = k_nearest_mean

    image.putdata(pixel_values.flatten().astype(np.uint8))


def median_filter(image: Image.Image, parameter):
    pixel_values = np.array(image.getdata(), dtype=np.uint8)
    pixel_values = pixel_values.reshape(image.height, image.width)
    kernel_size = parameter[0]

    # median filter algorithm
    pixel_values = signal.medfilt2d(pixel_values, kernel_size)

    image.putdata(pixel_values.flatten())
