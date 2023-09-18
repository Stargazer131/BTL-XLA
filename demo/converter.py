from PIL import Image
import numpy as np
from scipy import ndimage, signal
from demo import utility


def negative(image: Image.Image, parameter) -> None:
    pixel_values = np.array(image.getdata())  # Get the grayscale pixel values

    pixel_values = np.max(pixel_values) - pixel_values  # negative algorithm

    image.putdata(pixel_values)


def threshold(image: Image.Image, parameter) -> None:
    pixel_values = np.array(image.getdata())  # Get the grayscale pixel values
    thresh_hold_val = (np.max(pixel_values) + np.min(pixel_values)) / 2
    thresh_hold_val = int(thresh_hold_val)

    func = lambda x: 1 if x > thresh_hold_val else 0
    pixel_values = np.vectorize(func)(pixel_values)

    image.putdata(pixel_values)


def power_law(image: Image.Image, parameter) -> None:
    pixel_values = np.array(image.getdata())  # Get the grayscale pixel values
    print(pixel_values.shape)
    c, y = parameter[0], parameter[1]

    pixel_values = c * (pixel_values ** y)

    image.putdata(pixel_values)


def max_filter(image: Image.Image, parameter) -> None:
    pixel_values = np.array(image.getdata()).reshape(image.height, image.width)  # Get the grayscale pixel values
    k = parameter[0]

    pixel_values = ndimage.maximum_filter(pixel_values, size=(k, k), mode='constant', cval=0)

    image.putdata(pixel_values.flatten())


def min_filter(image: Image.Image, parameter) -> None:
    pixel_values = np.array(image.getdata()).reshape(image.height, image.width)  # Get the grayscale pixel values
    k = parameter[0]

    pixel_values = ndimage.minimum_filter(pixel_values, size=(k, k), mode='constant', cval=0)

    image.putdata(pixel_values.flatten())


def simple_average_filter(image: Image.Image, parameter) -> None:
    pixel_values = np.array(image.getdata()).reshape(image.height, image.width)  # Get the grayscale pixel values
    k = parameter[0]
    kernel = np.full((k, k), 1 / (k * k))

    pixel_values = ndimage.convolve(pixel_values, kernel, mode='constant', cval=0)

    image.putdata(pixel_values.flatten())


def weighted_average_filter(image: Image.Image, parameter) -> None:
    pixel_values = np.array(image.getdata()).reshape(image.height, image.width)  # Get the grayscale pixel values
    k = parameter[0]
    kernel = utility.create_custom_weighted_kernel(k)  # using the so-so same structure as gaussian kernel

    pixel_values = ndimage.convolve(pixel_values, kernel, mode='constant', cval=0)

    image.putdata(pixel_values.flatten())


def k_nearest_mean_filter(image: Image.Image, parameter) -> None:
    pixel_values = np.array(image.getdata()).reshape(image.height, image.width)  # Get the grayscale pixel values
    k, kernel_size, thresh_hold = parameter[0], parameter[1], parameter[2]

    for i in range(image.height):
        for j in range(image.width):
            neighbour = []
            center_value = pixel_values[i, j]
            for x in range(i - kernel_size // 2, i + kernel_size // 2 + 1):
                for y in range(j - kernel_size // 2, j + kernel_size // 2 + 1):
                    if 0 <= x < image.height and 0 <= y < image.width:
                        neighbour.append(pixel_values[x, y])
                    else:
                        neighbour.append(0)
            neighbour.sort(key=lambda val: abs(val-center_value))
            neighbour = np.array(neighbour)[:k]
            k_nearest_mean = int(np.mean(neighbour))
            if abs(center_value-k_nearest_mean) > thresh_hold:
                pixel_values[i, j] = k_nearest_mean

    image.putdata(pixel_values.flatten())


def median_filter(image: Image.Image, parameter) -> None:
    pixel_values = np.array(image.getdata()).reshape(image.height, image.width)  # Get the grayscale pixel values
    kernel_size = parameter[0]

    pixel_values = signal.medfilt2d(pixel_values, kernel_size)

    image.putdata(pixel_values.flatten())

