from PIL import Image
import numpy as np
from scipy import ndimage
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
    # func = lambda x: min(x, 255)
    # pixel_values = np.vectorize(func)(pixel_values)

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

