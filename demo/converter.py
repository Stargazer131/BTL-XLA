from PIL import Image
import numpy as np


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
    c, y = parameter[0], parameter[1]

    pixel_values = c * (pixel_values ** y)
    # func = lambda x: min(x, 255)
    # pixel_values = np.vectorize(func)(pixel_values)

    image.putdata(pixel_values)