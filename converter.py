from PIL import Image
import numpy as np
from scipy import ndimage, signal
import algorithm


# start of image enhancer (chap 3)

def negative(image: Image.Image, parameter: list):
    pixel_values = np.array(image.getdata(), dtype=np.uint8)

    # negative algorithm
    pixel_values = np.max(pixel_values) - pixel_values

    processed_image = image.copy()
    processed_image.putdata(pixel_values)
    return processed_image


def otsu(image: Image.Image, parameter: list):
    pixel_values = np.array(image.getdata())

    # threshold algorithm with otsu threshold
    otsu_threshold = algorithm.otsu(pixel_values)
    pixel_values = np.where(pixel_values > otsu_threshold, 255, 0).astype(np.uint8)

    processed_image = image.copy()
    processed_image.putdata(pixel_values)
    return processed_image


def power_law(image: Image.Image, parameter: list):
    pixel_values = np.array(image.getdata())
    c, y = parameter[0], parameter[1]

    # power law algorithm
    pixel_values = c * (pixel_values ** y)

    pixel_values = np.clip(pixel_values, 0, 255).astype(np.uint8)
    processed_image = image.copy()
    processed_image.putdata(pixel_values)
    return processed_image


def max_filter(image: Image.Image, parameter: list):
    pixel_values = np.array(image.getdata(), dtype=np.uint8)
    pixel_values = pixel_values.reshape(image.height, image.width)
    k = parameter[0]

    # max filter algorithm
    pixel_values = ndimage.maximum_filter(pixel_values, size=(k, k), mode='constant', cval=0)

    processed_image = image.copy()
    processed_image.putdata(pixel_values.flatten())
    return processed_image


def min_filter(image: Image.Image, parameter: list):
    pixel_values = np.array(image.getdata(), dtype=np.uint8)
    pixel_values = pixel_values.reshape(image.height, image.width)
    k = parameter[0]

    # min filter algorithm
    pixel_values = ndimage.minimum_filter(pixel_values, size=(k, k), mode='constant', cval=0)

    processed_image = image.copy()
    processed_image.putdata(pixel_values.flatten())
    return processed_image


def simple_average_filter(image: Image.Image, parameter: list):
    pixel_values = np.array(image.getdata(), dtype=np.float64)
    pixel_values = pixel_values.reshape(image.height, image.width)
    k = parameter[0]

    # simple average filter or box filter algorithm
    kernel = np.full((k, k), 1 / (k * k))
    pixel_values = ndimage.convolve(pixel_values, kernel, mode='constant', cval=0)

    pixel_values = np.round(pixel_values).astype(np.uint8)
    processed_image = image.copy()
    processed_image.putdata(pixel_values.flatten())
    return processed_image


def weighted_average_filter(image: Image.Image, parameter: list):
    pixel_values = np.array(image.getdata(), dtype=np.float64)
    pixel_values = pixel_values.reshape(image.height, image.width)
    k = parameter[0]

    # weighted average filter algorithm
    kernel = algorithm.create_custom_weighted_kernel(k)  # using the so-so same structure as gaussian kernel
    pixel_values = ndimage.convolve(pixel_values, kernel, mode='constant', cval=0)

    pixel_values = np.round(pixel_values).astype(np.uint8)
    processed_image = image.copy()
    processed_image.putdata(pixel_values.flatten())
    return processed_image


def k_nearest_mean_filter(image: Image.Image, parameter: list):
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

    pixel_values = pixel_values.flatten().astype(np.uint8)
    processed_image = image.copy()
    processed_image.putdata(pixel_values)
    return processed_image


def median_filter(image: Image.Image, parameter: list):
    pixel_values = np.array(image.getdata(), dtype=np.uint8)
    pixel_values = pixel_values.reshape(image.height, image.width)
    kernel_size = parameter[0]

    # median filter algorithm
    pixel_values = signal.medfilt2d(pixel_values, kernel_size)

    processed_image = image.copy()
    processed_image.putdata(pixel_values.flatten())
    return processed_image


# end of image enhancer (chap 3)

# start of edge detection (chap 4)

def laplacian_filter(image: Image.Image, parameter: list):
    pixel_values = np.array(image.getdata(), dtype=np.float64)
    pixel_values = pixel_values.reshape(image.height, image.width)
    kernel_type = parameter[0]

    # laplacian filter and enhancer
    if kernel_type == 'filter':
        kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float64)
    elif kernel_type == 'variant_filter':
        kernel = np.array([
            [0, 1, 0],
            [1, -8, 1],
            [0, 1, 0]
        ], dtype=np.float64)
    elif kernel_type == 'enhancer':
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float64)
    else:
        kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ], dtype=np.float64)
    pixel_values = ndimage.convolve(pixel_values, kernel, mode='constant', cval=0)
    pixel_values = np.clip(pixel_values, 0, 255).astype(np.uint8)

    processed_image = image.copy()
    processed_image.putdata(pixel_values.flatten())
    return processed_image


def canny(image: Image.Image, parameter: list):
    pixel_values = np.array(image.getdata(), dtype=np.float64)
    pixel_values = pixel_values.reshape(image.height, image.width)
    kernel_size, sigma, t2 = 3, 1.0, 85
    t1 = 2 * t2

    # canny algorithm
    # step 1, Smoothing
    pixel_values = algorithm.gaussian_filter(pixel_values, kernel_size, sigma)

    # step 2, Gradient operator
    def convert_angle(angle):
        if angle < 0:
            angle += 180
        if 22.5 <= angle < 67.5:
            return 45
        elif 67.5 <= angle < 112.5:
            return 90
        elif 112.5 <= angle <= 157.5:
            return 135
        else:
            return 0

    x_gradient = ndimage.convolve(pixel_values, algorithm.Matrix.x_sobel, mode='constant', cval=0)
    y_gradient = ndimage.convolve(pixel_values, algorithm.Matrix.y_sobel, mode='constant', cval=0)
    gradient = np.sqrt(x_gradient**2 + y_gradient**2)
    theta = np.arctan(y_gradient / (x_gradient + np.finfo(np.float64).eps))
    theta = np.vectorize(convert_angle)(theta)

    # step 3, Non-maximum suppression
    # max neighbour
    max_neighbour_0degree = np.maximum(
        np.roll(
            np.pad(pixel_values, pad_width=1, mode='constant', constant_values=0),
            shift=(0, 1), axis=(0, 1)
        ),
        np.roll(
            np.pad(pixel_values, pad_width=1, mode='constant', constant_values=0),
            shift=(0, -1), axis=(0, 1)
        )
    )[1:-1, 1:-1]

    max_neighbour_45degree = np.maximum(
        np.roll(
            np.pad(pixel_values, pad_width=1, mode='constant', constant_values=0),
            shift=(-1, 1), axis=(0, 1)
        ),
        np.roll(
            np.pad(pixel_values, pad_width=1, mode='constant', constant_values=0),
            shift=(1, -1), axis=(0, 1)
        )
    )[1:-1, 1:-1]

    max_neighbour_90degree = np.maximum(
        np.roll(
            np.pad(pixel_values, pad_width=1, mode='constant', constant_values=0),
            shift=(-1, 0), axis=(0, 1)
        ),
        np.roll(
            np.pad(pixel_values, pad_width=1, mode='constant', constant_values=0),
            shift=(1, 0), axis=(0, 1)
        )
    )[1:-1, 1:-1]

    max_neighbour_135degree = np.maximum(
        np.roll(
            np.pad(pixel_values, pad_width=1, mode='constant', constant_values=0),
            shift=(-1, -1), axis=(0, 1)
        ),
        np.roll(
            np.pad(pixel_values, pad_width=1, mode='constant', constant_values=0),
            shift=(1, 1), axis=(0, 1)
        )
    )[1:-1, 1:-1]

    max_0degree = np.where(theta == 0, gradient, 0)
    max_0degree = np.where(max_0degree > max_neighbour_0degree, max_0degree, 0)

    max_45degree = np.where(theta == 45, gradient, 0)
    max_45degree = np.where(max_45degree > max_neighbour_45degree, max_45degree, 0)

    max_90degree = np.where(theta == 90, gradient, 0)
    max_90degree = np.where(max_90degree > max_neighbour_90degree, max_90degree, 0)

    max_135degree = np.where(theta == 135, gradient, 0)
    max_135degree = np.where(max_135degree > max_neighbour_135degree, max_135degree, 0)

    suppressed = max_0degree + max_45degree + max_90degree + max_135degree

    # step 4, Hysteresis thresholding
    t1_higher = np.where(suppressed > t1, suppressed, 0)
    t2_higher_t1_lower = np.where((t2 <= suppressed) & (suppressed <= t1), suppressed, 0)

    mask_kernel = np.ones((3, 3))
    mask_kernel[1, 1] = 0
    mask_sum = ndimage.convolve(suppressed, mask_kernel, mode='constant', cval=0)

    t2_higher_t1_lower = np.where(mask_sum != 0, t2_higher_t1_lower, 0)
    suppressed = t1_higher + t2_higher_t1_lower
    pixel_values = np.clip(np.round(suppressed), 0, 255).astype(np.uint8)

    processed_image = image.copy()
    processed_image.putdata(pixel_values.flatten())
    return processed_image


def mask_one_dim(image: Image.Image, parameter: list):
    pixel_values = np.array(image.getdata(), dtype=np.uint8)
    pixel_values = pixel_values.reshape(image.height, image.width)

    # median filter algorithm
    pixel_values = np.clip(algorithm.mask1d(pixel_values), 0, 255).astype(np.uint8)

    processed_image = image.copy()
    processed_image.putdata(pixel_values.flatten())
    return processed_image


def robert(image: Image.Image, parameter: list):
    pixel_values = np.array(image.getdata(), dtype=np.uint8)
    pixel_values = pixel_values.reshape(image.height, image.width)

    # median filter algorithm
    pixel_values = np.clip(algorithm.robert(pixel_values), 0, 255).astype(np.uint8)

    processed_image = image.copy()
    processed_image.putdata(pixel_values.flatten())
    return processed_image


def sobel(image: Image.Image, parameter: list):
    pixel_values = np.array(image.getdata(), dtype=np.uint8)
    pixel_values = pixel_values.reshape(image.height, image.width)

    # median filter algorithm
    pixel_values = np.clip(algorithm.sobel(pixel_values), 0, 255).astype(np.uint8)

    processed_image = image.copy()
    processed_image.putdata(pixel_values.flatten())
    return processed_image


def prewitt(image: Image.Image, parameter: list):
    pixel_values = np.array(image.getdata(), dtype=np.uint8)
    pixel_values = pixel_values.reshape(image.height, image.width)

    # median filter algorithm
    pixel_values = np.clip(algorithm.prewitt(pixel_values), 0, 255).astype(np.uint8)

    processed_image = image.copy()
    processed_image.putdata(pixel_values.flatten())
    return processed_image
