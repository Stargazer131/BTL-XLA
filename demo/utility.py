import re
import inspect
import numpy as np


def get_classes_from_file(file_path):
    classes = []

    with open(file_path, 'r') as file:
        source_code = file.read()

    global_namespace = {}
    exec(source_code, global_namespace)

    for item in global_namespace.values():
        if inspect.isclass(item):
            classes.append(item)

    return classes


def split_camel_case(input_string):
    # Use regular expression to split CamelCase string
    parts = re.findall(r'[A-Z][a-z]*', input_string)
    return '_'.join(parts).lower()


def create_custom_weighted_kernel(k):
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
