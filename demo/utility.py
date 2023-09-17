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
    matrix_size = k
    # Create an empty matrix filled with zeros
    matrix = np.zeros((matrix_size, matrix_size))

    for i in range(matrix_size):
        for j in range(matrix_size):
            min_dist_to_edge_i = min(i, matrix_size - 1 - i)
            min_dist_to_edge_j = min(j, matrix_size - 1 - j)
            matrix[i, j] = 2 ** (min_dist_to_edge_i + min_dist_to_edge_j)

    matrix = matrix / np.sum(matrix)
    return matrix



