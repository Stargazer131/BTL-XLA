import re
import inspect


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
