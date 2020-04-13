def dict_append(dictionary, appending, ignore_keys=None):
    if ignore_keys is None:
        ignore_keys = list()
    for key, value in dictionary.items():
        if key in ignore_keys:
            continue
        if isinstance(value, dict):
            dictionary[key] = dict_append(value, appending[key], ignore_keys)
        else:
            dictionary[key].append(appending[key])
    return dictionary


def dict_op(dictionary, operation, ignore_keys=None):
    result = dict()
    if ignore_keys is None:
        ignore_keys = list()
    for key, value in dictionary.items():
        if key in ignore_keys:
            continue
        if isinstance(value, dict):
            result[key] = dict_op(value, operation, ignore_keys)
        else:
            result[key] = operation(value)
    return result


if __name__ == '__main__':
    import numpy as np
    a = {'c': {'b': 1}, 'd': 1}
    b = dict_op(a, lambda x: list())
    print(b)
    b = dict_append(b, a)
    b = dict_append(b, a)
    b = dict_op(b, np.array)
    print(b)
