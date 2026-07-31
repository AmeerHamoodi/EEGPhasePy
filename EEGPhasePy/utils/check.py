import numpy as np


def _is_array(value):
    '''
    Checks if a value is an array. \n

    Parameters
    ----------
    value : any
        Value to check

    Returns
    -------
    bool
        True if value is an array; False otherwise
    '''
    return hasattr(value, '__len__')


def _check_array_dimensions(test_array, target_shape_structures):
    '''
    Checks if two arrays share the same dimensions (i.e. both are 1D, 2D, 3D,
    etc). Raises value error if array doesn't match target dimension

    Parameters
    ----------
    test_array : array_like
        Array to test dimensions of
    target_shape_structure : array_like
        Array representing the target structures
    '''
    failed = True
    for struct in target_shape_structures:
        np_target_shape_structure = np.array(struct)
        if len(np.shape(test_array)) == len(np_target_shape_structure):
            failed = False

    if failed:
        raise ValueError("The provided array has the wrong dimension. " +
                         "Arrays can have the following dimensions:" +
                         "".join([" %dD" % len(_struct)
                                  for _struct in target_shape_structures]))


def _check_type(value: any, types: list):
    '''
    Checks if value matches a specific type(s). Raises type error if value
    doesn't match type

    Parameters
    ----------

    value : any
        Value to be checked
    types : array of "array" | "int" | "float" | "bool"
        Type value should match.
    '''

    for i, type_name in enumerate(types):
        if type_name == "array" and _is_array(value):
            break
        elif type_name == "int" and isinstance(value, int) and not \
                isinstance(value, bool):
            break
        elif type_name == "float" and isinstance(value, float):
            break
        elif type_name == "bool" and isinstance(value, bool):
            break
        elif i + 1 >= len(types):
            if len(types) == 1:
                type_to_message_map = {
                    "array": "an array",
                    "int": "an int",
                    "float": "a float",
                    "bool": "a bool"
                }
                raise TypeError("Value must be " +
                                type_to_message_map[type_name] + " type")
            else:
                raise TypeError("Value must be one of: " + ' or '.join(types))
