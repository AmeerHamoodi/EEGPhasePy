import numpy as np
from numbers import Real


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


def _check_real_number(value, name, non_negative=False):
    '''
    Validate and convert a finite real-valued scalar.

    Parameters
    ----------
    value : int | float
        Value to validate
    name : str
        Name used in validation error messages
    non_negative : bool
        Whether to require the value to be greater than or equal to zero

    Returns
    -------
    float
        The validated value converted to a Python float
    '''
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Real, np.integer, np.floating)):
        raise TypeError(f'{name} must be a real number')

    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f'{name} must be finite') from error

    if not np.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if non_negative and value < 0:
        raise ValueError(f'{name} must be non-negative')

    return value


def _check_real_array(value, name, contents='values'):
    '''
    Convert an array-like value and require finite real numeric elements.

    Parameters
    ----------
    value : array_like
        Array-like value to validate
    name : str
        Name used in validation error messages
    contents : str
        Description of the array elements for validation messages

    Returns
    -------
    numpy.ndarray
        The validated array
    '''
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f'{name} must be a rectangular numeric array') from error

    if not np.issubdtype(array.dtype, np.number) or \
            np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(
            f'{name} must contain real numeric {contents}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain only finite {contents}')

    return array


def _check_filter_coefficients(value, name='real_time_filter'):
    '''
    Validate FIR or (numerator, denominator) filter coefficients.

    Returns
    -------
    tuple
        The coefficient array, numerator length, and denominator length
    '''
    coefficients = _check_real_array(value, name, contents='coefficients')
    if coefficients.ndim == 1:
        numerator_length = len(coefficients)
        denominator_length = 1
    elif coefficients.ndim == 2 and coefficients.shape[0] == 2:
        numerator_length, denominator_length = map(len, coefficients)
    else:
        raise ValueError(
            f'{name} must be a 1D numerator or a 2-row '
            '(numerator, denominator) array')

    if numerator_length == 0 or denominator_length == 0:
        raise ValueError(f'{name} coefficients cannot be empty')

    return coefficients, numerator_length, denominator_length
