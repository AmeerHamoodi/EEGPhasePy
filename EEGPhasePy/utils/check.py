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


def _filter_form(coefficients):
    '''
    Classify a validated coefficient array as FIR, (b, a) or SOS.

    A 1D array is an FIR numerator. A 2D array with 6 columns whose leading
    denominator coefficients (column 3) are all 1, as ``scipy.signal``
    produces with ``output='sos'``, is a second-order-sections array. Any
    other 2-row array is a (numerator, denominator) pair

    Parameters
    ----------
    coefficients : numpy.ndarray
        Filter coefficients

    Returns
    -------
    str
        ``'fir'``, ``'ba'`` or ``'sos'``, or None if the shape matches none
    '''
    if coefficients.ndim == 1:
        return 'fir'
    if coefficients.ndim != 2:
        return None
    if coefficients.shape[1] == 6 and coefficients.shape[0] >= 1 and \
            np.all(coefficients[:, 3] == 1):
        return 'sos'
    if coefficients.shape[0] == 2:
        return 'ba'
    return None


def _check_filter_coefficients(value, name='real_time_filter'):
    '''
    Validate FIR, (numerator, denominator) or second-order-sections (SOS)
    filter coefficients, and check that IIR filters are stable.

    Returns
    -------
    tuple
        The coefficient array, numerator length, and denominator length.
        For SOS filters both lengths are the order of the equivalent
        (numerator, denominator) filter plus one, which is what
        ``scipy.signal.sosfiltfilt`` pads by
    '''
    coefficients = _check_real_array(value, name, contents='coefficients')
    form = _filter_form(coefficients)
    if form is None:
        raise ValueError(
            f'{name} must be a 1D numerator or a 2-row '
            '(numerator, denominator) array, or an (n_sections, 6) array '
            'of second-order sections')

    if form == 'fir':
        numerator_length = len(coefficients)
        denominator_length = 1
    elif form == 'ba':
        numerator_length, denominator_length = map(len, coefficients)
    else:
        numerator_length = denominator_length = 2 * len(coefficients) + 1

    if numerator_length == 0 or denominator_length == 0:
        raise ValueError(f'{name} coefficients cannot be empty')

    if form == 'ba':
        denominators = coefficients[1:]
    elif form == 'sos':
        denominators = coefficients[:, 3:]
    else:
        denominators = []

    for denominator in denominators:
        if denominator[0] == 0:
            raise ValueError(
                f'{name} has a leading denominator coefficient of 0')
        poles = np.roots(denominator)
        if len(poles) > 0 and np.max(np.abs(poles)) >= 1:
            message = (f'{name} is unstable: it has a pole at |z| = '
                       f'{np.max(np.abs(poles)):.4g}, on or outside the '
                       'unit circle, so filtering would produce '
                       'non-finite or meaningless values.')
            if form == 'ba':
                message += (
                    ' High-order IIR filters lose precision in '
                    '(numerator, denominator) form; design the filter as '
                    'second-order sections instead, e.g. '
                    "scipy.signal.butter(..., output='sos')")
            raise ValueError(message)

    return coefficients, numerator_length, denominator_length


def _check_triggers(value, n_samples, name='triggers'):
    '''
    Validate trigger sample indices against a recording length.

    Triggers must be whole numbers in the range [0, n_samples). Negative
    indices are rejected rather than counted from the end of the recording.
    Floats are accepted only if every value is a whole number.

    Parameters
    ----------
    value : array_like
        Trigger sample indices
    n_samples : int
        Number of samples in the recording the triggers index into
    name : str
        Name used in validation error messages

    Returns
    -------
    numpy.ndarray
        The triggers as an int64 array
    '''
    triggers = np.asarray(value)
    if triggers.size == 0:
        return triggers.astype(np.int64)

    if np.issubdtype(triggers.dtype, np.integer):
        triggers = triggers.astype(np.int64)
    elif np.issubdtype(triggers.dtype, np.floating):
        if not np.all(np.isfinite(triggers)) or \
                not np.all(triggers == np.round(triggers)):
            raise TypeError(
                f'{name} must be whole-number sample indices')
        triggers = triggers.astype(np.int64)
    else:
        raise TypeError(f'{name} must be integer sample indices')

    out_of_range = (triggers < 0) | (triggers >= n_samples)
    if np.any(out_of_range):
        bad = triggers[out_of_range]
        raise ValueError(
            f'{name} must be sample indices in the range [0, {n_samples}) '
            f'for data with {n_samples} samples; {len(bad)} are out of '
            f'range (first: {bad[0]})')

    return triggers
