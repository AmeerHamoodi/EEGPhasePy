import numpy as np

def _is_array(value):
  '''
  Checks if a value is an array. \n

  Parameters
  ----------
  value : any
      Value to check
  
  -------
  Returns
  -------
  bool
      True if value is an array; False otherwise
  '''
  return hasattr(value, '__len__')

def _check_array_dimensions(test_array, target_shape_structures):
  '''
  Checks if two arrays share the same dimensions (i.e. both are 1D, 2D, 3D, etc). Raises value error if 
  array doesn't match target dimension

  Parameters
  ----------

  test_array : array_like
      Array to test dimensions of
  target_shape_structure : array_like
      Array representing the target structures
  '''
  np_test_array = np.array(test_array)
  failed = True
  for struct in target_shape_structures:
    np_target_shape_structure = np.array(struct)
    if len(np_test_array.shape) == len(np_target_shape_structure):
      failed = False
  
  if failed:
    raise ValueError("The provided array has the wrong dimension. Arrays can have the following dimensions:" + "".join([" %dD" % len(_struct) for _struct in target_shape_structures]))

def _check_type(value, types):
  '''
  Checks if value matches a specific type(s). Raises type error if value doesn't match type

  Parameters
  ----------

  value : any
      Value to be checked
  types : array of "array" | "int"
      Type value should match.
  '''

  if types == "array" and not _is_array(value):
    raise TypeError("Value must be an array type")
  elif types == "int" and not isinstance(value, int):
    raise TypeError("Value must be an int type")
      
