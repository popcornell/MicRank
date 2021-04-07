import numpy as np


def uniquetol(array, tol, return_indexes=True):
    """
    similar to MATLAB uniquetol, returns unique elements and possibly their indexes from an array given some tolerance.
    could be made more efficient. For now this will suffice.
    Args:
        array: numpy array
        tol: float
        return_indexes: bool

    Returns:

    """

    assert array.ndim == 1
    sorted = np.sort(array, 0)
    indexes = []
    uniq_values = []
    for a_indx in range(len(array)):
        unique = False
        for s_index in range(len(sorted)):
            if np.abs(array[a_indx] - sorted[s_index]) > tol and array[a_indx] not in uniq_values:
                unique = True
        if unique:
            indexes.append(a_indx)
            uniq_values.append(array[a_indx])

    if return_indexes:
        return uniq_values, indexes
    else:
        return uniq_values




