import numpy as np


# shuffle rows of a and b in unison, returns new objects
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# shuffle columns of a and b in unison, returns new objects
# actually, not used anywhere
def shuffle_columns_in_unison(a, b):
    assert a.shape[1] == b.shape[1]
    p = np.random.permutation(a.shape[1])
    return a[:, p], b[:, p]
