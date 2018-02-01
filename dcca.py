import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.signal import detrend


def moving_detrend(x, win_len):
    stride, = x.strides
    x = as_strided(x, shape=(len(x) - win_len, win_len), strides=(stride, stride))
    return detrend(x, axis=1)


def fluctuation(x, y=None):
    if y is None:
        return np.mean(np.std(x, axis=1))
    return np.mean(np.mean(x * y, axis=1))


def dcca(x, y, win_len_list=None):
    """Estimate the DCCA coefficient for a pair of time series

    Parameters
    ------------
    x, y: array like
       Input data in any form that can be converted to 1D arrays, e.g., lists, tuples, and numpy.ndarrays.
       x and y must have the same length
    win_len_list: list of ints, optional
        A sequence of window lengths to segment the data for analysis.
        If not given, then it will be set to all values in the range from 4 to data.length/4.

    Return
    -------------
    output: 1d-array of floats
        The estimated DCCA coefficients

    Example
    -------------
    Computes the DCCA coefficient:
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n = 40
    >>> x = np.random.rand(n)
    >>> y = np.random.rand(n)
    >>> dcca(x, y)
    array([-0.03377323,  0.27173006,  0.06972629, -0.01924527,  0.14087188,
            0.27654658,  0.27014191])
    """

    # ignore harmless warnings
    import warnings
    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

    # convert input into ndarrays
    x = np.asarray(x)
    y = np.asarray(y)

    # check x and y arrays
    if x.ndim != y.ndim or x.ndim != 1:
        raise TypeError("expected 1D array for x and y")
    if x.size == 0:
        raise TypeError("expected non-empty arrays for x and y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected same-size arrays for x and y")
    if x.shape[0] <= 4:
        raise ValueError("not enough samples")

    if win_len_list is None:
        win_len_list = np.arange(4, x.shape[0] // 4 + 1, dtype=int)
    else:
        win_len_list = np.asarray(win_len_list, dtype=int)

    # check win_len_list
    if win_len_list.ndim != 1:
        raise ValueError("expected 1D array for win_len_list")
    if win_len_list.size == 0:
        raise TypeError("expected non-empty array for win_len_list")
    if np.any(win_len_list) >= x.shape[0]:
        raise ValueError("scale values should be less than %d" % x.shape[0])

    # replace each time series by a cumulative sum with subtracted mean
    x = np.cumsum(x - np.mean(x))
    y = np.cumsum(y - np.mean(y))

    # fluctuation arrays for x, y, and xy2
    Fxx = np.zeros(len(win_len_list))
    Fyy = np.ones_like(Fxx)
    Fxy2 = np.ones_like(Fxx)
    for j, n in enumerate(win_len_list):
        # get the detrended arrays for x and y. For convenience, xd and yd are matrices of size (N-n) by n.
        # The kth row of xd is the detrended array for the sequence x_k, ..., k_(k+n)
        xd = moving_detrend(x, n)
        yd = moving_detrend(y, n)
        # fluctuation coefficients for xd, yd, and xd * yd
        Fxx[j] = fluctuation(xd)
        Fyy[j] = fluctuation(yd)
        Fxy2[j] = fluctuation(xd, yd)
    rho_dcca = Fxy2 / (Fxx * Fyy)
    return rho_dcca, Fxx, Fyy, Fxy2
