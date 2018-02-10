import numpy as np
from numpy.lib.stride_tricks import as_strided

from progress_bar import progress


def detrend(y, x=None, deg=1, axis=0):
    # check arguments.
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if y.size == 0:
        raise TypeError("expected non-empty vector for y")
    if x is None:
        if y.ndim == 1:
            x = np.arange(y.size)
        else:
            x = np.arange(y.shape[1])
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.shape[0] != y.shape[axis]:
        raise TypeError("expected x and y to have same length")
    p = np.polyfit(x, y.T, deg)
    if y.ndim == 1:
        return np.polyval(p, x)
    ret = np.zeros_like(y)
    for a in p:
        ret = ret * x + a[:, None]
    return y - ret


def moving_detrend(x, box_size, deg=1):
    """
    Remove a polynomial trend from a time series using a moving window

    Parameters
    ----------------------
    x: array_like
        The time series. Input data in any form that can be converted to a 1-D ndarray, e.g., lists, and tuples
    box_size: int
        The size of the box to estimate and remove trends. It must be less than the size of the time series

    Return
    -----------------------
    ret: 2-D numpy.ndarray
        The detrended array
    """
    x = np.asarray(x)
    # check x and y arrays
    if x.ndim != 1:
        raise TypeError("expected an 1D array")
    if x.size == 0:
        raise TypeError("expected a non-empty array")
    if int(box_size) != box_size:
        raise TypeError("expected an integer value for box size")
    box_size = int(box_size)
    if box_size > x.size:
        raise TypeError("expected box_size to be less than the time series size")
    stride, = x.strides
    x = as_strided(x, shape=(len(x) - box_size, box_size), strides=(stride, stride))
    return detrend(x, deg=deg, axis=1)


def fluctuation(x_detrended, y_detrended=None):
    if y_detrended is None:
        return np.sqrt(np.mean(np.var(x_detrended, axis=1)))
    return np.mean(np.mean(x_detrended * y_detrended, axis=1))


def check_box_sizes(box_sizes_list, x, max_num_boxes=100):
    """ Check if box_size_list is valid
    Parameters
    --------------------
    box_sizes_list: None or array like
        If None, a box size list will be created
    x: 1-D numpy.ndarray
        The time series to be segmented
    max_num_boxes: int
        The max number of boxes to be included in the list
    """
    if box_sizes_list is None:
        # creates a valid list of box sizes
        beg = np.log10(4)
        end = np.log10(x.shape[0] // 4)
        box_sizes_list = np.logspace(beg, end, dtype=int)
        box_sizes_list = np.unique(box_sizes_list)
        index = np.linspace(0, len(box_sizes_list) - 1, max_num_boxes, dtype=int)
        index = np.unique(index)
        box_sizes_list = box_sizes_list[index]
    box_sizes_list = np.asarray(box_sizes_list, dtype=int)
    # check if the input is valid
    if box_sizes_list.ndim != 1:
        raise ValueError("expected 1-D array for box_sizes_list")
    if box_sizes_list.size == 0:
        raise TypeError("expected non-empty array for box_sizes_list")
    if np.any(box_sizes_list) >= x.shape[0]:
        raise ValueError("not enough samples")
    return box_sizes_list


def dfa(x, box_sizes_list, ignore_warnings=True, max_num_boxes=100):
    """Estimate the DFA coefficients for a pair of time series

    Parameters
    ------------
    x, y: array like
       Input data in any form that can be converted to 1-D numpy.ndarray, e.g., lists, tuples, and numpy.ndarrays.
       x and y must have the same length
    box_sizes_list: list of ints, optional
        A sequence of window lengths to segment the data for the fluctuation analysis.
        If not given, then it will be set to all values in the range from 4 to data.length/4.
    ignore_warnings: bool, optional
        If warning_filter is True, then all warnings from scipy will be suppressed

    Returns
    -------------
    box_sizes_list: 1-D numpy.ndarray of ints
        The box sizes used in the calculation
    flc_x: 1-D numpy.ndarray of floats
        The DFA coefficients of the time series
    """

    if ignore_warnings:
        # ignore harmless warnings
        import warnings
        warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

    # convert input into ndarrays
    x = np.asarray(x)

    # check x and y
    if x.ndim != 1:
        raise TypeError("expected an 1-D array for x")
    if x.size == 0:
        raise TypeError("expected a non-empty array for x")

    box_sizes_list = check_box_sizes(box_sizes_list, x, max_num_boxes=max_num_boxes)

    # replace each time series by a cumulative sum with subtracted mean
    x = np.cumsum(x - np.mean(x))
    f = np.zeros(len(box_sizes_list))
    for j, n in enumerate(box_sizes_list):
        # get the detrended arrays for x and y. For convenience, xd and yd are matrices of size (N-n) by n.
        # The kth row of xd is the detrended array for the sequence x_k, ..., k_(k+n)
        xd = moving_detrend(x, n)
        # fluctuation coefficients for xd, yd, and xd * yd
        f[j] = fluctuation(xd)
    return f


def dcca(x, y, box_sizes_list=None, max_num_boxes=100, deg=1, show_warnings=False):
    """Estimate the DCCA coefficients for a pair of time series

    Parameters
    ------------
    x, y: array like
       Input data in any form that can be converted to 1-D numpy.arrays, e.g., lists, tuples, and numpy.ndarrays.
       x and y must have the same length
    box_sizes_list: list of ints, optional
        A sequence of box sizes to segment the data for the fluctuation analysis.
        If not given, then it will be set to all values in the range from 4 to data.length/4.
    deg: degree of polynomial interpolation to detrend the signals
    ignore_warnings: bool, optional
        If warning_filter is True, then all warnings from scipy will be suppressed

    Returns
    -------------
    box_sizes_list: 1-D numpy.ndarray of ints
        The box sizes used in the calculation
    rho: 1-D numpy.ndarray of floats
        The DCCA coefficients
    flc_x: 1-D numpy.ndarray of floats
        The DFA coefficients for the first time series
    flc_y: 1-D numpy.ndarray of floats
        The DFA coefficients for the second time series
    flc_xy2: 1-D numpy.ndarray of floats
        The fluctuation coefficients for the product x*y
    """

    if not show_warnings:
        # ignore harmless warnings
        import warnings
        warnings.filterwarnings(action="ignore")

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

    box_sizes_list = check_box_sizes(box_sizes_list, x, max_num_boxes=max_num_boxes)

    # replace each time series by a cumulative sum with subtracted mean
    x = np.cumsum(x - np.mean(x))
    y = np.cumsum(y - np.mean(y))

    # fluctuation arrays for x, y, and xy2
    flc_x = np.zeros(len(box_sizes_list))
    flc_y = np.ones_like(flc_x)
    flc_xy2 = np.ones_like(flc_x)
    rho = np.ones_like(flc_x)
    for j, n in enumerate(box_sizes_list):
        progress(j+1, len(box_sizes_list))
        # get the detrended arrays for x and y. For convenience, xd and yd are matrices of size (N-n) by n.
        # The kth row of xd is the detrended array for the sequence x_k, ..., k_(k+n)
        xd = moving_detrend(x, n, deg=deg)
        yd = moving_detrend(y, n, deg=deg)
        # fluctuation coefficients for xd, yd, and xd * yd
        flc_x[j] = fluctuation(xd)
        flc_y[j] = fluctuation(yd)
        flc_xy2[j] = fluctuation(xd, yd)
        rho[j] = flc_xy2[j] / (flc_x[j] * flc_y[j])
    return box_sizes_list, rho, flc_x, flc_y, flc_xy2
