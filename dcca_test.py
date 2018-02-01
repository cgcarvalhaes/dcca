import os
import argparse

import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pylab as plt

from dcca import dcca


def valid_input(file_name):
    if os.path.isdir(file_name):
        msg = "Expecting a file name but got a directory: '{0}'".format(file_name)
        raise argparse.ArgumentTypeError(msg)

    if not os.path.isfile(file_name):
        msg = "File '{0}' does not exist.".format(file_name)
        raise argparse.ArgumentTypeError(msg)
    else:
        return file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("time_series_file", type=valid_input, help="The path to the time series file")
    parser.add_argument("expected_result_file", type=valid_input, help="The path to the file with expected results")
    args = parser.parse_args()

    x, y = np.loadtxt(args.time_series_file, unpack=True)
    win_len_list, Fxx_gilney, Fyy_gilney, Fxy2_gilney, rho_gilney = np.loadtxt(args.expected_result_file, unpack=True)

    rho, Fxx, Fyy, Fxy2 = dcca(x, y, win_len_list=win_len_list)

    fig, axes = plt.subplots(ncols=2, nrows=2)
    axes[0, 0].plot(win_len_list, Fxx_gilney, label="Gilney")
    axes[0, 0].plot(win_len_list, Fxx, label="Claudio")
    axes[0, 0].text(.5, .9, 'Fxx', horizontalalignment='center', transform=axes[0, 0].transAxes)
    axes[0, 0].legend(loc='lower right')

    axes[0, 1].plot(win_len_list, Fyy_gilney)
    axes[0, 1].plot(win_len_list, Fyy)
    axes[0, 1].text(.5, .9, 'Fyy', horizontalalignment='center', transform=axes[0, 1].transAxes)

    axes[1, 0].plot(win_len_list, Fxy2_gilney, lw=3)
    axes[1, 0].plot(win_len_list, Fxy2)
    axes[1, 0].text(.5, .9, 'Fxy2', horizontalalignment='center', transform=axes[1, 0].transAxes)
    axes[1, 0].axhline(ls="--", color='gray')

    axes[1, 1].semilogx(win_len_list, rho_gilney, lw=3)
    axes[1, 1].semilogx(win_len_list, rho)
    axes[1, 1].text(.5, .9, 'Rho', horizontalalignment='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axhline(ls="--", color='gray')
    axes[1, 1].set_ylim([-1, 1])

    plt.show()
