import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

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


def valid_output(file_name):
    if os.path.isdir(file_name):
        raise argparse.ArgumentTypeError("Invalid output %s" % file_name)
    if os.path.isfile(file_name):
        raise argparse.ArgumentTypeError("Output file %s already exists" % file_name)
    path = os.path.dirname(file_name) or '.'
    if not os.access(path, os.W_OK):
        raise argparse.ArgumentTypeError("Write privileges are not given on %s" % path)
    return file_name


def progress_bar(cnt, cnt_max, length=50):
    done = int(round(length * cnt / float(cnt_max)))
    done_percent = 100. * cnt / cnt_max
    sys.stdout.write('%s %.1f%%\r' % ('#' * done, done_percent))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+', type=valid_input, help="The path to the time series files")
    parser.add_argument("-o", "--output", type=valid_output, help="Output file to save the results")
    parser.add_argument("-b", "--boxes", nargs="*", type=int, help="A list of specific box sizes to be used in the "\
                                                                   "calculation")
    parser.add_argument("-m", "--max_num_boxes", type=int, default=100, help="Maximum number of boxes")
    parser.add_argument("-d", "--deg", type=int, default=1, help="The polynomial degree to detrend the signals")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-w", "--warnings", action="store_true", help="Output warning messages")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the output on the screen")
    args = parser.parse_args()

    if len(args.files) == 1:
        x, y = np.loadtxt(args.files[0], unpack=True)
    else:
        x = np.loadtxt(args.files[0])
        y = np.loadtxt(args.files[1])

    def verbose_print(a, b):
        if args.verbose:
            progress_bar(a, b)

    box_sizes_list, rho, Fxx, Fyy, Fxy2 = dcca(x, y,
                                               box_sizes_list=args.boxes,
                                               max_num_boxes=args.max_num_boxes,
                                               deg=args.deg,
                                               verbose_print=verbose_print,
                                               show_warnings=args.warnings)

    if args.output:
        data = np.stack((box_sizes_list, rho, Fxx, Fyy, Fxy2)).T
        np.savetxt(args.output, data, header="n rho Fxx Fyy Fxy2")

    if not args.output or args.plot:
        fig, axes = plt.subplots(nrows=2, sharex=True)
        axes[0].plot(box_sizes_list, Fxx, label="Fx(n)")
        axes[0].plot(box_sizes_list, Fyy, label="Fy(n)")
        axes[0].plot(box_sizes_list, Fxy2, label="Fxy2(n)")
        axes[0].set_xlabel("Box Size (n)")
        axes[0].text(.5, .9, 'DFA coefficients', horizontalalignment='center', transform=axes[0].transAxes)
        axes[0].legend(loc='upper left')
        axes[1].semilogx(box_sizes_list, rho)
        axes[1].set_xlabel("Box Size (n)")
        axes[1].text(.5, .9, 'rho_dcca (n)', horizontalalignment='center', transform=axes[1].transAxes)
        axes[1].axhline(ls="--", color='gray')
        axes[1].set_ylim([-1, 1])
        plt.show()

    print("\nComplete.")


if __name__ == "__main__":
    rc = main()
    exit(rc)
