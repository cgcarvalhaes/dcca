import argparse
import os
import sys

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
    path = os.path.dirname(file_name) or '.'
    if not os.access(path, os.W_OK):
        raise argparse.ArgumentTypeError("Write privileges are not given on %s" % path)
    return file_name


def get_yes_no_response(question):
    while True:
        try:
            response = input(question).lower()
        except KeyboardInterrupt:
            return False
        if response in ('yes', 'y', 'no', 'n'):
            return 'yes' if response[0] == 'y' else 'no'

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+', type=valid_input, help="The path to the time series files")
    parser.add_argument("-o", "--output", required=True, type=valid_output, help="Output file to save the results")
    parser.add_argument("-b", "--boxes", nargs="*", type=int, help="A list of specific box sizes to be used in the "\
                                                                   "calculation")
    parser.add_argument("-m", "--max_num_boxes", type=int, default=100, help="Maximum number of boxes")
    parser.add_argument("-d", "--deg", type=int, default=1, help="The polynomial degree to detrend the signals")
    parser.add_argument("-w", "--warnings", action="store_true", help="Output warning messages")
    args = parser.parse_args()

    if args.output and os.path.isfile(args.output):
        resp = get_yes_no_response("Output file '%s' already exists. Replace it [Yes/No]?" % args.output)
        if resp == 'no':
            exit(0)
    
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
                                               show_warnings=args.warnings)

    if args.output:
        data = np.stack((box_sizes_list, rho, Fxx, Fyy, Fxy2)).T
        np.savetxt(args.output, data, header="n rho Fxx Fyy Fxy2")

    print("\nComplete.")
