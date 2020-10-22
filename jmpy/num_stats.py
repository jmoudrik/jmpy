#!/usr/bin/env python3

import sys
import argparse
from itertools import chain

import jmpy.utils


def iter_lines(filename):
    with open(filename, 'r') as fin:
        return fin.readlines()


def iter_input(filename, silent):
    if filename == '-':
        filename = "STDIN"
        lines = sys.stdin
    else:
        lines = iter_lines(filename)
    for num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            yield float(line)
        except ValueError:
            if not silent:
                print("file '%s': skipping line %d: '%s%s'" % (
                    filename, num, line[:20], '...' if len(line) >= 23 else line[20:]), file=sys.stderr)


def iter_inputs(files, silent):
    return chain.from_iterable(iter_input(file, silent) for file in files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""
    Print statistics of numbers. Numbers are read from STDIN.
    This program stores all numbers in memory before the computation
    and is thus NOT suitable for large number of inputs.
    """)

    parser.add_argument("-s", "--sum-histogram", help="Plot histogram of sums of values in bins.", action="store_true")
    parser.add_argument("-c", "--count-histogram", help="Plot normal histogram of values.", action="store_true")
    parser.add_argument("--bins", type=str,
                        help="Specify numpy histogram bin algorithm to use (e.g. sturges, doane, fd).",
                        default='doane')
    parser.add_argument("-w", "--histogram-width", type=int, help="""
    Specify max width of the histogram plot.
    Just the '*', excluding labels on the left.
    """, default=50)
    parser.add_argument("--silent", help="""
    If set, program silently skips any lines that are not a number.
    """, action='store_true')
    parser.add_argument('FILENAME', nargs='*', help="""
    Files to read the numbers from. Assumes one-line per number ('\\n' separated).
    Not listing any files, or including FILENAME of '-' causes the program to read inputs from STDIN.
    """, default=[])
    args = parser.parse_args()

    files = args.FILENAME
    if not files:
        files = ['-']

    numbers = list(iter_inputs(files, args.silent))
    jmpy.utils.full_stats(numbers, sum_hist=args.sum_histogram, count_hist=args.count_histogram, bins=args.bins,
                          max_bin_width=args.histogram_width)
