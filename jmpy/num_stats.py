#!/usr/bin/env python3

import jmpy.utils
import sys

if __name__ == "__main__":
    numbers = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            numbers.append(float(line))
        except:
            print("skipping line '%s'"%(line[:5]), file=sys.stderr)

    def has_arg(l):
        return len(sys.argv) >= 2 and any(a in l for a in sys.argv[1:])

    plot = has_arg(['-h', '--hist'])

    jmpy.utils.full_stats(numbers, plot, bins='doane')

