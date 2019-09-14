#!/usr/bin/env python3

import jmpy.utils
import sys

if __name__ == "__main__":
    nums = []
    for line in sys.stdin:
        line = line.strip()
        try:
            nums.append(float(line))
        except:
            print("skipping line '%s'"%(line[:5]), file=sys.stderr)

    plot = len(sys.argv) >= 2 and sys.argv[1] in ['-h', '--hist']

    stats = jmpy.utils.num_stats(nums, plot)
    jmpy.utils.print_num_stats(stats)

