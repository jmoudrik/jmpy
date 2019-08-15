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


    stats = jmpy.utils.num_stats(nums)
    for k,(v, fmt) in stats.items():
        print("\t%s\t%s"%(k, fmt%v))
