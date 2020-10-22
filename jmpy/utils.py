import collections as _collections
import contextlib as _contextlib
import itertools as _itertools
import math as _math
import re as _re
import sys as _sys


def identity(x):
    """Return itself
    >>> identity(1)
    1
    """
    return x


def filter_null(iterable):
    """Filter out elements that do not evaluate to True
    >>> list(filter_null((0, None, 1, '', 'cherry')))
    [1, 'cherry']
    """
    return filter(identity, iterable)


def filter_both(predicate, iterable):
    """ Splits the iterable into two groups, based on the result of
    calling `predicate` on each element.

    WARN: Consumes the whole iterable in the process. This is the
    price for calling the `predicate` function only once for each
    element. (See itertools recipes for similar functionality without
    this requirement.)
    >>> filter_both(lambda x: x%2 == 0, range(4))
    ([0, 2], [1, 3])
    """
    yes, no = [], []
    for i in iterable:
        if predicate(i):
            yes.append(i)
        else:
            no.append(i)
    return yes, no


def flatten(iterables):
    """
    >>> list(flatten(((1, 2, 3), (4, 5, 6))))
    [1, 2, 3, 4, 5, 6]
    """
    for iterable in iterables:
        for element in iterable:
            yield element


def argmax(pairs):
    """Given an iterable of pairs (key, value), return the key corresponding to the greatest value.
    Raises `ValueError` on empty sequence.
    >>> argmax(zip(range(20), range(20, 0, -1)))
    0
    """
    return max(pairs, key=lambda x: x[1])[0]


def argmin(pairs):
    """Given an iterable of pairs (key, value), return the key corresponding to the smallest value.
    Raises `ValueError` on empty sequence.
    >>> argmin(zip(range(20), range(20, 0, -1)))
    19
    """
    return min(pairs, key=lambda x: x[1])[0]


def argmax_index(values):
    """Given an iterable of values, return the index of the (first) greatest value.
    Raises `ValueError` on empty sequence.
    >>> argmax_index([0, 4, 3, 2, 1, 4, 0])
    1
    """
    return argmax(zip(_itertools.count(), values))


def argmin_index(values):
    """Given an iterable of values, return the index of the (first) smallest value.
    Raises `ValueError` on empty sequence.
    >>> argmin_index([10, 4, 0, 2, 1, 0])
    2
    """
    return argmin(zip(_itertools.count(), values))


def bucket_by_key(iterable, key_fc):
    """
    Throws items in @iterable into buckets given by @key_fc function.
    e.g.
    >>> bucket_by_key([1, 2, -3, 4, 5, 6, -7, 8, -9], lambda num: 'neg' if num < 0 else 'nonneg')
    OrderedDict([('nonneg', [1, 2, 4, 5, 6, 8]), ('neg', [-3, -7, -9])])
    """
    buckets = _collections.OrderedDict()
    for item in iterable:
        buckets.setdefault(key_fc(item), []).append(item)
    return buckets


def first_true_pred(predicates, value):
    """Given a list of predicates and a value, return the index of first predicate,
    s.t. predicate(value) == True.
    If no such predicate found, raises IndexError.

    >>> first_true_pred([lambda x: x%2==0, lambda x: x%2==1], 13)
    1
    """
    for num, pred in enumerate(predicates):
        if pred(value):
            return num
    raise IndexError


def stderr(*args, **kwargs):
    kwargs['file'] = _sys.stderr
    print(*args, **kwargs)


def cache_into(factory, filename):
    """Simple pickle caching. Calls `factory`, stores result to `filename` pickle.
    Subsequent calls load the obj from the pickle instead of running the `factory` again."""
    import os
    import pickle

    if os.path.exists(filename):
        stderr("loading from '%s'" % filename)
        with open(filename, 'rb') as fin:
            return pickle.load(fin)

    obj = factory()
    stderr("saving to '%s'" % filename)
    with open(filename, 'wb') as fout:
        pickle.dump(obj, fout)
    return obj


def consuming_length(iterator):
    """Return length of an iterator, consuming its contents. O(1) memory.
    >>> consuming_length(range(10))
    10
    """
    cnt = 0
    for _ in iterator:
        cnt += 1
    return cnt


def simple_tokenize(txt, sep_rexp=r"\W"):
    """Iterates through tokens, kwarg `sep_rexp` specifies the whitespace.
    O(N) memory.
    >>> list(simple_tokenize('23_45 hello, how are  you?'))
    ['23_45', 'hello', 'how', 'are', 'you']
    """
    txt = _re.sub(sep_rexp, ' ', txt)
    for s in txt.split(' '):
        if s:
            yield s


def k_grams(iterable, k):
    """Returns iterator of k-grams of elements from `iterable`.
    >>> list(k_grams(range(4), 2))
    [(0, 1), (1, 2), (2, 3)]
    >>> list(k_grams((), 2))
    []
    >>> list(k_grams((1,), 2))
    []
    """
    it = iter(iterable)
    keep = tuple(_itertools.islice(it, k-1))
    # if we do not even have the starting k-1 elements, exit
    if len(keep) < k - 1:
        return

    # every remaining element will yield a k-gram
    for e in it:
        this = keep + (e,)
        yield this
        keep = this[1:]


def uniq(iterable, count=False):
    """
    Similar to unix `uniq`. Returns counts as well if `count` arg is True.
    Has O(1) memory footprint.

    >>> list(uniq([1, 1, 1, 2, 3, 3, 2, 2]))
    [1, 2, 3, 2]
    >>> list(uniq([1, 1, 1, 2, 3, 3, 2, 2], count=True))
    [(3, 1), (1, 2), (2, 3), (2, 2)]
    >>> list(uniq([1, None]))
    [1, None]
    >>> list(uniq([None]))
    [None]
    >>> list(uniq([]))
    []
    """

    def output(counter, element):
        if count:
            return counter, element
        return element

    it = iter(iterable)
    previous = None
    counter = 0
    first_run = True
    for element in it:
        if not first_run and element != previous:
            yield output(counter, previous)
            counter = 0

        counter += 1
        previous = element
        first_run = False

    if not first_run:
        yield output(counter, element)


def group_consequent(iterator, key=None):
    """
    Groups consequent elements from an iterable and returns them
    as a sequence.

    Has O(maximal groupsize) memory footprint.

    >>> list(group_consequent([0, 2, 1, 3, 2, 1], key=lambda x:x%2))
    [[0, 2], [1, 3], [2], [1]]
    >>> list(group_consequent([None, None]))
    [[None, None]]
    >>> [len(g) for g in group_consequent([1, 1, 1, 2, 3, 3, 2, 2])]
    [3, 1, 2, 2]
    """
    if key is None:
        key = lambda e: e

    prev_key = None
    first_run = True
    current_group = []
    for row in iterator:
        current_key = key(row)
        if not first_run and current_key != prev_key:
            yield current_group
            current_group = []

        current_group.append(row)
        first_run = False
        prev_key = current_key

    if current_group:
        yield current_group


def nonempty_strip(iterable):
    """
    >>> list(nonempty_strip(['little ', '    ', '\tpiggy\\n']))
    ['little', 'piggy']
    """
    for txt in iterable:
        txt = txt.strip()
        if txt:
            yield txt


def collapse_whitespace(txt):
    """
    >>> collapse_whitespace("bla   bla")
    'bla bla'
    """
    return _re.sub(r'\s+', r' ', txt)


@_contextlib.contextmanager
def timer(name='', verbose=True):
    import time

    ts = []

    def next():
        ts.append(time.time())

    try:
        yield next
    finally:
        next()

        diffs = []
        prev = ts[0]
        for i in range(1, len(ts)):
            diffs.append(ts[i] - prev)
            prev = ts[i]

        if verbose:
            stderr("Timer %s" % (repr(name)))
            stats = num_stats(diffs)
            units = {k: " s" if k != 'count' else ' iterations' for k in stats.keys()}
            print_num_stats(stats, units=units, file=_sys.stderr)


def num_stats(numbers, print=False, print_formats=None):
    """Computes stats of the `numbers`, returns an OrderedDict with value and suggested print format
    >>> num_stats(range(10))
    OrderedDict([('count', 10), ('sum', 45), ('mean', 4.5), ('sd', 2.8722813232690143), ('min', 0), ('1%', 0.09), ('5%', 0.45), ('25%', 2.25), ('50%', 4.5), ('75%', 6.75), ('95%', 8.549999999999999), ('99%', 8.91), ('max', 9)])
    >>> print_num_stats(num_stats(range(10)))
    count 10
    sum 45.000
    mean 4.500
    sd 2.872
    min 0.000
    1% 0.090
    5% 0.450
    25% 2.250
    50% 4.500
    75% 6.750
    95% 8.550
    99% 8.910
    max 9.000
    """
    import numpy

    def fl(num):
        return num

    nums = numpy.array(numbers)
    ret = _collections.OrderedDict()

    ret['count'] = len(numbers)

    if len(numbers):
        ret.update([
            ("sum", fl(nums.sum())),
            ("mean", fl(nums.mean())),
            ("sd", fl(numpy.std(nums))),
            ("min", fl(nums.min())),
            ("1%", fl(numpy.percentile(nums, 1))),
            ("5%", fl(numpy.percentile(nums, 5))),
            ("25%", fl(numpy.percentile(nums, 25))),
            ("50%", fl(numpy.median(nums))),
            ("75%", fl(numpy.percentile(nums, 75))),
            ("95%", fl(numpy.percentile(nums, 95))),
            ("99%", fl(numpy.percentile(nums, 99))),
            ("max", fl(nums.max()))])

        if print:
            print_num_stats(ret, formats=print_formats)
    return ret


def draw_console_histogram(counts, bins, nums, stats={}, max_bin_width=50, sum_hist=False, count_hist=True):
    import numpy
    # FIXME
    # problem: for small numbers, the %.2f format could hide
    # information
    # fix: the format for the bins should be variable based on
    # some information measure or something

    assert nums.min() == bins[0] and nums.max() == bins[-1]

    if sum_hist and (nums < 0.0).any():
        stderr("WARN: num_sums only makes sense for positive numbers.")
        stderr("WARN:   plotting counts instead")
        sum_hist = False
        count_hist = True

    histogram_to_print = []
    if count_hist:
        histogram_to_print.append(False)
    if sum_hist:
        histogram_to_print.append(True)

    def norm_to_width(counts):
        max_count = counts.max()
        if max_count > max_bin_width:
            norm = max_count / max_bin_width
            return counts / norm, max_count
        return counts, max_count

    def fmt_f(f):
        return "%.3f" % f

    def pad_right(c, size):
        return c + " " * max(0, size - len(c))

    def pad_left(c, size):
        return " " * max(0, size - len(c)) + c

    def reduce_info(stuff):
        return ''.join(stuff)

    def first_or(a, b, add=True):
        if a:
            return a + (("." if (b and a[-1] != '.') else '') if add else '')
        return b

    def stat(flag, key, left, right):
        if key not in stats:
            return ''
        val = stats[key]
        if not (left <= val <= right):
            return ''
        return flag

    col_pads = [pad_right, pad_left]

    def mkrow(s):
        return ["%s = %.3f" % (pad_left(key, 3), stats[key]) for key in s.split()]

    def print_row(l, pd=10):
        if not l:
            return

        for element in l:
            print(pad_right(element, (0 if len(l) == 1 else pd)), end='\t')
        print()

    size = counts.sum()
    print_row(["count  = %d" % size])
    if size == 0:
        return
    if stats:
        print_row(["sum  = %.3f" % stats['sum']])
        print_row(["mean   = %.3f ± %.3f" % (stats['mean'], stats['sd'])])
        print_row(["median = %.3f" % (stats['50%'])])
        print_row(mkrow("min 1% 5% 25%"))
        print_row(mkrow("max 99% 95% 75%"))

        if 'mean' in stats and 'sd' in stats:
            mean = stats['mean']
            sd = stats['sd']
            stats['sd_low'] = mean - sd
            stats['sd_hi'] = mean + sd

    stars, max_count = norm_to_width(counts)

    bin_sums = numpy.zeros(len(counts))
    for i in range(len(counts)):
        left, right = bins[i], bins[i + 1]

        # https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
        # the last histogram boundary is not half-open
        select_right = nums < right
        if i == len(counts) - 1:
            select_right = nums <= right

        bin_select = (nums >= left) * select_right
        bin_sums[i] = (nums * bin_select).sum()
    stars_sum, max_count_sum = norm_to_width(bin_sums)

    # num digits for count
    digits = len(str(max_count))
    # num digits for max size of bin (interval on the left)
    digits_bin = max(len(fmt_f(b)) for b in bins)

    infos = []
    for i in range(len(counts)):
        left, right = bins[i], bins[i + 1]
        info = first_or(''.join((stat('m', 'mean', left, right),
                                 stat('M', '50%', left, right))).strip(),
                        first_or(
                            first_or(first_or(stat('-', 'sd_low', left, right),
                                              stat('25', '25%', left, right)),
                                     first_or(stat('5', '5%', left, right),
                                              stat('1', '1%', left, right)),
                                     add=True),
                            first_or(first_or(stat('-', 'sd_hi', left, right),
                                              stat('75', '75%', left, right)),
                                     first_or(stat('95', '95%', left, right),
                                              stat('99', '99%', left, right)),
                                     add=True),
                            add=False)
                        )
        infos.append(info)

    info_len = max(map(len, infos))
    sum_norm = max(1, nums.sum())
    count_norm = max(1, counts.sum())

    for print_num_sum in histogram_to_print:
        print()

        count_cumsum = 0.0
        cumsum = 0.0
        legend = "%s, %s %s %s  %s" % (
            pad_left("<from", digits_bin),
            pad_left("to)", digits_bin),
            pad_left("#", digits),
            pad_left("", 1 + info_len),
            "statistic of bin SUM" if print_num_sum else "statistics of bin count")
        print(legend)
        print("-" * (len(legend) + int(legend.startswith(" "))))

        for i in range(len(counts)):
            left, right = bins[i], bins[i + 1]
            count_cumsum += counts[i]
            bin_sum = bin_sums[i]
            cumsum += bin_sum
            stars_float = stars[i]
            if print_num_sum:
                stars_float = stars_sum[i]

            stars_here = _math.floor(stars_float)
            # <stars, stars+1)
            extra = _math.ceil(stars_float) - stars_here
            assert extra <= 1
            row = "*" * stars_here + ('.' if extra > 0 else '')

            suff = ''
            is_last_one = i == len(counts) - 1
            round = lambda x: x  # _math.floor if not is_last_one else _math.ceil
            d_cum = round(100 * count_cumsum / count_norm)
            d_bin = 100 * counts[i] / count_norm
            if print_num_sum:
                d_cum = round(100 * cumsum / sum_norm)
                d_bin = 100 * bin_sum / sum_norm

            # just for pretty print
            # this can be lower because of double rounding
            if is_last_one:
                pass
                # d_cum = 100

            print("%s, %s %s %s %s %s %s" % (
                pad_left(fmt_f(left), digits_bin),
                pad_left(fmt_f(right), digits_bin),
                pad_left(str(counts[i]), digits),
                pad_left(infos[i], 1 + info_len),
                pad_left("%.0f%%" % (d_cum), 4),
                pad_left("%.0f%%" % (d_bin), 3),
                row))


def full_stats(numbers, bins='sturges', count_hist=True, **kwargs):
    import numpy
    nums = numpy.array(numbers)

    stats = num_stats(nums)

    counts, bins = numpy.histogram(nums, bins=bins)
    draw_console_histogram(counts, bins, nums, stats=stats, count_hist=count_hist, **kwargs)


def print_num_stats(stats, units=None, formats=None, file=None):
    """Utility function to print results of `num_stats` function.
    >>> print_num_stats(num_stats(range(10)), units={'count':'iterations'}, formats={'sum':'%.5f'})
    count 10 iterations
    sum 45.00000
    mean 4.500
    sd 2.872
    min 0.000
    1% 0.090
    5% 0.450
    25% 2.250
    50% 4.500
    75% 6.750
    95% 8.550
    99% 8.910
    max 9.000
    >>> print_num_stats(num_stats(range(10)), formats={'sum':'', '1%':'', '5%':''})
    count 10
    mean 4.500
    sd 2.872
    min 0.000
    25% 2.250
    50% 4.500
    75% 6.750
    95% 8.550
    99% 8.910
    max 9.000
    """

    def get_from(d, key, default):
        if d is None:
            return default
        return d.get(key, default)

    for key, value in stats.items():
        fmt = "%.3f"
        if isinstance(value, int):
            fmt = "%d"
        fmt = get_from(formats, key, fmt)
        if fmt == '':
            continue

        unit = get_from(units, key, '')
        if unit:
            unit = ' ' + unit

        pstr = "%s %s%s" % (key, fmt % value, unit)
        if file is None:
            print(pstr)
        else:
            print(pstr, file=file)


@_contextlib.contextmanager
def mod_stdout(transform, redirect_fn=_contextlib.redirect_stdout, print_fn=print):
    """A context manager that modifies every line printed to stdout.
    >>> with mod_stdout(lambda line: line.upper()):
    ...     print("this will be upper")
    THIS WILL BE UPPER
    """
    import io

    f = io.StringIO()
    with redirect_fn(f):
        yield
    out = f.getvalue()
    lines = out.split('\n')
    for num, line in enumerate(lines):
        if not (num == len(lines) - 1 and not line):
            print_fn(transform(line))


def prefix_stdout(prefix):
    """A context manager that prefixes every line printed to stout by `prefix`.
    >>> with prefix_stdout(" * "):
    ...     print("bullet")
     * bullet
    """
    return mod_stdout(lambda line: "%s%s" % (prefix, line))


if __name__ == "__main__":
    print("# test")
    with prefix_stdout("\t* "):
        print("bullet 1")
        print("bullet 2")
    print()

    with timer("Test") as start_iteration:
        for a in range(100):
            start_iteration()
            j = 0
            for i in range(10000):
                j += 10
            if a == 20:
                raise RuntimeError("ble")
