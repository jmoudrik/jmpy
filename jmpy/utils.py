import re as _re
import collections as _collections
import contextlib as _contextlib
import itertools as _itertools


def identity(x):
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
    import sys

    kwargs['file'] = sys.stderr
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
    cnt = 0
    for _ in iterator:
        cnt += 1
    return cnt


def simple_tokenize(txt, word_rexp=r"\W"):
    txt = _re.sub(word_rexp, ' ', txt)
    for s in txt.split(' '):
        if s:
            yield s


def k_grams(iterable, k):
    """Returns iterator of k-grams of elements from `iterable`.
    >>> list(k_grams(range(4), 2))
    [(0, 1), (1, 2), (2, 3)]
    >>> list(k_grams((), 2))
    []
    """
    it = iter(iterable)
    keep = tuple(next(it) for i in range(k - 1))
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
    import time, numpy

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
        da = numpy.array(diffs)

        if verbose:
            stderr("Timer %s, %d iterations:" % (repr(name), len(diffs)))
            stderr("total\t%.3f s" % (da.sum()))
            if diffs:
                stderr("avg\t%.3f s" % (da.mean()))
                stderr()
                stderr("min\t%.3f s" % (da.min()))
                stderr("50%% <=\t%.3f s" % (numpy.median(da)))
                stderr("95%% <=\t%.3f s" % (numpy.percentile(da, 95)))
                stderr("max\t%.3f s" % (da.max()))


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
