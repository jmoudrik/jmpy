import io
import os
import pickle
import re
import sys
import time
from contextlib import redirect_stdout, contextmanager

from itertools import count
import numpy as np


def identity(x):
    return x

def filter_null(iterable):
    return [ x for x in iterable if x ]

def filter_both(predicate, iterable):
    yes, no = [], []
    for i in iterable:
        if predicate(i):
            yes.append(i)
        else:
            no.append(i)
    return yes, no

def flatten_twice(list_of_lists_of_lists):
    return flatten(flatten( list_of_lists_of_lists ))

def argmax(pairs):
    """Given an iterable of pairs (key, value), return the key corresponding to the greatest value."""
    return max(pairs, key=lambda x:x[1])[0]

def argmin(pairs):
    return min(pairs, key=lambda x:x[1])[0]

def argmax_index(values):
    """Given an iterable of values, return the index of the greatest value."""
    return argmax(zip(count(), values))

def argmin_index(values):
    return argmin(zip(count(), values))

def bucket_by_key(iterable, key_fc):
    """
    Throws items in @iterable into buckets given by @key_fc function.
    e.g.
    >>> bucket_by_key([1,2,-3,4,5,6,-7,8,-9], lambda num: 'neg' if num < 0 else 'nonneg')
    {'neg': [-3, -7, -9], 'nonneg': [1, 2, 4, 5, 6, 8]}
    """
    buckets = {}
    for item in iterable:
        buckets.setdefault(key_fc(item), []).append(item)
    return buckets

def first_true_pred(predicates, value):
    """Given a list of predicates and a value, return the index of first predicate,
    s.t. predicate(value) == True. If no such predicate found, raises IndexError."""
    for num, pred in enumerate(predicates):
        if pred(value):
            return num
    raise IndexError



def stderr(*args, **kwargs):
    kwargs['file'] = sys.stderr
    print(*args, **kwargs)


def cache_into(factory, filename):
    if os.path.exists(filename):
        stderr("loading from '%s'" % filename)
        with open(filename, 'rb') as fin:
            return pickle.load(fin)

    obj = factory()
    stderr("saving to '%s'" % filename)
    with open(filename, 'wb') as fout:
        pickle.dump(obj, fout)
    return obj


def length(iterator):
    cnt = 0
    for _ in iterator:
        cnt += 1
    return cnt


def simple_tokenize(txt):
    txt = re.sub(r"\W", ' ', txt)
    for s in txt.split(' '):
        if s:
            yield s


def k_grams(iterable, k):
    it = iter(iterable)
    keep = tuple(next(it) for i in range(k - 1))
    for e in it:
        this = keep + (e,)
        yield this
        keep = this[1:]


assert list(k_grams([1, 2, 3], 2)) == [(1, 2), (2, 3)]


def flatten(iterable):
    for it in iterable:
        for i in it:
            yield i


def uniq(iterable, count=False):
    def output(counter, element):
        if count:
            return counter, element
        return element

    it = iter(iterable)
    previous = None
    counter = 0
    just_started = True
    for element in it:
        if not just_started and element != previous:
            yield output(counter, previous)
            counter = 0

        counter += 1
        previous = element
        just_started = False

    if previous is not None:
        yield output(counter, element)


assert list(uniq([1, 1, 1, 2, 3, 3, 2, 2])) == [1, 2, 3, 2]
assert list(uniq([1, 1, 1, 2, 3, 3, 2, 2], count=True)) == [(3, 1), (1, 2), (2, 3), (2, 2)]
assert list(uniq([])) == []


def group_consequent(iterator, key=None):
    if key is None:
        key = lambda e: e
    prev_key = None
    current_group = []
    for row in iterator:
        current_key = key(row)
        if current_key != prev_key and prev_key is not None:
            yield current_group
            current_group = []

        current_group.append(row)
        prev_key = current_key

    if current_group:
        yield current_group


assert [len(g) for g in group_consequent([1, 1, 1, 2, 3, 3, 2, 2])] == [3, 1, 2, 2]


def nonempty_strip(iterable):
    for txt in iterable:
        txt = txt.strip()
        if txt:
            yield txt


def collapse_whitespace(txt):
    return re.sub(r'\s+', r' ', txt)


def polish(txt):
    return collapse_whitespace(txt).strip()


def crop_long(txt, length):
    txtp = polish(txt)[:length]
    return "%dB: %s" % (len(txt), txtp)


@contextmanager
def timer(name='', verbose=True):
    ts = []

    def next():
        ts.append(time.time())

    yield next
    next()

    diffs = []
    prev = ts[0]
    for i in range(1, len(ts)):
        diffs.append(ts[i] - prev)
        prev = ts[i]
    da = np.array(diffs)

    if verbose:
        stderr("Timer %s, %d iterations:" % (repr(name), len(diffs)))
        stderr("total\t%.3f s" % (da.sum()))
        if diffs:
            stderr("avg\t%.3f s" % (da.mean()))
            stderr()
            stderr("min\t%.3f s" % (da.min()))
            stderr("50%% <=\t%.3f s" % (np.median(da)))
            stderr("95%% <=\t%.3f s" % (np.percentile(da, 95)))
            stderr("max\t%.3f s" % (da.max()))


@contextmanager
def mod_stdout(transform, redirect_fn=redirect_stdout, print_fn=print):
    f = io.StringIO()
    with redirect_fn(f):
        yield
    out = f.getvalue()
    lines = out.split('\n')
    for num, line in enumerate(lines):
        if not (num == len(lines) - 1 and not line):
            print_fn(transform(line))


def prefix_stdout(prefix):
    return mod_stdout(lambda line: "%s%s"%(prefix, line))


if __name__ == "__main__":
    print("# test")
    with mod_stdout(lambda t: "\t%s" % t):
        print("- bullet")
        print("- bullet 2")
    print()

    with timer("Test") as start_iteration:
        for a in range(100):
            start_iteration()
            j = 0
            for i in range(10000):
                j += 10
