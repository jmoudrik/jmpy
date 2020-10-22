# jmpy
jm's Python utils

## Instalation

```bash
sudo python3 setup.py install
```

## `num_stats` - detailed statistics in Your commandline

Apart from the utility functions below, this contains a cmd-line program `num_stats`
that prints statistics of numbers read from STDIN (or other files) to STDOUT.

Run `num_stats --help` for help.

For example:
```sh
# generate 300 random numbers first
$ python -c 'import numpy;print("\n".join(map(str,numpy.random.normal(size=300))))' > numbers
```
Then:
```
# pipe them into the num_stats program
$ cat numbers | num_stats -c
count  = 300	
sum    = 14.511	
mean   = 0.048 ± 1.036	
median = 0.060	
min = -3.473     1% = -2.260    5% = -1.610    25% = -0.652	
max = 2.878      99% = 2.322    95% = 1.628    75% = 0.796	

 <from,    to)  #      statistics of bin count
-----------------------------------------------
-3.473, -2.838  1       0%  0% .
-2.838, -2.203  4   1   2%  1% **.
-2.203, -1.568 15   5   7%  5% *********.
-1.568, -0.932 35   -  18% 12% ***********************.
-0.932, -0.297 58  25  38% 19% **************************************.
-0.297,  0.338 76  mM  63% 25% **************************************************
 0.338,  0.973 52  75  80% 17% **********************************.
 0.973,  1.608 42   -  94% 14% ***************************.
 1.608,  2.243 10  95  98%  3% ******.
 2.243,  2.878  7  99 100%  2% ****.
```

# Table of Contents

* [jmpy/utils](#jmpy/utils)
  * [identity](#jmpy/utils.identity)
  * [filter\_null](#jmpy/utils.filter_null)
  * [filter\_both](#jmpy/utils.filter_both)
  * [flatten](#jmpy/utils.flatten)
  * [argmax](#jmpy/utils.argmax)
  * [argmin](#jmpy/utils.argmin)
  * [argmax\_index](#jmpy/utils.argmax_index)
  * [argmin\_index](#jmpy/utils.argmin_index)
  * [bucket\_by\_key](#jmpy/utils.bucket_by_key)
  * [first\_true\_pred](#jmpy/utils.first_true_pred)
  * [cache\_into](#jmpy/utils.cache_into)
  * [consuming\_length](#jmpy/utils.consuming_length)
  * [simple\_tokenize](#jmpy/utils.simple_tokenize)
  * [k\_grams](#jmpy/utils.k_grams)
  * [uniq](#jmpy/utils.uniq)
  * [group\_consequent](#jmpy/utils.group_consequent)
  * [nonempty\_strip](#jmpy/utils.nonempty_strip)
  * [collapse\_whitespace](#jmpy/utils.collapse_whitespace)
  * [num\_stats](#jmpy/utils.num_stats)
  * [full\_stats](#jmpy/utils.full_stats)
  * [print\_num\_stats](#jmpy/utils.print_num_stats)
  * [mod\_stdout](#jmpy/utils.mod_stdout)
  * [prefix\_stdout](#jmpy/utils.prefix_stdout)

<a name="jmpy/utils"></a>
# jmpy/utils

<a name="jmpy/utils.identity"></a>
#### identity

```python
identity(x)
```

Return itself

```python
>>> identity(1)
1
```

<a name="jmpy/utils.filter_null"></a>
#### filter\_null

```python
filter_null(iterable)
```

Filter out elements that do not evaluate to True

```python
>>> list(filter_null((0, None, 1, '', 'cherry')))
[1, 'cherry']
```

<a name="jmpy/utils.filter_both"></a>
#### filter\_both

```python
filter_both(predicate, iterable)
```

Splits the iterable into two groups, based on the result of
calling `predicate` on each element.

WARN: Consumes the whole iterable in the process. This is the
price for calling the `predicate` function only once for each
element. (See itertools recipes for similar functionality without
this requirement.)

```python
>>> filter_both(lambda x: x%2 == 0, range(4))
([0, 2], [1, 3])
```

<a name="jmpy/utils.flatten"></a>
#### flatten

```python
flatten(iterables)
```

```python
>>> list(flatten(((1, 2, 3), (4, 5, 6))))
[1, 2, 3, 4, 5, 6]
```

<a name="jmpy/utils.argmax"></a>
#### argmax

```python
argmax(pairs)
```

Given an iterable of pairs (key, value), return the key corresponding to the greatest value.
Raises `ValueError` on empty sequence.

```python
>>> argmax(zip(range(20), range(20, 0, -1)))
0
```

<a name="jmpy/utils.argmin"></a>
#### argmin

```python
argmin(pairs)
```

Given an iterable of pairs (key, value), return the key corresponding to the smallest value.
Raises `ValueError` on empty sequence.

```python
>>> argmin(zip(range(20), range(20, 0, -1)))
19
```

<a name="jmpy/utils.argmax_index"></a>
#### argmax\_index

```python
argmax_index(values)
```

Given an iterable of values, return the index of the (first) greatest value.
Raises `ValueError` on empty sequence.

```python
>>> argmax_index([0, 4, 3, 2, 1, 4, 0])
1
```

<a name="jmpy/utils.argmin_index"></a>
#### argmin\_index

```python
argmin_index(values)
```

Given an iterable of values, return the index of the (first) smallest value.
Raises `ValueError` on empty sequence.

```python
>>> argmin_index([10, 4, 0, 2, 1, 0])
2
```

<a name="jmpy/utils.bucket_by_key"></a>
#### bucket\_by\_key

```python
bucket_by_key(iterable, key_fc)
```

Throws items in @iterable into buckets given by @key_fc function.
e.g.

```python
>>> bucket_by_key([1, 2, -3, 4, 5, 6, -7, 8, -9], lambda num: 'neg' if num < 0 else 'nonneg')
OrderedDict([('nonneg', [1, 2, 4, 5, 6, 8]), ('neg', [-3, -7, -9])])
```

<a name="jmpy/utils.first_true_pred"></a>
#### first\_true\_pred

```python
first_true_pred(predicates, value)
```

Given a list of predicates and a value, return the index of first predicate,
s.t. predicate(value) == True.
If no such predicate found, raises IndexError.

```python
>>> first_true_pred([lambda x: x%2==0, lambda x: x%2==1], 13)
1
```

<a name="jmpy/utils.cache_into"></a>
#### cache\_into

```python
cache_into(factory, filename)
```

Simple pickle caching. Calls `factory`, stores result to `filename` pickle.
Subsequent calls load the obj from the pickle instead of running the `factory` again.

<a name="jmpy/utils.consuming_length"></a>
#### consuming\_length

```python
consuming_length(iterator)
```

Return length of an iterator, consuming its contents. O(1) memory.

```python
>>> consuming_length(range(10))
10
```

<a name="jmpy/utils.simple_tokenize"></a>
#### simple\_tokenize

```python
simple_tokenize(txt, sep_rexp=r"\W")
```

Iterates through tokens, kwarg `sep_rexp` specifies the whitespace.
O(N) memory.

```python
>>> list(simple_tokenize('23_45 hello, how are  you?'))
['23_45', 'hello', 'how', 'are', 'you']
```

<a name="jmpy/utils.k_grams"></a>
#### k\_grams

```python
k_grams(iterable, k)
```

Returns iterator of k-grams of elements from `iterable`.

```python
>>> list(k_grams(range(4), 2))
[(0, 1), (1, 2), (2, 3)]
>>> list(k_grams((), 2))
[]
>>> list(k_grams((1,), 2))
[]
```

<a name="jmpy/utils.uniq"></a>
#### uniq

```python
uniq(iterable, count=False)
```

Similar to unix `uniq`. Returns counts as well if `count` arg is True.
Has O(1) memory footprint.

```python
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
```

<a name="jmpy/utils.group_consequent"></a>
#### group\_consequent

```python
group_consequent(iterator, key=None)
```

Groups consequent elements from an iterable and returns them
as a sequence.

Has O(maximal groupsize) memory footprint.

```python
>>> list(group_consequent([0, 2, 1, 3, 2, 1], key=lambda x:x%2))
[[0, 2], [1, 3], [2], [1]]
>>> list(group_consequent([None, None]))
[[None, None]]
>>> [len(g) for g in group_consequent([1, 1, 1, 2, 3, 3, 2, 2])]
[3, 1, 2, 2]
```

<a name="jmpy/utils.nonempty_strip"></a>
#### nonempty\_strip

```python
nonempty_strip(iterable)
```

```python
>>> list(nonempty_strip(['little ', '    ', '\tpiggy\\n']))
['little', 'piggy']
```

<a name="jmpy/utils.collapse_whitespace"></a>
#### collapse\_whitespace

```python
collapse_whitespace(txt)
```

```python
>>> collapse_whitespace("bla   bla")
'bla bla'
```

<a name="jmpy/utils.num_stats"></a>
#### num\_stats

```python
num_stats(numbers, print=False, print_formats=None)
```

Computes stats of the `numbers`, returns an OrderedDict with value and suggested print format

```python
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
```

<a name="jmpy/utils.full_stats"></a>
#### full\_stats

```python
full_stats(numbers, count_hist=True, sum_hist=False, bins='sturges', **kwargs)
```

Prints statistics of a list of `numbers` to console.

**Arguments**:

- `count_hist`: prints histogram.
- `sum_hist`: prints histogram, but of SUMS of the values in the bins.
- `bins`: numpy bins arguments

```python
>>> import numpy as np
>>> np.random.seed(666)
>>> first_peak = np.random.normal(size=100)
>>> second_peak = np.random.normal(loc=4,size=100)
>>> numbers = np.concatenate([first_peak, second_peak])
>>> full_stats(numbers)
count  = 200
sum    = 403.403
mean   = 2.017 ± 2.261
median = 1.874
min = -3.095	 1% = -1.870	 5% = -0.990	25% = -0.045
max = 7.217	99% = 6.063	95% = 5.404	75% = 4.089
<BLANKLINE>
<from,    to)  #       statistics of bin count
------------------------------------------------
-3.095, -1.949  1        0%  0% *
-1.949, -0.803 18   5.  10%  9% ******************
-0.803,  0.343 48   -.  34% 24% ************************************************
0.343,  1.488 28       48% 14% ****************************
1.488,  2.634 12   mM  54%  6% ************
2.634,  3.780 34       70% 17% **********************************
3.780,  4.926 39   -.  90% 20% ***************************************
4.926,  6.071 18  95.  99%  9% ******************
6.071,  7.217  2      100%  1% **
```

<a name="jmpy/utils.print_num_stats"></a>
#### print\_num\_stats

```python
print_num_stats(stats, units=None, formats=None, file=None)
```

Utility function to print results of `num_stats` function.

```python
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
```

<a name="jmpy/utils.mod_stdout"></a>
#### mod\_stdout

```python
@_contextlib.contextmanager
mod_stdout(transform, redirect_fn=_contextlib.redirect_stdout, print_fn=print)
```

A context manager that modifies every line printed to stdout.

```python
>>> with mod_stdout(lambda line: line.upper()):
...     print("this will be upper")
THIS WILL BE UPPER
```

<a name="jmpy/utils.prefix_stdout"></a>
#### prefix\_stdout

```python
prefix_stdout(prefix)
```

A context manager that prefixes every line printed to stout by `prefix`.

```python
>>> with prefix_stdout(" * "):
...     print("bullet")
 * bullet
```

