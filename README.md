# jmpy
jm's Python utils

## Instalation

```bash
sudo python3 setup.py install
```

## API

- [`argmax(pairs)`](#argmax(pairs))
- [`argmax_index(values)`](#argmax_index(values))
- [`argmin(pairs)`](#argmin(pairs))
- [`argmin_index(values)`](#argmin_index(values))
- [`bucket_by_key(iterable, key_fc)`](#bucket_by_key(iterable,-key_fc))
- [`cache_into(factory, filename)`](#cache_into(factory,-filename))
- [`collapse_whitespace(txt)`](#collapse_whitespace(txt))
- [`consuming_length(iterator)`](#consuming_length(iterator))
- [`filter_both(predicate, iterable)`](#filter_both(predicate,-iterable))
- [`filter_null(iterable)`](#filter_null(iterable))
- [`first_true_pred(predicates, value)`](#first_true_pred(predicates,-value))
- [`flatten(iterables)`](#flatten(iterables))
- [`group_consequent(iterator, key=None)`](#group_consequent(iterator,-key=none))
- [`identity(x)`](#identity(x))
- [`k_grams(iterable, k)`](#k_grams(iterable,-k))
- [`mod_stdout(transform, redirect_fn=<class 'contextlib.redirect_stdout'>, print_fn=<built-in function print>)`](#mod_stdout(transform,-redirect_fn=<class-'contextlib.redirect_stdout'>,-print_fn=<built-in-function-print>))
- [`nonempty_strip(iterable)`](#nonempty_strip(iterable))
- [`num_stats(numbers, plot=False)`](#num_stats(numbers,-plot=false))
- [`prefix_stdout(prefix)`](#prefix_stdout(prefix))
- [`simple_tokenize(txt, sep_rexp='\\W')`](#simple_tokenize(txt,-sep_rexp='\\w'))
- [`stderr(*args, **kwargs)`](#stderr(*args,-**kwargs))
- [`timer(name='', verbose=True)`](#timer(name='',-verbose=true))
- [`uniq(iterable, count=False)`](#uniq(iterable,-count=false))
- [`utils`](#utils)


### `argmax(pairs)`

Given an iterable of pairs (key, value), return the key corresponding to the greatest value.
Raises `ValueError` on empty sequence.
```python
>>> argmax(zip(range(20), range(20, 0, -1)))
0
```


### `argmax_index(values)`

Given an iterable of values, return the index of the (first) greatest value.
Raises `ValueError` on empty sequence.
```python
>>> argmax_index([0, 4, 3, 2, 1, 4, 0])
1
```


### `argmin(pairs)`

Given an iterable of pairs (key, value), return the key corresponding to the smallest value.
Raises `ValueError` on empty sequence.
```python
>>> argmin(zip(range(20), range(20, 0, -1)))
19
```


### `argmin_index(values)`

Given an iterable of values, return the index of the (first) smallest value.
Raises `ValueError` on empty sequence.
```python
>>> argmin_index([10, 4, 0, 2, 1, 0])
2
```


### `bucket_by_key(iterable, key_fc)`

Throws items in @iterable into buckets given by @key_fc function.
e.g.
```python
>>> bucket_by_key([1, 2, -3, 4, 5, 6, -7, 8, -9], lambda num: 'neg' if num < 0 else 'nonneg')
OrderedDict([('nonneg', [1, 2, 4, 5, 6, 8]), ('neg', [-3, -7, -9])])
```


### `cache_into(factory, filename)`

Simple pickle caching. Calls `factory`, stores result to `filename` pickle.
Subsequent calls load the obj from the pickle instead of running the `factory` again.


### `collapse_whitespace(txt)`

```python
>>> collapse_whitespace("bla   bla")
'bla bla'
```


### `consuming_length(iterator)`

Return length of an iterator, consuming its contents. O(1) memory.
```python
>>> consuming_length(range(10))
10
```


### `filter_both(predicate, iterable)`

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


### `filter_null(iterable)`

Filter out elements that do not evaluate to True
```python
>>> list(filter_null((0, None, 1, '', 'cherry')))
[1, 'cherry']
```


### `first_true_pred(predicates, value)`

Given a list of predicates and a value, return the index of first predicate,
s.t. predicate(value) == True.
If no such predicate found, raises IndexError.

```python
>>> first_true_pred([lambda x: x%2==0, lambda x: x%2==1], 13)
1
```


### `flatten(iterables)`

```python
>>> list(flatten(((1, 2, 3), (4, 5, 6))))
[1, 2, 3, 4, 5, 6]
```


### `group_consequent(iterator, key=None)`

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


### `identity(x)`

Return itself
```python
>>> identity(1)
1
```


### `k_grams(iterable, k)`

Returns iterator of k-grams of elements from `iterable`.
```python
>>> list(k_grams(range(4), 2))
[(0, 1), (1, 2), (2, 3)]
>>> list(k_grams((), 2))
[]
```


### `mod_stdout(transform, redirect_fn=<class 'contextlib.redirect_stdout'>, print_fn=<built-in function print>)`

A context manager that modifies every line printed to stdout.
```python
>>> with mod_stdout(lambda line: line.upper()):
...     print("this will be upper")
THIS WILL BE UPPER
```


### `nonempty_strip(iterable)`

```python
>>> list(nonempty_strip(['little ', '    ', '       piggy\n']))
['little', 'piggy']
```


### `num_stats(numbers, plot=False)`

Computes stats of the `numbers`, returns an OrderedDict with value and suggested print format
If `plot` is True, a histogram of the values is plotted (requires matplotlib).
```python
>>> num_stats(range(10))
OrderedDict([('count', 10), ('sum', 45), ('avg', 4.5), ('min', 0), ('50%', 4.5), ('95%', 8.5499999999999989), ('max', 9)])
>>> print_num_stats(num_stats(range(10)))
count 10
sum 45.000
avg 4.500
min 0.000
50% 4.500
95% 8.550
max 9.000
```


### `prefix_stdout(prefix)`

A context manager that prefixes every line printed to stout by `prefix`.
```python
>>> with prefix_stdout(" * "):
...     print("bullet")
 * bullet
```


### `simple_tokenize(txt, sep_rexp='\\W')`

Iterates through tokens, kwarg `sep_rexp` specifies the whitespace.
O(N) memory.
```python
>>> list(simple_tokenize('23_45 hello, how are  you?'))
['23_45', 'hello', 'how', 'are', 'you']
```






### `uniq(iterable, count=False)`

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


