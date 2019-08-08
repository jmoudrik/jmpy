# jmpy
jm's Python utils

## Instalation

```bash
sudo python3 setup.py install
```
## API

- [`argmax`](#argmax)
- [`argmax_index`](#argmax_index)
- [`argmin`](#argmin)
- [`argmin_index`](#argmin_index)
- [`bucket_by_key`](#bucket_by_key)
- [`cache_into`](#cache_into)
- [`collapse_whitespace`](#collapse_whitespace)
- [`consuming_length`](#consuming_length)
- [`filter_both`](#filter_both)
- [`filter_null`](#filter_null)
- [`first_true_pred`](#first_true_pred)
- [`flatten`](#flatten)
- [`group_consequent`](#group_consequent)
- [`identity`](#identity)
- [`k_grams`](#k_grams)
- [`mod_stdout`](#mod_stdout)
- [`nonempty_strip`](#nonempty_strip)
- [`prefix_stdout`](#prefix_stdout)
- [`simple_tokenize`](#simple_tokenize)
- [`stderr`](#stderr)
- [`timer`](#timer)
- [`uniq`](#uniq)
- [`utils`](#utils)


### `argmax`

Given an iterable of pairs (key, value), return the key corresponding to the greatest value.
Raises `ValueError` on empty sequence.
```python
>>> argmax(zip(range(20), range(20, 0, -1)))
0
```


### `argmax_index`

Given an iterable of values, return the index of the (first) greatest value.
Raises `ValueError` on empty sequence.
```python
>>> argmax_index([0, 4, 3, 2, 1, 4, 0])
1
```


### `argmin`

Given an iterable of pairs (key, value), return the key corresponding to the smallest value.
Raises `ValueError` on empty sequence.
```python
>>> argmin(zip(range(20), range(20, 0, -1)))
19
```


### `argmin_index`

Given an iterable of values, return the index of the (first) smallest value.
Raises `ValueError` on empty sequence.
```python
>>> argmin_index([10, 4, 0, 2, 1, 0])
2
```


### `bucket_by_key`

Throws items in @iterable into buckets given by @key_fc function.
e.g.
```python
>>> bucket_by_key([1, 2, -3, 4, 5, 6, -7, 8, -9], lambda num: 'neg' if num < 0 else 'nonneg')
OrderedDict([('nonneg', [1, 2, 4, 5, 6, 8]), ('neg', [-3, -7, -9])])
```


### `cache_into`

Simple pickle caching. Calls `factory`, stores result to `filename` pickle.
Subsequent calls load the obj from the pickle instead of running the `factory` again.


### `collapse_whitespace`

```python
>>> collapse_whitespace("bla   bla")
'bla bla'
```




### `filter_both`

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


### `filter_null`

Filter out elements that do not evaluate to True
```python
>>> list(filter_null((0, None, 1, '', 'cherry')))
[1, 'cherry']
```


### `first_true_pred`

Given a list of predicates and a value, return the index of first predicate,
s.t. predicate(value) == True.
If no such predicate found, raises IndexError.

```python
>>> first_true_pred([lambda x: x%2==0, lambda x: x%2==1], 13)
1
```




### `group_consequent`

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




### `k_grams`

Returns iterator of k-grams of elements from `iterable`.
```python
>>> list(k_grams(range(4), 2))
[(0, 1), (1, 2), (2, 3)]
>>> list(k_grams((), 2))
[]
```


### `mod_stdout`

A context manager that modifies every line printed to stdout.
```python
>>> with mod_stdout(lambda line: line.upper()):
...     print("this will be upper")
THIS WILL BE UPPER
```




### `prefix_stdout`

A context manager that prefixes every line printed to stout by `prefix`.
```python
>>> with prefix_stdout(" * "):
...     print("bullet")
 * bullet
```








### `uniq`

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


