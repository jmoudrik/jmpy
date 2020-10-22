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

