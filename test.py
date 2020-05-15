import numpy as np
import math


def doit():
    error = []
    a = []
    for i in range(10):
        error.append({'a': [i, i]})
    for i in range(10):
        a.append([1, 1])

    print(error)

    for i in range(10):
        print((np.array(error[i]['a']))-np.array(a[i]))


print(doit())
