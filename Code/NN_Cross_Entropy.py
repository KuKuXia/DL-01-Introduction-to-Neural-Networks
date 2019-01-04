# @Time: 2019/1/1 0001 19:52
# @Author: KuKuXia
# Note:

import numpy as np

def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

