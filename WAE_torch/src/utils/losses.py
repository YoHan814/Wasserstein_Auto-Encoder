import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch

# *** MMD penalty ***
# MMD loss between x and y
def k(x, y, diag = True):
    z_dim = y.shape[1]
    stat = 0.
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = scale*2*z_dim*2 # z_prior ~ N(0,2)
        kernel = (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)))
        if diag:
            stat += kernel.sum()
        else:
            stat += kernel.sum() - kernel.diag().sum()
    return stat
    
def mmd_penalty(x, y):
    n = len(x)
    return (k(x,x, False) + k(y,y, False))/(n*(n-1)) - 2*k(x,y, True)/(n*n)
