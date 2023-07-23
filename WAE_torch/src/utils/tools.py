import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gzip
import pickle

import torch

random.seed(1)
np.random.seed(1)
matplotlib.use('Agg')

# Calculating average loss
class inc_avg():
    def __init__(self):
        self.avg = 0.0
        self.weight = 0
        
    def append(self, dat, w = 1):
        self.weight += w
        self.avg = self.avg + (dat - self.avg)*w/self.weight

def plot_losses(losses_dict, keys=None, path=None, step=1):
    if keys is None:
        keys = list(losses_dict.keys())
        
    if len(keys) > 0:
        idxs = np.arange(step, len(list(losses_dict.values())[0])+1, step)
        idxs = np.append(1, idxs)
    
        for key in keys:
            plt.plot(idxs, np.array(losses_dict[key])[idxs-1], label=key)
        plt.xlabel("epochs")
        plt.ylabel("loss values")
        plt.legend()
        plt.savefig(path+".png")
        plt.close()