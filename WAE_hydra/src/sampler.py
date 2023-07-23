import torch

import torch
import torch.distributions as D
import math

def unif(x, y, a = -1, b = 1):
    return (b-a)*torch.rand(x, y) + a

def gaus(x, y):
    return torch.normal(0, math.sqrt(2), size = (x,y))

def h_sphere(x, y):
    xyz = torch.normal(0, 1, size = (x,y))
    return (xyz/xyz.norm(dim = 1).unsqueeze(1))

def multinomial(x, y):
    return torch.eye(y)[torch.randint(y,(x,))]
