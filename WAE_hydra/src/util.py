import os
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

def identity(e):
    return 1.0

def basic(e):
    return 1.0 / (1.0 + e)

def manual(e):
    return 1.0 * (0.5 ** (e >= 30)) * (0.2 ** (e >= 50)) * (0.1 ** (e >= 100))

def scheduler_parse(optimizer, lr_schedule):
    if lr_schedule == "basic":
        return optim.lr_scheduler.MultiplicativeLR(optimizer, basic)
    if lr_schedule == "manual":
        return optim.lr_scheduler.MultiplicativeLR(optimizer, manual)
    return optim.lr_scheduler.MultiplicativeLR(optimizer, identity)

# def scheduler_parse(optimizer, lr_schedule):
#     lamb = lambda e: 1.0
#     if lr_schedule == "basic":
#         lamb = lambda e: 1.0 / (1.0 + e)
#     if lr_schedule == "manual":
#         lamb = lambda e: 1.0 * (0.5 ** (e >= 30)) * (0.2 ** (e >= 50)) * (0.1 ** (e >= 100))
#     return optim.lr_scheduler.MultiplicativeLR(optimizer, lamb)

def init_params(model):
    for p in model.parameters():
        if(p.dim() > 1):
            # nn.init.xavier_normal_(p)
            nn.init.trunc_normal_(p, std = 0.01, a = -0.02, b = 0.02)
        else:
            nn.init.uniform_(p, 0.1, 0.2)
    return

def save_sample_images(save_path, name, epoch, img_list):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    fig = plt.figure(figsize=(4,4))
    for i in range(64):
        plt.subplot(8,8,i+1)
        if img_list.shape[1] == 1:
            plt.imshow(np.squeeze(np.transpose(img_list[i,:,:,:], (1,2,0))), cmap = 'gray')
        else:
            plt.imshow(np.squeeze(np.transpose(img_list[i,:,:,:], (1,2,0))))
        plt.axis('off')
    
    fig.tight_layout(pad = 0)
    fig.subplots_adjust(wspace=0.0, hspace = 0.0)
    plt.savefig('%s/%s-%03d.png' % (save_path, name, epoch + 1))
    return
