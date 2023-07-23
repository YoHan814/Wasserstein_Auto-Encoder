#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import os
import argparse
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader

from codebase import utils as ut
from utils import _h_A, CelebA
from codebase.models.mask_vae_celeba import CausalVAE
from codebase.models.mask_wae_celeba import CausalWAE
from torchvision.utils import save_image, make_grid

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epoch_max',   type=int, default=80,    help="Number of training epochs")
parser.add_argument('--iter_save',   type=int, default=10, help="Save model every n epochs")
parser.add_argument('--run',         type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',       type=int, default=1,     help="Flag for training")
parser.add_argument('--toy',       type=str, default="celeba",     help="Flag for toy")
parser.add_argument('--type', type=int, default=0, help="label type: smile(0), beard(1)")
parser.add_argument('--pretrain', type=str, default="y", help="Pretrain step: y, n")
parser.add_argument('--model', type=str, default="vae", help="Models: vae or wae")
parser.add_argument('--alpha', type=float, default=1.0, help="hyperparam. of l_u")
parser.add_argument('--beta', type=float, default=1.0, help="hyperparam. of l_m")
parser.add_argument('--lamb1', type=float, default=100.0, help="hyperparam. of mmd")
parser.add_argument('--lamb2', type=float, default=100.0, help="hyperparam. of hsic")
parser.add_argument('--checkpoint', type=int, default=0, help="checkpoint to resume training")
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def _sigmoid(x):
#     I = torch.eye(x.size()[0]).to(device)
#     x = torch.inverse(I + torch.exp(-x))
#     return x

layout = [
    ('model={:s}',  'causal' + args.model),
    ('run={:04d}', args.run),
    ('type={:d}', args.type),
    ('toy={:s}', str(args.toy))
]

model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)
if args.model.lower() == "vae":
    lvae = CausalVAE(name=model_name).to(device)
else:
    lvae = CausalWAE(name=model_name).to(device)

if not os.path.exists(f'./figs_{args.model}_celeba/run_{args.run:04d}'):
    os.makedirs(f'./figs_{args.model}_celeba/run_{args.run:04d}')

dataset_dir = '/home/hjlee/data/CelebA'
train_set = CelebA(dataset_dir, split=[0], label_type = args.type)
val_set = CelebA(dataset_dir, split=[1], label_type = args.type)
test_set = CelebA(dataset_dir, split=[2], label_type = args.type)

train_dataset = DataLoader(train_set, 64, shuffle=True, drop_last=True, num_workers=5)
val_dataset = DataLoader(val_set, 64, shuffle=False, num_workers=5)
test_dataset = DataLoader(test_set, 64, shuffle=False)

optimizer = torch.optim.Adam(lvae.parameters(), lr=1e-3, betas=(0.9, 0.999))
# beta = ut.DeterministicWarmup(n=100, t_max=1) # Linear warm-up from 0 to 1 over 50 epoch


# Pretrain A (10 epochs)
if args.pretrain.lower() == "y":
    u_loss = torch.nn.MSELoss()
    lamb_A, c_A = 0., 1.
    h_a_old = _h_A(lvae.dag.A, lvae.dag.A.size()[0]).item()
    for p in lvae.enc.parameters():
        p.requires_grad = False
    for p in lvae.dec.parameters():
        p.requires_grad = False
    for p in lvae.mask_z.parameters():
        p.requires_grad = False
    for _ in range(10):
        for u, l in train_dataset:
            optimizer.zero_grad()
            u = u.to(device)
            m_u = lvae.dag.mask_u(l.to(device)) # A^T u
            g_u = torch.sigmoid(m_u).to(device) # u_i = g_i(A_i o z)
            l_u = u_loss(g_u.squeeze(), l.to(device))
#             _, _, _, _, _, l_u, _ = lvae.negative_elbo_bound(u, l, sample = False, alpha=1., beta=1.)
            
            dag_param = lvae.dag.A
            #dag_reg = dag_regularization(dag_param)
            h_a = _h_A(dag_param, dag_param.size()[0])
            L = l_u + lamb_A*h_a + c_A*.5*h_a*h_a

            L.backward()
            optimizer.step()

        dag_param = lvae.dag.A.clone()
        h_a_new = _h_A(dag_param, dag_param.size()[0])
        if h_a_new.item() > 0.25 * h_a_old:
            c_A *= 10
        lamb_A += c_A * h_a_new.item()
        h_a_old = h_a_new.item()
    ut.save_model_by_name(lvae, 0)
else:
    file_path = f"checkpoints/model=causal{args.model}_run=0000_type=0_toy=celeba/model-00000.pt"
    state = torch.load(file_path)
    lvae.load_state_dict(state)
    print("Loaded from {}".format(file_path))
#     ut.load_model_by_name(lvae, 0)

print(lvae.dag.A)

if args.checkpoint > 0:
    ut.load_model_by_name(lvae, args.checkpoint)

# Training step
for p in lvae.enc.parameters():
    p.requires_grad = True
for p in lvae.dec.parameters():
    p.requires_grad = True
for p in lvae.mask_z.parameters():
    p.requires_grad = True

total_loss_arr, val_loss_arr = [], []
total_kl_arr, val_kl_arr = [], []
total_mask_kl_arr, val_mask_kl_arr = [], []
total_l_u_arr, val_l_u_arr = [], []
total_rec_arr, val_rec_arr = [], []
for epoch in range(args.checkpoint+1, args.epoch_max+1):
    lvae.train()
    total_loss = 0
    total_rec = 0
    total_kl = 0
    total_mask_kl = 0.
    total_l_u = 0.
    # h_a_old = 0.
    for u, l in train_dataset:
        optimizer.zero_grad()
        # u = torch.bernoulli(u.to(device).reshape(u.size(0), -1))
        u = u.to(device)
        # L, kl, rec, reconstructed_image, mask_kl, l_u, _ = lvae.negative_elbo_bound(u, l, sample = False, alpha=1., beta=1.)
        if args.model.lower() == "vae":
            L, kl, rec, reconstructed_image, mask_kl, l_u, _, _ = lvae.negative_elbo_bound(u, l, sample = False, alpha=args.alpha, beta=args.beta)
            reconstructed_image = torch.sigmoid(reconstructed_image)
        else:
            L, kl, rec, reconstructed_image, mask_kl, l_u, _, _ = lvae.negative_elbo_bound(u, l, sample = False, 
                                                                                        alpha=args.alpha, beta=args.beta,
                                                                                        lamb1=args.lamb1, lamb2=args.lamb2)
            reconstructed_image = (reconstructed_image + 1.)/2.
        
        dag_param = lvae.dag.A
        
        #dag_reg = dag_regularization(dag_param)
        h_a = _h_A(dag_param, dag_param.size()[0])
        L = L + 100 * h_a
   
        L.backward()
        optimizer.step()
        #optimizer.zero_grad()

        total_loss += L.item()
        total_kl += kl.item() 
        total_rec += rec.item()
        total_mask_kl += mask_kl.item()
        total_l_u += l_u.item()
        
        # save_image(u[0], 'figs_vae/reconstructed_image_true_{}.png'.format(epoch), normalize = True)
        # save_image(reconstructed_image[0], 'figs_vae/reconstructed_image_{}.png'.format(epoch), normalize = True)
    grid = make_grid(torch.cat(((u[:32]+1.)/2., reconstructed_image[:32]), dim=0), normalize=True)
    save_image(grid, f'figs_{args.model}_celeba/run_{args.run:04d}/recon_{epoch}.png')
    
    m = len(train_dataset)
    total_loss_arr.append(total_loss/m)
    total_kl_arr.append(total_kl/m)
    total_mask_kl_arr.append(total_mask_kl/m)
    total_l_u_arr.append(total_l_u/m)
    total_rec_arr.append(total_rec/m)
    
    print(f"[train] {epoch:02d} loss: {total_loss/m:.4f}\tkl: {total_kl/m:.4f}\trec: {total_rec/m:.4f}")
    
    lvae.eval()
    val_loss = 0
    val_rec = 0
    val_kl = 0
    val_mask_kl = 0.
    val_l_u = 0.
    # h_a_old = 0.
    with torch.no_grad():
        for u, l in val_dataset:
            # u = torch.bernoulli(u.to(device).reshape(u.size(0), -1))
            u = u.to(device)
            if args.model.lower() == "vae":
                L, kl, rec, reconstructed_image, mask_kl, l_u, _ = lvae.negative_elbo_bound(u, l, sample = False, alpha=1., beta=1.)
            else:
                L, kl, rec, reconstructed_image, mask_kl, l_u, _ = lvae.negative_elbo_bound(u, l, sample = False, 
                                                                                            alpha=1., beta=1., lamb1=100.0, lamb2=100.0)
            
            val_loss += L.item()
            val_kl += kl.item() 
            val_rec += rec.item()
            val_mask_kl += mask_kl.item()
            val_l_u += l_u.item()
        
        u, l = next(iter(val_dataset)) # first batch
        u = u.to(device)
        if args.model.lower() == "vae":
            _, _, _, recon, _, _, _ = lvae.negative_elbo_bound(u, l, sample = False, alpha=1., beta=1.)
            recon = torch.sigmoid(recon)
        else:
            _, _, _, recon, _, _, _ = lvae.negative_elbo_bound(u, l, sample = False, alpha=1., beta=1., lamb1=100.0, lamb2=100.0)
            recon = (recon + 1.)/2.
        
        grid = make_grid(torch.cat(((u[:32]+1.)/2., recon[:32]), dim=0), normalize=True)
        save_image(grid, f'figs_{args.model}_celeba/run_{args.run:04d}/val_{epoch}.png')
        
        m = len(val_dataset)
        val_loss_arr.append(val_loss/m)
        val_kl_arr.append(val_kl/m)
        val_mask_kl_arr.append(val_mask_kl/m)
        val_l_u_arr.append(val_l_u/m)
        val_rec_arr.append(val_rec/m)
        
    print(f"[val] {epoch:02d} loss: {val_loss/m:.4f}\tkl: {val_kl/m:.4f}\trec: {val_rec/m:.4f}")
    
    if epoch % args.iter_save == 0:
        ut.save_model_by_name(lvae, epoch)

np.save(f'./figs_{args.model}_celeba/run_{args.run:04d}/losses.npy',
        np.transpose(np.vstack((total_loss_arr, total_kl_arr, total_mask_kl_arr, total_l_u_arr, total_rec_arr))))
np.save(f'./figs_{args.model}_celeba/run_{args.run:04d}/val_losses.npy',
        np.transpose(np.vstack((val_loss_arr, val_kl_arr, val_mask_kl_arr, val_l_u_arr, val_rec_arr))))
