#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import numpy as np
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from scipy.special import gamma
# from torch.nn import functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# *** MMD penalty ***
# MMD loss between x and y
def k(x, y, diag=True):
    z_dim = y.shape[1]
    stat = 0.
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = scale * 2 * z_dim * 2  # z_prior ~ N(0,2)
        kernel = (C / (C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim=2)))
        if diag:
            stat += kernel.sum()
        else:
            stat += kernel.sum() - kernel.diag().sum()
    return stat

def mmd_penalty(x, y):
    n = len(x)
    return (k(x, x, False) + k(y, y, False)) / (n * (n - 1)) - 2 * k(x, y, True) / (n * n)

"""
Refers to original Tensorflow implementation: https://github.com/romain-lopez/HCV
Refers to original implementations
    - https://github.com/kacperChwialkowski/HSIC
    - https://cran.r-project.org/web/packages/dHSIC/index.html
"""
def bandwidth(d):
    gz = 2 * gamma(0.5 * (d + 1)) / gamma(0.5 * d)
    return 1. / (2. * gz ** 2)

def knl(x, y, gam=1.):
    dist_table = (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim=2)
    return (-gam * dist_table).exp().transpose(0, 1)

def hsic_penalty(x, y):
    dx = x.shape[1]
    dy = y.shape[1]

    xx = knl(x, x, gam=bandwidth(dx))
    yy = knl(y, y, gam=bandwidth(dy))

    res = ((xx * yy).mean()) + (xx.mean()) * (yy.mean())
    res -= 2 * ((xx.mean(dim=1)) * (yy.mean(dim=1))).mean()
    return res.clamp(min=1e-16).sqrt()


class CausalWAE(nn.Module):
    def __init__(self, nn='mask', name='vae', z_dim=128, z1_dim=4, z2_dim=32, inference = False):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.channel = 3
        # self.scale = np.array([[0,44], [100,40], [6.5, 3.5], [10,5]])
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Conv_Encoder(self.z_dim, y_dim=self.z1_dim) # e = f(x,u)
        self.dec = nn.Conv_Decoder_DAG(self.z_dim, self.z1_dim, self.z2_dim) # x = g_3(z2)
        self.z_u = [torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32, self.z2_dim)
        ).to(device) for j in range(self.z1_dim)] # z1 = g_1(u)
        self.dag = nn.DagLayer(self.z1_dim, self.z1_dim, i = inference)
        self.mask_z = nn.MaskLayer(self.z_dim, concept=z1_dim, z2_dim=z2_dim)

        self.u_loss = torch.nn.MSELoss()

    def negative_elbo_bound(self, x, label, mask = None, sample = False, adj = None, alpha=1., beta=1.,
                            lamb1=100., lamb2=100.):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        assert label.size()[1] == self.z1_dim

        q_m, q_v = self.enc.encode(x.to(device), y=label.to(device)) # B * 128; e_mean, e_var
        # Reshape B x 16 -> B x 4 x 4
        # deterministic encoder
        q_m, q_v = q_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]), torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(device)

        # decode_m: z = (I-A^t)^-1 epsilon + g1(u) / decode_v=1 ; deterministic
        decode_m, decode_v = self.dag.calculate_dag(q_m.to(device), torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(device))
        decode_m = decode_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim])
        z1_u = torch.cat([self.z_u[j](label.to(device)[:, j].unsqueeze(dim=-1)) for j in range(self.z1_dim)], dim=1)
        decode_m += z1_u.reshape([q_m.size()[0], self.z1_dim, self.z2_dim])
#         decode_m = torch.cat([self.z_u[j](label.to(device)[:, j].unsqueeze(dim=-1)) for j in range(self.z1_dim)], dim=1)
#         decode_m = decode_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim])
        if sample == False:
            if mask != None: # mask: index of concept to intervene
                # intervene z before masking layer
                z_mask = torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(device) * adj
                decode_m[:, mask, :] = z_mask[:, mask, :]
                # decode_v[:, mask, :] = z_mask[:, mask, :]
            m_zm = self.dag.mask_z(decode_m.to(device)).reshape([q_m.size()[0], self.z1_dim, self.z2_dim]) # A^T z
            m_zv = decode_v.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]) # all is 1
            m_u = self.dag.mask_u(label.to(device)) # A^T u

            # Apply mask layer(MLP) for each concept(dim=1)
            f_z = self.mask_z.mix(m_zm).reshape([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device) # g(A^T z)
#             e_tilde = Attention((I-A^t)^{-1}e, e)
#             e_tilde = self.attn.attention(decode_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(device),
#                                           q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(device))[0]          
            
            f_z1 = f_z + q_m # z_i = g_i(A_i o z) + e_i
            if mask != None:
                z_mask = torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(device) * adj
                f_z1[:, mask, :] = z_mask[:, mask, :]
                # m_zv[:, mask, :] = z_mask[:, mask, :]
            
            g_u = torch.sigmoid(m_u).to(device) # u_i = g_i(A_i o z)
            # z_given_dag = ut.conditional_sample_gaussian(f_z1, m_zv*lambdav)
        
        recon_logits = self.dec.decode(f_z1, label.to(device))
        recon = torch.tanh(recon_logits)

        rec = torch.mean(torch.sum((x - recon)**2, dim=[1,2,3]))

        # Z|u ~ N(u, I)
        cp_m = label.unsqueeze(2).expand(label.size()[0], self.z1_dim, self.z2_dim).to(device)
        cp_v = torch.ones([q_m.size()[0],self.z1_dim, self.z2_dim]).to(device)

        # D_KL(q(e|x,u) || p(e)) = sum of 0.5 * ((e - 0)^2 - 1)
        true_e = torch.randn_like(q_m.reshape(-1, self.z_dim)) # N x z_dim
        mmd = mmd_penalty(q_m.reshape(-1, self.z_dim).to(device), true_e.to(device))
        hsic = hsic_penalty(q_m.reshape(-1, self.z_dim).to(device), cp_m.reshape(-1, self.z_dim).to(device))

        # mask_kl2 = torch.zeros(1).to(device)
        # KL = sum of 0.5 * ((f_z1 - cp_m)^2 - 1)
        mask_kl = torch.zeros(1).to(device)
        for i in range(4):
#             mask_kl = mask_kl + 1*ut.kl_normal(f_z1[:,i,:].to(device), cp_v[:,i,:].to(device), cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
            mask_kl = mask_kl + 1*ut.kl_normal(f_z[:,i,:].to(device), cp_v[:,i,:].to(device),
                                               decode_m[:,i,:].to(device), decode_v[:,i,:].to(device))

        l_u = self.u_loss(g_u.squeeze(), label.to(device))
        mask_l = alpha*l_u + beta*torch.mean(mask_kl)
        loss = rec + lamb1*mmd + lamb2*hsic + mask_l
        
        return loss, lamb1*mmd + lamb2*hsic, rec, recon.reshape(x.size()), torch.mean(mask_kl), l_u, decode_m, cp_m

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries
