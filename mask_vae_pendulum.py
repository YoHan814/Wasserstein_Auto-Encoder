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
from torch.nn import functional as F
from scipy.special import gamma
from minepy import MINE
from utils import _h_A

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

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


def bandwidth(d):
    gz = 2 * gamma(0.5 * (d+1)) / gamma(0.5 * d)
    return 1. / (2. * gz**2)

def knl(x, y, gam=1.):
    dist_table = (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)
    return (-gam * dist_table).exp().transpose(0,1)

def hsic_penalty(x, y):
    dx = x.shape[1]
    dy = y.shape[1]

    xx = knl(x, x, gam=bandwidth(dx))
    yy = knl(y, y, gam=bandwidth(dy))

    res = ((xx*yy).mean()) + (xx.mean()) * (yy.mean())
    res -= 2*((xx.mean(dim=1))*(yy.mean(dim=1))).mean()
    return res.clamp(min = 1e-16).sqrt()




class CausalVAE(nn.Module):
    def __init__(self, nn='mask', name='vae', z_dim=16, z1_dim=4, z2_dim=4, 
                 inference = False, alpha=1, beta=1, r_dim = 96, channel = 4):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.channel = channel
        self.scale = np.array([[0,44],[100,40],[6.5, 3.5],[10,5]])
        self.r_dim = r_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.channel, r_dim = self.r_dim)
        self.dec = nn.Decoder_DAG(self.z_dim, self.z1_dim, self.z2_dim, r_dim = self.r_dim, channel = self.channel)
        self.dag = nn.DagLayer(self.z1_dim, self.z1_dim, i = inference)
        self.z_u = torch.nn.Sequential(
            torch.nn.Linear(self.z1_dim, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, self.z_dim)
        )
        
        self.attn = nn.Attention(self.z1_dim)
        self.mask_z = nn.MaskLayer(self.z_dim, concept = self.z1_dim, z1_dim = self.z1_dim)
        self.mask_u = nn.MaskLayer(self.z1_dim,z1_dim=1, concept = self.z1_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x, label, mask = None, sample = False, adj = None, alpha=1, beta=1, lambdav=0.001, mode = 1, get_mic = False):
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

        q_m, q_v = self.enc.encode(x.to(device)) # epsilon = h(x, u) + zeta
        q_m, q_v = q_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device)
        
        if mode == 1:
            decode_m, decode_v = self.dag.calculate_dag(q_m.to(device), torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device)) # z = (I-A^t)epsilon
            decode_m, decode_v = decode_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),decode_v
        # q_m == decode_m 같음
        elif mode == 2:
            decode_m, decode_v = self.dag.calculate_dag(q_m.to(device), torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device)) # z = (I-A^t)epsilon
            decode_m, decode_v = decode_m.reshape([q_m.size()[0], self.z1_dim,self.z2_dim]),decode_v
            z1_m = self.z_u(label.to(device)).reshape([q_m.size()[0], self.z1_dim, self.z2_dim]) # g1(u)
            decode_m += z1_m # z = (I - A^t)^-1 epsilon + g1(u)
        
        if sample == False:
            if mask != None and mask < 2:
                z_mask = torch.ones(q_m.size()[0], self.z1_dim,self.z2_dim).to(device)*adj
                decode_m[:, mask, :] = z_mask[:, mask, :]
                decode_v[:, mask, :] = z_mask[:, mask, :]
            m_zm = self.dag.mask_z(decode_m.to(device)).reshape([q_m.size()[0], self.z1_dim,self.z2_dim]) # A^t z
            m_zv = decode_v.reshape([q_m.size()[0], self.z1_dim,self.z2_dim])
            m_u = self.dag.mask_u(label.to(device)) # A^t u
            
            f_z = self.mask_z.mix(m_zm).reshape([q_m.size()[0], self.z1_dim,self.z2_dim]).to(device) # g(A o z)
            
            # e_tilde = self.attn.attention(decode_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device),
                                          # q_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device))[0]
            
            if mask != None and mask < 2:
                z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
                q_m[:, mask, :] = z_mask[:, mask, :]
              
            # f_z1 = f_z+e_tilde # g(A o z) + epsilon
            f_z1 = f_z + q_m# g(A o z) + epsilon
            if mask!= None and mask == 2 :
                z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
                f_z1[:, mask, :] = z_mask[:, mask, :]
                m_zv[:, mask, :] = z_mask[:, mask, :]
            if mask!= None and mask == 3 :
                z_mask = torch.ones(q_m.size()[0],self.z1_dim,self.z2_dim).to(device)*adj
                f_z1[:, mask, :] = z_mask[:, mask, :]
                m_zv[:, mask, :] = z_mask[:, mask, :]
            g_u = self.mask_u.mix(m_u).to(device) # sigmoid
            z_given_dag = ut.conditional_sample_gaussian(f_z1, m_zv*lambdav)  # f_z1 + N(0,1) * sqrt(lambdav)
        
        decoded_bernoulli_logits,x1,x2,x3,x4 = self.dec.decode_sep(z_given_dag.reshape([z_given_dag.size()[0], self.z_dim]), label.to(device))
        
        p_m, p_v = torch.zeros(q_m.size()), torch.ones(q_m.size())
        cp_m, cp_v = ut.condition_prior(self.scale, label, self.z2_dim)
        cp_v = torch.ones([q_m.size()[0],self.z1_dim,self.z2_dim]).to(device)
        cp_z = ut.conditional_sample_gaussian(cp_m.to(device), cp_v.to(device))
        
        
        kl = torch.zeros(1).to(device)
        kl = ut.kl_normal(q_m.view(-1,self.z_dim).to(device), q_v.view(-1,self.z_dim).to(device), 
                          p_m.view(-1,self.z_dim).to(device), p_v.view(-1,self.z_dim).to(device))
            
        for i in range(self.z1_dim):
            kl = kl + 1*ut.kl_normal(decode_m[:,i,:].to(device), # (I-A)^-1 \epsilon
                                     cp_v[:,i,:].to(device), # 의미 없음, 1로 주어짐
                                     cp_m[:,i,:].to(device), # 평균 레이블, 분산 1인 normal rv
                                     cp_v[:,i,:].to(device)) # 의미 없음, 1로 주어짐
        kl = torch.mean(kl)
        mask_kl = torch.zeros(1).to(device)
        mask_kl2 = torch.zeros(1).to(device)
        
        for i in range(4):
            # mask_kl = mask_kl + 1*ut.kl_normal(f_z1[:,i,:].to(device), cp_v[:,i,:].to(device),
            #                                    cp_m[:,i,:].to(device), cp_v[:,i,:].to(device)) # gamma*l_m

            mask_kl = mask_kl + 1*ut.kl_normal(decode_m[:,i,:].to(device), cp_v[:,i,:].to(device),
                                               f_z[:,i,:].to(device), cp_v[:,i,:].to(device)) # gamma*l_m

        u_loss = torch.nn.MSELoss()

       # mask_l = torch.mean(mask_kl) + u_loss(g_u, label.float().to(device)) # kl + l_u
        mask_l = torch.mean(mask_kl) + u_loss(g_u, label.float().to(device))
        
        h_a = torch.zeros(1).to(device)
        h_a = h_a + _h_A(self.dag.A.to(device), self.dag.A.size()[0])
        
        if mode == 1:
            rec = ut.log_bernoulli_with_logits(x, decoded_bernoulli_logits.reshape(x.size()))
            rec = -torch.mean(rec)
            
            penalty = kl + mask_l + 0.3*h_a
            nelbo = rec + penalty
                        
            if get_mic:
                # mine1 = MINE(alpha=0.6, c=15, est="mic_approx")
                # mine1.compute_score(decode_m.cpu().detach().numpy().reshape(1,-1)[0,:], cp_m.cpu().detach().numpy().reshape(1,-1)[0,:])
                # mic = mine1.mic()
                zz_to_mic = [decode_m[:, j, :].cpu().detach().numpy().reshape(1,-1)[0,:] for j in range(self.z1_dim)]
                ll_to_mic = [cp_m[:, j, :].cpu().detach().numpy().reshape(1,-1)[0,:] for j in range(self.z1_dim)]
                # zz_to_mic = decode_m.cpu().detach().numpy().reshape(1,-1)[0,:]
                # ll_to_mic = cp_m.cpu().detach().numpy().reshape(1,-1)[0,:]
                return nelbo, rec, penalty, decoded_bernoulli_logits.reshape(x.size()), z_given_dag, zz_to_mic, ll_to_mic
            else:
                return nelbo, rec, penalty, decoded_bernoulli_logits.reshape(x.size()), z_given_dag
            
        elif mode == 2:
            r_loss = torch.nn.MSELoss()
            rec = r_loss(x, torch.sigmoid(decoded_bernoulli_logits.reshape(x.size()).to(device)))
            mmd = mmd_penalty(q_m.view(-1,self.z_dim).to(device), torch.randn_like(q_m.view(-1,self.z_dim).to(device)))
            hsic = hsic_penalty(q_m.view(-1,self.z_dim).to(device), cp_m.view(-1,self.z_dim).to(device))
            
            penalty1 = alpha*mmd + beta*hsic
            penalty2 = mask_l + h_a*0.3 # kl + mask_l + h_a*0.3
            nelbo = rec + penalty1 + penalty2
            
            if get_mic:
                # mine1 = MINE(alpha=0.6, c=15, est="mic_approx")
                # mine1.compute_score(decode_m.cpu().detach().numpy().reshape(1,-1)[0,:], cp_m.cpu().detach().numpy().reshape(1,-1)[0,:])
                # mic = mine1.mic()
                zz_to_mic = [z1.cpu().detach().numpy() for z1 in torch.split(decode_m, self.z_dim//self.concept, dim = 1)]
                ll_to_mic = [uu.cpu().detach().numpy() for uu in torch.split(cp_m, self.z_dim//self.concept, dim = 1)]
                # zz_to_mic = decode_m.cpu().detach().numpy().reshape(1,-1)[0,:]
                # ll_to_mic = cp_m.cpu().detach().numpy().reshape(1,-1)[0,:]
                return nelbo, rec, penalty1, penalty2, decoded_bernoulli_logits.reshape(x.size()), z_given_dag, zz_to_mic, ll_to_mic
            else:
                return nelbo, rec, penalty1, penalty2, decoded_bernoulli_logits.reshape(x.size()), z_given_dag
        # mic(decoded_m, label)
        
    
    
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


    








