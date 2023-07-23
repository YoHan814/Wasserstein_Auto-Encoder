import os, sys
CODE_HOME = os.path.normpath(os.path.join(os.getcwd()))
sys.path.append(CODE_HOME)

import torch
import torch.nn as nn

from ..model_abstract import *
from ..util import init_params
# from .block import res_block, prob_block

class WAE_MMD_MNIST(WAE_MMD_abstract):
    def __init__(self, cfg, log, verbose = 1):
        super().__init__(cfg, log, verbose)
        d = 64
        self.encoder = nn.Sequential(
            nn.Conv2d(1, d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.Conv2d(d, d, kernel_size = 4, padding = 'same', bias = False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.Conv2d(d, 2*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),

            nn.Conv2d(2*d, 2*d, kernel_size = 4, padding = 'same', bias = False),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(49*2*d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(True),

            nn.Linear(d, self.z_dim)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 49*2*d),
            nn.Unflatten(1, (2*d, 7, 7)),
            
            nn.ConvTranspose2d(2*d, d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d, d//2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(d//2),
            nn.ReLU(True),

            nn.Conv2d(d//2, d//4, kernel_size = 4, padding = 'same', bias = False),
            nn.BatchNorm2d(d//4),
            nn.ReLU(True),
            
            # reconstruction
            nn.Conv2d(d//4, 1, kernel_size = 4, padding = 'same'),
            nn.Sigmoid(),
            
            )

        init_params(self.encoder)
        init_params(self.decoder)

        self.encoder_trainable = [self.encoder]
        self.decoder_trainable = [self.decoder]

class WAE_GAN_MNIST(WAE_GAN_abstract):
    def __init__(self, cfg, log, verbose = 1):
        super().__init__(cfg, log, verbose)
        d = 64
        self.encoder = nn.Sequential(
            nn.Conv2d(1, d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.Conv2d(d, d, kernel_size = 4, padding = 'same', bias = False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.Conv2d(d, 2*d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),

            nn.Conv2d(2*d, 2*d, kernel_size = 4, padding = 'same', bias = False),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(49*2*d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(True),

            nn.Linear(d, self.z_dim)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 49*2*d),
            nn.Unflatten(1, (2*d, 7, 7)),
            
            nn.ConvTranspose2d(2*d, d, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d, d//2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(d//2),
            nn.ReLU(True),

            nn.Conv2d(d//2, d//4, kernel_size = 4, padding = 'same', bias = False),
            nn.BatchNorm2d(d//4),
            nn.ReLU(True),
            
            # reconstruction
            nn.Conv2d(d//4, 1, kernel_size = 4, padding = 'same'),
            nn.Sigmoid(),
            )

        self.disc = nn.Sequential(
            nn.Linear(self.z_dim, d),
            nn.ReLU(True),

            nn.Linear(d, d),
            nn.ReLU(True),

            nn.Linear(d, d),
            nn.ReLU(True),

            nn.Linear(d, d),
            nn.ReLU(True),

            nn.Linear(d, d),
            nn.ReLU(True),

            nn.Linear(d, 1),
            )

        init_params(self.encoder)
        init_params(self.decoder)
        init_params(self.disc)

        self.encoder_trainable = [self.encoder]
        self.decoder_trainable = [self.decoder]
        self.disc_trainable = [self.disc]

class WAE_MMD_CelebA(WAE_MMD_abstract):
    def __init__(self, cfg, log, verbose = 1):
        super().__init__(cfg, log, verbose)
        d = 128
        self.encoder = nn.Sequential(
            nn.Conv2d(3, d, 5, 2, 2),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            
            nn.Conv2d(d, 2*d, 5, 2, 2),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),
            
            nn.Conv2d(2*d, 4*d, 5, 2, 2),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),
            
            nn.Conv2d(4*d, 8*d, 5, 2, 2),
            nn.BatchNorm2d(8*d),
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(4*4*8*d, self.z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 8*8*8*d), # 1024 x 8 x 8
            nn.ReLU(True),
            nn.Unflatten(1, (8*d,8,8)),

            nn.ConvTranspose2d(8*d, 4*d, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True), # 512 x 16 x 16
            nn.ConvTranspose2d(4*d, 2*d, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True), # 256 x 32 x 32
            nn.ConvTranspose2d(2*d, d, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(True), # 128 x 64 x 64
            nn.ConvTranspose2d(d, 3, 1, 1),
            nn.Tanh()
        )

        # init_params(self.encoder)
        # init_params(self.decoder)

        self.encoder_trainable = [self.encoder]
        self.decoder_trainable = [self.decoder]

class WAE_GAN_CelebA(WAE_GAN_abstract):
    def __init__(self, cfg, log, verbose = 1):
        super().__init__(cfg, log, verbose)
        d = 128
        self.encoder = nn.Sequential(
            nn.Conv2d(3, d, 5, 2, 2),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            
            nn.Conv2d(d, 2*d, 5, 2, 2),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),
            
            nn.Conv2d(2*d, 4*d, 5, 2, 2),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),
            
            nn.Conv2d(4*d, 8*d, 5, 2, 2),
            nn.BatchNorm2d(8*d),
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(4*4*8*d, self.z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 8*8*8*d), # 1024 x 8 x 8
            nn.ReLU(True),
            nn.Unflatten(1, (8*d,8,8)),

            nn.ConvTranspose2d(8*d, 4*d, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True), # 512 x 16 x 16
            nn.ConvTranspose2d(4*d, 2*d, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True), # 256 x 32 x 32
            nn.ConvTranspose2d(2*d, d, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(True), # 128 x 64 x 64
            nn.ConvTranspose2d(d, 3, 1, 1),
            nn.Tanh()
        )

        self.disc = nn.Sequential(
            nn.Linear(self.z_dim, d),
            nn.ReLU(True),
            nn.Linear(d, d),
            nn.ReLU(True),
            nn.Linear(d, d),
            nn.ReLU(True),
            nn.Linear(d, d),
            nn.ReLU(True),
            nn.Linear(d, 1)
        )

        # init_params(self.encoder)
        # init_params(self.decoder)
        # init_params(self.disc)

        self.encoder_trainable = [self.encoder]
        self.decoder_trainable = [self.decoder]
        self.disc_trainable = [self.disc]
