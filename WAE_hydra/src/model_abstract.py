import os, importlib

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from . import sampler
from .util import scheduler_parse, save_sample_images

class AutoEncoder_abstract(LightningModule):
    def __init__(self, cfg, log, verbose = 1):
        super().__init__()
        self.hydra_log = log
        if verbose == 1:
            self.hydra_log.info('------------------------------------------------------------')
            for key in cfg['train_info']:
                self.hydra_log.info('%s : %s' % (key, cfg['train_info'][key]))

            for key in cfg['path_info']:
                self.hydra_log.info('%s : %s' % (key, cfg['path_info'][key]))
        
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

        self.z_dim = int(cfg['train_info']['z_dim'])
        self.z_sampler = getattr(sampler, cfg['train_info']['z_sampler']) # generate prior

        self.lr = float(cfg['train_info']['lr'])
        self.beta1 = float(cfg['train_info']['beta1'])
        self.lr_scheduler = str(cfg['train_info']['lr_scheduler'])
        self.num_epoch = int(cfg['train_info']['epoch'])

        # self.tb_logs = cfg['path_info']['tb_logs']
        self.save_img_path = cfg['path_info']['save_img_path']

        self.get_recon_flag = True

        self.encoder_trainable = [self.encoder]
        self.decoder_trainable = [self.decoder]

    def log_architecture(self):
        self.hydra_log.info('------------------------------------------------------------')
        for net in self.encoder_trainable:
            self.hydra_log.info(net)
        for net in self.decoder_trainable:
            self.hydra_log.info(net)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def configure_optimizers(self):
        optimizer = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))
        if self.lr_scheduler is None:
            return {"optimizer": optimizer}
        scheduler = scheduler_parse(optimizer, self.lr_scheduler)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none").sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)

        # Progress Bar
        self.log("recon", loss, prog_bar=True, logger = False)
        # TensorBoard
        self.log("train/recon", loss, on_step = False, on_epoch = True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)

        # Progress Bar
        self.log("recon", loss, prog_bar=True, logger = False)
        # TensorBoard
        self.log("train/recon", loss, on_step = False, on_epoch = True)
        
        return loss

class WAE_MMD_abstract(AutoEncoder_abstract):
    def __init__(self, cfg, log, verbose = 1):
        super().__init__(cfg, log, verbose)
        self.z_sampler = getattr(sampler, cfg['train_info']['z_sampler'])
        self.lamb = float(cfg['train_info']['lambda'])

    def k(self, x, y, diag = True):
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = scale*2*self.z_dim*2
            kernel = (C/(C + (x.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(dim = 2)))
            if diag:
                stat += kernel.sum()
            else:
                stat += kernel.sum() - kernel.diag().sum()
        return stat
    
    def penalty_loss(self, x, y, n):
        return (self.k(x,x, False) + self.k(y,y, False))/(n*(n-1)) - 2*self.k(x,y, True)/(n*n)

    def _get_losses(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x = batch
        n = len(x)

        fake_latent = self.encode(x)
        prior_z = self.z_sampler(n, self.z_dim).type_as(fake_latent)
        x_hat = self.decode(fake_latent)

        loss = F.mse_loss(x, x_hat, reduction="none").sum(dim=[1, 2, 3]).mean(dim=[0])
        penalty = self.penalty_loss(fake_latent, prior_z, n)
        return loss, penalty

    def training_step(self, batch, batch_idx):
        loss, penalty = self._get_losses(batch)
        # tqdm_dict = {"recon":loss.detach(), "penalty":penalty.detach()}

        # Progress Bar
        self.log("recon", loss, prog_bar=True, logger = False)
        self.log("penalty", penalty, prog_bar=True, logger = False)

        # TensorBoard
        self.log("train/recon", loss, on_step = False, on_epoch = True)
        self.log("train/penalty", penalty, on_step = False, on_epoch = True)

        # return {"loss": loss + self.lamb * penalty, "progress_bar": tqdm_dict}
        return loss + self.lamb * penalty
    
    def validation_step(self, batch, batch_idx):
        loss, penalty = self._get_losses(batch)
        # tqdm_dict = {"recon":loss.detach(), "penalty":penalty.detach()}

        origin = None
        recon = None
        if self.get_recon_flag:
            self.hydra_log.debug(f'Epoch {self.current_epoch} - test loss: {loss:.4f} D: {penalty:.4f}')
            self.get_recon_flag = False
            x = batch
            with torch.no_grad():
                recon = self.decode(self.encode(x)).detach()
            origin = x.detach()

        # Progress Bar
        self.log("recon", loss, prog_bar=True, logger = False)
        self.log("penalty", penalty, prog_bar=True, logger = False)

        # TensorBoard
        self.log("test/recon", loss, on_step = False, on_epoch = True)
        self.log("test/penalty", penalty, on_step = False, on_epoch = True)

        # return {"loss": loss + self.lamb * penalty, "progress_bar": tqdm_dict}
        return {"loss": loss + self.lamb * penalty, "x":origin, "recon":recon}

    def validation_epoch_end(self, outputs) -> None:
        # sample reconstruction
        x = outputs[0]["x"]
        recon = outputs[0]["recon"]
        x_recon = torch.cat((x[0:32],recon[0:32]), dim = 0)
        self.get_recon_flag = True

        # sample generate
        z = self.z_sampler(64, self.z_dim).type_as(self.decoder[0].weight)
        gen_img = self.decode(z)
        if self.save_img_path is not None:
            save_sample_images(self.save_img_path, "recon", self.current_epoch, (((x_recon+1.)/2.).to('cpu').detach().numpy()[0:64]))
            save_sample_images(self.save_img_path, "gen", self.current_epoch, (((gen_img+1.)/2.).to('cpu').detach().numpy()[0:64]))

        # Tensorboard
        grid = torchvision.utils.make_grid((x_recon+1.)/2.)
        self.logger.experiment.add_image("reconstructed_images", grid, self.current_epoch)
        grid = torchvision.utils.make_grid((gen_img+1.)/2.)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

        
class WAE_GAN_abstract(AutoEncoder_abstract):
    def __init__(self, cfg, log, verbose = 1):
        super().__init__(cfg, log, verbose)
        try:
            self.gen_dataset = instantiate(cfg['train_info']['gen_data'])
            self.gen_dataloader = DataLoader(self.gen_dataset, 32, num_workers = 5, shuffle = True)
        except:
            pass

        self.z_sampler = getattr(sampler, cfg['train_info']['z_sampler'])
        self.lamb = float(cfg['train_info']['lambda'])

        self.lr_adv = float(cfg['train_info']['lr_adv'])
        self.beta1_adv = float(cfg['train_info']['beta1_adv'])

        self.disc = nn.Identity()
        self.disc_trainable = [self.disc]

    def log_architecture(self):
        self.hydra_log.info('------------------------------------------------------------')
        for net in self.encoder_trainable:
            self.hydra_log.info(net)
        for net in self.decoder_trainable:
            self.hydra_log.info(net)
        for net in self.disc_trainable:
            self.hydra_log.info(net)

    def discriminate(self, z):
        return self.disc(z)
    
    def _adv_loss(self, batch):
        x = batch
        q = self.encode(x)
        p = self.z_sampler(len(q), self.z_dim).type_as(q)
        pz = self.discriminate(p)
        qz = self.discriminate(q)
        return F.binary_cross_entropy_with_logits(pz, torch.ones_like(pz)) + F.binary_cross_entropy_with_logits(qz, torch.zeros_like(qz))

    def penalty_loss(self, q):
        qz = self.discriminate(q)
        return F.binary_cross_entropy_with_logits(qz, torch.ones_like(qz))

    def _get_losses(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x = batch  # When batch returns both image and label

        fake_latent = self.encode(x)
        x_hat = self.decode(fake_latent)

        loss = F.mse_loss(x, x_hat, reduction="none").sum(dim=[1, 2, 3]).mean(dim=[0])
        penalty = self.penalty_loss(fake_latent)
        return loss, penalty

    def configure_optimizers(self):
        opt1 = optim.Adam(sum([list(net.parameters()) for net in self.disc_trainable], []), lr = self.lr_adv, betas = (self.beta1_adv, 0.999))
        opt2 = optim.Adam(sum([list(net.parameters()) for net in self.encoder_trainable], []) + sum([list(net.parameters()) for net in self.decoder_trainable], []), lr = self.lr, betas = (self.beta1, 0.999))
        if self.lr_scheduler is None:
            return ({"optimizer": opt1}, {"optimizer":opt2})
        scheduler1 = scheduler_parse(opt1, self.lr_scheduler)
        scheduler2 = scheduler_parse(opt2, self.lr_scheduler)
        return ({"optimizer": opt1, "lr_scheduler": scheduler1}, {"optimizer":opt2, "lr_scheduler": scheduler2})

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            return self.lamb * self._adv_loss(batch)

        if optimizer_idx == 1:
            loss, penalty = self._get_losses(batch)

            if self.trainer.is_global_zero:
                # Progress Bar
                self.log("recon", loss, prog_bar=True, logger = False, rank_zero_only=True)
                self.log("penalty", penalty, prog_bar=True, logger = False, rank_zero_only=True)

                # TensorBoard
                self.log("train/recon", loss, on_step = False, on_epoch = True, rank_zero_only=True)
                self.log("train/penalty", penalty, on_step = False, on_epoch = True, rank_zero_only=True)

            return loss + self.lamb * penalty
    
    def validation_step(self, batch, batch_idx):
        loss, penalty= self._get_losses(batch)

        origin = None
        recon = None
        if self.get_recon_flag:
            self.hydra_log.debug(f'Epoch {self.current_epoch} - test loss: {loss:.4f} D: {penalty:.4f}')
            self.get_recon_flag = False
            x = batch
            with torch.no_grad():
                recon = self.decode(self.encode(x)).detach()
            origin = x.detach()
        
        # Progress Bar
        self.log("recon", loss, prog_bar=True, logger = False, sync_dist=True)
        if self.lamb > 0.0:
            self.log("penalty", penalty, prog_bar=True, logger = False, sync_dist=True)

        # TensorBoard
        self.log("test/recon", loss, on_step = False, on_epoch = True, sync_dist=True)
        if self.lamb > 0.0:
            self.log("test/penalty", penalty, on_step = False, on_epoch = True, sync_dist=True)

        # return loss + self.lamb * penalty
        return {"loss": loss + self.lamb * penalty, "x":origin, "recon":recon}

    def validation_epoch_end(self, outputs) -> None:
        # sample reconstruction
        x = outputs[0]["x"]
        recon = outputs[0]["recon"]
        x_recon = torch.cat((x[0:32],recon[0:32]), dim = 0)
        self.get_recon_flag = True

        # sample generate
        z = self.z_sampler(64, self.z_dim).type_as(self.decoder[0].weight)
        gen_img = self.decode(z)
        if self.save_img_path is not None:
            save_sample_images(self.save_img_path, "recon", self.current_epoch, (((x_recon+1.)/2.).to('cpu').detach().numpy()[0:64]))
            save_sample_images(self.save_img_path, "gen", self.current_epoch, (((gen_img+1.)/2.).to('cpu').detach().numpy()[0:64]))

        # Tensorboard
        grid = torchvision.utils.make_grid((x_recon+1.)/2.)
        self.logger.experiment.add_image("reconstructed_images", grid, self.current_epoch)
        grid = torchvision.utils.make_grid((gen_img+1.)/2.)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)