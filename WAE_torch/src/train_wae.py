import os, argparse
import numpy as np
from tqdm import tqdm

from utils.tools import *
from utils.losses import *
from utils.dataloader import *
from models.celeba import *

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

# Parsing arguments
parser = argparse.ArgumentParser(description="CelebA WAE Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-dir", type=str, default="/home/hjlee/data/CelebA")
parser.add_argument("--resolution", type=int, default=64)
parser.add_argument("--crop-size", type=int, default=140)
parser.add_argument("--model", type=str, default="wae_mmd", help="choose among 2 models: wae_mmd, wae_gan.")
parser.add_argument("--random-seed", type=int, default=2022)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--adv-learning-rate", type=float, default=1e-4)
parser.add_argument("--learning-schedule", type=str, default="none", help="learning rate scheduler: none, manual")
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument("--z-dim", type=int, default=64)
parser.add_argument("--h-dim", type=int, default=128)
parser.add_argument("--lambda_mmd", type=float, default=100.0)
parser.add_argument("--lambda_gan", type=float, default=1.0)
parser.add_argument("--checkpoint", type=int, default=0)
parser.add_argument("--cuda", type=int, default=0, help="Choose a gpu")
args = parser.parse_args()

# Make directories
os.makedirs(f"../outputs/CelebA/{args.model}/Images/Sample", exist_ok=True)
os.makedirs(f"../outputs/CelebA/{args.model}/Images/Recon", exist_ok=True)
os.makedirs(f"../outputs/CelebA/{args.model}/Models", exist_ok=True)
os.makedirs(f"../outputs/CelebA/{args.model}/Losses/train", exist_ok=True)
os.makedirs(f"../outputs/CelebA/{args.model}/Losses/val", exist_ok=True)

# Setting torch
torch.manual_seed(args.random_seed)
device = torch.device("cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu")

adv_train = 'gan' in args.model.lower() # Check adversarial training
penalty_keys = {"wae_mmd": ["mmd_penalty"], "wae_gan": ["gan_penalty"]}
disc_keys = {"wae_mmd": [], "wae_gan": ["disc_loss"]}

# initialize models (encoder, decoder)
encoder = Encoder(z_dim=args.z_dim, d=args.h_dim).to(device)
decoder = Decoder(z_dim=args.z_dim, d=args.h_dim).to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)
if args.learning_schedule == "manual": # set up schedulers
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (0.5**(epoch >= 30)) * (0.2**(epoch >= 50)) * (0.1**(epoch >= 100)))

# discriminator
if adv_train:
    disc = Discriminator().to(device)
    disc_optimizer = optim.Adam(disc.parameters(), lr=args.adv_learning_rate)
    if args.learning_schedule == "manual":
        disc_scheduler = optim.lr_scheduler.LambdaLR(disc_optimizer, lr_lambda=lambda epoch: (0.5**(epoch >= 30)) * (0.2**(epoch >= 50)) * (0.1**(epoch >= 100)))

# Load datasets
transform = transforms.Compose([
    # transforms.CenterCrop((140, 140)),
    transforms.CenterCrop((args.crop_size, args.crop_size)),
    transforms.Resize((args.resolution, args.resolution)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = CelebDataset(data_dir=args.data_dir, split=[0], transform=transform)
val_set = CelebDataset(data_dir=args.data_dir, split=[1], transform=transform)

# test_celeba_batcher = CelebDataset(data_dir=args.data_dir, split=[2], transform=transform)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)

print(f"Train data : {len(train_set)}\nValidation data : {len(val_set)}\nModel: {args.model}")

# initial empty lists for training progress
train_losses = {"recon": [], "mmd_penalty": [], "gan_penalty": [], "disc_loss": [], "gradient_penalty": []}
val_losses = {"recon": [], "mmd_penalty": [], "gan_penalty": [], "disc_loss": [], "gradient_penalty": [], "penalty": []}

z_sample = torch.randn(64, args.z_dim).to(device) # random z sample for generation
best_val_losses = np.Inf
best_epoch = 0

# Load checkpoint
if args.checkpoint != 0:
    # load model from checkpoints
    encoder.load_state_dict(torch.load(f"../outputs/CelebA/{args.model}/Models/netQ_{args.checkpoint}"))
    decoder.load_state_dict(torch.load(f"../outputs/CelebA/{args.model}/Models/netG_{args.checkpoint}"))
    if adv_train:
        disc.load_state_dict(torch.load(f"../outputs/CelebA/{args.model}/Models/netD_{args.checkpoint}"))

    # load best losses
    with open(f"../outputs/CelebA/{args.model}/Losses/train/losses.pkl", "rb") as f:
        train_losses = pickle.load(f)
    with open(f"../outputs/CelebA/{args.model}/Losses/val/losses.pkl", "rb") as f:
        val_losses = pickle.load(f)
    best_val_losses = np.min(np.array(val_losses["recon"]) + np.array(val_losses["penalty"]))
    best_epoch = np.argmin(np.array(val_losses["recon"]) + np.array(val_losses["penalty"]))


for epoch in tqdm(range(args.checkpoint, args.epochs)):
    train_recon_loss = inc_avg()
    train_gan_loss = inc_avg()
    train_mmd_loss = inc_avg()
    train_disc_loss = inc_avg()

    val_recon_loss = inc_avg()
    val_gan_loss = inc_avg()
    val_mmd_loss = inc_avg()
    val_disc_loss = inc_avg()
    val_penalty_loss = inc_avg()
    
    # =========================
    # Train step
    encoder.train()
    decoder.train()
    for _data in train_loader:
        real_data = torch.Tensor(_data).to(device)
        
        # Update encoder, decoder
        for p in list(encoder.parameters()) + list(decoder.parameters()):
            p.requires_grad = True
        if adv_train:
            for p in disc.parameters():
                p.requires_grad = False
        optimizer.zero_grad()
        
        # Reconstruction loss
        z_encoded = encoder(real_data)
        post_data = decoder(z_encoded)
        l2 = torch.sqrt(torch.sum((real_data-post_data)**2, dim=list(range(1, len(post_data.shape)))))
        recon = l2.mean()
        
        train_recon_loss.append(recon.cpu().item(), real_data.size(0)) # save loss

        # penalty
        z_prior = torch.randn(real_data.size(0), args.z_dim).to(device)
        if args.model.lower() == "wae_mmd":
            mmd = mmd_penalty(z_encoded, z_prior)
            penalty = args.lambda_mmd * mmd
            train_mmd_loss.append(mmd.cpu().item(), real_data.size(0)) # save loss
        else: # wae_gan
            qz = disc(z_encoded)
            gan_penalty = F.binary_cross_entropy_with_logits(qz, torch.ones_like(qz)) # gan loss for encoder, decoder
            penalty = args.lambda_gan * gan_penalty
            train_gan_loss.append(gan_penalty.cpu().item(), real_data.size(0)) # save loss
        
        losses = recon + penalty
        losses.backward()
        optimizer.step()

        if adv_train:
            for p in disc.parameters():
                p.requires_grad = True
            for p in list(encoder.parameters()) + list(decoder.parameters()):
                p.requires_grad = False
            
            disc_optimizer.zero_grad()
            z_prior = torch.randn(real_data.size(0), args.z_dim).to(device)
            
            z_encoded = encoder(real_data)
            pz = disc(z_prior)
            qz = disc(z_encoded)
            disc_loss = F.binary_cross_entropy_with_logits(pz, torch.ones_like(pz)) + \
                F.binary_cross_entropy_with_logits(qz, torch.zeros_like(qz)) # gan loss for discriminator
            adv_loss = disc_loss
            train_disc_loss.append(disc_loss.cpu().item(), real_data.size(0)) # save loss
            adv_loss.backward()
            disc_optimizer.step()
    
    if args.learning_schedule != "none":
        scheduler.step()
        if adv_train:
            disc_scheduler.step()

    # ===========================
    # Validation step
    encoder.eval()
    decoder.eval()
    for _data in val_loader:
        # reconstruction
        real_data = torch.Tensor(_data).to(device)
        z_encoded = encoder(real_data)
        post_data = decoder(z_encoded)
        l2 = torch.sqrt(torch.sum((real_data-post_data)**2, dim=list(range(1, len(post_data.shape)))))
        recon = l2.mean() # recon error

        val_recon_loss.append(recon.cpu().item(), real_data.size(0)) # save loss

        # penalty
        z_prior = torch.randn(real_data.size(0), args.z_dim).to(device)
        if args.model == "wae_mmd":
            mmd = mmd_penalty(z_encoded, z_prior)
            penalty = args.lambda_mmd * mmd
            val_mmd_loss.append(mmd.cpu().item(), real_data.size(0)) # save loss
        else: # wae_gan
            qz = disc(z_encoded)
            gan_penalty = F.binary_cross_entropy_with_logits(qz, torch.ones_like(qz))
            pz = disc(z_prior)
            disc_loss = F.binary_cross_entropy_with_logits(pz, torch.ones_like(pz)) + \
                F.binary_cross_entropy_with_logits(qz, torch.zeros_like(qz))
            penalty = args.lambda_gan * gan_penalty
            # save loss
            val_gan_loss.append(gan_penalty.cpu().item(), real_data.size(0)) # save loss
            val_disc_loss.append(disc_loss.cpu().item(), real_data.size(0)) # save loss

    # Save losses in each epoch
    train_losses["recon"].append(train_recon_loss.avg)
    train_losses["mmd_penalty"].append(train_mmd_loss.avg)
    train_losses["gan_penalty"].append(train_gan_loss.avg)
    train_losses["disc_loss"].append(train_disc_loss.avg)
    
    val_losses["recon"].append(val_recon_loss.avg)
    val_losses["mmd_penalty"].append(val_mmd_loss.avg)
    val_losses["gan_penalty"].append(val_gan_loss.avg)
    val_losses["disc_loss"].append(val_disc_loss.avg)

    # Save best model
    curr_val_losses = val_recon_loss.avg + val_penalty_loss.avg
    if best_val_losses > curr_val_losses:
        best_val_losses = curr_val_losses
        best_epoch = epoch
        torch.save(encoder.state_dict(), f"../outputs/CelebA/{args.model}/Models/best_enc")
        torch.save(decoder.state_dict(), f"../outputs/CelebA/{args.model}/Models/best_dec")
        if adv_train:
            torch.save(disc.state_dict(), f"../outputs/CelebA/{args.model}/Models/best_disc")

    # Save image
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            torchvision.utils.save_image((decoder(z_sample)+1)/2., f"../outputs/CelebA/{args.model}/Images/Sample/fake_{epoch+1}.png", nrow=8)
            torchvision.utils.save_image(torch.cat(((real_data[:32,:,:,:]+1)/2., (decoder(encoder(real_data))[:32,:,:,:]+1)/2.), dim=0),
                                        f"../outputs/CelebA/{args.model}/Images/Recon/recon_{epoch+1}.png", nrow=8)
        
        # torch.save(decoder.state_dict(), f"../outputs/CelebA/{args.model}/Models/dec_{epoch+1}")
        # torch.save(encoder.state_dict(), f"../outputs/CelebA/{args.model}/Models/enc_{epoch+1}")
        # if adv_train:
        #     torch.save(disc.state_dict(), f"../outputs/CelebA/{args.model}/Models/disc_{epoch+1}")

# Save losses history
# train
plot_losses(train_losses, keys=["recon"],
            path=f"../outputs/CelebA/{args.model}/Losses/train/recon")
plot_losses(train_losses, keys=penalty_keys[args.model],
            path=f"../outputs/CelebA/{args.model}/Losses/train/penalty")
plot_losses(train_losses, keys=disc_keys[args.model],
            path=f"../outputs/CelebA/{args.model}/Losses/train/disc")
# val
plot_losses(val_losses, keys=["recon"],
            path=f"../outputs/CelebA/{args.model}/Losses/val/recon")
plot_losses(val_losses, keys=penalty_keys[args.model],
            path=f"../outputs/CelebA/{args.model}/Losses/val/penalty")
plot_losses(val_losses, keys=disc_keys[args.model],
            path=f"../outputs/CelebA/{args.model}/Losses/val/disc")

f = open(f"../outputs/CelebA/{args.model}/Losses/train/losses.pkl", "wb")
pickle.dump(train_losses, f)
f.close()

f = open(f"../outputs/CelebA/{args.model}/Losses/val/losses.pkl", "wb")
pickle.dump(val_losses, f)
f.close()

print(f"Best model checkpoint: {best_epoch+1}")
