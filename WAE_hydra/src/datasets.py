import os
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ToyDataset(LightningDataModule):
    def __init__(self, train_mat, valid_mat, batch_size: int = 512):
        super().__init__()
        self.batch_size = batch_size
        self.train_mat = train_mat
        self.valid_mat = valid_mat
    def train_dataloader(self):
        return DataLoader(self.train_mat, self.batch_size, shuffle = True, drop_last=True)
    def val_dataloader(self):
        return DataLoader(self.valid_mat, self.batch_size, shuffle = False, drop_last=True)

class MNIST(Dataset):
    def __init__(self, data_home, train = True, output_channels = 1, portion = 1.0, class_no = False):
        # self.label = label
        self.class_no = class_no
        self.output_channels = output_channels
        if train:
            self.data = np.loadtxt('%s/mnist_train.csv' % data_home, delimiter=',', skiprows = 1)
        else:
            self.data = np.loadtxt('%s/mnist_test.csv' % data_home, delimiter=',', skiprows = 1)

        if portion < 1.0:
            k = int(np.shape(self.data)[0]*portion)
            self.data[k:, 0] = 10

        self.code = torch.from_numpy(np.concatenate([np.eye(10), np.zeros((1,10))], axis = 0)).type(torch.float32)
        
    def __len__(self):
        return np.shape(self.data)[0]
    
    def __getitem__(self, idx):
        # if self.label:
        if self.class_no:
            y = torch.Tensor([self.data[idx, 0]]).type(torch.long)
        else :
            y = self.code[self.data[idx, 0].astype(np.int)]
        if self.output_channels > 1:
            return [torch.from_numpy(self.data[idx, 1:785]/255).reshape((1,28,28)).type(torch.float32).repeat((self.output_channels,1,1)), y]
        else:
            return [torch.from_numpy(self.data[idx, 1:785]/255).reshape((1,28,28)).type(torch.float32), y]

class MNIST_pl(LightningDataModule):
    def __init__(self, data_dir: str = '.', batch_size: int = 100, class_no: bool = False):
        super().__init__()
        self.class_no = class_no
        self.data_dir = data_dir
        self.num_classes = 10
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = MNIST(self.data_dir, train=True, class_no = self.class_no)
        self.val_dataset = MNIST(self.data_dir, train=False, class_no = self.class_no)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, num_workers = 5, shuffle = True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, num_workers = 5, shuffle = False, drop_last=True)

class CelebA(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, transform=None, partition_file_path="list_eval_partition.csv"):
        self.partition_file = pd.read_csv('%s/%s' % (data_dir, partition_file_path))
        self.data_dir = data_dir
        self.split = split # 0: train / 1: validation / 2: test
        self.transform = transform
        self.partition_file_sub = self.partition_file[self.partition_file["partition"].isin(self.split)]
    
    def __len__(self):
        return len(self.partition_file_sub)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, 'img_align_celeba', self.partition_file_sub.iloc[idx, 0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

class CelebA_pl(LightningDataModule):
    def __init__(self, data_dir: str = '/home/hjlee/data/CelebA', batch_size: int = 100):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        transform = transforms.Compose([
            # transforms.CenterCrop((140, 140)),
            transforms.CenterCrop((128, 128)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.train_dataset = CelebA(self.data_dir, split=[0], transform=transform)
        self.val_dataset = CelebA(self.data_dir, split=[1], transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, num_workers = 5, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, num_workers = 5, shuffle = False)
