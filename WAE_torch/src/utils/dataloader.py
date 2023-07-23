import os, gzip, pickle
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# MNIST Dataset
class MNISTDataset(Dataset):
    def __init__(self, split, data_path='/home/hjlee/data/MNIST/mnist.pkl.gz', label=False):
        self.data_path = data_path
        self.label = label
        # split (0: train / 1: validation / 2: test)
        with gzip.open(self.data_path, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')[split]

    def __len__(self):
        return len(self.data[1])

    def __getitem__(self, idx):
        images, targets = self.data
        if not self.label:
            return torch.from_numpy(2.*images[idx, :]-1.) .reshape((1,28,28)).type(torch.float32)
        else:
            return torch.from_numpy(2.*images[idx, :]-1.) .reshape((1,28,28)).type(torch.float32), targets[idx]

# CelebA Dataset
# data source: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
class CelebDataset(Dataset):
    def __init__(self, data_dir, split, transform=None, partition_file_path="list_eval_partition.csv"):
        self.partition_file = pd.read_csv('%s/%s' % (data_dir, partition_file_path))
        self.data_dir = data_dir
        self.split = split # [0]: train / [1]: validation / [2]: test
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
