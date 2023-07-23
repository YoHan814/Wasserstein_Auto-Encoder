import torch
import torch.nn as nn
import torch.nn.functional as F

class res_block(nn.Module):
    def __init__(self, channel, kernel_size, leak = False):
        super().__init__()
        d = channel
        if leak:
            self.conv = nn.Sequential(
                nn.Conv2d(d, d, kernel_size = kernel_size, padding = 'same'),
                nn.BatchNorm2d(d),
                nn.LeakyReLU(True),

                nn.Conv2d(d, d, kernel_size = kernel_size, padding = 'same'),
                nn.BatchNorm2d(d),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(d, d, kernel_size = kernel_size, padding = 'same'),
                nn.BatchNorm2d(d),
                nn.ReLU(True),

                nn.Conv2d(d, d, kernel_size = kernel_size, padding = 'same'),
                nn.BatchNorm2d(d),
            )
        self.skip = nn.Conv2d(d, d, kernel_size = 1)

    def forward(self, x):
        return F.relu(self.skip(x) + self.conv(x))

class prob_block(nn.Module):
    def __init__(self, channel_in, channel_out, sep = False):
        super().__init__()
        self.sep = sep
        self.mu = nn.Linear(channel_in, channel_out)
        self.logvar = nn.Linear(channel_in, channel_out)

    # def forward(self, x):
    #     if self.sep:
    #         return self.mu(x), self.logvar(x)
    #     else:
    #         mean = self.mu(x)
    #         return mean + torch.exp(self.logvar(x)) * torch.randn_like(mean)

    def forward(self, x):
        mean = self.mu(x)
        return mean + torch.exp(self.logvar(x)) * torch.randn_like(mean)