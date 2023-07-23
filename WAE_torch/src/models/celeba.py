import torch.nn as nn

def init_params(m):
    stddev = 0.0099999
    if type(m) == nn.Conv2d:
        nn.init.trunc_normal_(m.weight, 0.0, stddev, -2*stddev, 2*stddev)
    elif type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, 0.0, stddev)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, 0.0, stddev)

class Encoder(nn.Module):
    def __init__(self, z_dim=64, d=128):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, d, 5, 2, 2),
            nn.BatchNorm2d(d),
            nn.ReLU(),
            nn.Conv2d(d, 2*d, 5, 2, 2),
            nn.BatchNorm2d(2*d),
            nn.ReLU(),
            nn.Conv2d(2*d, 4*d, 5, 2, 2),
            nn.BatchNorm2d(4*d),
            nn.ReLU(),
            nn.Conv2d(4*d, 8*d, 5, 2, 2),
            nn.BatchNorm2d(8*d),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*4*8*d, z_dim)
        )

        self.model.apply(init_params)

    def forward(self, inputs):
        return self.model(inputs)

class Decoder(nn.Module):
    def __init__(self, z_dim=64, d=128):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 8*8*8*d),
            nn.ReLU(True),
            nn.Unflatten(1, (8*d, 8, 8)),

            nn.ConvTranspose2d(8*d, 4*d, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(4*d),
            nn.ReLU(), # 512 x 16 x 16
            nn.ConvTranspose2d(4*d, 2*d, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(2*d),
            nn.ReLU(), # 256 x 32 x 32
            nn.ConvTranspose2d(2*d, d, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(), # 128 x 64 x 64
            nn.ConvTranspose2d(d, 3, 1, 1)
        )
        self.decoder.apply(init_params)
    
    def forward(self, x):
        return self.decoder(x)

class Discriminator(nn.Module):
    def __init__(self, z_dim=64, d=512):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )
    
        self.model.apply(init_params)

    def forward(self, input):
        output = self.model(input)
        return output.view(-1)