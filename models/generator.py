"""
Generator: concatenates a noise vector z with an attribute vector a, then
upsamples through transposed convolutions into a 64x64 RGB image.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_attrs=10, conv_dim=64, channels=3):
        super().__init__()
        input_dim = latent_dim + num_attrs

        self.net = nn.Sequential(
            # input: (input_dim) x 1 x 1
            nn.ConvTranspose2d(input_dim, conv_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(conv_dim * 8),
            nn.ReLU(inplace=True),
            # state: (conv_dim*8) x 4 x 4
            nn.ConvTranspose2d(conv_dim * 8, conv_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(inplace=True),
            # state: (conv_dim*4) x 8 x 8
            nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(inplace=True),
            # state: (conv_dim*2) x 16 x 16
            nn.ConvTranspose2d(conv_dim * 2, conv_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(inplace=True),
            # state: (conv_dim) x 32 x 32
            nn.ConvTranspose2d(conv_dim, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # output: (channels) x 64 x 64
        )

    def forward(self, z, attrs):
        x = torch.cat([z, attrs], dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return self.net(x)
