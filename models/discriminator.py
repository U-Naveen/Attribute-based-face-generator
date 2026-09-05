"""
Discriminator: a convolutional trunk shared by two heads -
1. real_fake_head: a single logit, is this image real or generated?
2. attr_head: one logit per attribute, does this image show each attribute?

Spectral normalization is used instead of batch norm to keep training
stable without needing careful batch-norm statistics for fake batches.
"""

import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self, num_attrs=10, conv_dim=64, channels=3):
        super().__init__()

        self.trunk = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, conv_dim, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (conv_dim) x 32 x 32
            spectral_norm(nn.Conv2d(conv_dim, conv_dim * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (conv_dim*2) x 16 x 16
            spectral_norm(nn.Conv2d(conv_dim * 2, conv_dim * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (conv_dim*4) x 8 x 8
            spectral_norm(nn.Conv2d(conv_dim * 4, conv_dim * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (conv_dim*8) x 4 x 4
        )

        self.real_fake_head = nn.Conv2d(conv_dim * 8, 1, 4, 1, 0, bias=False)
        self.attr_head = nn.Conv2d(conv_dim * 8, num_attrs, 4, 1, 0, bias=False)

    def forward(self, x):
        features = self.trunk(x)
        real_fake_logit = self.real_fake_head(features).view(x.size(0), -1)
        attr_logit = self.attr_head(features).view(x.size(0), -1)
        return real_fake_logit, attr_logit
