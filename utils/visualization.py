"""
Helpers for visually tracking training progress.
"""

import os

import torch
import torchvision.utils as vutils


def save_sample_grid(generator, fixed_z, fixed_attrs, epoch, step, sample_dir, device):
    """Generates a grid from a fixed noise+attribute batch so you can watch
    the SAME faces improve over training, instead of a new random batch
    each time (which makes progress hard to judge by eye)."""
    generator.eval()
    with torch.no_grad():
        fakes = generator(fixed_z.to(device), fixed_attrs.to(device))
    generator.train()

    fakes = (fakes + 1) / 2  # [-1, 1] -> [0, 1] for saving as an image
    os.makedirs(sample_dir, exist_ok=True)
    out_path = os.path.join(sample_dir, f"epoch{epoch:03d}_step{step:06d}.png")
    vutils.save_image(fakes, out_path, nrow=8, padding=2)
    return out_path


def attribute_sweep(generator, z, base_attrs, attr_index, device, steps=5):
    """Fixes noise z and all attributes except one, sweeping that single
    attribute from 0 to 1. If the model has learned disentangled control,
    only that one visual feature should change across the row."""
    generator.eval()
    frames = []
    with torch.no_grad():
        for i in range(steps):
            attrs = base_attrs.clone()
            attrs[:, attr_index] = i / (steps - 1)
            frames.append(generator(z.to(device), attrs.to(device)))
    generator.train()
    return torch.cat(frames, dim=0)
