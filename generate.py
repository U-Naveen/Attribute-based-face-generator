"""
Generate faces from a trained checkpoint with specific requested attributes.

Usage:
    python generate.py --checkpoint ./checkpoints/ckpt_epoch050.pt \
        --attrs "Male,Smiling,Eyeglasses" --num_samples 8 --out generated.png
"""

import argparse

import torch
import torchvision.utils as vutils

from config import config
from models.generator import Generator


def parse_attr_string(attr_string, selected_attrs):
    """Turns 'Male,Smiling,Eyeglasses' into a binary vector matching the
    order of config.selected_attrs. Unrecognized names are ignored with a warning."""
    requested = set(a.strip() for a in attr_string.split(",")) if attr_string else set()
    unknown = requested - set(selected_attrs)
    if unknown:
        print(f"Warning: ignoring unknown attribute name(s): {sorted(unknown)}")
    vector = [1.0 if attr in requested else 0.0 for attr in selected_attrs]
    return torch.tensor(vector).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--attrs", type=str, default="",
                         help="comma-separated attribute names to turn on, e.g. 'Male,Smiling'")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--out", type=str, default="generated.png")
    args = parser.parse_args()

    device = config.device
    G = Generator(config.latent_dim, config.num_attrs, config.g_conv_dim, config.channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    attr_vector = parse_attr_string(args.attrs, config.selected_attrs).to(device)
    attr_vector = attr_vector.repeat(args.num_samples, 1)
    z = torch.randn(args.num_samples, config.latent_dim, device=device)

    with torch.no_grad():
        fakes = G(z, attr_vector)
    fakes = (fakes + 1) / 2

    vutils.save_image(fakes, args.out, nrow=4, padding=2)
    print(f"Saved {args.num_samples} samples to {args.out}")


if __name__ == "__main__":
    main()
