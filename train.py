"""
Training loop for the attribute-guided conditional GAN.

Usage:
    python train.py
    python train.py --epochs 30 --batch_size 64
    python train.py --resume ./checkpoints/ckpt_epoch010.pt
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import config
from data.dataset import CelebADataset
from models.discriminator import Discriminator
from models.generator import Generator
from utils.losses import discriminator_loss, generator_loss
from utils.visualization import save_sample_grid


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.num_epochs)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument("--resume", type=str, default=None,
                         help="path to a checkpoint .pt file to resume from")
    return parser.parse_args()


def main():
    args = get_args()
    torch.manual_seed(config.seed)
    device = config.device
    print(f"Using device: {device}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    dataset = CelebADataset(
        img_dir=config.img_dir,
        attr_path=config.attr_path,
        selected_attrs=config.selected_attrs,
        image_size=config.image_size,
        split="train",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Loaded {len(dataset)} training images, {config.num_attrs} attributes")

    G = Generator(config.latent_dim, config.num_attrs, config.g_conv_dim, config.channels).to(device)
    D = Discriminator(config.num_attrs, config.d_conv_dim, config.channels).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(config.beta1, config.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(config.beta1, config.beta2))

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume}, continuing at epoch {start_epoch}")

    # Fixed batch used every time we save a sample grid, so progress is
    # comparable across steps instead of looking at different random faces.
    fixed_z = torch.randn(64, config.latent_dim)
    fixed_attrs = torch.randint(0, 2, (64, config.num_attrs)).float()

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        for real_images, real_attrs in loader:
            real_images = real_images.to(device)
            real_attrs = real_attrs.to(device)
            batch_size = real_images.size(0)

            # ---------------- Train Discriminator ----------------
            z = torch.randn(batch_size, config.latent_dim, device=device)
            target_attrs = torch.randint(0, 2, (batch_size, config.num_attrs), device=device).float()
            fake_images = G(z, target_attrs)

            real_logit, real_attr_logit = D(real_images)
            fake_logit, _ = D(fake_images.detach())

            d_adv_loss, d_attr_loss = discriminator_loss(
                real_logit, fake_logit, real_attr_logit, real_attrs, config.label_smoothing
            )
            d_loss = d_adv_loss + config.lambda_attr * d_attr_loss

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # ---------------- Train Generator ----------------
            z = torch.randn(batch_size, config.latent_dim, device=device)
            target_attrs = torch.randint(0, 2, (batch_size, config.num_attrs), device=device).float()
            fake_images = G(z, target_attrs)
            fake_logit, fake_attr_logit = D(fake_images)

            g_adv_loss, g_attr_loss = generator_loss(fake_logit, fake_attr_logit, target_attrs)
            g_loss = g_adv_loss + config.lambda_attr * g_attr_loss

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            if global_step % 100 == 0:
                print(
                    f"epoch {epoch} step {global_step} | "
                    f"D: {d_loss.item():.4f} (adv {d_adv_loss.item():.4f}, attr {d_attr_loss.item():.4f}) | "
                    f"G: {g_loss.item():.4f} (adv {g_adv_loss.item():.4f}, attr {g_attr_loss.item():.4f})"
                )

            if global_step % config.sample_every == 0:
                save_sample_grid(G, fixed_z, fixed_attrs, epoch, global_step, config.sample_dir, device)

            global_step += 1

        if (epoch + 1) % config.checkpoint_every == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"ckpt_epoch{epoch + 1:03d}.pt")
            torch.save({
                "epoch": epoch,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
