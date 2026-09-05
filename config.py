"""
Central configuration for the attribute-guided face generation project.
Edit values here rather than hardcoding them in train.py / generate.py.
"""

import torch


class Config:
    # ---- Data ----
    img_dir = "./data/celeba/img_align_celeba"
    attr_path = "./data/celeba/list_attr_celeba.txt"
    image_size = 64
    channels = 3

    # Subset of the 40 official CelebA attributes we condition on.
    # Keep this list small while you're getting things working -
    # more attributes = harder conditioning task.
    selected_attrs = [
        "Male",
        "Young",
        "Smiling",
        "Eyeglasses",
        "Bald",
        "Black_Hair",
        "Blond_Hair",
        "Brown_Hair",
        "Mustache",
        "Wearing_Hat",
    ]
    num_attrs = len(selected_attrs)

    # ---- Model ----
    latent_dim = 100
    g_conv_dim = 64
    d_conv_dim = 64

    # ---- Training ----
    batch_size = 128
    num_epochs = 50
    lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    lambda_attr = 1.0       # weight on the attribute loss relative to adversarial loss
    label_smoothing = 0.9   # one-sided smoothing for "real" labels, helps D not overpower G

    # ---- Misc ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    sample_every = 500      # save a sample grid every N steps
    checkpoint_every = 5    # save a checkpoint every N epochs
    sample_dir = "./samples"
    checkpoint_dir = "./checkpoints"
    seed = 42


config = Config()
