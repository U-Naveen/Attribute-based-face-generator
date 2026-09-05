"""
CelebA dataset loader.

Reads the official list_attr_celeba.txt annotation file (format:
line 1 = image count, line 2 = the 40 attribute names, remaining lines =
"filename val1 val2 ... val40" with vals in {-1, 1}) and returns
(image_tensor, attribute_vector) pairs for a chosen subset of attributes.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_path, selected_attrs, image_size=64,
                 split="train", split_ratio=0.98):
        self.img_dir = img_dir
        self.selected_attrs = selected_attrs

        with open(attr_path, "r") as f:
            lines = f.readlines()

        all_attr_names = lines[1].split()
        rows = [line.split() for line in lines[2:] if line.strip()]
        filenames = [row[0] for row in rows]

        values = np.array([[int(v) for v in row[1:]] for row in rows], dtype=np.float32)
        values = (values + 1) / 2  # map {-1, 1} -> {0, 1}

        missing = [a for a in selected_attrs if a not in all_attr_names]
        if missing:
            raise ValueError(
                f"Unknown attribute name(s) {missing}. "
                f"Available attributes: {all_attr_names}"
            )
        attr_idx = [all_attr_names.index(a) for a in selected_attrs]
        values = values[:, attr_idx]

        split_point = int(len(filenames) * split_ratio)
        if split == "train":
            self.filenames = filenames[:split_point]
            self.labels = values[:split_point]
        elif split == "val":
            self.filenames = filenames[split_point:]
            self.labels = values[split_point:]
        else:
            raise ValueError(f"Unknown split '{split}', expected 'train' or 'val'")

        # CelebA aligned images are 178x218; center-crop to a square before resizing
        # so faces aren't squished, then normalize to [-1, 1] to match the
        # generator's Tanh output.
        self.transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = torch.from_numpy(self.labels[idx])
        return image, label
