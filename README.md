# Attribute-guided face generation

A conditional GAN that generates synthetic faces with user-controlled
attributes (gender, age, hair color, smile, glasses, etc.), trained on
CelebA.

This code has been tested end-to-end (forward pass, backward pass,
optimizer step, checkpoint save/resume, and image generation) on a
synthetic dataset. You just need to plug in real CelebA data.

## How it works

- **Generator**: takes a random noise vector `z` and an attribute vector
  `a`, concatenates them, and upsamples through transposed convolutions
  into a 64x64 RGB image.
- **Discriminator**: a shared convolutional trunk with two output heads -
  one predicts real vs. fake, the other predicts which attributes are
  present. This lets the same network both judge realism and enforce
  that requested attributes actually show up.
- **Losses**: adversarial loss (BCE on the real/fake logit) trains
  realism; attribute loss (multi-label BCE) trains attribute
  consistency, computed on real images against true labels and on fake
  images against the attributes the generator was asked to produce.

## Project structure

```
attribute-guided-face-gan/
├── config.py              # all hyperparameters and paths, edit here
├── data/
│   └── dataset.py         # CelebADataset: pairs images with attribute vectors
├── models/
│   ├── generator.py       # Generator network
│   └── discriminator.py   # Discriminator network (real/fake + attribute heads)
├── utils/
│   ├── losses.py          # adversarial + attribute loss functions
│   └── visualization.py   # sample grid saving, attribute sweep
├── train.py                # training loop
├── generate.py              # generate faces from a trained checkpoint
├── checkpoints/            # saved model weights land here
├── samples/                # training-progress sample grids land here
└── requirements.txt
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download CelebA (aligned & cropped version) and the attribute
   annotation file `list_attr_celeba.txt`. The official source is the
   CelebA project page (search "CelebA dataset MMLAB"); a Kaggle mirror
   also works. Arrange the files like this:
   ```
   data/celeba/
   ├── img_align_celeba/       # all the .jpg images, unzipped
   └── list_attr_celeba.txt
   ```
   These paths match the defaults in `config.py` - edit `img_dir` and
   `attr_path` there if you place the data elsewhere.

3. (Optional) Edit `selected_attrs` in `config.py` to change which of
   the 40 CelebA attributes you condition on. The default 10 (gender,
   age, smile, glasses, baldness, three hair colors, mustache, hat)
   are visually distinct and a good starting set. Fewer attributes =
   an easier, more reliable training run.

## Training

```
python train.py
python train.py --epochs 30 --batch_size 64 --lr 2e-4
python train.py --resume ./checkpoints/ckpt_epoch010.pt
```

Sample image grids are written to `samples/` every `sample_every` steps
(see `config.py`) using a **fixed** noise+attribute batch, so you can
watch the same faces improve over training instead of comparing
different random faces each time. Checkpoints are written to
`checkpoints/` every `checkpoint_every` epochs.

Watch the printed loss values: if `D`'s adversarial loss collapses
toward 0 while `G`'s adversarial loss climbs, the discriminator is
overpowering the generator - try lowering D's learning rate, adding
more label smoothing, or training G more often than D.

## Generating faces

```
python generate.py --checkpoint ./checkpoints/ckpt_epoch050.pt \
    --attrs "Male,Smiling,Eyeglasses" --num_samples 8 --out generated.png
```

`--attrs` is a comma-separated list of attribute names (must match
entries in `config.selected_attrs`); any attribute not listed is left
off. Unrecognized names are skipped with a warning rather than failing.

## Suggested path forward

1. Get this training on real CelebA at 64x64 first, with the default
   10 attributes - confirm losses are stable and generated faces look
   face-like before doing anything else.
2. Use `utils.visualization.attribute_sweep` to fix noise `z` and sweep
   one attribute at a time - this is the fastest way to check whether
   the model has actually learned to isolate that attribute rather than
   entangling it with others.
3. Once stable, try 128x128 (you'll need to add another
   upsample/downsample layer to both networks), add more attributes, or
   move to an image-to-image approach like AttGAN/StarGAN if you want
   to edit existing photos rather than generate from pure noise.
