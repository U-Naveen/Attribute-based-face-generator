"""
Loss functions for the conditional GAN.

Adversarial loss: standard non-saturating GAN loss via BCEWithLogitsLoss
(numerically stable, works directly on raw discriminator logits).

Attribute loss: multi-label binary cross-entropy, applied to real images
against their true labels (trains D's classifier head) and to fake
images against the attribute vector the generator was asked to produce
(pushes G toward those attributes).
"""

import torch
import torch.nn as nn

_adv_criterion = nn.BCEWithLogitsLoss()
_attr_criterion = nn.BCEWithLogitsLoss()


def discriminator_loss(real_logit, fake_logit, real_attr_logit, real_attrs,
                        label_smoothing=0.9):
    real_labels = torch.full_like(real_logit, label_smoothing)
    fake_labels = torch.zeros_like(fake_logit)

    adversarial_loss = _adv_criterion(real_logit, real_labels) + \
        _adv_criterion(fake_logit, fake_labels)
    attribute_loss = _attr_criterion(real_attr_logit, real_attrs)

    return adversarial_loss, attribute_loss


def generator_loss(fake_logit, fake_attr_logit, target_attrs):
    real_labels = torch.ones_like(fake_logit)
    adversarial_loss = _adv_criterion(fake_logit, real_labels)
    attribute_loss = _attr_criterion(fake_attr_logit, target_attrs)
    return adversarial_loss, attribute_loss
