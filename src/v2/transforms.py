"""
v2 image pipeline.

Changes from v1 (src/transforms.py) and why:
- v1 squashes 600x450 dermoscopy images to 224x224, distorting lesion shape
  and throwing away detail. v2 trains on RandomResizedCrop at a higher
  resolution (384 by default) and evaluates on resize-shorter-side + center
  crop, so the aspect ratio is kept.
- Shades of Gray colour constancy (Finlayson & Trezzi 2004), applied the same
  way at train, eval and serve time. Dermatoscopes differ in illumination and
  white balance; ISIC 2018 test images come partly from other devices than
  HAM10000's training images, which is one plausible cause of v1's internal
  (0.72) vs external (0.65) balanced-accuracy drop. It is cheap and toggleable.
- Stronger augmentation: full dihedral symmetry (lesions have no "up"), mild
  affine, blur, and random erasing.
- Test-time augmentation over the 8 dihedral transforms, averaged in logit
  space so downstream temperature scaling still applies.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ShadesOfGray:
    """Colour constancy with Minkowski norm p (p=6 is the common choice for
    dermoscopy). Estimates the illuminant per channel and rescales so it is
    neutral. Deterministic, so it is safe to apply identically everywhere."""

    def __init__(self, power: int = 6):
        self.power = power

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img, dtype=np.float32)
        illum = np.power(np.mean(np.power(arr, self.power), axis=(0, 1)), 1.0 / self.power)
        norm = np.sqrt(np.sum(illum ** 2))
        if norm <= 1e-6:
            return img
        illum = illum / norm
        arr = arr / (illum * np.sqrt(3.0) + 1e-6)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class RandomDihedral:
    """One of the 8 rotations/flips of the square, uniformly."""

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        if k:
            img = img.rotate(90 * k, expand=True)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


def _prefix(color_constancy: bool) -> list:
    return [ShadesOfGray()] if color_constancy else []


def get_train_transforms(image_size: int = 384, color_constancy: bool = True):
    return T.Compose(_prefix(color_constancy) + [
        RandomDihedral(),
        T.RandomResizedCrop(image_size, scale=(0.35, 1.0), ratio=(0.75, 1.333)),
        T.RandomAffine(degrees=15, translate=(0.05, 0.05), shear=5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.12), value=0),
    ])


def get_eval_transforms(image_size: int = 384, color_constancy: bool = True):
    return T.Compose(_prefix(color_constancy) + [
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def dihedral_views(images: torch.Tensor, n_views: int = 8):
    """Yields up to 8 dihedral variants of an NCHW batch (square images)."""
    views = []
    for k in range(4):
        rotated = torch.rot90(images, k, dims=(2, 3))
        views.append(rotated)
        views.append(torch.flip(rotated, dims=(3,)))
    return views[:n_views]
