#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Callable, Union

from torch import Tensor
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

import random
import torchvision.transforms.functional as TF


def make_dataset(root, subset, use_every: int = 1) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))[::use_every]

class SliceDataset(Dataset):
    """
    - If augment=True and subset in {'train','val'}:
        * __len__ returns 2 * N  (N = #files)
        * First N indices -> original samples
        * Last  N indices -> augmented copies
    """
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False, use_every: int = 1):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset, use_every=use_every)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        # In test mode or when augmentation is disabled, length is unchanged
        if self.test_mode or not self.augmentation:
            return len(self.files)
        # In train/val with augmentation enabled, double the virtual length
        return 2 * len(self.files)

    def _apply_aug(self, img_pil: Image.Image, gt_pil: Image.Image) -> tuple[Image.Image, Image.Image]:
        # Horizontal flip (30%)
        if random.random() < 0.3:
            img_pil = TF.hflip(img_pil)
            gt_pil = TF.hflip(gt_pil)

        # Vertical flip (30%)
        if random.random() < 0.3:
            img_pil = TF.vflip(img_pil)
            gt_pil = TF.vflip(gt_pil)

        # Small rotation (always when augmenting)
        angle = random.uniform(-10, 10)
        img_pil = TF.rotate(img_pil, angle)
        gt_pil = TF.rotate(gt_pil, angle)

        # Gaussian Blur (40%)
        if random.random() < 0.4:
            sigma = random.uniform(0.3, 1.2)
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        # Add contrast (20%)
        if random.random() < 0.2:
            factor = random.uniform(0.8, 1.2) # factor <1 lowers, >1 raises contrast
            img_pil = TF.adjust_contrast(img_pil, factor)

        # Add brightness (20%)
        if random.random() < 0.2:
            factor = random.uniform(0.8, 1.2) # factor <1 darker, >1 brighter
            img_pil = TF.adjust_brightness(img_pil, factor)

        return img_pil, gt_pil

    def __getitem__(self, index):
        base_len = len(self.files)
        is_augmented_view = False
        base_index = index

        if not self.test_mode and self.augmentation:
            # First half: originals [0, base_len)
            # Second half: augmented copies [base_len, 2*base_len)
            if index >= base_len:
                is_augmented_view = True
                base_index = index - base_len

        img_path, gt_path = self.files[base_index]
        img_pil = Image.open(img_path).convert("L")
        gt_pil = Image.open(gt_path).convert("L") if not self.test_mode else None

        if is_augmented_view and not self.test_mode:
            img_pil, gt_pil = self._apply_aug(img_pil, gt_pil)

        img: Tensor = self.img_transform(img_pil)
        data_dict = {"images": img, "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(gt_pil)
            data_dict["gts"] = gt

        return data_dict
