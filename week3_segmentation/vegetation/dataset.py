"""
Vegetation Dataset for Binary Semantic Segmentation
===================================================

Loads RGB images and corresponding binary PNG masks:
  - Images:  preprocessed_dataset/vegetation_detection/{split}/images
  - Masks:   week3_segmentation/masks/vegetation/{split}/masks

Mask encoding:
  - 0 = non-vegetation (background)
  - 1 = vegetation (foreground)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import torch


class VegetationDataset(Dataset):
    """Binary vegetation segmentation dataset using precomputed masks."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        image_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.augment = augment

        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg"))
            + list(self.images_dir.glob("*.jpeg"))
            + list(self.images_dir.glob("*.png"))
        )

        aug_transforms: list[transforms.transforms.Transform] = []
        if augment:
            aug_transforms.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
                    ),
                ]
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                *aug_transforms,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / f"{img_path.stem}.png"

        # Image
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        # Binary mask (0/1 float tensor, shape [1, H, W])
        mask = self._load_mask(mask_path)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return img_tensor, mask_tensor

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """Load binary vegetation mask; any non-zero pixel is foreground."""
        if not mask_path.exists():
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)

        mask_img = Image.open(mask_path)
        if mask_img.mode != "L":
            mask_img = mask_img.convert("L")
        if mask_img.size != (self.image_size, self.image_size):
            mask_img = mask_img.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        mask = np.array(mask_img, dtype=np.float32)
        mask = (mask > 0).astype(np.float32)
        return mask


