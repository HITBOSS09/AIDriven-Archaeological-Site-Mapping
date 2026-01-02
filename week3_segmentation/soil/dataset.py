from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class SoilSegmentationDataset(Dataset):
    """Dataset for multiclass soil segmentation.

    Images are read from a separate images directory (usually
    `preprocessed_dataset/soil_detection/{split}/images`) while masks are in
    `week3_segmentation/masks/soil/{split}/masks` as single-channel PNGs with
    integer class values {0,1,2,3,4} and 255 for ignored boundary pixels.

    Masks are kept as integers and are NOT normalized or converted to floats.
    A light median filter is applied to each mask to smooth annotation noise.
    """

    def __init__(
        self,
        images_dir: Path | str,
        masks_dir: Path | str,
        image_size: Tuple[int, int] = (640, 640),
        augment: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = tuple(image_size)
        self.augment = augment

        # Pair images with masks by stem
        image_files = sorted(
            list(self.images_dir.glob("*.jpg"))
            + list(self.images_dir.glob("*.jpeg"))
            + list(self.images_dir.glob("*.png"))
        )

        self.samples: List[Tuple[Path, Path]] = []
        for img_path in image_files:
            mask_path = self.masks_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        # Quick masks validation (sample a subset) to ensure integer labels
        sampled = self.samples[:50]
        bad = False
        allowed = set([0, 1, 2, 3, 4, 255])
        for _, mpath in sampled:
            arr = self._load_mask(mpath)
            unique = set(np.unique(arr).tolist())
            if not unique.issubset(allowed):
                bad = True
                print(f"Warning: unexpected mask values {unique - allowed} in {mpath}")
        if bad:
            print("Warning: masks contain unexpected label values; ensure masks use 0-4 and 255 only.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        # Load as single-channel (L), apply small median filter to reduce
        # annotation noise, and return integer ndarray (uint8)
        m = Image.open(mask_path).convert("L")
        m = m.filter(ImageFilter.MedianFilter(size=3))
        arr = np.array(m, dtype=np.uint8)
        return arr

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = self._load_mask(mask_path)

        # Resize both to image_size using nearest for mask
        img = img.resize(self.image_size, resample=Image.BILINEAR)
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize(self.image_size, resample=Image.NEAREST)
        mask = np.array(mask_img, dtype=np.uint8)

        # Augmentations (paired)
        if self.augment:
            # Random horizontal flip
            if np.random.rand() < 0.5:
                img = TF.hflip(img)
                mask = np.fliplr(mask).copy()
            # Random vertical flip
            if np.random.rand() < 0.5:
                img = TF.vflip(img)
                mask = np.flipud(mask).copy()
            # Random crop
            w, h = img.size
            th, tw = int(0.9 * h), int(0.9 * w)
            if w > tw and h > th:
                i = np.random.randint(0, h - th + 1)
                j = np.random.randint(0, w - tw + 1)
                img = img.crop((j, i, j + tw, i + th))
                mask = mask[i : i + th, j : j + tw]
                img = img.resize(self.image_size, resample=Image.BILINEAR)
                mask = np.array(Image.fromarray(mask).resize(self.image_size, resample=Image.NEAREST), dtype=np.uint8)

        # Image -> tensor, normalized (ImageNet stats)
        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Mask -> LongTensor (required by CrossEntropyLoss). Keep 255 as ignore_index
        mask_t = torch.from_numpy(mask).long()

        return img_t, mask_t


if __name__ == "__main__":  # quick smoke test
    from pathlib import Path

    ds = SoilSegmentationDataset(
        images_dir=Path("preprocessed_dataset/soil_detection/train/images"),
        masks_dir=Path("week3_segmentation/masks/soil/train/masks"),
        augment=True,
    )
    print("Samples:", len(ds))
    if len(ds) > 0:
        x, y = ds[0]
        print(x.shape, y.shape, torch.unique(y)[:10])
