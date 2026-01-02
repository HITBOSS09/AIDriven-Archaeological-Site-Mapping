"""
VEGETATION BINARY SEMANTIC SEGMENTATION PIPELINE
================================================
Pure binary semantic segmentation of vegetation (foreground) vs non-vegetation
background using a U-Net model.

Inputs:
  - RGB images from preprocessed_dataset/vegetation_detection/
  - Binary masks from week3_segmentation/masks/vegetation/*/masks
    (0 = non-vegetation, 1 = vegetation)

Outputs:
  - Trained U-Net weights in vegetation_pipeline/models/
  - Metrics and coverage statistics in vegetation_pipeline/results/
  - Binary prediction visualizations in vegetation_pipeline/results/test_visualizations/
"""

import os
import sys
import argparse
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# UNet Architecture
# ============================================================================

class UNet(nn.Module):
    """Simple U-Net for binary segmentation."""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = self.conv_block(256, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder
        up3 = self.upconv3(b)
        d3 = self.dec3(torch.cat([up3, e3], dim=1))
        up2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))
        
        out = self.final_conv(d1)
        return out


# ============================================================================
# Dataset
# ============================================================================

class VegetationDataset(Dataset):
    """Binary vegetation segmentation dataset using precomputed masks.

    Images are read from the preprocessed vegetation dataset, while pixel-accurate
    masks are loaded from:
        week3_segmentation/masks/vegetation/{split}/masks

    Background is encoded as 0, vegetation as 1.
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        image_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.augment = augment

        # Get all images
        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg"))
            + list(self.images_dir.glob("*.png"))
            + list(self.images_dir.glob("*.jpeg"))
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / f"{img_path.stem}.png"

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        # Load binary mask from precomputed PNG
        mask = self._load_mask_from_png(mask_path)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return img_tensor, mask_tensor

    def _load_mask_from_png(self, mask_path: Path) -> np.ndarray:
        """Load binary vegetation mask from PNG file.

        Non-zero pixels are treated as vegetation (1.0), background is 0.0.
        """
        if not mask_path.exists():
            # Fallback to empty mask if missing
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)

        mask_img = Image.open(mask_path)
        if mask_img.mode != "L":
            mask_img = mask_img.convert("L")
        if mask_img.size != (self.image_size, self.image_size):
            mask_img = mask_img.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        mask = np.array(mask_img, dtype=np.float32)
        # Binarize: any non-zero value is treated as vegetation
        mask = (mask > 0).astype(np.float32)
        return mask


# ============================================================================
# Metrics & Losses
# ============================================================================

def compute_iou_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5):
    """Compute IoU and Dice for binary segmentation.

    Args:
        pred_mask: Probabilistic prediction in [0, 1], shape [H, W].
        gt_mask: Binary ground truth in {0, 1}, shape [H, W].
        threshold: Threshold for binarising predictions.
    """
    pred = (pred_mask > threshold).astype(np.float32)
    gt = gt_mask.astype(np.float32)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    iou = intersection / (union + 1e-7)

    dice = 2 * intersection / (np.sum(pred) + np.sum(gt) + 1e-7)

    return iou, dice


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Soft Dice loss for binary segmentation.

    Args:
        logits: Raw model outputs [B, 1, H, W].
        targets: Binary masks [B, 1, H, W].
    """
    probs = torch.sigmoid(logits)
    targets = targets.float()

    intersection = (probs * targets).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    dice = 2.0 * intersection / denom
    loss = 1.0 - dice
    return loss.mean()


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_one_epoch(model, loader, optimizer, bce_criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(loader, desc='Train', leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        # Combined BCE + Dice loss
        loss_bce = bce_criterion(logits, masks)
        loss_dice = dice_loss(logits, masks)
        loss = loss_bce + loss_dice
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(loader)


def evaluate(model, loader, bce_criterion, device, split_name: str = 'Val'):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc=split_name, leave=False):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss_bce = bce_criterion(logits, masks)
            loss_dice = dice_loss(logits, masks)
            loss = loss_bce + loss_dice

            pred_np = torch.sigmoid(logits).cpu().numpy()
            mask_np = masks.cpu().numpy()

            for i in range(pred_np.shape[0]):
                iou, dice = compute_iou_dice(pred_np[i, 0], mask_np[i, 0])

                running_iou += iou
                running_dice += dice
                n_batches += 1

            running_loss += loss.item()

    # Return Python floats to ensure JSON serialization works
    return {
        'loss': float(running_loss / max(len(loader), 1)),
        'iou': float(running_iou / max(n_batches, 1)),
        'dice': float(running_dice / max(n_batches, 1)),
    }


def compute_coverage_stats(model, loader, device, out_path: Path, threshold: float = 0.5) -> None:
    """Compute vegetation coverage percentage per image and dataset-wide stats.

    Coverage is defined as the fraction of pixels predicted/labelled as vegetation.
    Saves a JSON with per-image ground-truth and predicted coverage and dataset means.
    """
    model.eval()
    dataset = loader.dataset
    coverages = []

    with torch.no_grad():
        idx_global = 0
        for images, masks in tqdm(loader, desc='Coverage', leave=False):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            gt_np = masks.cpu().numpy()

            batch_size = images.size(0)
            for b in range(batch_size):
                gt = gt_np[b, 0]
                pred = probs[b, 0]

                gt_cov = float(gt.mean())  # fraction of pixels with vegetation
                pred_cov = float((pred > threshold).mean())

                # Use filename if available, otherwise fallback to index
                if hasattr(dataset, "image_files"):
                    img_name = Path(dataset.image_files[idx_global]).name
                else:
                    img_name = f"image_{idx_global}"

                coverages.append(
                    {
                        "image": img_name,
                        "gt_coverage": gt_cov,
                        "pred_coverage": pred_cov,
                    }
                )
                idx_global += 1

    if coverages:
        mean_gt = float(np.mean([c["gt_coverage"] for c in coverages]))
        mean_pred = float(np.mean([c["pred_coverage"] for c in coverages]))
    else:
        mean_gt = 0.0
        mean_pred = 0.0

    stats = {
        "per_image": coverages,
        "mean_gt_coverage": mean_gt,
        "mean_pred_coverage": mean_pred,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)


def sanity_check_vegetation_dataset(
    dataset: Dataset,
    out_path: Path,
    low_fg_threshold: float = 0.02,
    low_fg_ratio_threshold: float = 0.8,
) -> None:
    """Run sanity checks on vegetation masks before training.

    Computes per-image vegetation coverage (fraction of foreground pixels),
    counts empty masks, and logs dataset-wide statistics to JSON.

    Raises RuntimeError if the fraction of images with vegetation coverage
    below `low_fg_threshold` exceeds `low_fg_ratio_threshold`.
    """
    coverages: list[float] = []
    num_empty = 0
    num_low = 0

    total = len(dataset)
    for idx in range(total):
        _, mask = dataset[idx]  # mask shape [1, H, W]
        mask_np = mask.numpy()[0]
        cov = float(mask_np.mean())
        coverages.append(cov)
        if cov == 0.0:
            num_empty += 1
        if cov < low_fg_threshold:
            num_low += 1

    if coverages:
        mean_cov = float(np.mean(coverages))
        std_cov = float(np.std(coverages))
    else:
        mean_cov = 0.0
        std_cov = 0.0

    stats = {
        "num_images": total,
        "num_empty_masks": num_empty,
        "num_low_coverage_masks": num_low,
        "low_coverage_threshold": low_fg_threshold,
        "low_coverage_ratio_threshold": low_fg_ratio_threshold,
        "mean_coverage": mean_cov,
        "std_coverage": std_cov,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)

    low_ratio = (num_low / total) if total > 0 else 0.0
    if low_ratio > low_fg_ratio_threshold:
        raise RuntimeError(
            f"Vegetation foreground is < {low_fg_threshold*100:.1f}% for "
            f"{low_ratio*100:.1f}% of training images. "
            f"Please verify masks or adjust the dataset before training."
        )


def save_visualizations(model, loader, device, out_dir, max_samples=5):
    """Save sample predictions."""
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            preds = torch.sigmoid(logits)
            
            for b in range(images.size(0)):
                if saved >= max_samples:
                    return
                
                # Denormalize image
                img = images[b].cpu().permute(1, 2, 0).numpy()
                img = (img * np.array([0.229, 0.224, 0.225]) + 
                       np.array([0.485, 0.456, 0.406]))
                img = np.clip(img, 0, 1)
                
                gt = masks[b, 0].cpu().numpy()
                pred = preds[b, 0].cpu().numpy()
                
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img)
                axes[0].set_title('Input')
                axes[0].axis('off')
                
                axes[1].imshow(gt, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                axes[2].imshow(pred, cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                fig.suptitle('Vegetation Binary Segmentation')
                out_path = out_dir / f'sample_{saved}.png'
                fig.tight_layout()
                fig.savefig(out_path, dpi=100)
                plt.close(fig)
                saved += 1


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Vegetation Binary Semantic Segmentation')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    dataset_root = project_root / 'preprocessed_dataset' / 'vegetation_detection'
    masks_root = project_root / 'week3_segmentation' / 'masks' / 'vegetation'
    models_dir = script_dir / 'models'
    results_dir = script_dir / 'results'
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    train_dataset = VegetationDataset(
        dataset_root / "train" / "images",
        masks_root / "train" / "masks",
        image_size=args.img_size,
        augment=True,
    )
    val_dataset = VegetationDataset(
        dataset_root / "val" / "images",
        masks_root / "val" / "masks",
        image_size=args.img_size,
        augment=False,
    )
    test_dataset = VegetationDataset(
        dataset_root / "test" / "images",
        masks_root / "test" / "masks",
        image_size=args.img_size,
        augment=False,
    )

    # ------------------------------------------------------------------
    # Data sanity checks (vegetation coverage) BEFORE training
    # ------------------------------------------------------------------
    veg_stats_path = results_dir / "vegetation_data_stats.json"
    print(f"\nRunning vegetation data sanity checks (saving to {veg_stats_path})")
    sanity_check_vegetation_dataset(train_dataset, veg_stats_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"\nVEGETATION BINARY SEGMENTATION PIPELINE")
    print(f"=" * 60)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"=" * 60)
    
    # Model
    model = UNet(in_channels=3, out_channels=1).to(device)
    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_model_path = models_dir / 'best_vegetation_unet.pth'
    best_val_iou = -1.0
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_criterion, device)
        val_metrics = evaluate(model, val_loader, bce_criterion, device, split_name='Val')
        
        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Val   Loss: {val_metrics['loss']:.4f}, "
            f"IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}"
        )
        
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save({'model_state_dict': model.state_dict()}, best_model_path)
            print(f"✓ New best model saved (Val IoU = {best_val_iou:.4f})")
    
    # Final evaluation
    if best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"\nLoaded best model from {best_model_path}")
    
    test_metrics = evaluate(model, test_loader, bce_criterion, device, split_name='Test')
    
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS (VEGETATION BINARY SEGMENTATION)")
    print("=" * 60)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test IoU:  {test_metrics['iou']:.4f}")
    print(f"Test Dice: {test_metrics['dice']:.4f}")
    print("=" * 60)
    
    # Save visualizations
    vis_dir = results_dir / 'test_visualizations'
    print(f"\nSaving visualizations to {vis_dir}")
    save_visualizations(model, test_loader, device, vis_dir, max_samples=10)

    # Save metrics to JSON
    metrics_file = results_dir / 'final_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    # Compute and save vegetation coverage statistics
    coverage_file = results_dir / 'coverage_stats.json'
    print(f"\nComputing coverage statistics to {coverage_file}")
    compute_coverage_stats(model, test_loader, device, coverage_file)


if __name__ == '__main__':
    main()
