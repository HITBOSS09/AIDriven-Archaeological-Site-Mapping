"""
Week 3 Vegetation Pipeline - Training
=====================================

Binary semantic segmentation of vegetation vs background using a U-Net model.

This module is designed to run from the project root:

  python -m week3_segmentation.vegetation.train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from week3_segmentation.vegetation.dataset import VegetationDataset
from week3_segmentation.vegetation.model import UNet


def compute_iou_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5):
    """Compute IoU and Dice for binary segmentation."""
    pred = (pred_mask > threshold).astype(np.float32)
    gt = gt_mask.astype(np.float32)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    iou = intersection / (union + 1e-7)

    dice = 2 * intersection / (np.sum(pred) + np.sum(gt) + 1e-7)

    return iou, dice


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Soft Dice loss for binary segmentation."""
    probs = torch.sigmoid(logits)
    targets = targets.float()

    intersection = (probs * targets).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    dice = 2.0 * intersection / denom
    loss = 1.0 - dice
    return loss.mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    bce_criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss_bce = bce_criterion(logits, masks)
        loss_dice = dice_loss(logits, masks)
        loss = loss_bce + loss_dice
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    bce_criterion: nn.Module,
    device: torch.device,
    split_name: str = "Val",
) -> dict:
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

    # Ensure returned metrics are plain Python floats (JSON serializable)
    return {
        "loss": float(running_loss / max(len(loader), 1)),
        "iou": float(running_iou / max(n_batches, 1)),
        "dice": float(running_dice / max(n_batches, 1)),
    }


def save_visualizations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    max_samples: int = 5,
) -> None:
    """Save sample predictions as (image | GT | prediction) triplets."""
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

                img = images[b].cpu().permute(1, 2, 0).numpy()
                img = (
                    img * np.array([0.229, 0.224, 0.225])
                    + np.array([0.485, 0.456, 0.406])
                )
                img = np.clip(img, 0, 1)

                gt = masks[b, 0].cpu().numpy()
                pred = preds[b, 0].cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img)
                axes[0].set_title("Input")
                axes[0].axis("off")

                axes[1].imshow(gt, cmap="gray")
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(pred, cmap="gray")
                axes[2].set_title("Prediction")
                axes[2].axis("off")

                fig.suptitle("Vegetation Binary Segmentation")
                out_path = out_dir / f"sample_{saved}.png"
                fig.tight_layout()
                fig.savefig(out_path, dpi=100)
                plt.close(fig)
                saved += 1


def compute_coverage_stats_from_dataset(
    dataset: VegetationDataset,
    out_path: Path,
) -> None:
    """Compute vegetation coverage percentage per image and dataset mean."""
    coverages: List[float] = []

    for idx in range(len(dataset)):
        _, mask = dataset[idx]
        mask_np = mask.numpy()[0]
        coverages.append(float(mask_np.mean()))

    if coverages:
        mean_cov = float(np.mean(coverages))
    else:
        mean_cov = 0.0

    stats = {
        "num_images": len(dataset),
        "per_image_coverage": coverages,
        "mean_coverage": mean_cov,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)


def compute_foreground_ratio(dataset: VegetationDataset) -> float:
    """Compute overall foreground (vegetation) pixel ratio for a dataset.

    Returns fraction in [0,1] representing pixels marked as vegetation.
    """
    total_fg = 0.0
    total_pixels = 0

    # Use the dataset's internal loader to ensure same resizing/format
    for img_path in dataset.image_files:
        mask = dataset._load_mask(dataset.masks_dir / f"{Path(img_path).stem}.png")
        # mask is numpy array of 0/1 floats
        total_fg += float(mask.sum())
        total_pixels += mask.size

    if total_pixels == 0:
        return 0.0
    return float(total_fg / total_pixels)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Week 3 Vegetation Segmentation Training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths relative to project root (this file lives in week3_segmentation/vegetation)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    dataset_root = project_root / "preprocessed_dataset" / "vegetation_detection"
    masks_root = project_root / "week3_segmentation" / "masks" / "vegetation"
    models_dir = script_dir / "models"
    results_dir = script_dir / "results"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("\nWEEK 3 VEGETATION BINARY SEGMENTATION")
    print("=" * 60)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"Image size:    {args.img_size}x{args.img_size}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Epochs:        {args.epochs}")
    print("=" * 60)

    # Pre-training coverage stats on training set
    coverage_path = results_dir / "coverage_stats_pretrain.json"
    print(f"\nComputing vegetation coverage statistics to {coverage_path}")
    compute_coverage_stats_from_dataset(train_dataset, coverage_path)

    # Compute foreground ratio from training masks and derive pos_weight
    fg_ratio = compute_foreground_ratio(train_dataset)
    eps = 1e-6
    if fg_ratio <= 0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = (1.0 - fg_ratio) / (fg_ratio + eps)

    pos_weight_value = float(pos_weight_value)
    print(f"Foreground ratio (training masks): {fg_ratio:.6f}")
    print(f"Using BCE pos_weight={pos_weight_value:.4f} to address class imbalance")

    # Persist pos_weight and fg_ratio into the coverage stats JSON for traceability
    try:
        with coverage_path.open("r") as f:
            cov_stats = json.load(f)
    except Exception:
        cov_stats = {}
    cov_stats["foreground_ratio"] = float(fg_ratio)
    cov_stats["pos_weight"] = float(pos_weight_value)
    with coverage_path.open("w") as f:
        json.dump(cov_stats, f, indent=2)
    print(f"Updated coverage stats with pos_weight -> {coverage_path}")

    # Model, loss, optimizer
    model = UNet(in_channels=3, out_channels=1).to(device)
    # Use pos_weight to handle class imbalance (positive class = vegetation)
    pos_weight_tensor = torch.tensor(pos_weight_value, device=device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_model_path = models_dir / "best_vegetation_unet.pth"
    best_val_dice = -1.0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, bce_criterion, device)
        val_metrics = evaluate(model, val_loader, bce_criterion, device, split_name="Val")

        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Val   Loss: {val_metrics['loss']:.4f}, "
            f"IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(
                {"model_state_dict": model.state_dict()},
                best_model_path,
            )
            print(f"✓ New best model saved (Val Dice = {best_val_dice:.4f})")

    # Final evaluation on test set with best model
    if best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\nLoaded best model from {best_model_path}")

    test_metrics = evaluate(model, test_loader, bce_criterion, device, split_name="Test")

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS (VEGETATION BINARY SEGMENTATION)")
    print("=" * 60)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test IoU:  {test_metrics['iou']:.4f}")
    print(f"Test Dice: {test_metrics['dice']:.4f}")
    print("=" * 60)

    # Save visualizations
    vis_dir = results_dir / "test_visualizations"
    print(f"\nSaving visualizations to {vis_dir}")
    save_visualizations(model, test_loader, device, vis_dir, max_samples=10)

    # Save metrics to JSON
    metrics_file = results_dir / "final_metrics.json"
    # Include pos_weight used during training for traceability
    test_metrics["pos_weight"] = float(pos_weight_value)
    with metrics_file.open("w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    # Coverage stats on test set predictions/GT can be computed by reusing
    # compute_coverage_stats_from_dataset if needed; here we rely on pretrain stats.


if __name__ == "__main__":  # pragma: no cover
    main()


