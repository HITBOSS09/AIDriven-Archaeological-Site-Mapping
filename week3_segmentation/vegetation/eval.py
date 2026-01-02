"""
Week 3 Vegetation Pipeline - Evaluation
=======================================

Loads the best vegetation U-Net model and evaluates it on the test split.

Outputs (under week3_segmentation/vegetation/results/):
  - final_metrics.json with IoU and Dice
  - coverage_stats.json with vegetation coverage percentages
  - binary prediction visualizations (image | GT | prediction)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from week3_segmentation.vegetation.dataset import VegetationDataset
from week3_segmentation.vegetation.model import UNet
from week3_segmentation.vegetation.train import (
    compute_iou_dice,
    dice_loss,
)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    bce_criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Test", leave=False):
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

    # Ensure metrics are native Python floats (safe for JSON)
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


def compute_coverage_stats(
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    threshold: float = 0.5,
) -> None:
    """Compute vegetation coverage percentage per image and dataset mean."""
    coverages: List[float] = []

    model_device = device  # unused, but clarifies intent
    del model_device

    for images, masks in loader:
        masks = masks  # [B,1,H,W]
        mask_np = masks.numpy()
        for b in range(mask_np.shape[0]):
            cov = float(mask_np[b, 0].mean())
            coverages.append(cov)

    if coverages:
        mean_cov = float(np.mean(coverages))
    else:
        mean_cov = 0.0

    stats = {
        "num_images": len(coverages),
        "per_image_coverage": coverages,
        "mean_coverage": mean_cov,
        "threshold": threshold,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)


def main(argv: List[str] | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    dataset_root = project_root / "preprocessed_dataset" / "vegetation_detection"
    masks_root = project_root / "week3_segmentation" / "masks" / "vegetation"
    models_dir = script_dir / "models"
    results_dir = script_dir / "results"

    results_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    test_dataset = VegetationDataset(
        dataset_root / "test" / "images",
        masks_root / "test" / "masks",
        image_size=256,
        augment=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Load best model
    best_model_path = models_dir / "best_vegetation_unet.pth"
    if not best_model_path.exists():
        print(f"\n❌ Model not found at {best_model_path}")
        print("Run training first: python -m week3_segmentation.vegetation.train")
        return

    model = UNet(in_channels=3, out_channels=1).to(device)
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"✓ Loaded model from {best_model_path}")

    # Evaluate
    bce_criterion = nn.BCEWithLogitsLoss()
    print(f"\nEvaluating on {len(test_dataset)} test samples...")
    test_metrics = evaluate(model, test_loader, bce_criterion, device)

    print("\n" + "=" * 70)
    print("VEGETATION BINARY SEGMENTATION - TEST RESULTS")
    print("=" * 70)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test IoU:  {test_metrics['iou']:.4f}")
    print(f"Test Dice: {test_metrics['dice']:.4f}")
    print("=" * 70)

    # Visualizations
    vis_dir = results_dir / "test_visualizations"
    print(f"\nSaving visualizations to {vis_dir}...")
    save_visualizations(model, test_loader, device, vis_dir, max_samples=10)

    # Metrics JSON
    metrics_file = results_dir / "final_metrics.json"
    with metrics_file.open("w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"✓ Saved metrics to {metrics_file}")

    # Coverage statistics (GT-based)
    coverage_file = results_dir / "coverage_stats.json"
    print(f"\nComputing coverage statistics to {coverage_file}...")
    compute_coverage_stats(test_loader, device, coverage_file)


if __name__ == "__main__":  # pragma: no cover
    main()


