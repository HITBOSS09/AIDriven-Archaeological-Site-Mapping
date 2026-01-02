from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import SoilSegmentationDataset
from .model import build_deeplabv3
from .utils import per_class_iou, mean_iou


CLASS_NAMES: List[str] = [
    "background",
    "red soil",
    "black soil",
    "clay soil",
    "alluvial soil",
]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        logits = out['out'] if isinstance(out, dict) and 'out' in out else out
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device, num_classes=5):
    model.eval()
    total_loss = 0.0
    preds_all = []
    targets_all = []
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            out = model(imgs)
            outputs = out['out'] if isinstance(out, dict) and 'out' in out else out
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            pred = torch.argmax(outputs, dim=1)
            preds_all.append(pred.cpu())
            targets_all.append(masks.cpu())

    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    ious = per_class_iou(preds_all, targets_all, num_classes=num_classes, ignore_index=255)
    return total_loss / max(len(loader), 1), ious


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Soil multiclass segmentation training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    images_root = project_root / "preprocessed_dataset" / "soil_detection"
    masks_root = project_root / "week3_segmentation" / "masks" / "soil"

    models_dir = script_dir / "models"
    results_dir = script_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = SoilSegmentationDataset(images_root / "train" / "images", masks_root / "train" / "masks", image_size=(args.img_size, args.img_size), augment=True)
    val_ds = SoilSegmentationDataset(images_root / "val" / "images", masks_root / "val" / "masks", image_size=(args.img_size, args.img_size), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_deeplabv3(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_miou = -1.0
    best_ckpt = models_dir / "best_soil_segmentation.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_ious = evaluate(model, val_loader, device, num_classes=len(CLASS_NAMES))
        val_miou = mean_iou(val_ious)

        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mIoU={val_miou:.4f}")
        for idx, iou in enumerate(val_ious):
            name = CLASS_NAMES[idx]
            print(f"  class {idx} ({name}): iou={iou if not (iou != iou) else float('nan'):.4f}")

        # Save best by mIoU
        if not (val_miou != val_miou) and val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "epoch": epoch,
            }, best_ckpt)
            print(f"✓ New best model saved to {best_ckpt}")

    # After training, save final metrics on validation
    metrics_out = {"class_names": CLASS_NAMES, "per_class_iou": {}, "mean_iou": None}
    _, final_ious = evaluate(model, val_loader, device, num_classes=len(CLASS_NAMES))
    for idx, iou in enumerate(final_ious):
        metrics_out["per_class_iou"][CLASS_NAMES[idx]] = None if iou != iou else float(iou)
    miou_val = mean_iou(final_ious)
    metrics_out["mean_iou"] = None if miou_val != miou_val else float(miou_val)

    metrics_file = results_dir / "final_metrics.json"
    with metrics_file.open("w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"✓ Saved final metrics to {metrics_file}")


if __name__ == "__main__":  # pragma: no cover
    main()
