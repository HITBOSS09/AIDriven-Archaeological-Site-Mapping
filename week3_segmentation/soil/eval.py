from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import SoilSegmentationDataset
from .model import build_deeplabv3
from .utils import per_class_iou, mean_iou, plot_confusion_matrix
import numpy as np


CLASS_NAMES: List[str] = [
    "background",
    "red soil",
    "black soil",
    "clay soil",
    "alluvial soil",
]


def evaluate_and_save(model, loader, device, out_path: Path):
    model.eval()
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            out = model(imgs)
            outputs = out['out'] if isinstance(out, dict) and 'out' in out else out
            pred = torch.argmax(outputs, dim=1)
            preds_all.append(pred.cpu())
            targets_all.append(masks.cpu())

    preds_all = torch.cat(preds_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    ious = per_class_iou(preds_all, targets_all, num_classes=len(CLASS_NAMES), ignore_index=255)
    miou = mean_iou(ious)

    metrics = {"class_names": CLASS_NAMES, "per_class_iou": {}, "mean_iou": None}
    for idx, iou in enumerate(ious):
        metrics["per_class_iou"][CLASS_NAMES[idx]] = None if iou != iou else float(iou)
    metrics["mean_iou"] = None if miou != miou else float(miou)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    # Optional confusion matrix
    # Build confusion matrix excluding ignore_index
    num_classes = len(CLASS_NAMES)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    preds_np = preds_all.numpy()
    targs_np = targets_all.numpy()
    for p, t in zip(preds_np, targs_np):
        valid = t != 255
        p_valid = p[valid]
        t_valid = t[valid]
        for gt, pr in zip(t_valid.flatten(), p_valid.flatten()):
            cm[int(gt), int(pr)] += 1

    plot_confusion_matrix(cm, CLASS_NAMES, out_path.parent / "confusion_matrix.png")

    return metrics, cm


def main(argv: List[str] | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    images_root = project_root / "preprocessed_dataset" / "soil_detection"
    masks_root = project_root / "week3_segmentation" / "masks" / "soil"

    models_dir = script_dir / "models"
    results_dir = script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    test_ds = SoilSegmentationDataset(images_root / "test" / "images", masks_root / "test" / "masks", image_size=(640, 640), augment=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    best = models_dir / "best_soil_segmentation.pth"
    if not best.exists():
        print(f"Best model not found at {best}. Run training first.")
        return

    ckpt = torch.load(best, map_location=device)
    model = build_deeplabv3(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"✓ Loaded model from {best}")

    metrics_file = results_dir / "final_metrics.json"
    metrics, cm = evaluate_and_save(model, test_loader, device, metrics_file)
    print("\n" + "=" * 70)
    print("SOIL MULTICLASS SEGMENTATION - TEST RESULTS")
    print("=" * 70)
    print(f"Mean IoU: {metrics['mean_iou']}")
    for name, iou in metrics["per_class_iou"].items():
        print(f"  {name}: {iou}")
    print(f"\n✓ Saved metrics to {metrics_file}")
    print(f"✓ Saved confusion matrix to {metrics_file.parent / 'confusion_matrix.png'}")


if __name__ == "__main__":  # pragma: no cover
    main()
