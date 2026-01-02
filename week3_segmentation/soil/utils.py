from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import json
import numpy as np
import matplotlib.pyplot as plt
import torch


def per_class_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 5, ignore_index: int = 255) -> List[float]:
    """Compute per-class IoU for predictions and targets.

    preds: (N, H, W) int tensor of predicted class ids
    targets: (N, H, W) int tensor of ground-truth class ids (255 is ignore)
    Returns list of floats (IoU for classes 0..num_classes-1). If a class has
    zero union, IoU is reported as NaN (will be excluded from mean calculation).
    """
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    ious = []
    for cls in range(num_classes):
        inter = 0
        union = 0
        for p, t in zip(preds, targets):
            valid = t != ignore_index
            p_valid = p[valid]
            t_valid = t[valid]
            inter += int(((p_valid == cls) & (t_valid == cls)).sum())
            union += int(((p_valid == cls) | (t_valid == cls)).sum())

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(inter / union))

    return ious


def mean_iou(ious: List[float]) -> float:
    arr = np.array(ious, dtype=float)
    # ignore NaNs when computing mean
    valid = ~np.isnan(arr)
    if valid.sum() == 0:
        return float('nan')
    return float(np.nanmean(arr))


def save_metrics(out_path: Path, class_names: List[str], ious: List[float]) -> None:
    data = {"per_class_iou": {}, "mean_iou": None}
    for idx, iou in enumerate(ious):
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        data["per_class_iou"][name] = None if np.isnan(iou) else float(iou)

    data["mean_iou"] = None if np.isnan(mean_iou(ious)) else float(mean_iou(ious))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


__all__ = ["per_class_iou", "mean_iou", "save_metrics", "plot_confusion_matrix"]
