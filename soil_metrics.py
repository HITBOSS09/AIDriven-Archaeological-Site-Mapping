import torch
from typing import List, Tuple


def soil_iou_and_dice_multiclass(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 4,
) -> Tuple[float, float, List[float], List[float]]:
    """Compute mean IoU and Dice for multi-class soil segmentation.

    This function is **only** for the soil task.
    It assumes soil masks with integer labels 0..num_classes-1 and **no** vegetation mask.

    Args:
        logits: Model outputs of shape [B, C, H, W], C == num_classes.
        target: Ground-truth masks of shape [B, H, W] with values in {0, ..., num_classes-1}.
        num_classes: Number of soil classes (default 4).

    Returns:
        mean_iou: float
        mean_dice: float
        per_class_iou: list of IoU values for the classes that are present
        per_class_dice: list of Dice values for the classes that are present
    """
    if logits.ndim != 4:
        raise ValueError(f"logits must be 4D [B,C,H,W], got shape {tuple(logits.shape)}")
    if logits.shape[1] != num_classes:
        raise ValueError(
            f"logits channel dimension ({logits.shape[1]}) must equal num_classes ({num_classes})"
        )
    if target.ndim != 3:
        raise ValueError(f"target must be 3D [B,H,W], got shape {tuple(target.shape)}")

    pred = logits.argmax(dim=1)  # [B,H,W]
    target = target.long()

    per_class_iou: List[torch.Tensor] = []
    per_class_dice: List[torch.Tensor] = []

    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        pred_count = pred_c.sum().float()
        target_count = target_c.sum().float()

        if union == 0 and target_count == 0:
            # Class not present in this batch; skip it for mean metrics.
            continue

        iou_c = intersection / (union + 1e-6) if union > 0 else torch.tensor(0.0, device=logits.device)
        denom = pred_count + target_count
        dice_c = (
            2 * intersection / (denom + 1e-6)
            if denom > 0
            else torch.tensor(0.0, device=logits.device)
        )

        per_class_iou.append(iou_c)
        per_class_dice.append(dice_c)

    if per_class_iou:
        mean_iou = torch.stack(per_class_iou).mean().item()
        mean_dice = torch.stack(per_class_dice).mean().item()
    else:
        mean_iou = 0.0
        mean_dice = 0.0

    return (
        mean_iou,
        mean_dice,
        [float(v) for v in per_class_iou],
        [float(v) for v in per_class_dice],
    )
