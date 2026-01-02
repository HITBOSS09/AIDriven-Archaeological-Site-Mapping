import torch


def veg_iou_and_dice(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Compute IoU and Dice score for binary vegetation segmentation.

    This function is **only** for the vegetation task (vegetation vs background).

    Args:
        logits: Model outputs of shape [B, 1, H, W] (sigmoid) or [B, 2, H, W] (softmax).
        target: Ground-truth masks of shape [B, H, W] with values 0 (background) or 1 (vegetation).
        threshold: Threshold for binarizing sigmoid outputs when logits.shape[1] == 1.

    Returns:
        iou_veg (float), dice_veg (float)
    """
    if logits.ndim != 4:
        raise ValueError(f"logits must be 4D [B,C,H,W], got shape {tuple(logits.shape)}")

    # Convert logits to predicted class indices
    if logits.shape[1] == 1:
        pred = (torch.sigmoid(logits) > threshold).long().squeeze(1)  # [B,H,W]
    else:
        pred = logits.argmax(dim=1)  # [B,H,W]

    if target.ndim != 3:
        raise ValueError(f"target must be 3D [B,H,W], got shape {tuple(target.shape)}")

    target = target.long()

    # Vegetation is class 1
    pred_veg = pred == 1
    target_veg = target == 1

    intersection = (pred_veg & target_veg).sum().float()
    union = (pred_veg | target_veg).sum().float()

    if union == 0:
        iou = torch.tensor(0.0, device=logits.device)
    else:
        iou = intersection / (union + 1e-6)

    pred_veg_count = pred_veg.sum().float()
    target_veg_count = target_veg.sum().float()
    denom = pred_veg_count + target_veg_count
    if denom == 0:
        dice = torch.tensor(0.0, device=logits.device)
    else:
        dice = 2 * intersection / (denom + 1e-6)

    return float(iou), float(dice)
