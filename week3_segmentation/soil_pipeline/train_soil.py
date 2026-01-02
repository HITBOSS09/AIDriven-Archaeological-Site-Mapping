"""
SOIL IMAGE-LEVEL CLASSIFICATION PIPELINE
=========================================


Pure image-level classification for soil type recognition.
Each image is assigned exactly ONE soil label (Red Soil, Black Soil, Clay Soil, or Alluvial Soil).

CRITICAL: This is CLASSIFICATION, NOT segmentation.
- Segmentation was removed because each image contains ONE dominant soil type (no spatial mixing)
- Image-level classification directly optimizes for the task (one label per image)
- No pixel-level predictions, no masks, no majority voting

ACADEMIC JUSTIFICATION - Why Classification (Not Segmentation):
1. Dataset Structure: Each image contains ONE dominant soil type (no spatial boundaries between classes)
2. Task Requirement: Identify "what type of soil" (texture recognition), not "where are different types" (spatial localization)
3. Learning Objective: Classification learns texture-based discriminative features (global image properties)
4. Direct Optimization: Classification loss directly optimizes image-level accuracy (no proxy metrics)

ACADEMIC LIMITATIONS - Why Accuracy May Be Modest:
1. RGB Limitations: Soil types are distinguished by subtle texture/color differences that RGB images may not fully capture
   - Hyperspectral or multispectral imagery would provide richer discriminative information
   - RGB is a 3-channel representation of the visible spectrum, losing information available in other wavelengths
2. Dataset Ambiguity: Some soil images may contain transitional or mixed characteristics
   - Natural soil boundaries are gradual, not discrete
   - Lighting conditions, moisture content, and camera settings affect appearance
3. Small Dataset: Limited training samples (285 images total) may not capture full variability
   - Deep learning models typically benefit from larger datasets
   - Limited diversity in lighting, angles, and soil conditions

Model Configuration:
- Architecture: ResNet-50 (ImageNet pretrained)
- Input size: 224 × 224
- Task: Multiclass classification (4 classes)
- Loss: CrossEntropyLoss with class weighting
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
- Regularization: Class weighting, weight decay, data augmentation
- Early stopping: On validation accuracy (patience=5 epochs)

Outputs:
- Best model: week3_segmentation/soil_pipeline/models/best_soil_classifier.pth
- Metrics: week3_segmentation/soil_pipeline/results/metrics.json
- Confusion matrix: week3_segmentation/soil_pipeline/results/confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from tqdm import tqdm


# ============================================================================
# Dataset - Pure Image-Level Classification
# ============================================================================


class SoilClassificationDataset(Dataset):
    """
    Image-level classification dataset for soil types.
    
    ACADEMIC NOTE: This dataset loads ONE class label per image.
    - Labels are extracted from YOLO format files (class ID only, bounding boxes ignored)
    - Each image is assigned exactly ONE soil type (no multi-label, no segmentation)
    - This is the correct formulation because each image contains one dominant soil type
    """

    def __init__(
        self,
        images_dir: Path | str,
        labels_dir: Path | str,
        class_names: List[str],
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.image_size = image_size
        self.augment = augment

        self.samples: List[Tuple[Path, int]] = []

        # Load images and extract class labels (one per image)
        image_files = sorted(
            list(self.images_dir.glob("*.jpg"))
            + list(self.images_dir.glob("*.jpeg"))
            + list(self.images_dir.glob("*.png"))
        )

        for img_path in image_files:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            # Read class ID from first line (ignore bounding box coordinates)
            with label_path.open() as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                if not lines:
                    continue
                parts = lines[0].split()
                try:
                    class_id = int(parts[0])  # Extract class ID only
                except (ValueError, IndexError):
                    continue

            if 0 <= class_id < self.num_classes:
                self.samples.append((img_path, class_id))

        # Data augmentation for training (strong regularization)
        if augment:
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomRotation(degrees=15),
            ]
        else:
            aug_transforms = []

        # ImageNet normalization (standard for pretrained models)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                *aug_transforms,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean
                    std=[0.229, 0.224, 0.225],   # ImageNet std
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, torch.tensor(label, dtype=torch.long)


# ============================================================================
# Model - ResNet-50 Classification
# ============================================================================


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build ResNet-50 classification model for soil type recognition.
    
    ACADEMIC JUSTIFICATION - Why ResNet-50:
    - ResNet-50 provides good balance between capacity and efficiency
    - ImageNet pretraining provides strong feature representations for texture recognition
    - Classification head (fully connected layer) directly outputs class probabilities
    - No segmentation architecture (no encoder-decoder, no pixel-level outputs)
    
    ACADEMIC JUSTIFICATION - Why Classification (Not Segmentation):
    - Each image contains ONE dominant soil type (no spatial mixing)
    - Task requires texture-based learning (global features), not region-based learning
    - Direct optimization for image-level accuracy is more effective
    - Segmentation would require learning artificial boundaries that don't exist
    """
    model_name = model_name.lower()
    
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name in {"efficientnet_b0", "efficientnet-b0"}:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported: resnet18, resnet50, efficientnet_b0")

    return model


# ============================================================================
# Evaluation Metrics - Classification Only
# ============================================================================


def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate classification model (NOT segmentation).
    
    Returns classification metrics: accuracy, per-class precision/recall, confusion matrix.
    No pixel-level metrics (IoU, mIoU) - those are for segmentation tasks.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    num_classes = loader.dataset.num_classes
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf_mat[int(t), int(p)] += 1

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)

    # Per-class precision/recall from confusion matrix
    per_class = []
    for c in range(num_classes):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        per_class.append(
            {
                "precision": float(precision),
                "recall": float(recall),
                "support": int(conf_mat[c, :].sum()),
            }
        )

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "confusion_matrix": conf_mat,
        "per_class": per_class,
    }


def plot_confusion_matrix(
    conf_mat: np.ndarray,
    class_names: List[str],
    out_path: Path,
) -> None:
    """Plot and save confusion matrix for classification evaluation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_mat, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Soil Classification Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = conf_mat.max() / 2.0 if conf_mat.max() > 0 else 0.5
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(
                j,
                i,
                format(conf_mat[i, j], "d"),
                ha="center",
                va="center",
                color="white" if conf_mat[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================================================================
# Training Loop
# ============================================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train one epoch of classification model."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def compute_class_weights(dataset: SoilClassificationDataset) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset.
    
    ACADEMIC JUSTIFICATION - Why Class Weighting:
    - Addresses class imbalance in training data
    - Prevents model from being biased toward majority class
    - Improves recall for minority classes
    - Formula: weight[i] = total_samples / (num_classes * class_i_samples)
    """
    num_classes = dataset.num_classes
    counts = np.zeros(num_classes, dtype=np.int64)

    for _, label in dataset.samples:
        counts[int(label)] += 1

    total = counts.sum()
    if total == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    # Inverse frequency weighting
    weights = total / (num_classes * counts + 1e-6)
    weights = weights / weights.sum() * num_classes  # Normalize so average weight = 1.0

    return torch.tensor(weights, dtype=torch.float32)


def compute_soil_data_stats(
    dataset: SoilClassificationDataset,
    out_path: Path,
) -> dict:
    """Compute class distribution statistics."""
    num_classes = dataset.num_classes
    counts = np.zeros(num_classes, dtype=np.int64)

    for _, label in dataset.samples:
        counts[int(label)] += 1

    total = int(counts.sum())
    fractions = (counts / max(total, 1)).tolist()

    stats = {
        "num_classes": num_classes,
        "class_names": dataset.class_names,
        "counts": counts.tolist(),
        "total_samples": total,
        "fractions": fractions,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ============================================================================
# Main Training Function
# ============================================================================


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Soil 4-class Image Classification")
    parser.add_argument("--model", type=str, default="resnet50", help="resnet18, resnet50, or efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs (early stopping may stop earlier)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretraining")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    dataset_root = project_root / "preprocessed_dataset" / "soil_detection"
    models_dir = script_dir / "models"
    results_dir = script_dir / "results"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load class names from YOLO data.yaml
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    with data_yaml.open() as f:
        config = yaml.safe_load(f)

    class_names: List[str] = list(config.get("names", []))
    num_classes = len(class_names)
    if num_classes != 4:
        print(f"Warning: expected 4 classes, found {num_classes} in data.yaml")

    print("\n" + "=" * 70)
    print("SOIL IMAGE-LEVEL CLASSIFICATION TRAINING")
    print("=" * 70)
    print("\nSoil classes (class_id -> class_name mapping):")
    print("  This is the SINGLE SOURCE OF TRUTH for label mapping")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    print("\n⚠️ IMPORTANT: Streamlit app MUST use this exact order!")

    # Create datasets (classification, not segmentation)
    train_dataset = SoilClassificationDataset(
        dataset_root / "train" / "images",
        dataset_root / "train" / "labels",
        class_names=class_names,
        image_size=args.img_size,
        augment=True,  # Strong augmentation for regularization
    )
    val_dataset = SoilClassificationDataset(
        dataset_root / "val" / "images",
        dataset_root / "val" / "labels",
        class_names=class_names,
        image_size=args.img_size,
        augment=False,
    )
    test_dataset = SoilClassificationDataset(
        dataset_root / "test" / "images",
        dataset_root / "test" / "labels",
        class_names=class_names,
        image_size=args.img_size,
        augment=False,
    )

    # Compute class distribution and weights
    soil_stats_path = results_dir / "soil_data_stats.json"
    print(f"\nComputing class distribution (saving to {soil_stats_path})")
    soil_stats = compute_soil_data_stats(train_dataset, soil_stats_path)

    print("\nClass distribution (training set):")
    for idx, (count, frac) in enumerate(zip(soil_stats["counts"], soil_stats["fractions"])):
        print(f"  {idx} ({class_names[idx]}): {count} samples ({frac:.1%})")

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weights(train_dataset)
    print(f"\nClass weights (for loss function): {class_weights.tolist()}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffle for training
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

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model:           {args.model}")
    print(f"Train samples:   {len(train_dataset)}")
    print(f"Val samples:     {len(val_dataset)}")
    print(f"Test samples:    {len(test_dataset)}")
    print(f"Image size:      {args.img_size}×{args.img_size}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Max epochs:      {args.epochs}")
    print(f"Learning rate:   {args.lr}")
    print(f"Weight decay:    {args.weight_decay}")
    print(f"Early stopping:  Patience={args.patience} epochs")
    print("=" * 70)

    # Build model (ResNet-50 classification)
    model = build_model(args.model, num_classes, pretrained=not args.no_pretrained).to(device)
    
    # Loss function with class weighting
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # AdamW optimizer with weight decay (strong regularization)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_model_path = models_dir / "best_soil_classifier.pth"
    best_val_acc = -1.0
    patience_counter = 0

    # Training loop with early stopping
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_classification(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

        # Early stopping on validation accuracy
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "class_names": class_names,
                    "val_accuracy": best_val_acc,
                },
                best_model_path,
            )
            print(f"✓ New best model saved (Val accuracy = {best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered (no improvement for {args.patience} epochs)")
                break

    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    if best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=device)
        model_name = ckpt.get("model_name", args.model)
        class_names_ckpt = ckpt.get("class_names", class_names)
        model = build_model(model_name, len(class_names_ckpt), pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best model from {best_model_path}")
        
        # Safely format validation accuracy (handle missing key)
        val_acc = ckpt.get('val_accuracy', None)
        if val_acc is not None:
            print(f"Best validation accuracy: {val_acc:.4f}")
        else:
            print("Best validation accuracy: N/A (not saved in checkpoint)")

    test_metrics = evaluate_classification(model, test_loader, criterion, device)

    print("\n" + "=" * 70)
    print("TEST RESULTS (SOIL 4-CLASS CLASSIFICATION)")
    print("=" * 70)
    print(f"Test Loss:     {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print("\nPer-class metrics:")
    for idx, stats in enumerate(test_metrics["per_class"]):
        cname = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        print(
            f"  {idx} ({cname}): "
            f"precision={stats['precision']:.4f}, recall={stats['recall']:.4f}, "
            f"support={stats['support']}"
        )
    print("=" * 70)

    # Academic note about limitations
    print("\n" + "=" * 70)
    print("ACADEMIC NOTES")
    print("=" * 70)
    print("If accuracy is modest, consider:")
    print("1. RGB Limitations: Soil types distinguished by subtle texture/color differences")
    print("   - Hyperspectral imagery would provide richer discriminative information")
    print("   - RGB is 3-channel representation, losing information in other wavelengths")
    print("2. Dataset Ambiguity: Some images may contain transitional/mixed characteristics")
    print("   - Natural soil boundaries are gradual, not discrete")
    print("   - Lighting, moisture, camera settings affect appearance")
    print("3. Small Dataset: Limited training samples may not capture full variability")
    print("   - Deep learning typically benefits from larger datasets")
    print("   - Limited diversity in conditions")
    print("=" * 70)

    # Save metrics
    conf_mat = test_metrics["confusion_matrix"]
    metrics_out = {
        "loss": test_metrics["loss"],
        "accuracy": test_metrics["accuracy"],
        "class_names": class_names,
        "per_class": test_metrics["per_class"],
        "confusion_matrix": conf_mat.tolist(),
        "training_config": {
            "model": args.model,
            "epochs": epoch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "class_weights": class_weights.tolist(),
        },
    }

    metrics_file = results_dir / "metrics.json"
    with metrics_file.open("w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n✓ Saved metrics to {metrics_file}")

    cm_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(conf_mat, class_names, cm_path)
    print(f"✓ Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
