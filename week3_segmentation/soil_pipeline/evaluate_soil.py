"""
SOIL 4-CLASS IMAGE CLASSIFICATION EVALUATION
============================================

Loads the best soil classification model and evaluates it on the test split.

Outputs (in week3_segmentation/soil_pipeline/results/):
  - metrics.json with accuracy, per-class precision/recall, confusion matrix
  - confusion_matrix.png visualisation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from soil_pipeline.train_soil import (
    SoilClassificationDataset,
    build_model,
    evaluate_classification,
    plot_confusion_matrix,
)


def main(argv: List[str] | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    dataset_root = project_root / "preprocessed_dataset" / "soil_detection"
    models_dir = script_dir / "models"
    results_dir = script_dir / "results"

    results_dir.mkdir(parents=True, exist_ok=True)

    # Load class names from data.yaml
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    with data_yaml.open() as f:
        config = yaml.safe_load(f)

    class_names: List[str] = list(config.get("names", []))
    num_classes = len(class_names)

    print("\nSoil classes:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    # Test dataset
    test_dataset = SoilClassificationDataset(
        dataset_root / "test" / "images",
        dataset_root / "test" / "labels",
        class_names=class_names,
        image_size=224,
        augment=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Load best model checkpoint
    best_model_path = models_dir / "best_soil_classifier.pth"
    if not best_model_path.exists():
        print(f"\n❌ Model not found at {best_model_path}")
        print("Run training first: python -m week3_segmentation.soil_pipeline.train_soil")
        return

    ckpt = torch.load(best_model_path, map_location=device)
    model_name = ckpt.get("model_name", "resnet18")
    ckpt_class_names = ckpt.get("class_names", class_names)

    model = build_model(model_name, len(ckpt_class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"✓ Loaded model '{model_name}' from {best_model_path}")

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    print(f"\nEvaluating on {len(test_dataset)} test samples...")
    test_metrics = evaluate_classification(model, test_loader, criterion, device)

    print("\n" + "=" * 70)
    print("SOIL 4-CLASS CLASSIFICATION - TEST RESULTS")
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

    # Save metrics
    conf_mat = test_metrics["confusion_matrix"]
    metrics_out = {
        "loss": test_metrics["loss"],
        "accuracy": test_metrics["accuracy"],
        "class_names": class_names,
        "per_class": test_metrics["per_class"],
        "confusion_matrix": conf_mat.tolist(),
    }

    metrics_file = results_dir / "metrics.json"
    with metrics_file.open("w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n✓ Saved metrics to {metrics_file}")

    # Save confusion matrix figure
    cm_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(conf_mat, class_names, cm_path)
    print(f"✓ Saved confusion matrix plot to {cm_path}")


if __name__ == "__main__":  # pragma: no cover
    main()


