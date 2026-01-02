"""
VEGETATION EVALUATION SCRIPT
============================
Load best vegetation model and run evaluation on test split for binary semantic
segmentation of vegetation vs background.

Metrics:
  - IoU
  - Dice

Outputs:
  - Binary prediction visualizations
  - coverage_stats.json with vegetation coverage statistics
"""

import sys
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from training script
from vegetation_pipeline.train_vegetation import (
    VegetationDataset,
    UNet,
    evaluate,
    save_visualizations,
    compute_coverage_stats,
)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    dataset_root = project_root / 'preprocessed_dataset' / 'vegetation_detection'
    masks_root = project_root / 'week3_segmentation' / 'masks' / 'vegetation'
    models_dir = script_dir / 'models'
    results_dir = script_dir / 'results'
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test dataset (using precomputed masks)
    test_dataset = VegetationDataset(
        dataset_root / "test" / "images",
        masks_root / "test" / "masks",
        image_size=256,
        augment=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Load model
    model = UNet(in_channels=3, out_channels=1).to(device)
    best_model_path = models_dir / 'best_vegetation_unet.pth'
    
    if not best_model_path.exists():
        print(f"\n❌ Model not found at {best_model_path}")
        print("Run training first: python vegetation_pipeline/train_vegetation.py --epochs 30")
        return
    
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"✓ Loaded model from {best_model_path}")
    
    # Evaluate
    import torch.nn as nn
    bce_criterion = nn.BCEWithLogitsLoss()

    print(f"\nEvaluating on {len(test_dataset)} test samples...")
    test_metrics = evaluate(model, test_loader, bce_criterion, device, split_name='Test')
    
    # Print results
    print("\n" + "=" * 70)
    print("VEGETATION BINARY SEGMENTATION - TEST RESULTS")
    print("=" * 70)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test IoU:  {test_metrics['iou']:.4f}")
    print(f"Test Dice: {test_metrics['dice']:.4f}")
    print("=" * 70)
    
    # Save visualizations
    vis_dir = results_dir / 'test_visualizations'
    print(f"\nSaving visualizations to {vis_dir}...")
    save_visualizations(model, test_loader, device, vis_dir, max_samples=10)

    # Save metrics
    metrics_file = results_dir / 'final_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"✓ Saved metrics to {metrics_file}")

    # Coverage statistics
    coverage_file = results_dir / 'coverage_stats.json'
    print(f"\nComputing coverage statistics to {coverage_file}...")
    compute_coverage_stats(model, test_loader, device, coverage_file)
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
