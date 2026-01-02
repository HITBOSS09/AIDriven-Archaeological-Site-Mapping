"""
Convert soil detection dataset from YOLO format to folder-based classification format.

This script creates a folder structure suitable for image classification:
  soil_classification/
  ├── train/
  │   ├── red_soil/
  │   ├── black_soil/
  │   ├── clay_soil/
  │   └── alluvial_soil/
  ├── val/
  │   └── (same structure)
  └── test/
      └── (same structure)

Each image is placed in the folder corresponding to its class label.
This structure is optional - the classification pipeline can also read directly from
YOLO format labels (which is what train_soil.py does by default).

Usage:
  python convert_soil_to_classification_dataset.py
"""

from pathlib import Path
import shutil
import yaml
from collections import defaultdict


def sanitize_class_name(name: str) -> str:
    """Convert class name to folder-safe name (lowercase, underscores)."""
    return name.lower().replace(" ", "_")


def convert_dataset(
    source_root: Path,
    target_root: Path,
    class_names: list[str],
) -> dict:
    """
    Convert YOLO-format dataset to folder-based classification dataset.
    
    Args:
        source_root: Root of preprocessed_dataset/soil_detection
        target_root: Root of new soil_classification directory
        class_names: List of class names (must match YOLO class IDs)
    
    Returns:
        Statistics dictionary with counts per class per split
    """
    stats = defaultdict(lambda: defaultdict(int))
    
    # Create class name mapping
    class_id_to_folder = {
        idx: sanitize_class_name(name) for idx, name in enumerate(class_names)
    }
    
    for split in ["train", "val", "test"]:
        source_images_dir = source_root / split / "images"
        source_labels_dir = source_root / split / "labels"
        
        if not source_images_dir.exists():
            print(f"⚠ Warning: {source_images_dir} does not exist, skipping {split}")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # Process each image
        image_files = list(source_images_dir.glob("*.jpg")) + \
                     list(source_images_dir.glob("*.jpeg")) + \
                     list(source_images_dir.glob("*.png"))
        
        for img_path in image_files:
            label_path = source_labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                print(f"⚠ Warning: No label file for {img_path.name}, skipping")
                continue
            
            # Read first class ID from label file
            with label_path.open() as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                if not lines:
                    continue
                parts = lines[0].split()
                try:
                    class_id = int(parts[0])
                except (ValueError, IndexError):
                    continue
            
            if class_id not in class_id_to_folder:
                print(f"⚠ Warning: Invalid class_id {class_id} for {img_path.name}, skipping")
                continue
            
            # Determine target folder
            class_folder = class_id_to_folder[class_id]
            target_dir = target_root / split / class_folder
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy image to class folder
            target_path = target_dir / img_path.name
            shutil.copy2(img_path, target_path)
            
            stats[split][class_folder] += 1
        
        # Print statistics for this split
        print(f"  {split} statistics:")
        total = sum(stats[split].values())
        for class_folder, count in sorted(stats[split].items()):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"    {class_folder}: {count} images ({percentage:.1f}%)")
    
    return dict(stats)


def validate_dataset(target_root: Path, class_names: list[str]) -> None:
    """
    Validate the converted dataset:
    - Check for empty classes
    - Check for mixed-class folders
    - Report class imbalance
    """
    print("\n" + "="*60)
    print("DATASET VALIDATION")
    print("="*60)
    
    expected_classes = {sanitize_class_name(name) for name in class_names}
    issues = []
    
    for split in ["train", "val", "test"]:
        split_dir = target_root / split
        if not split_dir.exists():
            continue
        
        print(f"\n{split.upper()} split:")
        found_classes = set()
        
        for class_folder in split_dir.iterdir():
            if not class_folder.is_dir():
                continue
            
            found_classes.add(class_folder.name)
            
            # Check if folder is empty
            images = list(class_folder.glob("*.jpg")) + \
                    list(class_folder.glob("*.jpeg")) + \
                    list(class_folder.glob("*.png"))
            
            if len(images) == 0:
                issues.append(f"{split}/{class_folder.name} is empty")
                print(f"  ❌ {class_folder.name}: EMPTY")
            else:
                print(f"  ✅ {class_folder.name}: {len(images)} images")
        
        # Check for unexpected classes
        unexpected = found_classes - expected_classes
        if unexpected:
            issues.append(f"{split} contains unexpected class folders: {unexpected}")
        
        # Check for missing classes
        missing = expected_classes - found_classes
        if missing:
            issues.append(f"{split} is missing class folders: {missing}")
    
    # Check for class imbalance
    train_dir = target_root / "train"
    if train_dir.exists():
        class_counts = {}
        for class_folder in train_dir.iterdir():
            if class_folder.is_dir():
                images = list(class_folder.glob("*.jpg")) + \
                        list(class_folder.glob("*.jpeg")) + \
                        list(class_folder.glob("*.png"))
                class_counts[class_folder.name] = len(images)
        
        if class_counts:
            total = sum(class_counts.values())
            max_count = max(class_counts.values())
            max_frac = max_count / total if total > 0 else 0
            
            print(f"\nClass imbalance check (train split):")
            print(f"  Maximum class fraction: {max_frac:.1%}")
            if max_frac > 0.7:
                issues.append(f"Severe class imbalance: max class is {max_frac:.1%} of dataset")
                print(f"  ⚠ Warning: Class imbalance exceeds 70/30 threshold")
            else:
                print(f"  ✅ Class distribution is acceptable")
    
    # Summary
    print("\n" + "="*60)
    if issues:
        print("VALIDATION ISSUES FOUND:")
        for issue in issues:
            print(f"  ❌ {issue}")
        print("\n⚠ Please review and fix issues before training.")
    else:
        print("✅ VALIDATION PASSED: Dataset structure is correct")
    print("="*60)


def main():
    """Main execution function."""
    project_root = Path(__file__).resolve().parent
    source_root = project_root / "preprocessed_dataset" / "soil_detection"
    target_root = project_root / "soil_classification"
    
    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")
    
    # Load class names from data.yaml
    data_yaml = source_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    
    with data_yaml.open() as f:
        config = yaml.safe_load(f)
    
    class_names = list(config.get("names", []))
    if len(class_names) != 4:
        print(f"⚠ Warning: Expected 4 classes, found {len(class_names)}")
    
    print("="*60)
    print("SOIL DATASET CONVERSION: YOLO → CLASSIFICATION FORMAT")
    print("="*60)
    print(f"\nSource: {source_root}")
    print(f"Target: {target_root}")
    print(f"\nClasses:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name} → {sanitize_class_name(name)}/")
    
    # Ask for confirmation if target exists
    if target_root.exists():
        response = input(f"\n⚠ Target directory {target_root} exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
        shutil.rmtree(target_root)
    
    # Convert dataset
    stats = convert_dataset(source_root, target_root, class_names)
    
    # Validate
    validate_dataset(target_root, class_names)
    
    print(f"\n✅ Conversion complete!")
    print(f"Dataset saved to: {target_root}")
    print("\nNote: The classification training script (train_soil.py) can work with")
    print("either format. This folder structure is optional but may be useful for")
    print("other frameworks or manual inspection.")


if __name__ == "__main__":
    main()

