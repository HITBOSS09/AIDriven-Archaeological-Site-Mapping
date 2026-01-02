
#Week 2: Dataset Preprocessing and Validation Script

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import hashlib
import json
from collections import defaultdict, Counter
import yaml


class DatasetPreprocessor:
    #Handles dataset preprocessing and validation
    
    def __init__(self, dataset_path, output_path, target_size=(640, 640)):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.target_size = target_size
        self.stats = {
            'total_images': 0,
            'corrupted_images': [],
            'duplicate_images': [],
            'missing_labels': [],
            'invalid_annotations': [],
            'class_distribution': defaultdict(int),
            'split_distribution': {}
        }
        
    def validate_image(self, image_path):
        """Validate if image can be loaded and is not corrupted"""
        try:
            img = Image.open(image_path)
            img.verify()  # Verify image integrity
            img = Image.open(image_path)  # Reopen after verify
            img.load()  # Actually load the image data
            return True, img
        except Exception as e:
            return False, str(e)
    
    def compute_image_hash(self, image_path):
        """Compute hash of image for duplicate detection"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    def validate_yolo_annotation(self, label_path, img_width, img_height):
        """Validate YOLO format annotation file (supports both detection and segmentation)"""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            valid_boxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    return False, "Invalid format: expected at least 5 values per line"
                
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                
                # Check format: detection (4 values) or segmentation (multiple pairs)
                if len(coords) == 4:
                    # Detection format: x_center, y_center, width, height
                    x_center, y_center, width, height = coords
                    if not all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
                        return False, "Coordinates not normalized (0-1 range)"
                    valid_boxes.append((class_id, *coords))
                elif len(coords) % 2 == 0:
                    # Segmentation format: x1, y1, x2, y2, ..., xn, yn
                    if not all(0 <= val <= 1 for val in coords):
                        return False, "Coordinates not normalized (0-1 range)"
                    valid_boxes.append((class_id, *coords))
                else:
                    return False, f"Invalid format: odd number of coordinate values ({len(coords)})"
            
            return True, valid_boxes
        except Exception as e:
            return False, str(e)
    
    def preprocess_image(self, image_path, output_path):
        """Resize and normalize image for YOLOv8"""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return False, "Failed to load image"
            
            # Resize to target size
            img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            # Save preprocessed image
            cv2.imwrite(str(output_path), img_resized)
            return True, img_resized.shape
        except Exception as e:
            return False, str(e)
    
    def find_duplicates(self, image_paths):
        """Find duplicate images based on content hash"""
        hash_to_paths = defaultdict(list)
        
        for img_path in image_paths:
            img_hash = self.compute_image_hash(img_path)
            if img_hash:
                hash_to_paths[img_hash].append(img_path)
        
        duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
        return duplicates
    
    def process_split(self, split_name, source_images_dir, source_labels_dir):
        """Process a single dataset split (train/val/test)"""
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} split...")
        print(f"{'='*60}")
        
        # Create output directories
        output_images_dir = self.output_path / split_name / 'images'
        output_labels_dir = self.output_path / split_name / 'labels'
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(Path(source_images_dir).glob('*.jpg')) + \
                     list(Path(source_images_dir).glob('*.png')) + \
                     list(Path(source_images_dir).glob('*.jpeg'))
        
        print(f"Found {len(image_files)} images")
        
        # Find duplicates
        print("Checking for duplicates...")
        duplicates = self.find_duplicates(image_files)
        if duplicates:
            print(f"⚠ Found {len(duplicates)} sets of duplicate images")
            self.stats['duplicate_images'].extend(duplicates.values())
        
        # Process each image
        valid_count = 0
        for img_path in image_files:
            # Validate image
            is_valid, result = self.validate_image(img_path)
            if not is_valid:
                print(f"✗ Corrupted image: {img_path.name} - {result}")
                self.stats['corrupted_images'].append(str(img_path))
                continue
            
            # Check for corresponding label
            label_path = Path(source_labels_dir) / f"{img_path.stem}.txt"
            if not label_path.exists():
                print(f"⚠ Missing label: {img_path.name}")
                self.stats['missing_labels'].append(str(img_path))
                continue
            
            # Validate annotation
            img = result
            is_valid_annot, annot_result = self.validate_yolo_annotation(
                label_path, img.width, img.height
            )
            
            if not is_valid_annot:
                print(f"✗ Invalid annotation: {img_path.name} - {annot_result}")
                self.stats['invalid_annotations'].append(str(label_path))
                continue
            
            # Count classes
            for box in annot_result:
                class_id = box[0]
                self.stats['class_distribution'][class_id] += 1
            
            # Preprocess and save image
            output_img_path = output_images_dir / img_path.name
            success, shape = self.preprocess_image(img_path, output_img_path)
            
            if not success:
                print(f"✗ Failed to preprocess: {img_path.name} - {shape}")
                continue
            
            # Copy label file
            shutil.copy2(label_path, output_labels_dir / label_path.name)
            
            valid_count += 1
        
        print(f"\n✓ Processed {valid_count}/{len(image_files)} valid images")
        self.stats['split_distribution'][split_name] = valid_count
        self.stats['total_images'] += valid_count
        
        return valid_count
    
    def generate_report(self, report_path):
        """Generate comprehensive dataset quality report"""
        report = {
            'dataset_summary': {
                'total_valid_images': self.stats['total_images'],
                'target_image_size': f"{self.target_size[0]}x{self.target_size[1]}",
                'split_distribution': self.stats['split_distribution']
            },
            'data_quality': {
                'corrupted_images_count': len(self.stats['corrupted_images']),
                'duplicate_sets_count': len(self.stats['duplicate_images']),
                'missing_labels_count': len(self.stats['missing_labels']),
                'invalid_annotations_count': len(self.stats['invalid_annotations'])
            },
            'class_distribution': dict(self.stats['class_distribution']),
            'issues': {
                'corrupted_images': self.stats['corrupted_images'],
                'duplicate_images': [list(map(str, dup)) for dup in self.stats['duplicate_images']],
                'missing_labels': self.stats['missing_labels'],
                'invalid_annotations': self.stats['invalid_annotations']
            }
        }
        
        # Save JSON report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("DATASET QUALITY REPORT")
        print("="*60)
        print(f"\n Total Valid Images: {report['dataset_summary']['total_valid_images']}")
        print(f" Standardized Size: {report['dataset_summary']['target_image_size']}")
        print(f"\n Split Distribution:")
        for split, count in report['dataset_summary']['split_distribution'].items():
            percentage = (count / self.stats['total_images'] * 100) if self.stats['total_images'] > 0 else 0
            print(f"   {split.capitalize()}: {count} images ({percentage:.1f}%)")
        
        print(f"\n Data Quality:")
        print(f"    Valid Images: {self.stats['total_images']}")
        print(f"    Corrupted: {report['data_quality']['corrupted_images_count']}")
        print(f"    Duplicates: {report['data_quality']['duplicate_sets_count']} sets")
        print(f"    Missing Labels: {report['data_quality']['missing_labels_count']}")
        print(f"   Invalid Annotations: {report['data_quality']['invalid_annotations_count']}")
        
        print(f"\n Class Distribution:")
        for class_id, count in sorted(report['class_distribution'].items()):
            print(f"   Class {class_id}: {count} instances")
        
        print(f"\n✓ Report saved to: {report_path}")
        print("="*60)
        
        return report


def merge_datasets(soil_dataset_path, veg_dataset_path, output_path):
    """Merge soil and vegetation datasets into unified structure"""
    print("\n" + "="*60)
    print("MERGING DATASETS")
    print("="*60)
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read original configs
    with open(Path(soil_dataset_path) / 'data.yaml', 'r') as f:
        soil_config = yaml.safe_load(f)
    
    with open(Path(veg_dataset_path) / 'data.yaml', 'r') as f:
        veg_config = yaml.safe_load(f)
    
    # Create merged class list
    merged_classes = soil_config['names'] + veg_config['names']
    num_classes = len(merged_classes)
    
    print(f"\n Merging Classes:")
    for idx, class_name in enumerate(merged_classes):
        print(f"   Class {idx}: {class_name}")
    
    # Create merged data.yaml
    merged_config = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': num_classes,
        'names': merged_classes
    }
    
    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(merged_config, f, sort_keys=False)
    
    print(f"\n✓ Created merged data.yaml with {num_classes} classes")
    print(f"✓ Output path: {output_path}")
    
    return merged_config


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("WEEK 2: DATASET PREPROCESSING & VALIDATION")
    print("="*60)

    # Define paths relative to the project root (this file lives at project root)
    project_root = Path(__file__).resolve().parent
    base_path = project_root / "dataset"
    soil_dataset = base_path / "Soil detection.v3i.yolov8"
    veg_dataset = base_path / "vegetation segmentation.v4i.yolov8"
    output_base = project_root / "preprocessed_dataset"
    
    # Process Soil Detection Dataset
    print("\n" + " PROCESSING SOIL DETECTION DATASET")
    soil_output = output_base / "soil_detection"
    soil_preprocessor = DatasetPreprocessor(soil_dataset, soil_output, target_size=(640, 640))
    
    for split in ['train', 'valid', 'test']:
        images_dir = soil_dataset / split / 'images'
        labels_dir = soil_dataset / split / 'labels'
        if images_dir.exists() and labels_dir.exists():
            # Map 'valid' to 'val' for YOLOv8 convention
            output_split = 'val' if split == 'valid' else split
            soil_preprocessor.process_split(output_split, images_dir, labels_dir)
    
    soil_report = soil_preprocessor.generate_report(soil_output / "quality_report.json")
    
    # Process Vegetation Segmentation Dataset
    print("\n\n" + " PROCESSING VEGETATION SEGMENTATION DATASET")
    veg_output = output_base / "vegetation_detection"
    veg_preprocessor = DatasetPreprocessor(veg_dataset, veg_output, target_size=(640, 640))
    
    for split in ['train', 'valid', 'test']:
        images_dir = veg_dataset / split / 'images'
        labels_dir = veg_dataset / split / 'labels'
        if images_dir.exists() and labels_dir.exists():
            output_split = 'val' if split == 'valid' else split
            veg_preprocessor.process_split(output_split, images_dir, labels_dir)
    
    veg_report = veg_preprocessor.generate_report(veg_output / "quality_report.json")
    
    # Update data.yaml files for each dataset
    for dataset_output, original_yaml in [(soil_output, soil_dataset / 'data.yaml'),
                                           (veg_output, veg_dataset / 'data.yaml')]:
        with open(original_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update paths to be absolute
        config['path'] = str(dataset_output.absolute())
        config['train'] = 'train/images'
        config['val'] = 'val/images'
        config['test'] = 'test/images'
        
        with open(dataset_output / 'data.yaml', 'w') as f:
            yaml.dump(config, f, sort_keys=False)
    
    print("\n\n" + "="*60)
    print("✓ PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\n📁 Preprocessed datasets saved to:")
    print(f"   • Soil Detection: {soil_output}")
    print(f"   • Vegetation Detection: {veg_output}")
    print(f"\n📊 Quality reports generated:")
    print(f"   • {soil_output / 'quality_report.json'}")
    print(f"   • {veg_output / 'quality_report.json'}")
    

if __name__ == "__main__":
    main()
