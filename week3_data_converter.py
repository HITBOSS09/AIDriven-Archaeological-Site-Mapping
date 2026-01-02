"""
Week 3: YOLO Annotation to Segmentation Mask Converter

Converts YOLO format annotations (polygons and bounding boxes) to 
semantic segmentation masks for U-Net and DeepLabV3+ training.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import json


class YOLOToMaskConverter:
    """Convert YOLO annotations to segmentation masks"""
    
    def __init__(self, dataset_path, output_path, img_size=(640, 640)):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.img_size = img_size
        self.stats = {
            'total_processed': 0,
            'polygons_converted': 0,
            'bboxes_converted': 0,
            'splits': {}
        }
    
    def yolo_polygon_to_mask(self, polygon_points, img_size, class_id):
        """
        Convert YOLO polygon format to segmentation mask
        
        Args:
            polygon_points: List of normalized (x, y) coordinates
            img_size: (height, width) of output mask
            class_id: Integer class ID (1-indexed for mask)
        """
        mask = np.zeros(img_size, dtype=np.uint8)
        
        # Convert normalized coordinates to pixel coordinates
        points = []
        for i in range(0, len(polygon_points), 2):
            x = int(polygon_points[i] * img_size[1])
            y = int(polygon_points[i + 1] * img_size[0])
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Fill polygon with class_id
        cv2.fillPoly(mask, [points], class_id + 1)  # +1 to make background 0
        
        return mask
    
    def yolo_bbox_to_mask(self, bbox, img_size, class_id):
        """
        Convert YOLO bounding box to segmentation mask
        
        Args:
            bbox: [x_center, y_center, width, height] normalized
            img_size: (height, width) of output mask
            class_id: Integer class ID (1-indexed for mask)
        """
        mask = np.zeros(img_size, dtype=np.uint8)
        
        x_center, y_center, width, height = bbox
        
        # Convert to pixel coordinates
        x_center_px = int(x_center * img_size[1])
        y_center_px = int(y_center * img_size[0])
        width_px = int(width * img_size[1])
        height_px = int(height * img_size[0])
        
        # Calculate corners
        x1 = max(0, x_center_px - width_px // 2)
        y1 = max(0, y_center_px - height_px // 2)
        x2 = min(img_size[1], x_center_px + width_px // 2)
        y2 = min(img_size[0], y_center_px + height_px // 2)
        
        # Fill rectangle with class_id
        mask[y1:y2, x1:x2] = class_id + 1  # +1 to make background 0
        
        return mask
    
    def convert_annotation(self, label_path, img_size, is_segmentation=True):
        """
        Convert single annotation file to mask
        
        Args:
            label_path: Path to YOLO .txt annotation file
            img_size: (height, width) of output mask
            is_segmentation: True for polygon format, False for bbox format
        """
        mask = np.zeros(img_size, dtype=np.uint8)
        
        if not label_path.exists():
            return mask
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            if is_segmentation and len(coords) > 4:
                # Polygon format (segmentation)
                poly_mask = self.yolo_polygon_to_mask(coords, img_size, class_id)
                # For multi-class, take the max (later class overwrites)
                mask = np.maximum(mask, poly_mask)
                self.stats['polygons_converted'] += 1
            elif len(coords) == 4:
                # Bounding box format (detection)
                bbox_mask = self.yolo_bbox_to_mask(coords, img_size, class_id)
                mask = np.maximum(mask, bbox_mask)
                self.stats['bboxes_converted'] += 1
        
        return mask
    
    def process_split(self, split_name, images_dir, labels_dir, output_masks_dir, is_segmentation=True):
        """Process all images in a split"""
        print(f"\nProcessing {split_name} split...")
        
        # Get all images
        image_files = list(Path(images_dir).glob('*.jpg')) + \
                     list(Path(images_dir).glob('*.png')) + \
                     list(Path(images_dir).glob('*.jpeg'))
        
        output_masks_dir.mkdir(parents=True, exist_ok=True)
        
        split_count = 0
        for img_path in tqdm(image_files, desc=f"{split_name}"):
            # Load image to get size
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_size = (img.shape[0], img.shape[1])
            
            # Get corresponding label
            label_path = Path(labels_dir) / f"{img_path.stem}.txt"
            
            # Convert to mask
            mask = self.convert_annotation(label_path, img_size, is_segmentation)
            
            # Save mask
            mask_filename = f"{img_path.stem}.png"
            mask_path = output_masks_dir / mask_filename
            cv2.imwrite(str(mask_path), mask)
            
            split_count += 1
        
        self.stats['splits'][split_name] = split_count
        self.stats['total_processed'] += split_count
        print(f"✓ Processed {split_count} masks for {split_name}")
    
    def convert_dataset(self, dataset_name, is_segmentation=True):
        """
        Convert entire dataset (train/val/test splits)
        
        Args:
            dataset_name: 'soil' or 'vegetation'
            is_segmentation: True for polygon masks, False for bounding boxes
        """
        print(f"\n{'='*60}")
        print(f"Converting {dataset_name.upper()} dataset to masks")
        print(f"{'='*60}")
        
        # Reset stats
        self.stats = {
            'total_processed': 0,
            'polygons_converted': 0,
            'bboxes_converted': 0,
            'splits': {}
        }
        
        # Process each split
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_path / split / 'images'
            labels_dir = self.dataset_path / split / 'labels'
            output_masks_dir = self.output_path / split / 'masks'
            
            if images_dir.exists() and labels_dir.exists():
                self.process_split(split, images_dir, labels_dir, output_masks_dir, is_segmentation)
        
        # Save statistics
        stats_path = self.output_path / 'conversion_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Conversion Summary for {dataset_name.upper()}")
        print(f"{'='*60}")
        print(f"Total masks created: {self.stats['total_processed']}")
        print(f"Polygons converted: {self.stats['polygons_converted']}")
        print(f"Bounding boxes converted: {self.stats['bboxes_converted']}")
        for split, count in self.stats['splits'].items():
            print(f"  {split}: {count} masks")
        print(f"✓ Statistics saved to: {stats_path}")


def create_dataset_info(masks_path, original_data_yaml, output_yaml):
    """Create dataset info YAML for segmentation"""
    with open(original_data_yaml, 'r') as f:
        original_config = yaml.safe_load(f)
    
    # Create new config for segmentation
    seg_config = {
        'dataset_path': str(masks_path),
        'train': str(masks_path / 'train'),
        'val': str(masks_path / 'val'),
        'test': str(masks_path / 'test'),
        'num_classes': original_config['nc'] + 1,  # +1 for background
        'class_names': ['background'] + original_config['names'],
        'image_size': [640, 640],
        'original_dataset': str(original_data_yaml)
    }
    
    with open(output_yaml, 'w') as f:
        yaml.dump(seg_config, f, sort_keys=False)
    
    print(f"✓ Created dataset config: {output_yaml}")
    return seg_config


def main():
    """Main conversion pipeline"""
    print("\n" + "="*60)
    print("WEEK 3: YOLO TO SEGMENTATION MASK CONVERTER")
    print("="*60)

    # Paths relative to project root (this file lives at project root)
    project_root = Path(__file__).resolve().parent
    preprocessed_path = project_root / "preprocessed_dataset"
    output_base = project_root / "week3_segmentation" / "masks"
    
    # Convert Vegetation Dataset (Segmentation - polygons)
    print("\n[1/2] Converting Vegetation Segmentation Dataset...")
    veg_dataset = preprocessed_path / "vegetation_detection"
    veg_output = output_base / "vegetation"
    
    veg_converter = YOLOToMaskConverter(veg_dataset, veg_output)
    veg_converter.convert_dataset('vegetation', is_segmentation=True)
    
    # Create dataset config
    create_dataset_info(
        veg_output,
        veg_dataset / 'data.yaml',
        veg_output / 'dataset_info.yaml'
    )
    
    # Convert Soil Detection Dataset (Detection - bounding boxes)
    print("\n[2/2] Converting Soil Detection Dataset...")
    soil_dataset = preprocessed_path / "soil_detection"
    soil_output = output_base / "soil"
    
    soil_converter = YOLOToMaskConverter(soil_dataset, soil_output)
    soil_converter.convert_dataset('soil', is_segmentation=False)
    
    # Create dataset config
    create_dataset_info(
        soil_output,
        soil_dataset / 'data.yaml',
        soil_output / 'dataset_info.yaml'
    )
    
    print("\n" + "="*60)
    print("✓ MASK CONVERSION COMPLETE")
    print("="*60)
    print(f"\n📁 Masks saved to:")
    print(f"  • Vegetation: {veg_output}")
    print(f"  • Soil: {soil_output}")
    print(f"\n✓ Ready for model training!")


if __name__ == "__main__":
    main()
