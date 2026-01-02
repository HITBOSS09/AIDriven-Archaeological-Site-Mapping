# Soil Pipeline Refactoring Summary

## Overview

This document summarizes the refactoring from **soil semantic segmentation** to **soil image classification**.

## Changes Made

### 1. Analysis Document Created
- **File**: `SOIL_CLASSIFICATION_ANALYSIS.md`
- **Content**: Comprehensive analysis explaining why segmentation is unsuitable for single-class-per-image classification tasks
- **Key Points**:
  - Segmentation assumes spatial boundaries (none exist in our dataset)
  - Majority voting amplifies pixel-level errors
  - Classification directly optimizes for image-level accuracy
  - Texture-based learning (classification) vs region-based learning (segmentation)

### 2. Training Pipeline Updated
- **File**: `week3_segmentation/soil_pipeline/train_soil.py`
- **Changes**:
  - Added ResNet-50 support (was only ResNet18 and EfficientNet-B0)
  - Default model changed to ResNet-50
  - Added academic justification comments in code
  - Already implements image-level classification (no changes needed to core logic)

### 3. Streamlit UI Refactored
- **File**: `streamlit_app.py`
- **Changes**:
  - `load_soil_model()`: Now loads classification model (ResNet-50/EfficientNet-B0) instead of segmentation model
  - `run_soil_pipeline()`: Returns `(class_id, confidence)` instead of segmentation mask
  - Removed `majority_vote()` function (no longer needed)
  - Removed `soil_map_to_color_image()` function (no segmentation visualization)
  - UI updated to show: "Predicted Soil Type: X" with confidence score
  - Removed segmentation visualization options for soil
  - Removed pixel-level ground truth upload for soil
  - Added academic justification comments throughout

### 4. Dataset Conversion Script Created
- **File**: `convert_soil_to_classification_dataset.py`
- **Purpose**: Optional script to convert YOLO format to folder-based structure
- **Note**: The training pipeline already reads from YOLO format labels (extracting class ID only), so this is optional

### 5. Obsolete Code Removed
- Removed segmentation-related functions from Streamlit app
- Removed placeholder functions that returned segmentation masks
- Added comments explaining why functions were removed

## Model Paths

### Old (Segmentation - DO NOT USE)
- `week3_segmentation/soil/models/best_soil_segmentation.pth`

### New (Classification - USE THIS)
- `week3_segmentation/soil_pipeline/models/best_soil_classifier.pth`

## Training Command

```bash
python -m week3_segmentation.soil_pipeline.train_soil \
    --model resnet50 \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.001
```

## Key Differences

| Aspect | Old (Segmentation) | New (Classification) |
|--------|-------------------|---------------------|
| Model | DeepLabV3 | ResNet-50/EfficientNet-B0 |
| Output | Per-pixel class map | Single class ID + confidence |
| Aggregation | Majority voting | Direct prediction |
| Optimization | Pixel-level accuracy | Image-level accuracy |
| Ground Truth | Pixel-level masks | Image-level labels |
| UI Display | Segmentation map overlay | "Predicted: X (confidence: Y%)" |

## Academic Justification

See `SOIL_CLASSIFICATION_ANALYSIS.md` for detailed justification. Summary:

1. **Task Mismatch**: Segmentation assumes spatial boundaries; our dataset has one class per image (no boundaries)
2. **Learning Objective**: Classification learns texture-based features (appropriate for soil types); segmentation learns spatial boundaries (not needed)
3. **Optimization**: Classification directly optimizes image-level accuracy; segmentation optimizes pixel-level accuracy (proxy metric)
4. **Stability**: Classification gives stable predictions; segmentation + majority voting amplifies errors

## Vegetation Pipeline (Unchanged)

The vegetation pipeline remains as **segmentation** because:
- It requires spatial localization (where is vegetation?)
- Multiple regions per image (vegetation and non-vegetation coexist)
- Task requires pixel-level predictions

## Testing Checklist

- [x] Analysis document created
- [x] Training script updated with ResNet-50
- [x] Streamlit UI refactored
- [x] Obsolete code removed
- [ ] Train new classification model
- [ ] Test Streamlit UI with new model
- [ ] Verify predictions are correct and stable

## Next Steps

1. Train the classification model using the updated training script
2. Verify the model path in Streamlit matches the saved model
3. Test the UI with sample images
4. Remove old segmentation model files if desired (backup first)

---

**Refactoring Date**: Generated during project refactoring  
**Status**: Code changes complete, ready for testing

