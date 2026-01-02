---

## Deployment on Render.com

This project is ready for deployment as a Streamlit web service on Render.com.

**Steps:**
1. Push the repository to GitHub.
2. Create a new Web Service on Render.com.
3. Select Python environment (Python 3.10 recommended).
4. Add the following build and start commands:
  - Build: `pip install -r requirements.txt`
  - Start: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
5. Ensure `.streamlit/config.toml` disables CORS and XSRF protection.
6. Confirm that model files (e.g., `week3_segmentation/soil_pipeline/models/best_soil_classifier.pth`) are present and accessible via relative paths.

**Notes:**
- Inference runs on CPU only; slow inference is expected for large models.
- No GPU, database, or background workers required.
- If the model file is missing, the app will show a clear error message.

**Limitations:**
- Soil classification may be slow due to CPU inference.
- Vegetation pipeline is heuristic and does not require a GPU.

**Checklist for Successful Deployment:**
- [x] requirements.txt includes only CPU-compatible packages
- [x] render.yaml or dashboard steps provided
- [x] .streamlit/config.toml disables CORS/XSRF
- [x] No hardcoded ports or absolute paths
- [x] Model loading uses relative paths and shows errors if missing
- [x] Streamlit UI works as expected

---
# Environmental Analysis AI Project

## Overview

This project implements two independent computer vision pipelines:
1. **Vegetation Segmentation**: Pixel-level binary segmentation (where is vegetation?)
2. **Soil Classification**: Image-level 4-class classification (what type of soil?)

---

## Task Formulation Justification

### Why Soil Uses Classification (Not Segmentation)

**Important**: The soil pipeline uses **image-level classification**, not semantic segmentation. This is a deliberate design choice based on the dataset characteristics and task requirements.

**Academic Justification**:
- **Dataset Structure**: Each image contains ONE dominant soil type (no spatial mixing of classes)
- **Task Requirement**: We need to identify "what type of soil is in this image" (texture recognition), not "where are different soil types located" (spatial localization)
- **Learning Objective**: Classification models learn texture-based discriminative features (global image properties), which is appropriate for soil type recognition
- **Optimization**: Direct optimization for image-level accuracy is more effective than segmentation + majority voting

**Why Segmentation Was Unsuitable**:
- Segmentation assumes spatial boundaries between classes (none exist in our dataset)
- Segmentation + majority voting amplifies pixel-level errors and leads to unstable predictions
- The model was forced to learn artificial boundaries where none exist

**See `SOIL_CLASSIFICATION_ANALYSIS.md` for detailed analysis.**

### Why Vegetation Uses Segmentation

Vegetation remains as segmentation because:
- It requires spatial localization ("where is the vegetation?")
- Multiple regions per image (vegetation and non-vegetation coexist)
- Task requires pixel-level predictions for visualization and analysis

---

# Week 2: Dataset Preparation

Week 2 : datasets are preprocessed, validated, and ready for training in Week 3.

---

# Core Deliverables:
1. **`dataset_preprocessing.py`** - Automated preprocessing and validation script
2. **`augmentation_config.yaml`** - Training augmentation configuration
3. **`WEEK2_SUMMARY.md`** - Executive summary with all statistics
4. **`WEEK2_DOCUMENTATION.md`** - Detailed technical documentation
5. **`preprocessed_dataset/`** - Processed datasets ready for training

---

#  Dataset Summary

| Dataset                     | Images | Classes | Format       | Status |
|-----------------------------|--------|---------|--------------|--------|
| **Soil Detection**          | 285    | 4       | Detection    | Ready  |
| **Vegetation Segmentation** | 1,068  | 1       | Segmentation | Ready  |
| **Total**                   | 1,353  | 5       | Mixed        | Ready  |

---

##  Quick Start (Week 3)

### Train Soil Classification & Vegetation Segmentation (main pipelines)

- Soil 4-class **image-level classification** (ResNet-50/EfficientNet-B0):
  ```bash
  python -m week3_segmentation.soil_pipeline.train_soil --model resnet50 --epochs 30
  ```
  **Note**: This is classification (one label per image), NOT segmentation. See task formulation justification above.

- Vegetation **binary semantic segmentation** (U-Net):
  ```bash
  python -m week3_segmentation.vegetation_pipeline.train_vegetation
  ```

> YOLOv8 is used only for an **optional soil detection baseline** under
> `experiments/yolo_baseline/soil_detection_pipeline.py` and is not required
> for the main pipelines.

---

## 📁 File Structure

```
C:\Users\Hitanshu\Downloads\Infosys\
│
├── README.md                        ← You are here
├── WEEK2_SUMMARY.md                 ← Executive summary with statistics
├── WEEK2_DOCUMENTATION.md           ← Detailed technical documentation
├── dataset_preprocessing.py         ← Preprocessing automation script
├── augmentation_config.yaml         ← Training configuration
│
├── preprocessed_dataset/
│   ├── soil_detection/              ← Ready for training
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   ├── data.yaml
│   │   └── quality_report.json
│   │
│   └── vegetation_detection/        ← Ready for training
│       ├── train/
│       ├── val/
│       ├── test/
│       ├── data.yaml
│       └── quality_report.json
│
└── dataset/                         ← Original datasets (unchanged)
    ├── Soil detection.v3i.yolov8/
    └── vegetation segmentation.v4i.yolov8/
```

---

## ✅ Quality Metrics

### Data Quality: 100%
- ✓ Zero corrupted images
- ✓ Zero missing labels
- ✓ All annotations validated
- ✓ Duplicates identified and documented

### Processing Status:
- ✓ All 1,353 images processed
- ✓ All images resized to 640×640
- ✓ Train/val/test splits organized
- ✓ YOLOv8 configuration files created

---

## 📚 Documentation Guide

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **README.md** | Quick reference | Start here |
| **WEEK2_SUMMARY.md** | Complete overview with stats | Before Week 3 |
| **WEEK2_DOCUMENTATION.md** | Technical deep dive | When implementing |
| **quality_report.json** | Detailed data metrics | When analyzing data |

---

## 🎯 Week 2 Checklist

- [x] Image annotation (pre-annotated datasets used)
- [x] Dataset preprocessing (resizing, normalization)
- [x] Data cleaning (validation, duplicate detection)
- [x] Quality assurance (comprehensive validation)
- [x] Dataset splitting (train/val/test organization)
- [x] Augmentation strategy (configuration created)
- [x] Documentation (complete technical specs)
- [x] Quality reports (JSON reports generated)

---

## 🔍 Key Statistics

**Soil Detection:**
- 285 images @ 640×640
- 4 classes (Alluvial, Black, Clay, Red Soil)
- 348 bounding box annotations
- 69.8% train / 20.4% val / 9.8% test

**Vegetation Segmentation:**
- 1,068 images @ 640×640
- 1 class (vegetation)
- 2,815 segmentation polygons
- 88.5% train / 7.8% val / 3.7% test

---

## 🛠️ Troubleshooting

**Issue:** Import errors when running scripts  
Install the core dependencies (YOLO/ultralytics is only needed for the optional baseline):
```bash
pip install opencv-python pillow numpy pyyaml torch torchvision tqdm matplotlib
```

**Issue:** GPU not detected
```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU available
```

**Issue:** Need to re-run preprocessing
```bash
python dataset_preprocessing.py
```

---

## 📞 Support Files

- **Preprocessing Script:** `dataset_preprocessing.py`
- **Augmentation Config:** `augmentation_config.yaml`
- **Quality Reports:** `preprocessed_dataset/*/quality_report.json`
- **Dataset Configs:** `preprocessed_dataset/*/data.yaml`

---

## 🎓 Next Steps

1.  Week 2: Dataset preparation **← COMPLETE**
2.  Week 3: Model training **← READY TO START**
3.  Week 4: Model evaluation
4.  Week 5: Model deployment

---

## 💡 Tips for Week 3

1. **Start Small:** Use `yolov8n` for fast experiments
2. **Monitor Training:** Watch loss curves and validation metrics
3. **GPU Memory:** Adjust batch size if out-of-memory errors occur
4. **Early Stopping:** Patience=50 will stop if no improvement
5. **Visualize Results:** YOLOv8 automatically saves training plots

---

**Status:**  Week 2 Complete  
**Ready for:** Week 3 Training  
**Quality Score:** 100%  
**Datasets Processed:** 2 (Soil + Vegetation)  
**Total Images:** 1,353  
**All Systems:** GO ✅

---

## Recent Updates

### Soil Pipeline Refactoring (Classification)

The soil pipeline has been refactored from segmentation to classification:
- **Old Approach**: DeepLabV3 segmentation + majority voting (unstable, incorrect predictions)
- **New Approach**: ResNet-50/EfficientNet-B0 classification (direct image-level prediction)
- **Model Path**: `week3_segmentation/soil_pipeline/models/best_soil_classifier.pth`
- **See**: `SOIL_CLASSIFICATION_ANALYSIS.md` and `REFACTORING_SUMMARY.md` for details

### Documentation

- **Analysis Document**: `SOIL_CLASSIFICATION_ANALYSIS.md` - Detailed explanation of task formulation
- **Refactoring Summary**: `REFACTORING_SUMMARY.md` - Summary of code changes
- **Dataset Conversion**: `convert_soil_to_classification_dataset.py` - Optional folder-based dataset structure
