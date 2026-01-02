"""
Environmental Analysis Dashboard - Two Independent Pipelines

PART A: SOIL (Image-Level Classification Only)
PART B: VEGETATION (Pixel-Level Segmentation Only)
"""

import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import io

# PyTorch imports
import torch
import torchvision.transforms.functional as TF

# ============================================================================
# PART A: SOIL CLASSIFICATION PIPELINE (COMPLETELY INDEPENDENT)
# ============================================================================

# Soil class name mapping - MUST match training order from data.yaml
# Training order (from preprocessed_dataset/soil_detection/data.yaml):
#   0: Alluvial Soil
#   1: Black Soil
#   2: Clay Soil
#   3: Red Soil
SOIL_CLASS_NAMES = {
    0: "Alluvial Soil",
    1: "Black Soil",
    2: "Clay Soil",
    3: "Red Soil",
}


@st.cache_resource
def _load_soil_classification_model(path: str = "week3_segmentation/soil_pipeline/models/best_soil_classifier.pth"):
    """
    Load the trained soil CLASSIFICATION model.
    
    Returns: (model, class_names_list, class_name_mapping_dict)
    """
    from week3_segmentation.soil_pipeline.train_soil import build_model

    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load soil classification model at {path}: {e}\n"
            f"Train the model first: python -m week3_segmentation.soil_pipeline.train_soil"
        )

    if isinstance(ckpt, dict):
        model_name = ckpt.get("model_name", "resnet50")
        class_names = ckpt.get("class_names", None)
        num_classes = len(class_names) if class_names is not None else 4
        state_dict = ckpt.get("model_state_dict", ckpt)
    else:
        raise ValueError("Unsupported checkpoint format")

    model = build_model(model_name, num_classes, pretrained=False)
    new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state)
    model.to("cpu")
    model.eval()

    # Use class_names from checkpoint (matches training order from data.yaml)
    # This is the SINGLE SOURCE OF TRUTH - matches training order exactly
    if class_names is None:
        # Fallback to SOIL_CLASS_NAMES if checkpoint doesn't have class_names
        # This should not happen if model was trained with current training script
        class_names = [SOIL_CLASS_NAMES.get(i, f"Class {i}") for i in range(num_classes)]
    
    # Build mapping dict from checkpoint class_names (ensures consistency)
    # This mapping MUST match the training order printed during training
    class_name_mapping = {idx: name for idx, name in enumerate(class_names)}
    
    return model, class_names, class_name_mapping


def _create_soil_augmentations(image: Image.Image) -> list[Image.Image]:
    """
    Create test-time augmentations for soil classification.
    Returns: List of augmented images (original + 3 augmentations)
    """
    augmentations = [image]  # Original
    
    # Horizontal flip
    augmentations.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    
    # Brightness +10%
    enhancer = ImageEnhance.Brightness(image)
    augmentations.append(enhancer.enhance(1.10))
    
    # Contrast +10%
    enhancer = ImageEnhance.Contrast(image)
    augmentations.append(enhancer.enhance(1.10))
    
    return augmentations


def _preprocess_soil_image(image: Image.Image) -> torch.Tensor:
    """Preprocess single image for soil classification model."""
    target_size = (224, 224)
    img_resized = image.resize(target_size, resample=Image.BILINEAR)
    img_t = TF.to_tensor(img_resized)
    img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img_t.unsqueeze(0)  # Add batch dimension


def _run_soil_classification_with_tta(image: Image.Image) -> tuple[int, float, np.ndarray, bool, dict]:
    """
    Run soil classification with Test-Time Augmentation (TTA) and confidence gating.
    
    Returns:
        (predicted_class_id, confidence_score, all_probabilities, is_gated, class_name_mapping)
        - predicted_class_id: Class index (0-3) or -1 if gated
        - confidence_score: Highest probability
        - all_probabilities: Array of probabilities for all classes
        - is_gated: True if confidence gating triggered (output should be "Mixed/Transitional")
        - class_name_mapping: Dict mapping class_id -> class_name (from checkpoint)
    """
    model, class_names, class_name_mapping = _load_soil_classification_model()
    
    # Create augmentations
    augmented_images = _create_soil_augmentations(image)
    
    # Run inference on each augmentation
    all_probs = []
    with torch.no_grad():
        for aug_img in augmented_images:
            img_t = _preprocess_soil_image(aug_img)
            logits = model(img_t)  # shape: (1, num_classes)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy()[0])
    
    # Average probabilities across augmentations
    avg_probs = np.mean(all_probs, axis=0)
    
    # Get top two probabilities
    sorted_indices = np.argsort(avg_probs)[::-1]
    p1 = float(avg_probs[sorted_indices[0]])  # Highest
    p2 = float(avg_probs[sorted_indices[1]])  # Second highest
    pred_class_id = int(sorted_indices[0])
    
    # Confidence gating
    is_gated = (p1 < 0.55) or ((p1 - p2) < 0.10)
    
    if is_gated:
        pred_class_id = -1  # Special value for "Mixed/Transitional"
    
    return pred_class_id, p1, avg_probs, is_gated, class_name_mapping


def _format_soil_output(class_id: int, confidence: float, is_gated: bool, class_name_mapping: dict) -> tuple[str, str]:
    """
    Format soil classification output with human-readable labels.
    
    NEVER shows "Class 0", "Class 1" - always uses explicit soil type names.
    Uses class_name_mapping from checkpoint (single source of truth).
    """
    if is_gated or class_id == -1:
        soil_name = "Mixed / Transitional Soil"
        confidence_text = f"Confidence: {confidence:.2%} (Low confidence - multiple classes possible)"
    else:
        # Safety assertion: ensure class_id is valid
        assert class_id in class_name_mapping, f"Invalid class_id: {class_id}. Must be in {list(class_name_mapping.keys())}"
        soil_name = class_name_mapping[class_id]  # Use mapping from checkpoint (matches training order)
        confidence_text = f"Confidence: {confidence:.2%}"
    
    return soil_name, confidence_text


# ============================================================================
# PART B: VEGETATION SEGMENTATION PIPELINE (COMPLETELY INDEPENDENT)
# ============================================================================

def _predict_vegetation_mask(image: Image.Image) -> np.ndarray:
    """
    Generate vegetation segmentation mask (placeholder for YOLOv8-SEG).
    Returns: Binary mask (H, W) with values 0 (background) or 1 (vegetation)
    """
    # TODO: Replace with actual YOLOv8-SEG model inference
    arr = np.array(image.convert("RGB"))
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    veg_score = g.astype(float) - 0.5 * (r.astype(float) + b.astype(float))
    mask = (veg_score > 10).astype(np.uint8)
    return mask


def _compute_vegetation_coverage(mask: np.ndarray) -> float:
    """
    Compute vegetation coverage ratio.
    Returns: Coverage ratio (0.0 to 1.0)
    """
    if mask.size == 0:
        return 0.0
    vegetation_pixels = (mask > 0).sum()
    total_pixels = mask.size
    return float(vegetation_pixels) / float(total_pixels)


def _overlay_vegetation_mask(image: Image.Image, mask: np.ndarray, color=(0, 255, 0), alpha: float = 0.5) -> Image.Image:
    """Overlay vegetation segmentation mask on image (vegetation-specific function)."""
    base = image.convert("RGBA")
    h, w = mask.shape
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay_arr = np.array(overlay)
    color_rgba = (*color, int(255 * alpha))
    overlay_arr[mask.astype(bool)] = color_rgba
    overlay = Image.fromarray(overlay_arr)
    blended = Image.alpha_composite(base, overlay)
    return blended


def _compute_vegetation_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU for vegetation segmentation (vegetation-specific metric)."""
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    if union == 0:
        return float("nan")
    return float(intersection) / float(union)


def _compute_vegetation_precision_recall(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    """Compute precision and recall for vegetation segmentation (vegetation-specific metrics)."""
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)
    tp = np.logical_and(pred_bool, gt_bool).sum()
    fp = np.logical_and(pred_bool, ~gt_bool).sum()
    fn = np.logical_and(~pred_bool, gt_bool).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return precision, recall


# ============================================================================
# STREAMLIT UI (Separated by Pipeline)
# ============================================================================

def main():
    st.set_page_config(page_title="AI Driven Archaeological Site Mapping", layout="wide")
    
    st.markdown("#  AI Driven Archaeological Site Mapping")
    st.write("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
        st.markdown("---")
        
        st.subheader("Select Analysis Type")
        analysis_type = st.radio(
            "Choose pipeline:",
            ("Soil Classification", "Vegetation Segmentation")
        )
        st.markdown("---")
        
        # Pipeline-specific options
        if analysis_type == "Vegetation Segmentation":
            alpha = 0.5
            show_overlay = True
            gt_veg_file = None

        else:
            # Soil options (minimal - classification doesn't need many options)
            gt_veg_file = None
            alpha = 0.5  # Not used
            show_overlay = False  # Not used
            coverage_threshold = 0.15  # Not used
        
        st.markdown("---")
        run_button = st.button("Run Analysis", type="primary")
    
    # Main content
    if uploaded is None:
        st.info("👆 Please upload an image to begin analysis.")
        return
    
    image = Image.open(uploaded).convert("RGB")
    st.write("### Input Image")
    st.image(image, use_column_width=True)
    
    if not run_button:
        st.info("👆 Press **Run Analysis** to process the image.")
        return
    
    # ========================================================================
    # SOIL CLASSIFICATION PIPELINE
    # ========================================================================
    if analysis_type == "Soil Classification":
        with st.spinner("Running soil classification with TTA..."):
            class_id, confidence, all_probs, is_gated, class_name_mapping = _run_soil_classification_with_tta(image)
            soil_name, confidence_text = _format_soil_output(class_id, confidence, is_gated, class_name_mapping)
        
        st.write("---")
        st.markdown("##  Soil Classification Results")
        
        # Debug visibility (temporary - can be removed later)
        st.caption(f"Debug → class_id={class_id}")
        
        # Main result
        if is_gated:
            st.warning(f"⚠️ **Predicted Soil Type:** {soil_name}")
            st.caption(confidence_text)
        else:
            st.success(f"✅ **Predicted Soil Type:** {soil_name}")
            st.write(f"**{confidence_text}**")
    
    # ========================================================================
    # VEGETATION SEGMENTATION PIPELINE
    # ========================================================================
    
    else:  # Vegetation Segmentation
        with st.spinner("Running vegetation segmentation..."):
            veg_mask = _predict_vegetation_mask(image)
            coverage = _compute_vegetation_coverage(veg_mask)
            
        
        st.write("---")
        st.markdown("##  Vegetation Segmentation Results")
        
        # Main result
       
        
        st.write(f"**Vegetation Coverage:** {coverage:.2%}")
        
        # Segmentation overlay
        if show_overlay:
            st.write("### Segmentation Overlay")
            overlay_img = _overlay_vegetation_mask(image, veg_mask, color=(0, 255, 0), alpha=alpha)
            st.image(overlay_img, use_column_width=True, caption="Green overlay indicates vegetation pixels")
            
            # Download button
            buf = io.BytesIO()
            overlay_img.save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download overlay", data=buf, file_name="vegetation_overlay.png", mime="image/png")
        
if __name__ == "__main__":
    main()
