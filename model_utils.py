"""
Model utilities for Traffic Mask R-CNN.
Handles model architecture definition, checkpoint loading, inference preprocessing,
and visual rendering of segmentation masks, bounding boxes, and label tags.
"""

import os
import time
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# 8 traffic classes + 1 background class (matches training notebook cell 18 & 30)
CLASS_NAMES: List[str] = [
    'background',     # 0
    'bicycle',        # 1 (COCO cat 0 + 1)
    'bus',            # 2 (COCO cat 1 + 1)
    'car',            # 3 (COCO cat 2 + 1)
    'motorcycle',     # 4 (COCO cat 3 + 1)
    'person',         # 5 (COCO cat 4 + 1)
    'stop sign',      # 6 (COCO cat 5 + 1)
    'traffic light',  # 7 (COCO cat 6 + 1)
    'truck'           # 8 (COCO cat 7 + 1)
]

NUM_CLASSES: int = len(CLASS_NAMES)  # 9

# Distinct palette for each traffic class (RGBA friendly)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    'background': (100, 116, 139),     # Slate
    'bicycle': (59, 130, 246),         # Vibrant Blue
    'bus': (245, 158, 11),             # Amber
    'car': (16, 185, 129),             # Emerald Green
    'motorcycle': (139, 92, 246),      # Purple
    'person': (236, 72, 153),          # Pink
    'stop sign': (239, 68, 68),        # Crimson Red
    'traffic light': (234, 179, 8),    # Yellow
    'truck': (6, 182, 212),            # Cyan
}

DEFAULT_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "traffic_maskrcnn_best.pth"
)

# Global cached model instance & metadata
_CACHED_MODEL: Optional[torch.nn.Module] = None
_MODEL_METADATA: Dict[str, Any] = {}


def build_maskrcnn_model(num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """
    Constructs the Mask R-CNN ResNet-50 FPN architecture matching the Colab training script.
    Replaces both the classification/box head and the mask predictor head for 9 classes.
    """
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    # 1. Replace the bounding-box prediction head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 2. Replace the instance-mask prediction head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


def load_traffic_model(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    force_reload: bool = False
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Loads and caches the fine-tuned Traffic Mask R-CNN model from the checkpoint file.
    Validates state dict keys and returns the model alongside checkpoint metadata.
    """
    global _CACHED_MODEL, _MODEL_METADATA

    if _CACHED_MODEL is not None and not force_reload:
        return _CACHED_MODEL, _MODEL_METADATA

    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Please ensure 'models/traffic_maskrcnn_best.pth' is present."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_maskrcnn_model(num_classes=NUM_CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    metadata: Dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
    }

    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            metadata["epoch"] = checkpoint["epoch"]
        if "loss" in checkpoint:
            metadata["loss"] = float(checkpoint["loss"])
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strict load to guarantee architecture parity
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    metadata["missing_keys_count"] = len(missing)
    metadata["unexpected_keys_count"] = len(unexpected)

    model.to(device)
    model.eval()

    _CACHED_MODEL = model
    _MODEL_METADATA = metadata

    return model, metadata


def predict_image(
    image: Image.Image,
    model: Optional[torch.nn.Module] = None,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Runs inference on a PIL Image and returns structured detection and segmentation results.

    Args:
        image: Input PIL Image (converted to RGB).
        model: Loaded Mask R-CNN model (uses cached model if None).
        score_threshold: Minimum confidence score to retain detection (0.05 - 1.0).
        mask_threshold: Binarization cutoff for soft mask probabilities.
        device: Device to execute model on.

    Returns:
        Dictionary with image dimensions, detection items, masks, counts, and timing.
    """
    if model is None:
        model, _ = load_traffic_model()

    if device is None:
        device = next(model.parameters()).device

    # Ensure 3-channel RGB image
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size

    # Convert to float32 Tensor [0.0, 1.0] shape (3, H, W)
    img_tensor = TF.to_tensor(rgb_image).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        predictions = model([img_tensor])[0]
    inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    boxes = predictions["boxes"].detach().cpu().numpy()
    labels = predictions["labels"].detach().cpu().numpy()
    scores = predictions["scores"].detach().cpu().numpy()
    raw_masks = predictions["masks"].detach().cpu().numpy()  # Shape: (N, 1, H, W)

    detections = []
    binary_masks = []
    class_counts = {name: 0 for name in CLASS_NAMES if name != 'background'}

    for idx, (box, label, score, raw_mask) in enumerate(zip(boxes, labels, scores, raw_masks)):
        if score < score_threshold:
            continue

        label_idx = int(label)
        class_name = CLASS_NAMES[label_idx] if 0 <= label_idx < len(CLASS_NAMES) else f"class_{label_idx}"

        # Soft mask thresholding -> binary boolean mask
        bin_mask = raw_mask[0] >= mask_threshold
        mask_pixel_count = int(np.sum(bin_mask))

        xmin, ymin, xmax, ymax = [float(coord) for coord in box]
        # Clamp to image boundaries
        xmin = max(0.0, min(float(width), xmin))
        ymin = max(0.0, min(float(height), ymin))
        xmax = max(0.0, min(float(width), xmax))
        ymax = max(0.0, min(float(height), ymax))

        color_rgb = CLASS_COLORS.get(class_name, (120, 120, 120))
        color_hex = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"

        detection_item = {
            "id": idx + 1,
            "class_id": label_idx,
            "class_name": class_name,
            "score": round(float(score), 4),
            "score_pct": f"{float(score) * 100:.1f}%",
            "box": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)],
            "box_area": int(round((xmax - xmin) * (ymax - ymin))),
            "mask_pixels": mask_pixel_count,
            "color": color_hex,
            "color_rgb": list(color_rgb)
        }

        detections.append(detection_item)
        binary_masks.append(bin_mask)
        if class_name in class_counts:
            class_counts[class_name] += 1

    return {
        "width": width,
        "height": height,
        "detections": detections,
        "binary_masks": binary_masks,
        "total_detected": len(detections),
        "class_counts": class_counts,
        "inference_time_ms": inference_time_ms,
        "score_threshold": score_threshold,
        "mask_threshold": mask_threshold
    }


def render_annotated_image(
    image: Image.Image,
    prediction_result: Dict[str, Any],
    draw_boxes: bool = True,
    draw_masks: bool = True,
    draw_labels: bool = True,
    mask_opacity: float = 0.45,
    selected_classes: Optional[List[str]] = None
) -> Image.Image:
    """
    Composites instance segmentation masks, colored bounding boxes, and label tags onto the image.

    Args:
        image: Original PIL Image.
        prediction_result: Output dictionary from predict_image().
        draw_boxes: Whether to render bounding box rectangles.
        draw_masks: Whether to render translucent segmentation masks.
        draw_labels: Whether to render label badges with confidence scores.
        mask_opacity: Alpha opacity for segmentation masks (0.0 - 1.0).
        selected_classes: Optional list of class names to display (filters others).

    Returns:
        Rendered PIL Image in RGB format.
    """
    base_image = image.convert("RGBA")
    width, height = base_image.size

    detections = prediction_result.get("detections", [])
    binary_masks = prediction_result.get("binary_masks", [])

    # Filter by selected_classes if provided
    active_indices = []
    for i, det in enumerate(detections):
        if selected_classes is None or det["class_name"] in selected_classes:
            active_indices.append(i)

    # 1. Overlay segmentation masks
    if draw_masks and binary_masks and active_indices:
        mask_layer = np.zeros((height, width, 4), dtype=np.uint8)

        for idx in active_indices:
            det = detections[idx]
            b_mask = binary_masks[idx]
            color = det.get("color_rgb", [255, 0, 0])
            alpha = int(max(0.0, min(1.0, mask_opacity)) * 255)

            # Apply color and alpha to positive mask pixels
            mask_layer[b_mask] = [color[0], color[1], color[2], alpha]

        mask_pil = Image.fromarray(mask_layer, mode="RGBA")
        base_image = Image.alpha_composite(base_image, mask_pil)

    # 2. Draw bounding boxes and label badges
    draw = ImageDraw.Draw(base_image)

    # Select font
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, int(height * 0.022)))
    except IOError:
        font = ImageFont.load_default()

    box_thickness = max(2, int(min(width, height) * 0.004))

    for idx in active_indices:
        det = detections[idx]
        xmin, ymin, xmax, ymax = det["box"]
        color_rgb = tuple(det.get("color_rgb", [255, 0, 0]))
        color_rgba = color_rgb + (255,)

        if draw_boxes:
            draw.rectangle(
                [xmin, ymin, xmax, ymax],
                outline=color_rgba,
                width=box_thickness
            )

        if draw_labels:
            label_text = f"{det['class_name']} {det['score_pct']}"
            
            # Text bounding box calculation for badge background
            try:
                text_bbox = draw.textbbox((xmin, ymin), label_text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_w, text_h = draw.textsize(label_text, font=font)

            badge_padding = 4
            badge_y1 = max(0, ymin - text_h - (badge_padding * 2))
            badge_y2 = badge_y1 + text_h + (badge_padding * 2)
            badge_x1 = xmin
            badge_x2 = min(width, xmin + text_w + (badge_padding * 2))

            # Draw label badge background with rounded corners
            badge_bg = color_rgb + (220,)
            draw.rectangle(
                [badge_x1, badge_y1, badge_x2, badge_y2],
                fill=badge_bg
            )

            # Text color (white with high contrast)
            text_x = badge_x1 + badge_padding
            text_y = badge_y1 + badge_padding
            draw.text((text_x, text_y), label_text, fill=(255, 255, 255, 255), font=font)

    return base_image.convert("RGB")
