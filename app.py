"""
Flask Web Application for Traffic Mask R-CNN.
Provides endpoints for image upload, segmentation, bounding box detection,
and model metadata inspection.
"""

import os
import io
import base64
import logging
from typing import Optional
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

from model_utils import (
    CLASS_NAMES,
    CLASS_COLORS,
    NUM_CLASSES,
    DEFAULT_CHECKPOINT_PATH,
    load_traffic_model,
    predict_image,
    render_annotated_image
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrafficMaskRCNN")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB maximum upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

SAMPLES_DIR = os.path.join(app.root_path, "static", "samples")

# Preload model on startup if checkpoint is present
MODEL_READY = False
MODEL_LOAD_ERROR: Optional[str] = None

try:
    logger.info("Initializing Traffic Mask R-CNN model...")
    load_traffic_model()
    MODEL_READY = True
    logger.info("Model loaded successfully and ready for inference.")
except Exception as exc:
    MODEL_LOAD_ERROR = str(exc)
    logger.warning("Model could not be preloaded: %s", exc)


def is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(img: Image.Image, format: str = "JPEG", quality: int = 90) -> str:
    """Converts a PIL Image to a base64 Data URL string."""
    buffer = io.BytesIO()
    if format.upper() == "JPEG":
        img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        img.save(buffer, format=format)
        mime = f"image/{format.lower()}"
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64_str}"


@app.route("/")
def index():
    """Renders the main Traffic Detection & Segmentation dashboard."""
    return render_template("index.html")


@app.route("/api/model_info", methods=["GET"])
def model_info():
    """Returns metadata regarding the trained model checkpoint and classes."""
    try:
        _, meta = load_traffic_model()
        return jsonify({
            "status": "ready",
            "model_ready": True,
            "classes": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "class_colors": {k: f"#{v[0]:02x}{v[1]:02x}{v[2]:02x}" for k, v in CLASS_COLORS.items()},
            "epoch": meta.get("epoch"),
            "loss": meta.get("loss"),
            "device": meta.get("device"),
            "checkpoint_path": os.path.basename(meta.get("checkpoint_path", DEFAULT_CHECKPOINT_PATH))
        })
    except Exception as exc:
        return jsonify({
            "status": "error",
            "model_ready": False,
            "error": str(exc),
            "classes": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "class_colors": {k: f"#{v[0]:02x}{v[1]:02x}{v[2]:02x}" for k, v in CLASS_COLORS.items()}
        }), 200


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Lists available sample traffic images for instant 1-click evaluation."""
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for fname in sorted(os.listdir(SAMPLES_DIR)):
            if is_allowed_file(fname):
                samples.append({
                    "name": fname,
                    "url": f"/static/samples/{fname}",
                    "title": fname.replace("_", " ").rsplit(".", 1)[0].title()
                })
    return jsonify({"samples": samples})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Inference endpoint. Accepts an uploaded file or sample name and returns
    annotated image and detailed detection metadata.
    """
    # 1. Parse parameters
    try:
        confidence = float(request.form.get("confidence", 0.5))
        confidence = max(0.05, min(0.99, confidence))
    except (ValueError, TypeError):
        confidence = 0.5

    try:
        mask_opacity = float(request.form.get("mask_opacity", 0.45))
        mask_opacity = max(0.0, min(1.0, mask_opacity))
    except (ValueError, TypeError):
        mask_opacity = 0.45

    draw_boxes = request.form.get("draw_boxes", "true").lower() == "true"
    draw_masks = request.form.get("draw_masks", "true").lower() == "true"
    draw_labels = request.form.get("draw_labels", "true").lower() == "true"

    selected_classes_raw = request.form.get("selected_classes")
    selected_classes = None
    if selected_classes_raw:
        selected_classes = [c.strip() for c in selected_classes_raw.split(",") if c.strip()]

    # 2. Acquire image source
    image = None
    source_name = "custom_upload"

    # Check for sample image reference
    sample_name = request.form.get("sample_name")
    if sample_name:
        safe_sample = secure_filename(sample_name)
        sample_path = os.path.join(SAMPLES_DIR, safe_sample)
        if not os.path.isfile(sample_path):
            return jsonify({"error": f"Sample image '{safe_sample}' was not found on the server."}), 404
        try:
            image = Image.open(sample_path)
            source_name = safe_sample
        except Exception as exc:
            return jsonify({"error": f"Failed opening sample image: {exc}"}), 400

    # Check for uploaded file
    elif "image" in request.files:
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file was selected for upload."}), 400
        if not is_allowed_file(file.filename):
            return jsonify({
                "error": f"Unsupported image format. Allowed formats are: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            }), 400
        try:
            image_bytes = file.read()
            if len(image_bytes) == 0:
                return jsonify({"error": "Uploaded file is empty (0 bytes)."}), 400
            image = Image.open(io.BytesIO(image_bytes))
            source_name = secure_filename(file.filename)
        except UnidentifiedImageError:
            return jsonify({"error": "The uploaded file could not be recognized as a valid image."}), 400
        except Exception as exc:
            return jsonify({"error": f"Failed reading uploaded image: {exc}"}), 400
    else:
        return jsonify({"error": "No image file or sample name was provided in the request."}), 400

    # 3. Verify model readiness
    try:
        model, meta = load_traffic_model()
    except Exception as exc:
        return jsonify({
            "error": f"Model failed to load: {exc}. Ensure 'models/traffic_maskrcnn_best.pth' exists."
        }), 500

    # 4. Run inference
    try:
        result = predict_image(
            image=image,
            model=model,
            score_threshold=confidence,
            mask_threshold=0.5
        )

        # Render composite annotated image
        annotated_pil = render_annotated_image(
            image=image,
            prediction_result=result,
            draw_boxes=draw_boxes,
            draw_masks=draw_masks,
            draw_labels=draw_labels,
            mask_opacity=mask_opacity,
            selected_classes=selected_classes
        )

        orig_b64 = image_to_base64(image, format="JPEG", quality=92)
        annotated_b64 = image_to_base64(annotated_pil, format="JPEG", quality=92)

        return jsonify({
            "success": True,
            "source_name": source_name,
            "width": result["width"],
            "height": result["height"],
            "detections": result["detections"],
            "total_detected": result["total_detected"],
            "class_counts": result["class_counts"],
            "inference_time_ms": result["inference_time_ms"],
            "original_image": orig_b64,
            "annotated_image": annotated_b64,
            "confidence_threshold": confidence,
            "mask_opacity": mask_opacity
        })

    except Exception as exc:
        logger.exception("Inference error occurred:")
        return jsonify({"error": f"An error occurred during inference: {str(exc)}"}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "The uploaded file exceeds the 16 MB limit."}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found."}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Traffic Mask R-CNN Web Application on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
