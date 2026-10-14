"""
Flask Web Application for Traffic Mask R-CNN.
Provides endpoints for image upload, segmentation, bounding box detection,
and model metadata inspection.
"""

import os
import io
import base64
import json
import uuid
import logging
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

# Optional requests library; fallback to urllib if not installed
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Attempt importing local model utilities (available in local dev with PyTorch)
try:
    from model_utils import (
        CLASS_NAMES,
        CLASS_COLORS,
        NUM_CLASSES,
        DEFAULT_CHECKPOINT_PATH,
        load_traffic_model,
        predict_image,
        render_annotated_image
    )
    LOCAL_MODEL_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as _import_exc:
    LOCAL_MODEL_AVAILABLE = False
    # Standard class definitions for fallback/metadata reporting
    CLASS_NAMES = [
        'background', 'bicycle', 'bus', 'car', 'motorcycle',
        'person', 'stop sign', 'traffic light', 'truck'
    ]
    NUM_CLASSES = len(CLASS_NAMES)
    CLASS_COLORS = {
        'background': (100, 116, 139),
        'bicycle': (59, 130, 246),
        'bus': (245, 158, 11),
        'car': (16, 185, 129),
        'motorcycle': (139, 92, 246),
        'person': (236, 72, 153),
        'stop sign': (239, 68, 68),
        'traffic light': (234, 179, 8),
        'truck': (6, 182, 212),
    }
    DEFAULT_CHECKPOINT_PATH = ""

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrafficMaskRCNN")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB maximum upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

SAMPLES_DIR = os.path.join(app.root_path, "static", "samples")

# Remote inference backend URL (e.g. Hugging Face Space, Render, Cloud Run)
INFERENCE_BACKEND_URL = os.environ.get("INFERENCE_BACKEND_URL", "").strip().rstrip("/")

MODEL_READY = False
MODEL_LOAD_ERROR: Optional[str] = None

if INFERENCE_BACKEND_URL:
    logger.info("Configured for remote inference gateway: %s", INFERENCE_BACKEND_URL)
    MODEL_READY = True
elif LOCAL_MODEL_AVAILABLE:
    try:
        logger.info("Initializing Traffic Mask R-CNN model locally...")
        load_traffic_model()
        MODEL_READY = True
        logger.info("Local model loaded successfully and ready for inference.")
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        logger.warning("Local model could not be preloaded: %s", exc)
else:
    MODEL_LOAD_ERROR = "PyTorch/model_utils not available locally and INFERENCE_BACKEND_URL is not configured."
    logger.warning(MODEL_LOAD_ERROR)


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


def _fetch_remote_json(url: str, timeout: int = 5) -> Dict[str, Any]:
    """Fetches JSON from a remote URL using requests or urllib."""
    if HAS_REQUESTS:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "TrafficMaskRCNN-Gateway/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def _forward_prediction_to_remote(
    image: Image.Image,
    source_name: str,
    confidence: float,
    mask_opacity: float,
    draw_boxes: bool,
    draw_masks: bool,
    draw_labels: bool,
    selected_classes: Optional[List[str]] = None
):
    """Forwards image and prediction settings to the remote INFERENCE_BACKEND_URL."""
    target_url = f"{INFERENCE_BACKEND_URL}/api/predict"

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    image_bytes = buf.getvalue()

    fields = {
        "confidence": str(confidence),
        "mask_opacity": str(mask_opacity),
        "draw_boxes": "true" if draw_boxes else "false",
        "draw_masks": "true" if draw_masks else "false",
        "draw_labels": "true" if draw_labels else "false",
    }
    if selected_classes:
        fields["selected_classes"] = ",".join(selected_classes)

    try:
        if HAS_REQUESTS:
            files = {"image": (source_name or "input.jpg", io.BytesIO(image_bytes), "image/jpeg")}
            resp = requests.post(target_url, data=fields, files=files, timeout=60)
            status_code = resp.status_code
            try:
                resp_json = resp.json()
            except Exception:
                resp_json = {"error": resp.text}
        else:
            boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
            body = bytearray()
            for name, val in fields.items():
                body.extend(f"--{boundary}\r\n".encode("utf-8"))
                body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
                body.extend(f"{val}\r\n".encode("utf-8"))
            filename = source_name or "input.jpg"
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode("utf-8"))
            body.extend(b"Content-Type: image/jpeg\r\n\r\n")
            body.extend(image_bytes)
            body.extend(b"\r\n")
            body.extend(f"--{boundary}--\r\n".encode("utf-8"))

            req = urllib.request.Request(
                target_url,
                data=bytes(body),
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "User-Agent": "TrafficMaskRCNN-Gateway/1.0"
                },
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    status_code = resp.status
                    resp_json = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as http_err:
                status_code = http_err.code
                try:
                    resp_json = json.loads(http_err.read().decode("utf-8"))
                except Exception:
                    resp_json = {"error": str(http_err)}

        if status_code == 200:
            return jsonify(resp_json)
        else:
            err_msg = resp_json.get("error", f"Remote server returned HTTP {status_code}")
            return jsonify({"error": f"Remote backend error: {err_msg}"}), status_code

    except Exception as exc:
        logger.exception("Failed to forward prediction to remote backend:")
        return jsonify({
            "error": f"Failed to communicate with remote inference backend at {INFERENCE_BACKEND_URL}: {str(exc)}"
        }), 502


@app.route("/")
def index():
    """Renders the main Traffic Detection & Segmentation dashboard."""
    return render_template("index.html")


@app.route("/api/model_info", methods=["GET"])
def model_info():
    """Returns metadata regarding the trained model checkpoint and classes."""
    # 1. If remote inference backend is configured, attempt querying it
    if INFERENCE_BACKEND_URL:
        try:
            remote_url = f"{INFERENCE_BACKEND_URL}/api/model_info"
            remote_info = _fetch_remote_json(remote_url, timeout=5)
            if "classes" not in remote_info:
                remote_info["classes"] = CLASS_NAMES
            if "num_classes" not in remote_info:
                remote_info["num_classes"] = NUM_CLASSES
            if "class_colors" not in remote_info:
                remote_info["class_colors"] = {k: f"#{v[0]:02x}{v[1]:02x}{v[2]:02x}" for k, v in CLASS_COLORS.items()}
            return jsonify(remote_info)
        except Exception as exc:
            logger.warning("Remote backend %s not reachable: %s", INFERENCE_BACKEND_URL, exc)
            return jsonify({
                "status": "warning",
                "model_ready": False,
                "backend": "remote",
                "backend_url": INFERENCE_BACKEND_URL,
                "error": f"Remote inference service not responding: {exc}",
                "classes": CLASS_NAMES,
                "num_classes": NUM_CLASSES,
                "class_colors": {k: f"#{v[0]:02x}{v[1]:02x}{v[2]:02x}" for k, v in CLASS_COLORS.items()}
            }), 200

    # 2. Local mode with PyTorch
    if LOCAL_MODEL_AVAILABLE:
        try:
            _, meta = load_traffic_model()
            return jsonify({
                "status": "ready",
                "model_ready": True,
                "backend": "local",
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
                "backend": "local",
                "error": str(exc),
                "classes": CLASS_NAMES,
                "num_classes": NUM_CLASSES,
                "class_colors": {k: f"#{v[0]:02x}{v[1]:02x}{v[2]:02x}" for k, v in CLASS_COLORS.items()}
            }), 200

    # 3. Neither local model nor remote backend available
    return jsonify({
        "status": "error",
        "model_ready": False,
        "error": "PyTorch is not available and INFERENCE_BACKEND_URL is not configured.",
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
    annotated image and detailed detection metadata. Supports both local PyTorch
    inference and remote INFERENCE_BACKEND_URL gateway forwarding.
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

    # 3. Mode A: Forward to remote inference service if INFERENCE_BACKEND_URL is set
    if INFERENCE_BACKEND_URL:
        return _forward_prediction_to_remote(
            image=image,
            source_name=source_name,
            confidence=confidence,
            mask_opacity=mask_opacity,
            draw_boxes=draw_boxes,
            draw_masks=draw_masks,
            draw_labels=draw_labels,
            selected_classes=selected_classes
        )

    # 4. Mode B: Local PyTorch inference
    if not LOCAL_MODEL_AVAILABLE:
        return jsonify({
            "error": (
                "Inference service unavailable: PyTorch is not available in this environment "
                "and INFERENCE_BACKEND_URL is not configured. For Vercel deployment, "
                "please configure the INFERENCE_BACKEND_URL environment variable."
            )
        }), 503

    try:
        model, meta = load_traffic_model()
    except Exception as exc:
        return jsonify({
            "error": f"Model failed to load: {exc}. Ensure 'models/traffic_maskrcnn_best.pth' exists."
        }), 500

    try:
        result = predict_image(
            image=image,
            model=model,
            score_threshold=confidence,
            mask_threshold=0.5
        )

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
