"""
Unit and Integration Tests for Traffic Mask R-CNN Application.
Verifies model architecture, checkpoint parity, class mapping, inference pipeline,
and Flask web endpoints.
"""

import os
import sys
import io
import pytest
from PIL import Image
import torch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_utils import (
    CLASS_NAMES,
    NUM_CLASSES,
    CLASS_COLORS,
    DEFAULT_CHECKPOINT_PATH,
    build_maskrcnn_model,
    load_traffic_model,
    predict_image,
    render_annotated_image
)
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_class_definitions():
    """Verify class count and exact class names matching training notebook."""
    assert NUM_CLASSES == 9
    expected_classes = [
        'background',
        'bicycle',
        'bus',
        'car',
        'motorcycle',
        'person',
        'stop sign',
        'traffic light',
        'truck'
    ]
    assert CLASS_NAMES == expected_classes
    for cls_name in expected_classes:
        assert cls_name in CLASS_COLORS


def test_model_architecture():
    """Verify head modifications for 9 classes."""
    model = build_maskrcnn_model(num_classes=NUM_CLASSES)
    assert model.roi_heads.box_predictor.cls_score.out_features == 9
    assert model.roi_heads.box_predictor.bbox_pred.out_features == 9 * 4
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 9


def test_checkpoint_loading_and_metadata():
    """Verify loading from models/traffic_maskrcnn_best.pth and metadata matches."""
    assert os.path.exists(DEFAULT_CHECKPOINT_PATH), "Checkpoint file must exist"
    model, meta = load_traffic_model(checkpoint_path=DEFAULT_CHECKPOINT_PATH, force_reload=True)
    assert model is not None
    assert meta["num_classes"] == 9
    assert meta["epoch"] == 10
    # Expected best loss from Colab training loop
    assert abs(meta["loss"] - 0.202925398) < 1e-4
    assert meta["missing_keys_count"] == 0
    assert meta["unexpected_keys_count"] == 0


def test_predict_image_dummy():
    """Verify predict_image handles an RGB image and returns valid structure."""
    model, _ = load_traffic_model()
    dummy = Image.new("RGB", (320, 240), color=(100, 150, 200))
    result = predict_image(dummy, model=model, score_threshold=0.1)

    assert result["width"] == 320
    assert result["height"] == 240
    assert "detections" in result
    assert "binary_masks" in result
    assert "class_counts" in result
    assert "inference_time_ms" in result
    assert result["inference_time_ms"] > 0


def test_predict_and_render_sample():
    """Verify predict and render on real sample image."""
    sample_path = "static/samples/sample_traffic_1.jpg"
    if not os.path.exists(sample_path):
        pytest.skip("Sample image not found, skipping visual test")

    img = Image.open(sample_path)
    model, _ = load_traffic_model()
    result = predict_image(img, model=model, score_threshold=0.3)

    assert result["width"] == img.width
    assert result["height"] == img.height
    assert len(result["detections"]) == len(result["binary_masks"])

    # Test visual rendering
    annotated = render_annotated_image(
        image=img,
        prediction_result=result,
        draw_boxes=True,
        draw_masks=True,
        draw_labels=True,
        mask_opacity=0.5
    )
    assert isinstance(annotated, Image.Image)
    assert annotated.size == img.size


def test_api_index(client):
    """Verify GET / renders homepage."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Traffic Mask R-CNN" in resp.data


def test_api_model_info(client):
    """Verify GET /api/model_info returns ready status and classes."""
    resp = client.get("/api/model_info")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["model_ready"] is True
    assert data["num_classes"] == 9
    assert "bicycle" in data["classes"]
    assert "car" in data["classes"]
    assert data["epoch"] == 10


def test_api_samples(client):
    """Verify GET /api/samples returns sample images list."""
    resp = client.get("/api/samples")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "samples" in data
    assert len(data["samples"]) >= 1


def test_api_predict_sample(client):
    """Verify POST /api/predict using sample_name."""
    resp = client.post("/api/predict", data={
        "sample_name": "sample_traffic_1.jpg",
        "confidence": "0.3",
        "mask_opacity": "0.4"
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "original_image" in data
    assert "annotated_image" in data
    assert data["original_image"].startswith("data:image/jpeg;base64,")
    assert data["annotated_image"].startswith("data:image/jpeg;base64,")
    assert "detections" in data
    assert data["total_detected"] >= 1


def test_api_predict_uploaded_file(client):
    """Verify POST /api/predict with file upload."""
    img_byte_arr = io.BytesIO()
    test_img = Image.new("RGB", (200, 200), color=(120, 180, 240))
    test_img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    resp = client.post("/api/predict", data={
        "image": (img_byte_arr, "test.jpg"),
        "confidence": "0.5"
    }, content_type="multipart/form-data")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["width"] == 200
    assert data["height"] == 200


def test_api_predict_invalid_file(client):
    """Verify POST /api/predict returns 400 for text/non-image upload."""
    resp = client.post("/api/predict", data={
        "image": (io.BytesIO(b"Not an image"), "test.txt"),
    }, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_api_predict_missing_input(client):
    """Verify POST /api/predict returns 400 when no input is provided."""
    resp = client.post("/api/predict", data={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()
