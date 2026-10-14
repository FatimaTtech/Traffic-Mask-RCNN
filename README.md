# Traffic Mask R-CNN: Detection & Instance Segmentation

A modern, responsive web application for detecting and segmenting traffic objects in images using a fine-tuned **Mask R-CNN (ResNet-50-FPN)** deep learning model.

---

## 🚦 System Architecture & Model Specifications

- **Base Architecture**: `torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)`
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Bounding Box Head**: `FastRCNNPredictor(in_features=1024, num_classes=9)`
- **Mask Predictor Head**: `MaskRCNNPredictor(in_channels=256, dim_reduced=256, num_classes=9)`
- **Classes**: 9 Total (1 Background + 8 Traffic Categories)

| Class Index | Label Name | COCO Category ID (`label = cat_id + 1`) | Palette Color |
| :---: | :--- | :---: | :---: |
| **0** | `background` | *Background* | Slate (`#64748b`) |
| **1** | `bicycle` | `0` | Blue (`#3b82f6`) |
| **2** | `bus` | `1` | Amber (`#f59e0b`) |
| **3** | `car` | `2` | Emerald (`#10b981`) |
| **4** | `motorcycle` | `3` | Purple (`#8b5cf6`) |
| **5** | `person` | `4` | Pink (`#ec4899`) |
| **6** | `stop sign` | `5` | Red (`#ef4444`) |
| **7** | `traffic light` | `6` | Yellow (`#eab308`) |
| **8** | `truck` | `7` | Cyan (`#06b6d4`) |

### Model Checkpoints
- [`models/traffic_maskrcnn_best.pth`](models/traffic_maskrcnn_best.pth): Fine-tuned checkpoint saved at **Epoch 10** with total loss **0.2029** (contains `model_state_dict` and `optimizer_state_dict`).
- [`models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth`](models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth): Original PyTorch COCO-pretrained weights (preserved untouched).
- [`results/training_history.csv`](results/training_history.csv): 10-epoch training loss history from Google Colab.
- [`maskrcnn.ipynb`](maskrcnn.ipynb): Original training and validation notebook.

---

## 📁 Repository Structure

```
Traffic-Mask-RCNN/
├── app.py                      # Flask backend & REST API (/api/predict, /api/model_info, /api/samples)
├── model_utils.py              # Architecture definition, checkpoint loading, inference & rendering
├── requirements.txt            # Python dependencies
├── README.md                   # Setup and usage documentation
├── static/
│   ├── css/style.css           # Modern dark-theme stylesheet
│   ├── js/app.js               # Frontend controller (upload, sliders, side-by-side view)
│   └── samples/                # Extracted sample traffic images for instant 1-click evaluation
├── templates/
│   └── index.html              # Interactive dashboard UI
├── tests/
│   └── test_model.py           # Comprehensive pytest suite (12 tests)
├── models/                     # Checkpoints (preserved)
│   ├── traffic_maskrcnn_best.pth
│   └── maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
└── results/                    # Colab training records (preserved)
    └── training_history.csv
```

---

## 🚀 Setup & Launch Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.
- PowerShell or Command Prompt (Windows), or Terminal (Linux/macOS).

### 1. Clone / Open the Directory
Open your terminal and navigate to the project directory:
```powershell
cd c:\Users\fatim\Downloads\Traffic-Mask-RCNN
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If you encounter execution policy restrictions in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install the required packages from `requirements.txt`:
```powershell
pip install -r requirements.txt
```
*Note: If you have an NVIDIA GPU and want GPU acceleration, install the CUDA-enabled PyTorch build corresponding to your CUDA version from [pytorch.org](https://pytorch.org).*

### 4. Run the Test Suite
Verify that the model architecture, checkpoint, inference pipeline, and endpoints pass all unit tests:
```powershell
python -m pytest tests/test_model.py -v
```
All 12 tests should report `PASSED`.

### 5. Launch the Web Application
Start the Flask server:
```powershell
python app.py
```

Open your web browser and navigate to:
```
http://localhost:5000
```

---

## 🖥️ Web Interface Features

1. **Drag-and-Drop & File Picker**: Upload standard images (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`) up to 16 MB.
2. **Pre-Loaded Sample Scenes**: Click sample chips (e.g. *Sample 1: Bicycle Scene* or *Sample 2: Urban Traffic*) for instant testing.
3. **Interactive Detection Controls**:
   - **Confidence Threshold Slider**: Adjust detection cutoff from `0.05` (more detections) to `0.95` (high precision).
   - **Mask Transparency Slider**: Control mask fill opacity from `0%` (transparent) to `100%` (solid).
   - **Layer Toggles**: Individually enable or disable instance masks, bounding boxes, or class badges.
4. **Side-by-Side Visualization**: Compare the original uploaded image directly against the model's predicted instance masks, colored bounding boxes, and confidence badges.
5. **Detection Metrics & Table**:
   - Instant metrics bar (detected count, inference time in ms, resolution, source name).
   - Category distribution chips with counts.
   - Comprehensive table detailing each detected object, bounding box `[xmin, ymin, xmax, ymax]`, box area in pixels, and mask pixel counts.
6. **Save Annotated Image**: 1-click download of the composite annotated result.

---

## 🌐 REST API Endpoints

### `POST /api/predict`
Executes inference on an image.
- **Form Data**:
  - `image`: Uploaded image file (multipart).
  - `sample_name`: (Alternative) Name of sample image in `static/samples/`.
  - `confidence`: (Optional float, default `0.5`) Minimum confidence score.
  - `mask_opacity`: (Optional float, default `0.45`) Alpha transparency of mask.
  - `draw_boxes`: (Optional bool, default `true`).
  - `draw_masks`: (Optional bool, default `true`).
  - `draw_labels`: (Optional bool, default `true`).
- **Response**: JSON with `detections`, `original_image` (base64), `annotated_image` (base64), `class_counts`, and `inference_time_ms`.

### `GET /api/model_info`
Returns current model status, active device (CPU/CUDA), classes list, epoch, and training loss.

### `GET /api/samples`
Returns the list of available preloaded sample images.

---

## ⚠️ Model Limitations & Practical Considerations

1. **Dataset Scale**: The model was trained on a custom traffic dataset with 168 training images and 72 validation images over 20 total epochs (final loss `0.2029`). Because of the compact dataset size:
   - Frequent classes (such as `bicycle` and `car`) show strong confidence scores.
   - Infrequent or underrepresented categories (such as `bus`, `truck`, or `stop sign` in unusual orientations or low-light conditions) may require lowering the confidence slider (e.g. to `0.25 - 0.35`) to detect.
2. **Dense Occlusions**: In heavily congested traffic where vehicles overlap closely, bounding boxes may group nearby vehicles or segment boundaries may overlap.
3. **Inference Latency on CPU**: Running a full ResNet-50-FPN Mask R-CNN model on CPU typically takes between 250ms and 800ms depending on CPU speed and image resolution. For real-time video streaming or high-throughput batching, a CUDA-compatible GPU is recommended.
