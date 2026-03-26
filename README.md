# Fire & Smoke Detection Toolkit (PyTorch)

A practical PyTorch project for fire/smoke analysis on images and videos.

This repository includes two pipelines:

1. **Multi-label image classification** (`fire`, `smoke`) with optional Grad-CAM overlays.
2. **Object detection** (bounding boxes) using Faster R-CNN trained from YOLO-format labels.

The code is designed to run on CPU, CUDA, and Apple Silicon (`mps`) when available.

## Key Features

- End-to-end training and inference for fire/smoke **classification**.
- End-to-end training and inference for fire/smoke **detection**.
- CSV builders for multiple dataset layouts:
  - Class-folder datasets (`only_fire`, `only_smoke`, `fire_and_smoke`, `none`)
  - YOLO labels + split files (`train.txt`, `val.txt`, `test.txt`)
- Frame-level video inference with temporal smoothing.
- Optional Grad-CAM visualization for classifier predictions.
- Training logs and artifact export (`best.pt`, `last.pt`, metrics/history CSV/JSON).

## Repository Structure

```text
Fire_Smoke_R_CNN/
  src/
    train.py
    predict_image.py
    predict_video.py
    train_faster_rcnn.py
    predict_image_faster_rcnn.py
    predict_video_faster_rcnn.py
    build_csv_from_class_folders.py
    build_csv_from_yolo_splits.py
    build_csv_from_yolo_dirs.py
  data_csv/                    # Generated classification CSV files (train/val/test)
  dataset/                     # Local dataset working folder (ignored by git except split manifests)
  input/                       # Example input assets
  predictions/                 # Inference outputs
  runs/                        # Training artifacts
  requirements.txt
  README.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Optional: Verify device backend

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('mps available:', torch.backends.mps.is_available())
print('cuda available:', torch.cuda.is_available())
PY
```

## Pipeline A: Multi-Label Classification

### 1) Prepare CSV files

Expected CSV schema:

- `path` (image path, relative or absolute)
- `fire` (0/1)
- `smoke` (0/1)

#### Option A: Build CSV from class folders

```bash
python src/build_csv_from_class_folders.py \
  --dataset-root /path/to/dataset_root \
  --output-dir data_csv
```

Supported class folder keywords include:

- `only_fire`, `fire_only` -> `(fire=1, smoke=0)`
- `only_smoke`, `smoke_only` -> `(fire=0, smoke=1)`
- `fire_and_smoke`, `both`, `fire_smoke` -> `(1, 1)`
- `none`, `normal`, `background`, `no_fire_no_smoke` -> `(0, 0)`

Split folders are inferred from names such as `train`, `val`, `validation`, `test`, etc.

#### Option B: Build CSV from YOLO split files

```bash
python src/build_csv_from_yolo_splits.py \
  --dataset-root /path/to/dataset_root \
  --train-list /path/to/dataset_root/train.txt \
  --val-list /path/to/dataset_root/val.txt \
  --test-list /path/to/dataset_root/test.txt \
  --fire-class-ids 0 \
  --smoke-class-ids 1 \
  --output-dir data_csv
```

### 2) Train classifier

```bash
python src/train.py \
  --train-csv data_csv/train.csv \
  --val-csv data_csv/val.csv \
  --dataset-root /path/to/dataset_root \
  --output-dir runs/dfire_cnn \
  --model-name resnet18 \
  --epochs 20 \
  --batch-size 16 \
  --image-size 224 \
  --lr 1e-4
```

Supported classifier backbones:

- `resnet18` (default)
- `resnet34`
- `mobilenet_v3_small`

### 3) Predict on images

```bash
python src/predict_image.py \
  --checkpoint runs/dfire_cnn/best.pt \
  --input /path/to/image_or_folder \
  --output-dir predictions/images \
  --recursive
```

Optional Grad-CAM:

```bash
python src/predict_image.py \
  --checkpoint runs/dfire_cnn/best.pt \
  --input /path/to/image_or_folder \
  --output-dir predictions/images_cam \
  --recursive \
  --grad-cam \
  --cam-target both
```

### 4) Predict on videos

```bash
python src/predict_video.py \
  --checkpoint runs/dfire_cnn/best.pt \
  --input /path/to/video.mp4 \
  --output predictions/demo_out.mp4
```

Useful options:

- `--frame-stride 2` for faster processing
- `--alpha 0.8` for EMA smoothing
- `--min-consecutive 3` to reduce flicker/false alarms
- `--grad-cam` to overlay heatmaps
- `--display` for live preview (optional)

### Classification outputs

Typical files under `runs/<exp_name>/`:

- `best.pt`
- `last.pt`
- `history.csv`
- `best_metrics.json`
- `config.json`

Visualization helper:

```bash
python src/plot_history.py --history runs/dfire_cnn/history.csv
```

## Pipeline B: Faster R-CNN Detection

This pipeline trains and runs bounding-box detection from YOLO labels.

### 1) Train detector

```bash
python src/train_faster_rcnn.py \
  --dataset-root /path/to/dataset_root \
  --train-list /path/to/dataset_root/train.txt \
  --val-list /path/to/dataset_root/val.txt \
  --class-names fire,smoke \
  --output-dir runs/faster_rcnn \
  --epochs 20 \
  --batch-size 2
```

### 2) Predict on images

```bash
python src/predict_image_faster_rcnn.py \
  --checkpoint runs/faster_rcnn/best.pt \
  --input /path/to/image_or_folder \
  --output-dir predictions/faster_rcnn_images \
  --recursive \
  --score-threshold 0.40 \
  --nms-iou 0.50
```

### 3) Predict on videos

```bash
python src/predict_video_faster_rcnn.py \
  --checkpoint runs/faster_rcnn/best.pt \
  --input /path/to/video.mp4 \
  --output predictions/faster_rcnn_video_out.mp4 \
  --score-threshold 0.40 \
  --nms-iou 0.50
```

Detector inference exports:

- Annotated images/videos
- Per-image/per-frame summary CSV
- Full detection CSV (box coordinates, class, score)

## Dataset Notes

- `dataset/` is intended as a local workspace for raw data and split files.
- Keep heavyweight raw images/labels outside git history.
- Use generated CSVs in `data_csv/` for classification training.

## Troubleshooting

- If `.mp4` writing fails, try `.avi` output.
- If `mps` is unavailable on Apple Silicon, update macOS/PyTorch and recreate the virtual environment.
- If `torch`/`torchvision` mismatch errors appear, reinstall both packages in the same environment.

## License

Add your preferred license (for example MIT) before publishing publicly.
