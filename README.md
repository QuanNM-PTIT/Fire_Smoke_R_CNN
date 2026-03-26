# Fire & Smoke Project

## Faster R-CNN Detection (BBox)

`train_faster_rcnn.py` cần `train.txt` và `val.txt`. Dataset hiện tại chỉ có thư mục `train/` và `test/`, nên cần tạo các file danh sách này.

### Bước 1: Tạo `train.txt`, `val.txt`, `test.txt` (đường dẫn tương đối)

```bash
python - <<'PY'
from pathlib import Path
import random

seed = 42
val_ratio = 0.15
root = Path("dataset")

train_imgs = sorted((root / "train" / "images").glob("*"))
train_imgs = [p for p in train_imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

random.Random(seed).shuffle(train_imgs)
n_val = max(1, int(len(train_imgs) * val_ratio))
val_imgs = train_imgs[:n_val]
train_imgs = train_imgs[n_val:]

def to_rel(p: Path) -> str:
    return str(p.relative_to(root)).replace("\\", "/")

(root / "train.txt").write_text("\n".join(to_rel(p) for p in train_imgs) + "\n", encoding="utf-8")
(root / "val.txt").write_text("\n".join(to_rel(p) for p in val_imgs) + "\n", encoding="utf-8")

test_imgs = sorted((root / "test" / "images").glob("*"))
test_imgs = [p for p in test_imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
(root / "test.txt").write_text("\n".join(to_rel(p) for p in test_imgs) + "\n", encoding="utf-8")

print("Done:", root / "train.txt", root / "val.txt", root / "test.txt")
print("Train:", len(train_imgs), "Val:", len(val_imgs), "Test:", len(test_imgs))
PY
```

### Bước 2: Train Faster R-CNN

```bash
python src/train_faster_rcnn.py \
  --dataset-root dataset \
  --train-list dataset/train.txt \
  --val-list dataset/val.txt \
  --class-names fire,smoke \
  --output-dir runs/faster_rcnn \
  --epochs 20 \
  --batch-size 2
```

Model tốt nhất:

- `runs/faster_rcnn/best.pt`

### Bước 3: Predict ảnh (vẽ bbox)

```bash
python src/predict_image_faster_rcnn.py \
  --checkpoint runs/faster_rcnn/best.pt \
  --input input/img \
  --recursive \
  --output-dir predictions/faster_rcnn_images
```

### Bước 4: Predict video (vẽ bbox)

```bash
python src/predict_video_faster_rcnn.py \
  --checkpoint runs/faster_rcnn/best.pt \
  --input input/vid/video.mp4 \
  --output predictions/faster_rcnn_video_out.mp4
```

## 5) Ghi chú vận hành

- Nếu class id của dataset khác mặc định, chỉnh lại tham số `--fire-class-ids` và `--smoke-class-ids` ở bước tạo CSV.
- Faster R-CNN nặng hơn CNN classification, nên bắt đầu với `--batch-size 1` hoặc `2`.
- Nếu lỗi codec khi ghi `.mp4`, đổi output sang `.avi`.
