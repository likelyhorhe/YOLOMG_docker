# YOLOMG
Codes and dataset for the paper "YOLOMG: Vision-based Drone-to-Drone Detection with Appearance and Pixel-Level Motion Fusion"

## Dataset
ARD100 dataset (100 aerial drone videos, Phantom series)
- [BaiduYun](https://pan.baidu.com/s/1ycAoKbzQ1rlzvKr8VRakgw?pwd=1x2z) (code: 1x2z)

![Dataset Example Images](data/ARD100_samples_show.png "Example Images")

---

## Quick Start — Docker (recommended)

**Requirements:** Docker + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
git clone <repo>
cd ARD100
./run.sh
```

The script will ask where your dataset is, save the config, and launch training + TensorBoard automatically.

- Training results → `runs/train/` (on your machine)
- TensorBoard → `http://localhost:6006`

---

## Full Pipeline

```
train_videos/  test_videos/          ← source .mp4 files
      │               │
      ▼               ▼
 [Step 1] Extract frames
      │               │
      ▼               ▼
train_images/    test_images/        ← per-video folders of .jpg frames
                      │
                      ▼
               [Step 2] Generate motion masks
                      │
                      ▼
                  masks/             ← per-video folders of grayscale mask .jpg
                      │
                      ▼
               [Step 3] Build dataset
                      │
                      ▼
          datasets/ARD100_mask32/
            images/  masks/  Annotations/
                      │
                      ▼
               [Step 4] Create splits + labels + path lists
                      │
                      ▼
          train.txt  val.txt  test.txt
          train2.txt val2.txt
                      │
                      ▼
               [Step 5] Train        → runs/train/.../weights/best.pt
                      │
                      ▼
               [Step 6] Validate
                      │
                      ▼
               [Step 7] Inference
```

---

## Step 1 — Extract frames

```bash
python3 YOLOMG_extract_frames.py
```

Edit `video_folder` and `image_folder` inside the script for train vs test videos.

---

## Step 2 — Generate motion masks

```bash
python3 test_code/generate_mask5.py
```

Uses 4-frame multi-difference with optical flow motion compensation. To process training videos, change `set0` to `sets` inside the script.

---

## Step 3 — Build dataset

```bash
python3 test_code/generate_dataset.py
```

Assembles `datasets/ARD100_mask32/images/`, `masks/`, `Annotations/`. Skips frames where drone bounding box area < 25 px².

---

## Step 4 — Create splits, labels, and path lists

**4a. Split train / val / test**
```bash
python3 data/split_train_val.py \
    --xml_path /path/to/datasets/ARD100_mask32/Annotations \
    --txt_path /path/to/datasets/ARD100_mask32/ImageSets/Main
```

**4b. Convert VOC XML → YOLO labels**
```bash
python3 data/voc2yolo.py
```

**4c. Generate path lists**
```bash
python3 data/voc_label.py
```

Produces `train.txt`, `train2.txt`, `val.txt`, `val2.txt`, `test.txt` inside the dataset folder.

---

## Step 5 — Train

**Via Docker (recommended):**
```bash
./run.sh
```

**Manually — Single GPU:**
```bash
python3 train.py --data data/ARD100_mask32.yaml --cfg models/dual_uav2.yaml \
    --weights yolov5s.pt --batch-size 8 --epochs 100 --imgsz 1280 \
    --name ARD100_mask32-1280
```

**Manually — Multi-GPU (DDP):**
```bash
python -m torch.distributed.run --nproc_per_node=2 --master_port 12345 train.py \
    --data data/ARD100_mask32.yaml --cfg models/dual_uav2.yaml \
    --weights yolov5s.pt --batch-size 16 --epochs 100 --imgsz 1280 \
    --name ARD100_mask32-1280 --device 0,1
```

---

## Step 6 — Validate

```bash
python3 val.py --weights runs/train/ARD100_mask32-1280/weights/best.pt \
    --data data/ARD100_mask32.yaml --task val --conf-thres 0.001 --imgsz 1280 --batch-size 8
```

> `--conf-thres 0.001` is intentionally low for full mAP curve computation.

---

## Step 7 — Inference

```bash
python3 dualdetector.py
```

---

## NPS Dataset (alternative)

1. Place data under `datasets/NPS3/` with subfolders: `images/`, `mask3/`, `annotations/`, `ImageSets/Main/`
2. Generate masks: `python3 test_code/generate_mask3.py`
3. Convert annotations: `python3 data/voc2yolo.py`
4. Create path lists: `python3 data/voc_label.py`
5. Train:
```bash
python3 train.py --data data/NPS3.yaml --cfg models/NPS_uav_s.yaml \
    --weights yolov5s.pt --batch-size 8 --epochs 100 --imgsz 1280 --name NPS-1280
```
