# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**YOLOMG** is a vision-based drone-to-drone detection system that fuses two input streams:
- **Appearance stream:** Standard RGB images
- **Motion stream:** Pixel-level motion masks (from optical flow + multi-frame differencing)

The model is a modified YOLOv5 with dual backbones fused via a custom `Concat3` attention layer (`models/common.py`). The primary dataset is **ARD100** (100 aerial videos of Phantom-series drones).

## Commands

### Training
```bash
# Single GPU
python3 train.py --data data/ARD100_mask32.yaml --cfg models/dual_uav2.yaml \
  --weights yolov5s.pt --batch-size 8 --epochs 100 --imgsz 1280 --name ARD100_mask32-1280

# Multi-GPU (DDP)
python -m torch.distributed.run --nproc_per_node=4 --master_port 12345 train.py \
  --data data/ARD100_mask32.yaml --cfg models/dual_uav2.yaml --weights yolov5s.pt \
  --batch-size 16 --epochs 100 --imgsz 1280 --name ARD100_mask32-1280 --device 0,1,2,3
```

### Validation
```bash
python3 val.py --weights runs/train/ARD100_mask32-1280/weights/best.pt \
  --data data/ARD100_mask32.yaml --task val --conf-thres 0.001 --imgsz 1280 --batch-size 8
```

### Inference / Demo
```bash
python3 dualdetector.py
```

### Data Preparation
```bash
# 1. Extract frames from MP4 videos
python3 YOLOMG_extract_frames.py

# 2. Generate motion masks (multi-frame difference with motion compensation)
python3 test_code/generate_mask5.py

# 3. Create dataset splits and annotation files
python3 test_code/generate_dataset.py
python3 data/voc_label.py   # appearance image lists
python3 data/voc_label2.py  # motion mask image lists
```

## Architecture

### Dual-Stream Model (`models/`)
- **`dual_uav2.yaml`** — Main model config. Two backbones (`backbone1depth: 2` for appearance, separate depth for motion), fused via `Concat3` attention at multiple stages, then shared FPN neck + detection head.
- **`yolo.py`** — `Model` class builds from YAML. `Detect` handles anchor-based outputs. `forward(x1, x2)` takes separate appearance and motion tensors.
- **`common.py`** — Defines `Concat3` (spatial + channel attention fusion), `CARAFE` (upsampling), `CBAM`, `ChannelAttention`, `SpatialAttention`.

**Fusion mechanism (`Concat3`):** Combines features as `x1*(2-weight) + x2*weight` where `weight` is a learned spatial+channel attention on the motion stream.

### Data Pipeline (`utils/datasets.py`)
- `LoadImagesAndMasks` — Dual-stream loader pairing appearance images (from `train.txt`) with motion masks (from `train2.txt`).
- `create_dataloader()` — Returns dataloaders for both streams together.

### Dataset Config (`data/ARD100_mask32.yaml`)
Points to two parallel directory trees: one for RGB images, one for motion masks. The `.yaml` defines `train`, `val`, `test` paths for **both** streams (appearance + mask).

### Motion Mask Generation (`test_code/`)
- **`MOD_Functions.py`** — Core optical flow utilities: `motion_compensate()`, `ECC_stablize()`, `affine_stablize()` (Lucas-Kanade tracker).
- **`FD5_mask.py`** — Computes multi-frame difference masks: compares frame at `t` vs `t-2` and `t-4` after camera motion compensation. Saves grayscale masks.
- `generate_mask5.py` orchestrates the full video → mask pipeline.

### Training Outputs
Checkpoints and logs saved to `runs/train/<name>/`:
- `weights/best.pt` — Best checkpoint by mAP
- `weights/last.pt` — Latest checkpoint
- TensorBoard logs, metric plots, confusion matrix

## Key Design Notes

- The model detects a **single class** (Drone). `nc: 1` in all model YAMLs.
- Default image size is **1280×1280** (not YOLOv5's standard 640) to preserve small drone targets.
- Anchors in `dual_uav2.yaml` are tuned for small aerial targets, not COCO defaults.
- `val.py` uses `--conf-thres 0.001` during evaluation (very low threshold for mAP computation); use higher (e.g. `0.1`) for actual inference.
- `dualdetector.py` provides a clean `Yolov5Detector` class wrapping preprocessing + inference + NMS for deployment use.
