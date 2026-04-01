#!/bin/bash
set -e

CONFIG_FILE="$(cd "$(dirname "$0")" && pwd)/config.env"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_NAME="ARD100_mask32"
IMAGE_NAME="yolomg"
CONTAINER_NAME="yolomg_train"

# ─── Colors ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║         YOLOMG — Docker Trainer          ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ─── Load or create config ────────────────────────────────────────────────────
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    echo -e "${GREEN}✔ Config loaded:${NC} $CONFIG_FILE"
else
    echo -e "${YELLOW}⚙ First run — let's set things up.${NC}"
fi

# ─── Dataset path ─────────────────────────────────────────────────────────────
if [ -z "$DATASET_PATH" ]; then
    echo ""
    echo "Where is your dataset directory?"
    echo "  It should contain: train.txt, val.txt, images/, labels/ etc."
    echo "  Example: /data/ARD100_mask32  or  /home/user/datasets/ARD100_mask32"
    echo ""
    read -rp "Dataset path: " INPUT_PATH

    # Strip trailing slash
    INPUT_PATH="${INPUT_PATH%/}"

    if [ ! -d "$INPUT_PATH" ]; then
        echo -e "${RED}✘ Directory not found: $INPUT_PATH${NC}"
        exit 1
    fi

    DATASET_PATH="$INPUT_PATH"
    echo "DATASET_PATH=\"$DATASET_PATH\"" > "$CONFIG_FILE"
    echo -e "${GREEN}✔ Saved to config.env${NC}"
else
    echo -e "   Dataset: ${GREEN}$DATASET_PATH${NC}"
    echo ""
    read -rp "Change dataset path? (y/N): " CHANGE
    if [[ "$CHANGE" =~ ^[Yy]$ ]]; then
        read -rp "New dataset path: " INPUT_PATH
        INPUT_PATH="${INPUT_PATH%/}"
        if [ ! -d "$INPUT_PATH" ]; then
            echo -e "${RED}✘ Directory not found: $INPUT_PATH${NC}"
            exit 1
        fi
        DATASET_PATH="$INPUT_PATH"
        echo "DATASET_PATH=\"$DATASET_PATH\"" > "$CONFIG_FILE"
        echo -e "${GREEN}✔ Updated config.env${NC}"
    fi
fi

# ─── GPU count ────────────────────────────────────────────────────────────────
echo ""
echo "How many GPUs?"
echo "  1) Single GPU"
echo "  2) Two GPUs (DDP)"
read -rp "Choice [1/2] (default: 1): " GPU_CHOICE
GPU_CHOICE="${GPU_CHOICE:-1}"

# ─── Update paths in yaml and txt files ───────────────────────────────────────
echo ""
echo "─── Updating dataset paths ───────────────────────"

YAML="$REPO_DIR/data/ARD100_mask32.yaml"
CONTAINER_DATASET="/dataset"

# Update yaml to use container path
CURRENT_PREFIX=$(grep "train:" "$YAML" | head -1 | awk '{print $2}' | grep -oP "^.*${DATASET_NAME}" || true)
if [ -n "$CURRENT_PREFIX" ] && [ "$CURRENT_PREFIX" != "$CONTAINER_DATASET" ]; then
    sed -i "s|$CURRENT_PREFIX|$CONTAINER_DATASET|g" "$YAML"
    echo -e "${GREEN}✔ yaml updated${NC}"
else
    echo "  yaml already correct"
fi

# Update txt files to use container path
TXT_FILES=("train.txt" "train2.txt" "val.txt" "val2.txt" "test.txt")
for f in "${TXT_FILES[@]}"; do
    FPATH="$DATASET_PATH/$f"
    [ ! -f "$FPATH" ] && continue
    FIRST_LINE=$(head -1 "$FPATH")
    [ -z "$FIRST_LINE" ] && echo "  SKIP: $f is empty" && continue
    CURRENT_TXT_PREFIX=$(echo "$FIRST_LINE" | grep -oP "^.*${DATASET_NAME}" || true)
    if [ -n "$CURRENT_TXT_PREFIX" ] && [ "$CURRENT_TXT_PREFIX" != "$CONTAINER_DATASET" ]; then
        sed -i "s|$CURRENT_TXT_PREFIX|$CONTAINER_DATASET|g" "$FPATH"
        echo -e "${GREEN}✔ $f updated${NC}"
    else
        echo "  $f already correct"
    fi
done

# ─── Build Docker image ───────────────────────────────────────────────────────
echo ""
echo "─── Docker image ─────────────────────────────────"
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Building image (first time, may take a few minutes)..."
    docker build -t "$IMAGE_NAME" - < "$REPO_DIR/Dockerfile"
    echo -e "${GREEN}✔ Image built${NC}"
else
    echo -e "  Image ${GREEN}$IMAGE_NAME${NC} already exists"
    read -rp "Rebuild? (y/N): " REBUILD
    if [[ "$REBUILD" =~ ^[Yy]$ ]]; then
        docker build -t "$IMAGE_NAME" - < "$REPO_DIR/Dockerfile"
        echo -e "${GREEN}✔ Image rebuilt${NC}"
    fi
fi

# ─── Stop existing container ──────────────────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ""
    echo "Stopping existing container..."
    docker rm -f "$CONTAINER_NAME" &>/dev/null
fi

# ─── Launch ───────────────────────────────────────────────────────────────────
echo ""
echo "─── Launching ────────────────────────────────────"
echo "  Dataset: $DATASET_PATH → /dataset (inside container)"
echo "  Results: $REPO_DIR/runs → /app/runs (inside container)"
echo "  TensorBoard: http://localhost:6006"
echo ""

RUNS_DIR="$REPO_DIR/runs"
mkdir -p "$RUNS_DIR"

if [ "$GPU_CHOICE" = "2" ]; then
    TRAIN_CMD="python -m torch.distributed.run --nproc_per_node=2 --master_port 12345 train.py \
        --data data/ARD100_mask32.yaml --cfg models/dual_uav2.yaml \
        --weights yolov5s.pt --batch-size 16 --epochs 100 --imgsz 1280 \
        --name ARD100_mask32-1280 --device 0,1"
    GPU_FLAG="--gpus all"
else
    TRAIN_CMD="python3 train.py \
        --data data/ARD100_mask32.yaml --cfg models/dual_uav2.yaml \
        --weights yolov5s.pt --batch-size 8 --epochs 100 --imgsz 1280 \
        --name ARD100_mask32-1280 --device 0"
    GPU_FLAG="--gpus device=0"
fi

docker run -d \
    --name "$CONTAINER_NAME" \
    $GPU_FLAG \
    --shm-size=16g \
    -v "$DATASET_PATH":/dataset \
    -v "$RUNS_DIR":/app/runs \
    -p 6006:6006 \
    "$IMAGE_NAME" \
    bash -c "tensorboard --logdir /app/runs/train --host 0.0.0.0 --port 6006 & $TRAIN_CMD"

echo -e "${GREEN}✔ Container started: $CONTAINER_NAME${NC}"
echo ""
echo "Useful commands:"
echo "  Logs:         docker logs -f $CONTAINER_NAME"
echo "  Stop:         docker stop $CONTAINER_NAME"
echo "  Shell:        docker exec -it $CONTAINER_NAME bash"
echo "  TensorBoard:  http://localhost:6006"
echo ""
