#!/bin/bash

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
YAML="$REPO_DIR/data/ARD100_mask32.yaml"
DATASET_DIR="$REPO_DIR/datasets/ARD100_mask32"
DATASET_NAME="ARD100_mask32"

TXT_FILES=("train.txt" "train2.txt" "val.txt" "val2.txt" "test.txt")

ERRORS=0

echo "=== Dataset check for YOLOMG ==="
echo "Repo:    $REPO_DIR"
echo "Dataset: $DATASET_DIR"
echo ""

# --- 1. Check yaml exists ---
echo "[1] Checking yaml..."
if [ ! -f "$YAML" ]; then
    echo "  ERROR: $YAML not found"
    exit 1
fi
echo "  OK: $YAML"

# --- 2. Check and fix paths in yaml ---
echo ""
echo "[2] Checking paths in yaml..."
CORRECT_YAML_PREFIX="$DATASET_DIR"

# Extract current prefix in yaml (everything up to and including ARD100_mask32)
CURRENT_YAML_PREFIX=$(grep "train:" "$YAML" | head -1 | awk '{print $2}' | grep -oP "^.*${DATASET_NAME}")

if [ -z "$CURRENT_YAML_PREFIX" ]; then
    echo "  ERROR: could not parse paths in yaml"
    ERRORS=$((ERRORS + 1))
elif [ "$CURRENT_YAML_PREFIX" != "$CORRECT_YAML_PREFIX" ]; then
    echo "  MISMATCH: yaml has $CURRENT_YAML_PREFIX"
    echo "  Updating to $CORRECT_YAML_PREFIX ..."
    sed -i "s|$CURRENT_YAML_PREFIX|$CORRECT_YAML_PREFIX|g" "$YAML"
    echo "  Updated."
else
    echo "  OK: yaml paths correct"
fi

# --- 3. Check txt files exist ---
echo ""
echo "[3] Checking txt files..."
for f in "${TXT_FILES[@]}"; do
    FPATH="$DATASET_DIR/$f"
    if [ ! -f "$FPATH" ]; then
        echo "  ERROR: $FPATH not found"
        ERRORS=$((ERRORS + 1))
    else
        echo "  OK: $f ($(wc -l < "$FPATH") lines)"
    fi
done

# --- 4. Check and fix paths inside txt files ---
echo ""
echo "[4] Checking paths inside txt files..."
CORRECT_TXT_PREFIX="$DATASET_DIR"

for f in "${TXT_FILES[@]}"; do
    FPATH="$DATASET_DIR/$f"
    [ ! -f "$FPATH" ] && continue

    FIRST_LINE=$(head -1 "$FPATH")
    if [ -z "$FIRST_LINE" ]; then
        echo "  SKIP: $f is empty"
        continue
    fi

    # Extract current prefix (everything up to and including ARD100_mask32)
    CURRENT_PREFIX=$(echo "$FIRST_LINE" | grep -oP "^.*${DATASET_NAME}")

    if [ -z "$CURRENT_PREFIX" ]; then
        echo "  WARNING: $f — cannot parse path: $FIRST_LINE"
        ERRORS=$((ERRORS + 1))
        continue
    fi

    if [ "$CURRENT_PREFIX" != "$CORRECT_TXT_PREFIX" ]; then
        echo "  MISMATCH in $f: found $CURRENT_PREFIX"
        echo "  Updating to $CORRECT_TXT_PREFIX ..."
        sed -i "s|$CURRENT_PREFIX|$CORRECT_TXT_PREFIX|g" "$FPATH"
        echo "  Updated."
    else
        echo "  OK: $f"
    fi
done

# --- 5. Spot-check that first image in train.txt exists ---
echo ""
echo "[5] Spot-checking first image in train.txt..."
TRAIN_TXT="$DATASET_DIR/train.txt"
if [ -f "$TRAIN_TXT" ] && [ -s "$TRAIN_TXT" ]; then
    FIRST_IMG=$(head -1 "$TRAIN_TXT")
    if [ -f "$FIRST_IMG" ]; then
        echo "  OK: $FIRST_IMG"
    else
        echo "  WARNING: $FIRST_IMG not found — dataset images may be missing"
        ERRORS=$((ERRORS + 1))
    fi
fi

# --- Summary ---
echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo "=== All checks passed. Ready to train. ==="
else
    echo "=== $ERRORS error(s) found. Fix them before training. ==="
fi
