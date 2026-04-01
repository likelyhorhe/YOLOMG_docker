#!/bin/bash
set -e

echo "=== YOLOMG setup for Jetson Orin AGX (JetPack 6) ==="
cd ~/ARD100

# --- 1. cuSPARSELt stub (required by Jetson PyTorch, not in JetPack 6) ---
echo "[1/5] Installing libcusparseLt stub..."
if [ ! -f /usr/local/cuda/lib64/libcusparseLt.so.0 ]; then
    cat > /tmp/cusparseLt_stub.c << 'EOF'
#include <stddef.h>
typedef int cusparseStatus_t;
typedef void* cusparseLtHandle_t;
typedef void* cusparseLtMatDescriptor_t;
typedef void* cusparseLtMatmulDescriptor_t;
typedef void* cusparseLtMatmulAlgSelection_t;
typedef void* cusparseLtMatmulPlan_t;
typedef void* cudaStream_t;
cusparseStatus_t cusparseLtInit(cusparseLtHandle_t* h) { return 0; }
cusparseStatus_t cusparseLtDestroy(const cusparseLtHandle_t* h) { return 0; }
cusparseStatus_t cusparseLtDenseDescriptorInit(const cusparseLtHandle_t* h, cusparseLtMatDescriptor_t* d, long r, long c, long ld, unsigned a, int t, int o) { return 0; }
cusparseStatus_t cusparseLtStructuredDescriptorInit(const cusparseLtHandle_t* h, cusparseLtMatDescriptor_t* d, long r, long c, long ld, unsigned a, int t, int o, int f) { return 0; }
cusparseStatus_t cusparseLtMatDescriptorDestroy(const cusparseLtMatDescriptor_t* d) { return 0; }
cusparseStatus_t cusparseLtMatDescSetAttribute(const cusparseLtHandle_t* h, cusparseLtMatDescriptor_t* d, int a, const void* v, size_t s) { return 0; }
cusparseStatus_t cusparseLtMatDescGetAttribute(const cusparseLtHandle_t* h, const cusparseLtMatDescriptor_t* d, int a, void* v, size_t s) { return 0; }
cusparseStatus_t cusparseLtMatmulDescriptorInit(const cusparseLtHandle_t* h, cusparseLtMatmulDescriptor_t* d, int oa, int ob, const cusparseLtMatDescriptor_t* a, const cusparseLtMatDescriptor_t* b, const cusparseLtMatDescriptor_t* c, const cusparseLtMatDescriptor_t* dd, int ct) { return 0; }
cusparseStatus_t cusparseLtMatmulDescSetAttribute(const cusparseLtHandle_t* h, cusparseLtMatmulDescriptor_t* d, int a, const void* v, size_t s) { return 0; }
cusparseStatus_t cusparseLtMatmulDescGetAttribute(const cusparseLtHandle_t* h, const cusparseLtMatmulDescriptor_t* d, int a, void* v, size_t s) { return 0; }
cusparseStatus_t cusparseLtMatmulAlgSelectionInit(const cusparseLtHandle_t* h, cusparseLtMatmulAlgSelection_t* a, const cusparseLtMatmulDescriptor_t* d, int alg) { return 0; }
cusparseStatus_t cusparseLtMatmulAlgSetAttribute(const cusparseLtHandle_t* h, cusparseLtMatmulAlgSelection_t* a, int attr, const void* v, size_t s) { return 0; }
cusparseStatus_t cusparseLtMatmulAlgGetAttribute(const cusparseLtHandle_t* h, const cusparseLtMatmulAlgSelection_t* a, int attr, void* v, size_t s) { return 0; }
cusparseStatus_t cusparseLtMatmulGetWorkspace(const cusparseLtHandle_t* h, const cusparseLtMatmulPlan_t* p, size_t* s) { return 0; }
cusparseStatus_t cusparseLtMatmulPlanInit(const cusparseLtHandle_t* h, cusparseLtMatmulPlan_t* p, const cusparseLtMatmulDescriptor_t* d, const cusparseLtMatmulAlgSelection_t* a) { return 0; }
cusparseStatus_t cusparseLtMatmulPlanDestroy(const cusparseLtMatmulPlan_t* p) { return 0; }
cusparseStatus_t cusparseLtMatmul(const cusparseLtHandle_t* h, const cusparseLtMatmulPlan_t* p, const void* a, const void* b, const void* c, void* d, void* w, cudaStream_t* s, int n) { return 0; }
cusparseStatus_t cusparseLtMatmulSearch(const cusparseLtHandle_t* h, cusparseLtMatmulPlan_t* p, const void* a, const void* b, const void* c, void* d, void* w, cudaStream_t* s, int n) { return 0; }
cusparseStatus_t cusparseLtSpMMAPrune(const cusparseLtHandle_t* h, const cusparseLtMatmulDescriptor_t* d, const void* i, void* o, int t, cudaStream_t s) { return 0; }
cusparseStatus_t cusparseLtSpMMAPrune2(const cusparseLtHandle_t* h, const cusparseLtMatDescriptor_t* s, int a, int op, const void* i, void* o, int t, cudaStream_t st) { return 0; }
cusparseStatus_t cusparseLtSpMMACompressedSize(const cusparseLtHandle_t* h, const cusparseLtMatmulPlan_t* p, size_t* s) { return 0; }
cusparseStatus_t cusparseLtSpMMACompressedSize2(const cusparseLtHandle_t* h, const cusparseLtMatDescriptor_t* d, size_t* s) { return 0; }
cusparseStatus_t cusparseLtSpMMACompress(const cusparseLtHandle_t* h, const cusparseLtMatmulPlan_t* p, const void* i, void* o, cudaStream_t s) { return 0; }
cusparseStatus_t cusparseLtSpMMACompress2(const cusparseLtHandle_t* h, const cusparseLtMatDescriptor_t* d, int a, int op, const void* i, void* o, cudaStream_t s) { return 0; }
EOF
    gcc -shared -fPIC -o /tmp/libcusparseLt.so.0 /tmp/cusparseLt_stub.c
    sudo cp /tmp/libcusparseLt.so.0 /usr/local/cuda/lib64/libcusparseLt.so.0
    echo "  Stub installed."
else
    echo "  Already exists, skipping."
fi

# --- 2. Create venv with access to system Jetson PyTorch ---
echo "[2/5] Creating virtual environment..."
python3 -m venv venv --system-site-packages
source venv/bin/activate

# --- 3. Upgrade pip ---
echo "[3/5] Upgrading pip..."
pip install --upgrade pip

# --- 4. Install requirements (excluding torch/torchvision — use system Jetson build) ---
echo "[4/5] Installing requirements..."
sed -i 's/numpy>=1.18.5$/numpy>=1.18.5,<2/' requirements.txt
pip install "numpy>=1.18.5,<2" --force-reinstall
grep -v -E "^torch|^torchvision" requirements.txt > /tmp/requirements_no_torch.txt
pip install -r /tmp/requirements_no_torch.txt

# --- 5. Verify ---
echo "[5/5] Verifying..."
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
import numpy as np
print('NumPy:', np.__version__)
"

echo ""
echo "=== Setup complete! ==="
echo "To activate: source ~/ARD100/venv/bin/activate"
echo "To train:"
echo "  python3 train.py --data data/ARD100_mask32.yaml --cfg models/dual_uav2.yaml \\"
echo "    --weights yolov5s.pt --batch-size 8 --epochs 100 --imgsz 1280 \\"
echo "    --name ARD100_mask32-1280 --device 0"
