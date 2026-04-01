#!/bin/bash

echo "=== Jetson Orin AGX Diagnostics ==="
echo ""

# JetPack version
echo "[1] JetPack / L4T version:"
cat /etc/nv_tegra_release 2>/dev/null || echo "  not found"
echo ""

# CUDA version
echo "[2] CUDA:"
nvcc --version 2>/dev/null || echo "  nvcc not found"
ls /usr/local/cuda* -d 2>/dev/null || echo "  /usr/local/cuda not found"
echo ""

# CUDA libraries
echo "[3] CUDA libraries in /usr/local/cuda/lib64:"
ls /usr/local/cuda/lib64/libcusparse* 2>/dev/null || echo "  not found"
ls /usr/local/cuda/lib64/libcusparseLt* 2>/dev/null || echo "  libcusparseLt not found"
echo ""

# Tegra libraries
echo "[4] Tegra libraries:"
ls /usr/lib/aarch64-linux-gnu/tegra/libcuda* 2>/dev/null || echo "  not found"
echo ""

# Python
echo "[5] Python:"
python3 --version
which python3
echo ""

# System PyTorch
echo "[6] System PyTorch (pip3):"
pip3 show torch 2>/dev/null | grep -E "Name|Version|Location" || echo "  not found"
echo ""

# Test system torch import
echo "[7] System torch CUDA test:"
python3 -c "
import torch
print('  Version:', torch.__version__)
print('  CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  Device:', torch.cuda.get_device_name(0))
" 2>&1 | sed 's/^/  /'
echo ""

# LD_LIBRARY_PATH
echo "[8] LD_LIBRARY_PATH:"
echo "  ${LD_LIBRARY_PATH:-<empty>}"
echo ""

# GPU info
echo "[9] GPU / NVIDIA driver:"
nvidia-smi 2>/dev/null || echo "  nvidia-smi not found"
cat /proc/driver/nvidia/version 2>/dev/null || true
echo ""

# Find libcusparseLt anywhere
echo "[10] Searching for libcusparseLt.so anywhere:"
find /usr /opt -name "libcusparseLt*" 2>/dev/null || echo "  not found"

echo ""
echo "=== Done ==="
