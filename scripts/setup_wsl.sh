#!/usr/bin/env bash
# One-time setup of Ubuntu 24.04 on WSL2 for building, testing, profiling and
# benchmarking minicompiler, including the CUDA backend.
#
# Installs NVIDIA's CUDA 13.1 toolkit from the WSL-Ubuntu repository. It is the
# toolkit only: on WSL the GPU driver comes from Windows, and installing the
# `cuda` or `cuda-drivers` packages would conflict with it. Packages that are
# already installed are left as they are.
#
#   wsl -d Ubuntu-24.04 -e bash scripts/setup_wsl.sh
set -euo pipefail

cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake ninja-build \
    libeigen3-dev libgtest-dev graphviz linux-tools-generic \
    python3.12-venv python3.12-dev \
    cuda-toolkit-13-1

/usr/local/cuda-13.1/bin/nvcc --version | tail -1
echo "Setup finished."
