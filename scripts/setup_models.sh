#!/usr/bin/env bash
set -euo pipefail
camera_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$camera_root"
python3 -m venv --without-pip .venv-models
python3 -m pip --python .venv-models/bin/python install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
python3 -m pip --python .venv-models/bin/python install ultralytics==8.4.177 onnx==1.23.2 onnxruntime==1.31.0 onnxslim==0.1.98
mkdir -p third_party
if [[ ! -f third_party/onnxruntime-linux-x64-1.31.0/include/onnxruntime_cxx_api.h ]]; then
    camera_archive="$(mktemp)"
    trap 'rm -f "$camera_archive"' EXIT
    curl -fL --retry 2 https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-linux-x64-1.31.0.tgz -o "$camera_archive"
    tar -xzf "$camera_archive" -C third_party
fi
.venv-models/bin/python scripts/prepare_models.py
