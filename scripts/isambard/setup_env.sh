#!/usr/bin/env bash
# setup_env.sh: build the probe's Python environment on Isambard-AI and fetch OpenVLA-7B.
#
# Run on a LOGIN node: it only downloads and installs (allowed there); no GPU work.
# The pins reproduce the environment the dissertation's predictions were logged
# under (probe_predictions.csv records torch 2.11.0+cu128, transformers 4.40.1,
# bitsandbytes 0.50.2). Every pinned package ships an aarch64 wheel for CPython 3.12.
#
#   bash scripts/isambard/setup_env.sh
set -euo pipefail

ROOT=/projects/b5cs/SGVLA
UV=$HOME/.local/bin/uv
export UV_CACHE_DIR=${SCRATCHDIR}/uv-cache
mkdir -p "$ROOT/logs" "$ROOT/hf" "$UV_CACHE_DIR"
exec > >(tee -a "$ROOT/logs/setup_env.log") 2>&1
echo "=== setup_env.sh $(date -Is) on $(hostname) ($(uname -m))"

if [ ! -x "$ROOT/venv/bin/python" ]; then
    "$UV" venv --python 3.12 "$ROOT/venv"
fi
PY=$ROOT/venv/bin/python
"$UV" pip install --python "$PY" torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$PY" \
    transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 \
    huggingface_hub==0.23.4 accelerate==0.30.1 bitsandbytes==0.50.2 \
    'protobuf>=6.31.1,<7' pandas scipy statsmodels matplotlib pillow pytest sentencepiece

"$PY" - <<'EOF'
import torch, transformers, tokenizers, timm, bitsandbytes, numpy, accelerate
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("transformers", transformers.__version__, "| tokenizers", tokenizers.__version__, "| timm", timm.__version__,
      "| bitsandbytes", bitsandbytes.__version__, "| accelerate", accelerate.__version__, "| numpy", numpy.__version__)
from transformers import AutoModelForVision2Seq  # noqa: F401  (removed in transformers 5.x)
print("AutoModelForVision2Seq import ok")
EOF

echo "=== downloading openvla/openvla-7b $(date -Is)"
HF_HOME=$ROOT/hf "$ROOT/venv/bin/huggingface-cli" download openvla/openvla-7b --exclude "*.bin" "*.pt"
du -sh "$ROOT/hf"
echo "=== done $(date -Is)"
