#!/usr/bin/env bash
# setup_env.sh: build the `sgvla` conda environment on termitech and fetch OpenVLA-7B.
#
# The pins reproduce the environment the dissertation's predictions were logged
# under (probe_predictions.csv records torch 2.11.0+cu128, transformers 4.40.1,
# bitsandbytes 0.50.2). Hugging Face is not reachable from this host, so weights
# come through the hf-mirror.com endpoint into a cache on /data.
#
#   tmux new -d -s sgvla-setup 'bash ~/code/SGVLA/scripts/termitech/setup_env.sh'
set -euo pipefail

ENV_NAME=sgvla
ROOT=/data/sgx/SGVLA
LOG_DIR=$ROOT/logs
mkdir -p "$LOG_DIR" "$ROOT/hf"
exec > >(tee -a "$LOG_DIR/setup_env.log") 2>&1
echo "=== setup_env.sh $(date -Is)"

source ~/miniforge3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    mamba create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"
python -V

python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --only-binary=:all: \
    transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 \
    huggingface_hub==0.23.4 accelerate==0.30.1 bitsandbytes==0.50.2 \
    'protobuf>=6.31.1,<7'
python -m pip install pandas scipy statsmodels matplotlib pillow pytest sentencepiece

python - <<'EOF'
import torch, transformers, tokenizers, timm, bitsandbytes, numpy, accelerate
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available(),
      "| devices", torch.cuda.device_count())
print("transformers", transformers.__version__, "| tokenizers", tokenizers.__version__, "| timm", timm.__version__,
      "| bitsandbytes", bitsandbytes.__version__, "| accelerate", accelerate.__version__, "| numpy", numpy.__version__)
from transformers import AutoModelForVision2Seq  # noqa: F401  (removed in transformers 5.x)
print("AutoModelForVision2Seq import ok")
EOF

echo "=== downloading openvla/openvla-7b via hf-mirror $(date -Is)"
export HF_HOME=$ROOT/hf HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download openvla/openvla-7b --exclude "*.bin" "*.pt"
du -sh "$HF_HOME"
echo "=== done $(date -Is)"
