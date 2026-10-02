#!/bin/bash
# smoke.sbatch: load OpenVLA-7B on one GH200, print the action space, run the readout gate.
#   sbatch scripts/isambard/smoke.sbatch            (from /projects/b5cs/SGVLA/code)
#SBATCH --job-name=sgvla-smoke
#SBATCH --output=/projects/b5cs/SGVLA/logs/%x_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:30:00
set -eu

cd /projects/b5cs/SGVLA/code
source scripts/isambard/env.sh
# Weights are already in the cache; keep compute nodes off the network.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python scripts/run_gpu.py smoke --precision "${PRECISION:-nf4}" --n 20
