# env.sh: paths and environment for running the probe on termitech.
#   source scripts/termitech/env.sh
# huggingface.co is unreachable from this host; hf-mirror.com serves the same files.
export SGVLA_ROOT=/data/sgx/SGVLA
export SGVLA_DATA=$SGVLA_ROOT/v2
export HF_HOME=$SGVLA_ROOT/hf
export HF_ENDPOINT=https://hf-mirror.com
source ~/miniforge3/etc/profile.d/conda.sh
conda activate sgvla
