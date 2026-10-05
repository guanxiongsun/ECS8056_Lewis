#!/usr/bin/env bash
# run_btp_prefix.sh: BtP plan B2 (prefix control), one shard per GPU, after run_btp.sh finishes.
#
#   bash scripts/termitech/run_btp_prefix.sh "1 3 4 5 6 7"
set -u
GPUS=(${1:?usage: run_btp_prefix.sh "<gpu ids>"})
N=${#GPUS[@]}
while pgrep -u "$(whoami)" -f "[r]un_btp.sh" > /dev/null; do sleep 30; done
cd ~/code/SGVLA
source scripts/termitech/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=$SGVLA_ROOT/runs/btp
mkdir -p "$OUT" "$SGVLA_ROOT/logs"
for k in "${!GPUS[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES=${GPUS[$k]}
    python scripts/run_gpu.py prefix --precision bf16 --mirror --shard "$k" --nshards "$N" --out "$OUT/prefix_bf16_a100.csv"
    python scripts/run_gpu.py prefix --precision nf4 --mirror --shard "$k" --nshards "$N" --out "$OUT/prefix_nf4_a100.csv"
    echo "=== prefix shard $k (GPU ${GPUS[$k]}) done $(date -Is)"
  ) > "$SGVLA_ROOT/logs/btp_prefix_shard$k.log" 2>&1 &
done
wait
echo "=== all prefix shards done $(date -Is)"
