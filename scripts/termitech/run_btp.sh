#!/usr/bin/env bash
# run_btp.sh: the BtP A100 runs (docs/btp_experiments_plan.md, B4.1 and X2), one shard per GPU.
#
#   bash scripts/termitech/run_btp.sh "1 3 4 5 6 7"       # GPUs to use, one shard each
#
# Every job runs at batch size 1 and logs full action distributions (--save-dist).
# Shards resume, so the script can simply be re-run after an interruption.
set -u
GPUS=(${1:?usage: run_btp.sh "<gpu ids>"})
N=${#GPUS[@]}
cd ~/code/SGVLA
source scripts/termitech/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=$SGVLA_ROOT/runs/btp
mkdir -p "$OUT" "$SGVLA_ROOT/logs"
for k in "${!GPUS[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES=${GPUS[$k]}
    run() { python scripts/run_gpu.py "$@" --shard "$k" --nshards "$N" --save-dist; }
    # B4.1: A100 bf16 baseline on the replay and ladder stimuli, plus same-twin placebo pairs.
    run replay --precision bf16 --out "$OUT/replay_bf16_a100.csv"
    run ladder --precision bf16 --mirror --placebo --out "$OUT/ladder_bf16_a100.csv"
    # X2: natural frames with raw and lower-cased instructions, both precisions, same GPUs.
    run natural --precision bf16 --out "$OUT/natural_bf16_a100.csv"
    run natural --precision bf16 --lowercase --out "$OUT/natural_lower_bf16_a100.csv"
    run natural --precision nf4 --out "$OUT/natural_nf4_a100.csv"
    run natural --precision nf4 --lowercase --out "$OUT/natural_lower_nf4_a100.csv"
    echo "=== shard $k (GPU ${GPUS[$k]}) done $(date -Is)"
  ) > "$SGVLA_ROOT/logs/btp_shard$k.log" 2>&1 &
done
wait
echo "=== all shards done $(date -Is)"
