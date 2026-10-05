#!/usr/bin/env bash
# run_btp_round2.sh: BtP plan B5 (pre-processing), B6 (screened-out scenes) and B4 (compute ladder),
# one shard per GPU, after the prefix runs finish. Every A100 row is compared with B4.1 (bf16, eager,
# same GPUs); the replay subset is the four conditions the comparisons need.
#
#   bash scripts/termitech/run_btp_round2.sh "1 3 4 5 6 7"
set -u
GPUS=(${1:?usage: run_btp_round2.sh "<gpu ids>"})
N=${#GPUS[@]}
while pgrep -u "$(whoami)" -f "[r]un_btp_prefix.sh" > /dev/null; do sleep 30; done
cd ~/code/SGVLA
source scripts/termitech/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=$SGVLA_ROOT/runs/btp
PRE=$SGVLA_ROOT/v2_pre
CONDS=baseline,neutral,mirror,mirror_neutral
test -d "$PRE/official" && test -d "$PRE/bridgeorig" || { echo "pre-processed images missing in $PRE"; exit 1; }
for k in "${!GPUS[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES=${GPUS[$k]}
    run() { python scripts/run_gpu.py "$@" --shard "$k" --nshards "$N"; }
    # B5: OpenVLA's evaluation pre-processing applied to our frames, and the bridge_orig approximation
    for path in official bridgeorig; do
      SGVLA_IMAGE_ROOT=$PRE/$path run replay --precision bf16 --conditions $CONDS --out "$OUT/replay_bf16_a100_pre_$path.csv"
      SGVLA_IMAGE_ROOT=$PRE/$path run ladder --precision bf16 --out "$OUT/ladder_bf16_a100_pre_$path.csv"
    done
    # B6: screened-out and approved composites, both scored by recorded arrangement
    run manifest --precision bf16 --decisions rejected,approved --out "$OUT/manifest_bf16_a100.csv"
    # B4: compute ladder (same stimuli as B4.1)
    for prec in fp4_official int8 fp16 fp32; do
      run replay --precision $prec --conditions $CONDS --out "$OUT/replay_${prec}_a100.csv"
      run ladder --precision $prec --out "$OUT/ladder_${prec}_a100.csv"
    done
    run replay --precision bf16 --attn sdpa --conditions $CONDS --out "$OUT/replay_bf16sdpa_a100.csv"
    run ladder --precision bf16 --attn sdpa --out "$OUT/ladder_bf16sdpa_a100.csv"
    run replay --precision bf16 --conditions $CONDS --out "$OUT/replay_bf16_a100_repeat.csv"
    echo "=== round2 shard $k (GPU ${GPUS[$k]}) done $(date -Is)"
  ) > "$SGVLA_ROOT/logs/btp_round2_shard$k.log" 2>&1 &
done
wait
echo "=== all round2 shards done $(date -Is)"
