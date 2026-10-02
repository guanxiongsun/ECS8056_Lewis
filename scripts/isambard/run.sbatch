#!/bin/bash
# run.sbatch: one run_gpu.py subcommand, sharded across the job's GPUs (one shard per GPU).
#
#   sbatch --gpus=4 --time=01:00:00 scripts/isambard/run.sbatch replay --precision nf4 \
#          --out /projects/b5cs/SGVLA/runs/replay_nf4.csv
#
# Each shard writes <out>.shard<i>.csv and skips work already logged, so a job
# that hits its time limit can simply be resubmitted.
#SBATCH --job-name=sgvla-run
#SBATCH --output=/projects/b5cs/SGVLA/logs/%x_%j.out
#SBATCH --gpus=4
#SBATCH --time=01:00:00
set -eu

cd /projects/b5cs/SGVLA/code
source scripts/isambard/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CMD=${1:?usage: run.sbatch <subcommand> [run_gpu.py options]}
shift
N=${SLURM_GPUS_ON_NODE:-1}
echo "=== $CMD on $N GPU(s), job ${SLURM_JOB_ID} $(date -Is)"
for i in $(seq 0 $((N - 1))); do
    srun --ntasks=1 --gpus=1 --exclusive \
        python scripts/run_gpu.py "$CMD" --shard "$i" --nshards "$N" "$@" \
        > "$SGVLA_ROOT/logs/${CMD}_${SLURM_JOB_ID}_shard${i}.log" 2>&1 &
done
wait
echo "=== done $(date -Is)"
