#!/bin/bash
#SBATCH --job-name=wisdm_snn_supervised
#SBATCH --output=slurm_logs/supervised-%j.out
#SBATCH --error=slurm_logs/supervised-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
# 2080 Ti (~11GB): shorter queues on Unity gpu-preempt; use gpu:a100:1 if OOM
#SBATCH --gres=gpu:2080ti:1

set -euo pipefail

REPO_DIR="/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR"
cd "${REPO_DIR}"

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Working dir: $(pwd)"
nvidia-smi || true

source .venv/bin/activate
mkdir -p log results plot slurm_logs run_logs

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="run_logs/wisdm_supervised_${RUN_TS}.log"

python main.py \
  --dataset wisdm \
  --backbone SFCN \
  --lr 1e-3 \
  --tau 0.75 \
  --thresh 0.5 \
  --wisdm_data_dir ./data/wisdm \
  --n_epoch 100 \
  --early_stop_patience 0 \
  --early_stop_min_delta 0.0 \
  2>&1 | tee "${RUN_LOG}"

echo "Job finished at: $(date)"
echo "Run log saved to: ${RUN_LOG}"
