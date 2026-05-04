#!/bin/bash
#SBATCH --job-name=wisdm_simclr_resnet_top5
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_simclr_resnet_top5.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_simclr_resnet_top5.err
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# 2080 Ti (~11GB): shorter queues on Unity gpu-preempt than A100; use gpu:a100:1 if OOM
#SBATCH --gres=gpu:2080ti:1
## SBATCH -A <ACCOUNT>

set -euo pipefail
cd /work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR
source .venv/bin/activate
# so Slurm .out files update as the job runs (otherwise Python may buffer for a long time)
export PYTHONUNBUFFERED=1

python main_ssl.py \
  --dataset wisdm --cases random --device Phones \
  --framework simclr --backbone SResNet1D --criterion NTXent \
  --tau 0.75 --thresh 0.5 \
  --simclr_contrastive instance \
  --wisdm_feat fdiff \
  --wisdm_top_k_classes 5 \
  --aug1 jit_scal --aug2 perm_jit \
  --p 128 --phid 128 \
  --pretrain_epochs 70 --lincls_epochs 70 \
  --early_stop_patience 0 \
  --batch_size 64 --lr 1e-3 --lr_cls 1e-3
