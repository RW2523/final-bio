#!/bin/bash
#SBATCH --job-name=wisdm_aug_resnet
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_augpred_resnet.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_augpred_resnet.err
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# 2080 Ti (~11GB): shorter queues on Unity gpu-preempt; use gpu:a100:1 if OOM
#SBATCH --gres=gpu:2080ti:1
## SBATCH -A <ACCOUNT>

set -euo pipefail
cd /work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR
source .venv/bin/activate

python main_ssl.py \
  --dataset wisdm --cases random --device Phones \
  --framework augpred --backbone SResNet1D \
  --wisdm_feat fdiff \
  --tau 0.75 --thresh 0.5 \
  --augpred_transforms t_flip,perm,t_warp \
  --pretrain_epochs 70 --lincls_epochs 70 \
  --early_stop_patience 0 \
  --batch_size 64 --lr 1e-3 --lr_cls 1e-3
