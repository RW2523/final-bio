#!/bin/bash
#SBATCH --job-name=wisdm_simclr_fcn
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_simclr_fcn.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_simclr_fcn.err
# 2080 Ti: CUDA OK for current PyTorch; shorter queues than A100. Use gpu:a100:1 if OOM.
# SFCN = spiking FCN (aligns with strong WISDM AugPred runs). Larger batch = more
# NTXent negatives; SupCon also sees more same-class pairs per step. Reduce batch if OOM.
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:2080ti:1
## SBATCH -A <ACCOUNT>

set -euo pipefail
cd /work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR
source .venv/bin/activate

python main_ssl.py \
  --dataset wisdm --cases random --device Phones \
  --framework simclr --backbone SFCN --criterion NTXent \
  --tau 0.75 --thresh 0.5 \
  --simclr_contrastive supcon \
  --supcon_temperature 0.07 \
  --wisdm_feat fdiff \
  --aug1 jit_scal --aug2 perm_jit \
  --p 128 --phid 128 \
  --pretrain_epochs 100 --lincls_epochs 80 \
  --early_stop_patience 0 \
  --batch_size 128 --lr 1e-3 --lr_cls 1.5e-3
