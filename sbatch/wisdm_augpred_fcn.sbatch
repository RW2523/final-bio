#!/bin/bash
#SBATCH --job-name=wisdm_aug_fcn
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_augpred_fcn.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_augpred_fcn.err
# 2080 Ti: CUDA OK for current PyTorch; shorter queues than A100. Use gpu:a100:1 if OOM.
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2080ti:1
## SBATCH -A <ACCOUNT>         # uncomment if your site requires a Slurm account / QoS

set -euo pipefail
cd /work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR

# # module load cuda/12.1
# # source /path/to/conda.sh && conda activate your_env

python main_ssl.py \
  --dataset wisdm --cases random --device Phones \
  --framework augpred --backbone FCN \
  --wisdm_feat fdiff \
  --augpred_transforms t_flip,perm,t_warp \
  --pretrain_epochs 70 --lincls_epochs 70 \
  --early_stop_patience 0 \
  --batch_size 64 --lr 1e-3 --lr_cls 1e-3
