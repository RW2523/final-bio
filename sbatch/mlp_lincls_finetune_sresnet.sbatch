#!/bin/bash
# MLP head + backbone finetune on WISDM Watch (SimCLR ckpt → beat frozen-probe ceiling).
# Submit:  sbatch sbatch/mlp_lincls_finetune_sresnet.sbatch
# Override: BACKBONE=SFCN sbatch --export=ALL,CHECKPOINT=/path/to.pt sbatch/mlp_lincls_finetune_sresnet.sbatch
#
#SBATCH --job-name=mlp_ft_sr
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_mlp_ft_sr.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_mlp_ft_sr.err
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
##SBATCH -A <YOUR_ACCOUNT>

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# GPU id inside the step (slurm often sets SLURM_STEP_GPUS or CUDA_VISIBLE_DEVICES)
: "${CUDA_ID:=0}"

# Pretrain ckpt (default is set in mlp_lincls_from_ckpt.sh if unset)
export CHECKPOINT="${CHECKPOINT:-}"
export BACKBONE="${BACKBONE:-SResNet1D}"
export CUDA_ID
export FT_SCOPE="${FT_SCOPE:-all}"
export LR_CLS="${LR_CLS:-5e-4}"
export BACKBONE_LR="${BACKBONE_LR:-5e-5}"
export WISDM_DATA_DIR="${WISDM_DATA_DIR:-/home/sriramkannan_umass_edu/690R-BioMarkers/wisdm-dataset}"
export WISDM_DIR="${WISDM_DIR:-$WISDM_DATA_DIR}"
export LINCLS_EPOCHS="${LINCLS_EPOCHS:-80}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"

# Optional: extra CLI, e.g.  EXTRA_FLAGS="--lincls_finetune_scope last_block"
export EXTRA_FLAGS="${EXTRA_FLAGS:-}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

exec bash "$HERE/mlp_lincls_from_ckpt.sh"
