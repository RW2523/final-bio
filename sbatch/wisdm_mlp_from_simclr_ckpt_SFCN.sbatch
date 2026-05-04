#!/bin/bash
# MLP downstream probe starting from a SimCLR **pretrain** checkpoint for WISDM + SFCN.
#
# Required:
#   export PRETRAIN_CKPT=/abs/path/to/results/pretrain_....pt
#
# Optional overrides (examples):
#   export WISDM_DEVICE=Watch
#   export LINCLS_EPOCHS=80
#   export LR_CLS=1e-3
#   export FINETUNE_BB=1              # sets --lincls_finetune_backbone
#   export FINETUNE_SCOPE=last_block
#   export BB_LR=1e-4
#   export ES_PATIENCE=8
#   export ES_MIN_DELTA=0.002
#
#SBATCH --job-name=mlp_sf_sfcn
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_mlp_sf_sfcn.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_mlp_sf_sfcn.err
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
## SBATCH -A <ACCOUNT>

set -euo pipefail
cd /work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR
mkdir -p run_logs

if [[ -z "${PRETRAIN_CKPT:-}" ]]; then
  echo "ERROR: set PRETRAIN_CKPT to an absolute path like:" >&2
  echo "  export PRETRAIN_CKPT=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/results/pretrain_....pt" >&2
  exit 2
fi

if [[ -f "/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/.snn_env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/.snn_env/bin/activate"
fi

WISDM_DEVICE="${WISDM_DEVICE:-Watch}"
WISDM_FEAT="${WISDM_FEAT:-fdiff}"
LINCLS_EPOCHS="${LINCLS_EPOCHS:-80}"
LR_CLS="${LR_CLS:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
P_DIM="${P_DIM:-128}"
PHID="${PHID:-128}"
SUPCON_TEMP="${SUPCON_TEMP:-0.07}"

EXTRA_FINETUNE=()
if [[ "${FINETUNE_BB:-0}" == "1" ]]; then
  EXTRA_FINETUNE+=(--lincls_finetune_backbone --lincls_finetune_scope "${FINETUNE_SCOPE:-last_block}" --lincls_backbone_lr "${BB_LR:-1e-4}")
fi

exec python mlp_lincls_from_simclr_ckpt.py \
  --checkpoint "$PRETRAIN_CKPT" \
  --backbone SFCN \
  --framework simclr \
  --dataset wisdm --cases random \
  --device "$WISDM_DEVICE" \
  --wisdm_feat "$WISDM_FEAT" \
  --wisdm_data_dir "/home/sriramkannan_umass_edu/690R-BioMarkers/wisdm-dataset" \
  --batch_size "$BATCH_SIZE" \
  --p "$P_DIM" --phid "$PHID" \
  --aug1 jit_scal --aug2 perm_jit \
  --simclr_contrastive supcon --supcon_temperature "$SUPCON_TEMP" \
  --lincls_head mlp \
  --lincls_epochs "$LINCLS_EPOCHS" \
  --lr_cls "$LR_CLS" \
  --lincls_hidden_dim "${H1:-512}" \
  --lincls_hidden_dim2 "${H2:-256}" \
  --lincls_dropout "${DROP:-0.2}" \
  --batch_log_every "${BATCH_LOG_EVERY:-10}" \
  --early_stop_patience "${ES_PATIENCE:-0}" \
  --early_stop_min_delta "${ES_MIN_DELTA:-0.0}" \
  "${EXTRA_FINETUNE[@]}"
