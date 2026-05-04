#!/usr/bin/env bash
# MLP eval from SimCLR pretrain checkpoint (mlp_lincls_from_simclr_ckpt.py).
# shellcheck disable=SC2086,SC2206
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${WISDM_DATA_DIR:=/home/sriramkannan_umass_edu/690R-BioMarkers/wisdm-dataset}"
: "${CHECKPOINT:=$ROOT/results/pretrain_try_scheduler_simclr_pretrain_wisdm_eps60_lr0.001_bs64_aug1jit_scal_aug2perm_jit_dim-pdim128-128_EMA0.996_criterion_NTXent_lambda1_1.0_lambda2_1.0_tempunit_tsfm_sclr_supcon22.pt}"
: "${BACKBONE:=SResNet1D}"
: "${CUDA_ID:=0}"
: "${BATCH_SIZE:=64}"
: "${LINCLS_EPOCHS:=80}"
: "${LR_CLS:=5e-4}"
: "${BACKBONE_LR:=5e-5}"
# last_block = finetune SResNet layer4 / SFCN conv_block3; all = full backbone
: "${FT_SCOPE:=all}"
: "${EARLY_STOP_PATIENCE:=20}"
# shellcheck disable=SC2034
: "${EXTRA_FLAGS:=}"

# Prefer project-local envs first.
if [ -f "$ROOT/.snn_env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.snn_env/bin/activate"
elif [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
elif [ -f "$ROOT/.clhar_env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.clhar_env/bin/activate"
elif [ -f "/work/pi_dagarwal_umass_edu/project_1/swetha/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "/work/pi_dagarwal_umass_edu/project_1/swetha/.venv/bin/activate"
else
  echo "No env found (.snn_env/.venv/.clhar_env)." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
mkdir -p run_logs

WISDM_DIR="${WISDM_DIR:-$WISDM_DATA_DIR}"
if [ ! -d "$WISDM_DIR/raw/watch/accel" ] || [ ! -d "$WISDM_DIR/raw/phone/accel" ]; then
  echo "Invalid WISDM_DIR: $WISDM_DIR" >&2
  exit 2
fi
if [ ! -f "$CHECKPOINT" ]; then
  echo "Missing CHECKPOINT: $CHECKPOINT" >&2
  exit 2
fi

# Match your frozen MLP run, plus finetune flags (override with env if needed)
ARGS=(
  python "$ROOT/mlp_lincls_from_simclr_ckpt.py"
  --cuda "$CUDA_ID"
  --checkpoint "$CHECKPOINT"
  --framework simclr
  --backbone "$BACKBONE"
  --wisdm_data_dir "$WISDM_DIR"
  --device Watch
  --wisdm_feat fdiff
  --batch_size "$BATCH_SIZE"
  --p 128 --phid 128
  --aug1 jit_scal --aug2 perm_jit
  --criterion NTXent
  --simclr_contrastive supcon
  --supcon_temperature 0.07
  --lincls_epochs "$LINCLS_EPOCHS"
  --lr_cls "$LR_CLS"
  --lincls_head mlp
  --lincls_hidden_dim 512
  --lincls_hidden_dim2 256
  --lincls_dropout 0.2
  --lincls_finetune_backbone
  --lincls_finetune_scope "$FT_SCOPE"
  --lincls_backbone_lr "$BACKBONE_LR"
  --early_stop_patience "$EARLY_STOP_PATIENCE"
  --early_stop_min_delta 0.001
  --batch_log_every 10
  --scheduler
  --logdir log/
)
# shellcheck disable=SC2206
extra=($EXTRA_FLAGS)
if [ ${#extra[@]} -gt 0 ]; then
  ARGS+=("${extra[@]}")
fi

echo "Running: ${ARGS[*]}"
exec "${ARGS[@]}"
