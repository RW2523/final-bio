#!/usr/bin/env bash
# WISDM SSL + linear eval runner (SNN_HAR/main_ssl.py).
# Uses shared WISDM path by default.
# shellcheck disable=SC2086,SC2206
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Shared WISDM path requested by user.
WISDM_SHARED_DIR="/home/sriramkannan_umass_edu/690R-BioMarkers/wisdm-dataset"

: "${FRAMEWORK:=simclr}"
: "${BACKBONE:=SResNet1D}"
: "${WISDM_DEVICE:=Phones}"
: "${WISDM_FEAT:=fdiff}"
: "${BATCH_LOG_EVERY:=10}"
: "${EXTRA_FLAGS:=}"

# Allow all backbones implemented in SNN_HAR/main_ssl.py.
case "$BACKBONE" in
  SFCN|SDCL|SLSTM|SNN_AE|SNN_CNN_AE|SNN_Transformer|SResNet1D|iSpikformer|FCN|DCL|LSTM|AE|CNN_AE|Transformer) ;;
  *)
    echo "Unsupported BACKBONE=$BACKBONE" >&2
    exit 2
    ;;
esac

# TS-TCC in this codebase is intended for FCN/SFCN-style encoders.
if [ "$FRAMEWORK" = "tstcc" ] && [ "$BACKBONE" != "SFCN" ]; then
  echo "TS-TCC supports SFCN in this spiking dispatcher; forcing BACKBONE=SFCN." >&2
  BACKBONE=SFCN
fi

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

# Defaults tuned for stable WISDM SSL runs.
case "$FRAMEWORK" in
  simclr)
    PRE=100; LIN=80; BS=128; LR=1e-3; LRCLS=1.5e-3; AUG1=jit_scal; AUG2=perm_jit; EXTRA_FW=(--criterion NTXent --simclr_contrastive supcon --supcon_temperature 0.07)
    ;;
  byol)
    PRE=80; LIN=80; BS=128; LR=1e-3; LRCLS=1e-3; AUG1=jit_scal; AUG2=perm_jit; EXTRA_FW=(--criterion cos_sim)
    ;;
  simsiam)
    PRE=80; LIN=80; BS=128; LR=1e-3; LRCLS=1e-3; AUG1=jit_scal; AUG2=perm_jit; EXTRA_FW=(--criterion cos_sim)
    ;;
  nnclr)
    PRE=100; LIN=80; BS=128; LR=1e-3; LRCLS=1e-3; AUG1=jit_scal; AUG2=perm_jit; EXTRA_FW=(--criterion NTXent --mmb_size 1024)
    ;;
  tstcc)
    PRE=60; LIN=80; BS=128; LR=3e-4; LRCLS=3e-4; AUG1=t_warp; AUG2=negate; EXTRA_FW=(--criterion NTXent)
    ;;
  augpred)
    PRE=100; LIN=80; BS=128; LR=1e-3; LRCLS=1e-3; AUG1=jit_scal; AUG2=perm_jit; EXTRA_FW=(--use_augpred --augpred_lambda 0.5 --criterion NTXent)
    ;;
  *)
    echo "Unknown FRAMEWORK=$FRAMEWORK" >&2
    exit 2
    ;;
esac

# Allow sbatch --export overrides after framework defaults are set.
PRE="${PRETRAIN_EPOCHS:-$PRE}"
LIN="${LINCLS_EPOCHS:-$LIN}"
BS="${BATCH_SIZE:-$BS}"
LR="${LR:-$LR}"
LRCLS="${LRCLS:-$LRCLS}"

ARGS=(
  python main_ssl.py
  --dataset wisdm
  --cases random
  --device "$WISDM_DEVICE"
  --wisdm_data_dir "$WISDM_SHARED_DIR"
  --wisdm_feat "$WISDM_FEAT"
  --framework "$FRAMEWORK"
  --backbone "$BACKBONE"
  --tau 0.75 --thresh 0.5
  --batch_log_every "$BATCH_LOG_EVERY"
  --aug1 "$AUG1" --aug2 "$AUG2"
  --p 128 --phid 128
  --pretrain_epochs "$PRE" --lincls_epochs "$LIN"
  --batch_size "$BS" --lr "$LR" --lr_cls "$LRCLS"
  --early_stop_patience 0
)
ARGS+=("${EXTRA_FW[@]}")
extra=($EXTRA_FLAGS)
if [ ${#extra[@]} -gt 0 ]; then
  ARGS+=("${extra[@]}")
fi

echo "Running: ${ARGS[*]}"
exec "${ARGS[@]}"
