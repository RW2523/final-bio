#!/usr/bin/env bash
# Research runner: cloned SSL entrypoint with switchable preprocessing and spike encodings.
# shellcheck disable=SC2086,SC2206
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WISDM_SHARED_DIR="/home/sriramkannan_umass_edu/690R-BioMarkers/wisdm-dataset"

: "${FRAMEWORK:=simclr}"
: "${BACKBONE:=SResNet1D}"
: "${WISDM_DEVICE:=Watch}"
: "${WISDM_FEAT:=raw}"
: "${INPUT_ENCODING:=step_forward}"          # none | poisson_rate | step_forward | moving_window | hybrid_ds_rate | legacy options
: "${BATCH_LOG_EVERY:=10}"
: "${EXTRA_FLAGS:=}"

# Research preprocessing knobs.
: "${BIO_SOURCE_HZ:=20}"
: "${BIO_RESAMPLE_HZ:=30}"
: "${BIO_WINDOW_SECONDS:=10}"
: "${BIO_STRIDE_SECONDS:=5}"
: "${BIO_FILTER_MODE:=motion_gravity}"       # raw | motion | gravity | motion_gravity | motion_gravity_accel
: "${BIO_GRAVITY_CUTOFF_HZ:=0.25}"
: "${BIO_MOTION_LOW_HZ:=0.25}"
: "${BIO_MOTION_HIGH_HZ:=15}"

# Research encoding knobs.
: "${SF_THRESHOLD:=0.15}"
: "${MW_WINDOW:=8}"
: "${MW_THRESHOLD:=0.15}"
: "${ENCODING_NORM_EPS:=1e-6}"

case "$BACKBONE" in
  SFCN|SDCL|SLSTM|SNN_AE|SNN_CNN_AE|SNN_Transformer|SResNet1D|iSpikformer|FCN|DCL|LSTM|AE|CNN_AE|Transformer|ViT1D) ;;
  *)
    echo "Unsupported BACKBONE=$BACKBONE" >&2
    exit 2
    ;;
esac

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/scratch/login/matplotlib-${USER:-codex}}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p run_logs

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

PRE="${PRETRAIN_EPOCHS:-$PRE}"
LIN="${LINCLS_EPOCHS:-$LIN}"
BS="${BATCH_SIZE:-$BS}"
LR="${LR:-$LR}"
LRCLS="${LRCLS:-$LRCLS}"

ARGS=(
  python main_ssl_research.py
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
  --input_encoding "$INPUT_ENCODING"
  --bio_source_hz "$BIO_SOURCE_HZ"
  --bio_resample_hz "$BIO_RESAMPLE_HZ"
  --bio_window_seconds "$BIO_WINDOW_SECONDS"
  --bio_stride_seconds "$BIO_STRIDE_SECONDS"
  --bio_filter_mode "$BIO_FILTER_MODE"
  --bio_gravity_cutoff_hz "$BIO_GRAVITY_CUTOFF_HZ"
  --bio_motion_low_hz "$BIO_MOTION_LOW_HZ"
  --bio_motion_high_hz "$BIO_MOTION_HIGH_HZ"
  --sf_threshold "$SF_THRESHOLD"
  --mw_window "$MW_WINDOW"
  --mw_threshold "$MW_THRESHOLD"
  --encoding_norm_eps "$ENCODING_NORM_EPS"
)
ARGS+=("${EXTRA_FW[@]}")
extra=($EXTRA_FLAGS)
if [ ${#extra[@]} -gt 0 ]; then
  ARGS+=("${extra[@]}")
fi

echo "Running: ${ARGS[*]}"
exec "${ARGS[@]}"
