#!/usr/bin/env bash
# WISDM fused-12 SSL + linear eval runner (SNN_HAR/wisdm12_ssl/run_ssl_wisdm12.py).
# shellcheck disable=SC2086,SC2206
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WISDM_SHARED_DIR="/home/sriramkannan_umass_edu/690R-BioMarkers/wisdm-dataset"

: "${FRAMEWORK:=simclr}"
: "${BACKBONE:=SResNet1D}"
: "${BATCH_LOG_EVERY:=10}"
: "${EXTRA_FLAGS:=}"
: "${CUDA_ID:=0}"
: "${RUN_WITH_SRUN:=1}"

case "$BACKBONE" in
  SFCN|SNN_Transformer|Transformer|SResNet1D) ;;
  *)
    echo "Unsupported BACKBONE=$BACKBONE" >&2
    exit 2
    ;;
esac
case "$FRAMEWORK" in
  simclr|byol) ;;
  *)
    echo "Unsupported FRAMEWORK=$FRAMEWORK (allowed: simclr|byol)." >&2
    exit 2
    ;;
esac

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

WISDM_DIR="${WISDM_DIR:-${WISDM_DATA_DIR:-$WISDM_SHARED_DIR}}"
if [ ! -d "$WISDM_DIR/raw/watch/accel" ] || [ ! -d "$WISDM_DIR/raw/phone/accel" ]; then
  echo "Invalid WISDM_DIR: $WISDM_DIR (need raw/watch/accel and raw/phone/accel)" >&2
  exit 2
fi

PRE="${PRETRAIN_EPOCHS:-70}"
LIN="${LINCLS_EPOCHS:-70}"
BS="${BATCH_SIZE:-64}"

EXTRA_FW=()
if [ "$FRAMEWORK" = "byol" ]; then
  EXTRA_FW+=(--criterion cos_sim)
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "CUDA_VISIBLE_DEVICES preset by scheduler: ${CUDA_VISIBLE_DEVICES}"
else
  # Match common old behavior: expose the selected device id.
  export CUDA_VISIBLE_DEVICES="$CUDA_ID"
  echo "CUDA_VISIBLE_DEVICES not set; forcing to ${CUDA_VISIBLE_DEVICES}"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

# Hard fail if CUDA isn't available in this allocation.
python - <<'PY'
import sys
import torch
ok = torch.cuda.is_available()
print(f"[gpu-check] torch.cuda.is_available()={ok} count={torch.cuda.device_count()}")
if not ok:
    sys.exit("ERROR: No CUDA device available in this job allocation.")
PY

ARGS=(
  python "$ROOT/wisdm12_ssl/run_ssl_wisdm12.py"
  --cuda "$CUDA_ID"
  --wisdm_data_dir "$WISDM_DIR"
  --framework "$FRAMEWORK"
  --backbone "$BACKBONE"
  --batch_size "$BS"
  --pretrain_epochs "$PRE"
  --lincls_epochs "$LIN"
  --batch_log_every "$BATCH_LOG_EVERY"
)
ARGS+=("${EXTRA_FW[@]}")
extra=($EXTRA_FLAGS)
if [ ${#extra[@]} -gt 0 ]; then
  ARGS+=("${extra[@]}")
fi

echo "Running: ${ARGS[*]}"
if [ "$RUN_WITH_SRUN" = "1" ] && command -v srun >/dev/null 2>&1; then
  exec srun --ntasks=1 "${ARGS[@]}"
fi
exec "${ARGS[@]}"
