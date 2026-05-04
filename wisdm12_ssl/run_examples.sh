#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash SNN_HAR/wisdm12_ssl/run_examples.sh "/path/to/wisdm-dataset"
#
# Defaults:
#   sample rate = 20 Hz
#   window = 10 s (200 samples)
#   stride = 5 s (100 samples)
#   channels = 12 (phone+watch accel+gyro xyz)

WISDM_DIR="${1:-${WISDM_DATA_DIR:-}}"
if [[ -z "${WISDM_DIR}" ]]; then
  echo "Provide WISDM root as arg or set WISDM_DATA_DIR."
  exit 1
fi

PY="python SNN_HAR/wisdm12_ssl/run_ssl_wisdm12.py --wisdm_data_dir \"${WISDM_DIR}\" --batch_size 64 --pretrain_epochs 70 --lincls_epochs 70"

# SimCLR
eval "${PY} --framework simclr --backbone SFCN"
eval "${PY} --framework simclr --backbone Transformer"
eval "${PY} --framework simclr --backbone SResNet1D"

# BYOL
eval "${PY} --framework byol --backbone SFCN --criterion cos_sim"
eval "${PY} --framework byol --backbone Transformer --criterion cos_sim"
eval "${PY} --framework byol --backbone SResNet1D --criterion cos_sim"
