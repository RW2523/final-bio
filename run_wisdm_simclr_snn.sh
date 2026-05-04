#!/usr/bin/env bash
# Self-supervised spiking ResNet-1D on WISDM (phone IMU 6ch): SimCLR (NTXent) pretrain + linear probe.
# Put WISDM AR v1.1 at ./dataset/WISDM_ar_v1.1 or set WISDM_DATA_DIR (or --wisdm_data_dir /path).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

# Optional: export WISDM_DATA_DIR=/path/to/wisdm-dataset
python main_ssl.py \
  --dataset wisdm --device Phones --cases random \
  --framework simclr --backbone SResNet1D --criterion NTXent \
  --simclr_contrastive instance \
  --wisdm_feat fdiff \
  --tau 0.75 --thresh 0.5 \
  --aug1 jit_scal --aug2 perm_jit --p 128 --phid 128 \
  --pretrain_epochs 50 --lincls_epochs 50 \
  --batch_size 64 --lr 1e-3 --lr_cls 1e-3
