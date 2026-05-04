#!/bin/bash
# 12-channel WISDM SSL on gpu-preempt / qos=short (4 hours).
# Submit examples:
#   sbatch --export=ALL,FRAMEWORK=simclr,BACKBONE=SResNet1D,WISDM_DIR=/path/to/wisdm-dataset SNN_HAR/wisdm12_ssl/wisdm12_ssl_short_preempt.sbatch
#   sbatch --export=ALL,FRAMEWORK=byol,BACKBONE=SFCN,WISDM_DIR=/path/to/wisdm-dataset SNN_HAR/wisdm12_ssl/wisdm12_ssl_short_preempt.sbatch
#
# Allowed BACKBONE values (new runner):
#   SFCN | SNN_Transformer | Transformer | SResNet1D
# Allowed FRAMEWORK values:
#   simclr | byol
#
#SBATCH --job-name=w12_ssl
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_w12_ssl.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_w12_ssl.err
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
ROOT=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR
cd "$ROOT"
mkdir -p run_logs
export FRAMEWORK="${FRAMEWORK:-simclr}"
export BACKBONE="${BACKBONE:-SResNet1D}"
export WISDM_DIR="${WISDM_DIR:-/home/sriramkannan_umass_edu/690R-BioMarkers/wisdm-dataset}"
export WISDM_DATA_DIR="$WISDM_DIR"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-70}"
export LINCLS_EPOCHS="${LINCLS_EPOCHS:-70}"
export CUDA_ID="${CUDA_ID:-0}"
exec bash "$ROOT/sbatch/snnhar_ssl_run_wisdm12.sh"
