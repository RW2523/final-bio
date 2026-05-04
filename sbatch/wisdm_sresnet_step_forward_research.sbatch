#!/bin/bash
# Best logged SResNet1D-style simclr setup + step-forward encoding on the copied research pipeline.

#SBATCH --job-name=wisdm_sr_step
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_sr_step.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_sr_step.err
#SBATCH --partition=gpu-preempt
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:2080ti:1

set -euo pipefail
cd /work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR

export FRAMEWORK="${FRAMEWORK:-simclr}"
export BACKBONE="SResNet1D"
export WISDM_DEVICE="${WISDM_DEVICE:-Watch}"
export WISDM_FEAT="${WISDM_FEAT:-fdiff}"
export INPUT_ENCODING="step_forward"

export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-60}"
export LINCLS_EPOCHS="${LINCLS_EPOCHS:-80}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LR="${LR:-1e-3}"
export LRCLS="${LRCLS:-1e-3}"

export BIO_SOURCE_HZ="${BIO_SOURCE_HZ:-20}"
export BIO_RESAMPLE_HZ="${BIO_RESAMPLE_HZ:-20}"
export BIO_WINDOW_SECONDS="${BIO_WINDOW_SECONDS:-10}"
export BIO_STRIDE_SECONDS="${BIO_STRIDE_SECONDS:-5}"
export BIO_FILTER_MODE="${BIO_FILTER_MODE:-raw}"
export BIO_GRAVITY_CUTOFF_HZ="${BIO_GRAVITY_CUTOFF_HZ:-0.25}"
export BIO_MOTION_LOW_HZ="${BIO_MOTION_LOW_HZ:-0.25}"
export BIO_MOTION_HIGH_HZ="${BIO_MOTION_HIGH_HZ:-15}"
export SF_THRESHOLD="${SF_THRESHOLD:-0.15}"
export ENCODING_NORM_EPS="${ENCODING_NORM_EPS:-1e-6}"

export EXTRA_FLAGS="${EXTRA_FLAGS:-} --lincls_finetune_backbone --lincls_finetune_scope last_block --lincls_backbone_lr 5e-5 --early_stop_patience 8 --early_stop_min_delta 0.002"

echo "Running SResNet1D with INPUT_ENCODING=${INPUT_ENCODING}"
bash sbatch/snnhar_ssl_run_research.sh
