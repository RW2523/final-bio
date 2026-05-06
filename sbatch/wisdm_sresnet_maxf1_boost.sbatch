#!/bin/bash
# Aggressive macro-F1 optimization recipe for WISDM (SResNet1D).

#SBATCH --job-name=wisdm_sr_maxf1
#SBATCH --output=/home/sriramkannan_umass_edu/690R-BioMarkers/final-bio/slurm_%j_sr_maxf1.out
#SBATCH --error=/home/sriramkannan_umass_edu/690R-BioMarkers/final-bio/slurm_%j_sr_maxf1.err
#SBATCH --partition=gpu-preempt
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:2080ti:1

set -euo pipefail
cd /home/sriramkannan_umass_edu/690R-BioMarkers/final-bio

export FRAMEWORK="${FRAMEWORK:-simclr}"
export BACKBONE="SResNet1D"
export WISDM_DEVICE="${WISDM_DEVICE:-Watch}"
export WISDM_FEAT="${WISDM_FEAT:-raw}"
export INPUT_ENCODING="${INPUT_ENCODING:-hybrid_ds_rate}"

export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-120}"
export LINCLS_EPOCHS="${LINCLS_EPOCHS:-120}"
export BIO_SOURCE_HZ="${BIO_SOURCE_HZ:-20}"
export BIO_RESAMPLE_HZ="${BIO_RESAMPLE_HZ:-30}"
export BIO_WINDOW_SECONDS="${BIO_WINDOW_SECONDS:-10}"
export BIO_STRIDE_SECONDS="${BIO_STRIDE_SECONDS:-5}"
export BIO_FILTER_MODE="${BIO_FILTER_MODE:-motion_gravity}"
export BIO_GRAVITY_CUTOFF_HZ="${BIO_GRAVITY_CUTOFF_HZ:-0.25}"
export BIO_MOTION_LOW_HZ="${BIO_MOTION_LOW_HZ:-0.25}"
export BIO_MOTION_HIGH_HZ="${BIO_MOTION_HIGH_HZ:-15}"
export ENCODING_NORM_EPS="${ENCODING_NORM_EPS:-1e-6}"
export SF_THRESHOLD="${SF_THRESHOLD:-0.15}"
export MW_WINDOW="${MW_WINDOW:-8}"
export MW_THRESHOLD="${MW_THRESHOLD:-0.15}"

export EXTRA_FLAGS="${EXTRA_FLAGS:---lincls_loss ce --lincls_label_smoothing 0.05 --lincls_scheduler onecycle --lincls_grad_clip 1.0 --lincls_select_metric macrof1 --lincls_logit_adjust_tau 0.25 --lincls_calibrate_temperature --lincls_temp_min 0.7 --lincls_temp_max 1.8 --lincls_temp_steps 12 --lincls_finetune_backbone --lincls_finetune_scope last_block --lincls_backbone_lr 2e-4 --early_stop_patience 12 --early_stop_min_delta 0.02 --report_sparsity}"

echo "Running max-F1 SResNet1D recipe with INPUT_ENCODING=${INPUT_ENCODING}"
echo "EXTRA_FLAGS=${EXTRA_FLAGS}"
bash sbatch/snnhar_ssl_run_research.sh
