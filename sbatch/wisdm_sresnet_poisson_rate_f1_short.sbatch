#!/bin/bash
# Upgraded SResNet1D + Poisson rate encoding with F1-oriented settings.

#SBATCH --job-name=wisdm_sr_pois_f1
#SBATCH --output=/home/sriramkannan_umass_edu/690R-BioMarkers/final-bio/slurm_%j_sr_pois_f1.out
#SBATCH --error=/home/sriramkannan_umass_edu/690R-BioMarkers/final-bio/slurm_%j_sr_pois_f1.err
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --time=04:00:00
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
export INPUT_ENCODING="poisson_rate"

export BIO_SOURCE_HZ="${BIO_SOURCE_HZ:-20}"
export BIO_RESAMPLE_HZ="${BIO_RESAMPLE_HZ:-30}"
export BIO_WINDOW_SECONDS="${BIO_WINDOW_SECONDS:-10}"
export BIO_STRIDE_SECONDS="${BIO_STRIDE_SECONDS:-5}"
export BIO_FILTER_MODE="${BIO_FILTER_MODE:-motion_gravity}"
export BIO_GRAVITY_CUTOFF_HZ="${BIO_GRAVITY_CUTOFF_HZ:-0.25}"
export BIO_MOTION_LOW_HZ="${BIO_MOTION_LOW_HZ:-0.25}"
export BIO_MOTION_HIGH_HZ="${BIO_MOTION_HIGH_HZ:-15}"
export ENCODING_NORM_EPS="${ENCODING_NORM_EPS:-1e-6}"

export EXTRA_FLAGS="${EXTRA_FLAGS:---lincls_loss cb_focal --focal_gamma 2.0 --cb_beta 0.999 --report_sparsity}"

echo "Running upgraded SResNet1D with INPUT_ENCODING=${INPUT_ENCODING}, BIO_FILTER_MODE=${BIO_FILTER_MODE}"
echo "EXTRA_FLAGS=${EXTRA_FLAGS}"
bash sbatch/snnhar_ssl_run_research.sh
