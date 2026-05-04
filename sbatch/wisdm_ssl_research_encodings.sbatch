#!/bin/bash
# Research ablation launcher for the copied preprocessing + encoding pipeline.
# Override knobs with: sbatch --export=ALL,INPUT_ENCODING=poisson_rate,BIO_FILTER_MODE=raw,...
# Recommended starting points:
#   INPUT_ENCODING=poisson_rate
#   INPUT_ENCODING=step_forward
#   INPUT_ENCODING=moving_window
#   BIO_FILTER_MODE=motion_gravity
#   BIO_RESAMPLE_HZ=30
#   WISDM_FEAT=raw
#
# Temporal contrast references:
#   Petro, Kasabov, Kiss. IEEE TNNLS 2020. doi:10.1109/TNNLS.2019.2906158
# Rate coding survey:
#   Auge et al. Neural Processing Letters 2021. doi:10.1007/s11063-021-10562-2

#SBATCH --job-name=wisdm_ssl_research
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_ssl_research.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_ssl_research.err
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:2080ti:1

set -euo pipefail
cd /work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR

: "${FRAMEWORK:=simclr}"
: "${BACKBONE:=SResNet1D}"
: "${WISDM_DEVICE:=Watch}"
: "${WISDM_FEAT:=raw}"
: "${INPUT_ENCODING:=step_forward}"

: "${BIO_SOURCE_HZ:=20}"
: "${BIO_RESAMPLE_HZ:=30}"
: "${BIO_WINDOW_SECONDS:=10}"
: "${BIO_STRIDE_SECONDS:=5}"
: "${BIO_FILTER_MODE:=motion_gravity}"
: "${BIO_GRAVITY_CUTOFF_HZ:=0.25}"
: "${BIO_MOTION_LOW_HZ:=0.25}"
: "${BIO_MOTION_HIGH_HZ:=15}"

: "${SF_THRESHOLD:=0.15}"
: "${MW_WINDOW:=8}"
: "${MW_THRESHOLD:=0.15}"

echo "==== RESEARCH SSL CONFIG ===="
echo "FRAMEWORK=${FRAMEWORK} BACKBONE=${BACKBONE} DEVICE=${WISDM_DEVICE}"
echo "WISDM_FEAT=${WISDM_FEAT} INPUT_ENCODING=${INPUT_ENCODING}"
echo "BIO_RESAMPLE_HZ=${BIO_RESAMPLE_HZ} FILTER=${BIO_FILTER_MODE} WINDOW=${BIO_WINDOW_SECONDS}s STRIDE=${BIO_STRIDE_SECONDS}s"
echo "SF_THRESHOLD=${SF_THRESHOLD} MW_WINDOW=${MW_WINDOW} MW_THRESHOLD=${MW_THRESHOLD}"
echo "============================="

bash sbatch/snnhar_ssl_run_research.sh
