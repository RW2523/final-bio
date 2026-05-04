#!/bin/bash
# Transformer + encoding sweep on the copied research pipeline.
# Default sweep: none, poisson_rate, step_forward, moving_window.
# Override with:
#   sbatch --export=ALL,ENCODINGS="poisson_rate step_forward moving_window" sbatch/wisdm_transformer_all_encodings_research.sbatch

#SBATCH --job-name=wisdm_tr_allenc
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_tr_allenc.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_tr_allenc.err
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:2080ti:1

set -euo pipefail
cd /work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR

: "${FRAMEWORK:=simclr}"
: "${BACKBONE:=Transformer}"
: "${WISDM_DEVICE:=Watch}"
: "${WISDM_FEAT:=raw}"
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
: "${ENCODING_NORM_EPS:=1e-6}"
: "${ENCODINGS:=none poisson_rate step_forward moving_window}"

if [[ "${BACKBONE}" != "Transformer" ]]; then
  echo "This sbatch is intended for BACKBONE=Transformer. Got BACKBONE=${BACKBONE}." >&2
  exit 2
fi

echo "==== TRANSFORMER ENCODING SWEEP ===="
echo "FRAMEWORK=${FRAMEWORK} BACKBONE=${BACKBONE} DEVICE=${WISDM_DEVICE}"
echo "WISDM_FEAT=${WISDM_FEAT} FILTER=${BIO_FILTER_MODE} RESAMPLE=${BIO_RESAMPLE_HZ}Hz"
echo "ENCODINGS=${ENCODINGS}"
echo "===================================="

for INPUT_ENCODING in ${ENCODINGS}; do
  echo
  echo ">>>> Running INPUT_ENCODING=${INPUT_ENCODING}"
  export FRAMEWORK BACKBONE WISDM_DEVICE WISDM_FEAT
  export INPUT_ENCODING
  export BIO_SOURCE_HZ BIO_RESAMPLE_HZ BIO_WINDOW_SECONDS BIO_STRIDE_SECONDS
  export BIO_FILTER_MODE BIO_GRAVITY_CUTOFF_HZ BIO_MOTION_LOW_HZ BIO_MOTION_HIGH_HZ
  export SF_THRESHOLD MW_WINDOW MW_THRESHOLD ENCODING_NORM_EPS
  bash sbatch/snnhar_ssl_run_research.sh
done
