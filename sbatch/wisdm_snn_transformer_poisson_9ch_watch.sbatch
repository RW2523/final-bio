#!/bin/bash
# WISDM Watch + accel+gyro merged (6 raw axes) -> research stack -> **9 channels** at each timestep:
#   motion_acc (3) + gravity estimate (3) + motion_gyr (3)  with  BIO_FILTER_MODE=motion_gravity
# SimCLR (default) + fdiff stream + Poisson rate input encoding + SNN_Transformer (vendored iSpikformer).
# Requires spikingjelly (pip install -r requirements.txt).

#SBATCH --job-name=wisdm_st9_pois
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_st9_pois.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_st9_pois.err
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

export FRAMEWORK="${FRAMEWORK:-simclr}"
export BACKBONE="SNN_Transformer"
export WISDM_DEVICE="${WISDM_DEVICE:-Watch}"
export WISDM_FEAT="${WISDM_FEAT:-fdiff}"
export INPUT_ENCODING="poisson_rate"

export BIO_SOURCE_HZ="${BIO_SOURCE_HZ:-20}"
export BIO_RESAMPLE_HZ="${BIO_RESAMPLE_HZ:-30}"
export BIO_WINDOW_SECONDS="${BIO_WINDOW_SECONDS:-10}"
export BIO_STRIDE_SECONDS="${BIO_STRIDE_SECONDS:-5}"
# Locked to 9-channel layout (see data_preprocess_wisdm_research.estimate_n_feature).
export BIO_FILTER_MODE="${BIO_FILTER_MODE:-motion_gravity}"
export BIO_GRAVITY_CUTOFF_HZ="${BIO_GRAVITY_CUTOFF_HZ:-0.25}"
export BIO_MOTION_LOW_HZ="${BIO_MOTION_LOW_HZ:-0.25}"
export BIO_MOTION_HIGH_HZ="${BIO_MOTION_HIGH_HZ:-15}"
export ENCODING_NORM_EPS="${ENCODING_NORM_EPS:-1e-6}"

# Optional: export EXTRA_FLAGS="--ispf_dim 512 --ispf_depths 2 --ispf_heads 8 --ispf_num_steps 4 --ispf_encoder conv"

echo "Running Watch 9ch (motion_gravity) SNN_Transformer + poisson_rate; BIO_FILTER_MODE=${BIO_FILTER_MODE}"
bash sbatch/snnhar_ssl_run_research.sh
