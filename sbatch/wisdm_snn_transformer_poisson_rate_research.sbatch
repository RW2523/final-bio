#!/bin/bash
# WISDM + SimCLR (default) + research Poisson rate input encoding + SNN_Transformer (vendored iSpikformer).
# Requires spikingjelly (pip install -r requirements.txt).

#SBATCH --job-name=wisdm_st_pois
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_st_pois.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_st_pois.err
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
# Per-channel first difference on the stream before windowing (same default as wisdm_simclr_SNN_Transformer.sbatch / sresnet *_research jobs). Use WISDM_FEAT=raw to disable.
export WISDM_FEAT="${WISDM_FEAT:-fdiff}"
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

# Optional: tune iSpikformer / SNN_Transformer width and depth (parsed by main_ssl_research -> main_ssl parser).
# export EXTRA_FLAGS="--ispf_dim 512 --ispf_depths 2 --ispf_heads 8 --ispf_num_steps 4 --ispf_encoder conv"

echo "Running SNN_Transformer with INPUT_ENCODING=${INPUT_ENCODING}"
bash sbatch/snnhar_ssl_run_research.sh
