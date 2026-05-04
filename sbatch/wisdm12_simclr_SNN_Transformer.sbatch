#!/bin/bash
# Auto-generated: WISDM12 + simclr + SNN_Transformer
#SBATCH --job-name=w12_sc_st
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_w12_sc_st.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_w12_sc_st.err
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
cd /work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR
mkdir -p run_logs

export FRAMEWORK=simclr
export BACKBONE=SNN_Transformer
export WISDM_DIR="${WISDM_DIR:-/home/sriramkannan_umass_edu/690R-BioMarkers/wisdm-dataset}"
export WISDM_DATA_DIR="$WISDM_DIR"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-70}"
export LINCLS_EPOCHS="${LINCLS_EPOCHS:-70}"
export CUDA_ID="${CUDA_ID:-0}"

exec bash /work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/sbatch/snnhar_ssl_run_wisdm12.sh
