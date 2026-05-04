#!/bin/bash
# Auto-generated: WISDM + nnclr + SNN_Transformer
#SBATCH --job-name=w_nc_st
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_w_nc_st.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_w_nc_st.err
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
export FRAMEWORK=nnclr
export BACKBONE=SNN_Transformer
export WISDM_DEVICE=${WISDM_DEVICE:-Phones}
export WISDM_FEAT=${WISDM_FEAT:-fdiff}
exec bash /work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/sbatch/snnhar_ssl_run.sh
