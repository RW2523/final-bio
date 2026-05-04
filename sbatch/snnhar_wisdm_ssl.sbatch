#!/bin/bash
# Generic SNN_HAR WISDM SSL dispatcher (spiking backbones only).
# Example:
#   sbatch --export=ALL,FRAMEWORK=simclr,BACKBONE=SResNet1D sbatch/snnhar_wisdm_ssl.sbatch
# Shared dataset path is fixed in snnhar_ssl_run.sh.
#SBATCH --job-name=snnhar_wisdm_ssl
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_snnhar_ssl.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_snnhar_ssl.err
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
export WISDM_DEVICE="${WISDM_DEVICE:-Phones}"
export WISDM_FEAT="${WISDM_FEAT:-fdiff}"
exec bash "$ROOT/sbatch/snnhar_ssl_run.sh"
