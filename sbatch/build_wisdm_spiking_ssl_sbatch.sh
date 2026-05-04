#!/usr/bin/env bash
# Build one sbatch per framework x backbone for WISDM SSL.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RUN_SH="$HERE/snnhar_ssl_run.sh"

frameworks=(simclr byol simsiam nnclr tstcc augpred)
backbones=(SFCN SDCL SLSTM SNN_AE SNN_CNN_AE SNN_Transformer SResNet1D iSpikformer)

short_fw() {
  case "$1" in
    simclr) echo sc ;;
    byol) echo by ;;
    simsiam) echo ss ;;
    nnclr) echo nc ;;
    tstcc) echo tt ;;
    augpred) echo ap ;;
  esac
}
short_bb() {
  case "$1" in
    SFCN) echo sf ;;
    SDCL) echo sd ;;
    SLSTM) echo sl ;;
    SNN_AE) echo sa ;;
    SNN_CNN_AE) echo sc ;;
    SNN_Transformer) echo st ;;
    SResNet1D) echo sr ;;
    iSpikformer) echo is ;;
  esac
}

count=0
for fw in "${frameworks[@]}"; do
  for bb in "${backbones[@]}"; do
    # TS-TCC is constrained to SFCN in run script; skip redundant files.
    if [ "$fw" = "tstcc" ] && [ "$bb" != "SFCN" ]; then
      continue
    fi
    fs="$(short_fw "$fw")"
    bs="$(short_bb "$bb")"
    out="$HERE/wisdm_${fw}_${bb}.sbatch"
    cat > "$out" <<EOF
#!/bin/bash
# Auto-generated: WISDM + ${fw} + ${bb}
#SBATCH --job-name=w_${fs}_${bs}
#SBATCH --output=${ROOT}/run_logs/slurm_%j_w_${fs}_${bs}.out
#SBATCH --error=${ROOT}/run_logs/slurm_%j_w_${fs}_${bs}.err
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
cd ${ROOT}
mkdir -p run_logs
export FRAMEWORK=${fw}
export BACKBONE=${bb}
export WISDM_DEVICE=\${WISDM_DEVICE:-Phones}
export WISDM_FEAT=\${WISDM_FEAT:-fdiff}
exec bash ${RUN_SH}
EOF
    chmod +x "$out"
    count=$((count+1))
  done
done

echo "Generated ${count} WISDM spiking SSL sbatch files in ${HERE}."
