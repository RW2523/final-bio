#!/usr/bin/env bash
# Submit all framework x spiking-backbone WISDM SSL jobs as separate Slurm dispatches.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Ensure matrix files exist (or are refreshed).
bash "$HERE/build_wisdm_spiking_ssl_sbatch.sh" >/dev/null

frameworks=(simclr byol simsiam nnclr tstcc augpred)
backbones=(SFCN SDCL SLSTM SNN_AE SNN_CNN_AE SNN_Transformer SResNet1D iSpikformer)

count=0
for fw in "${frameworks[@]}"; do
  for bb in "${backbones[@]}"; do
    # TS-TCC currently maps to SFCN only in snnhar_ssl_run.sh.
    if [ "$fw" = "tstcc" ] && [ "$bb" != "SFCN" ]; then
      continue
    fi
    job_file="$HERE/wisdm_${fw}_${bb}.sbatch"
    if [ -f "$job_file" ]; then
      sbatch "$job_file"
      count=$((count + 1))
    fi
  done
done

echo "Submitted ${count} WISDM SSL jobs (spiking backbones)."
