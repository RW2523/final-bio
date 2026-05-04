#!/bin/bash
#SBATCH --job-name=wisdm_ssl_ablate
#SBATCH --output=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_ssl_ablate.out
#SBATCH --error=/work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR/run_logs/slurm_%j_ssl_ablate.err
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
# 2080 Ti (~11GB): shorter queues on Unity gpu-preempt; use gpu:a100:1 if OOM
#SBATCH --gres=gpu:2080ti:1

set -euo pipefail
cd /work/pi_dagarwal_umass_edu/project_1/swetha/SNN_HAR
source .venv/bin/activate

# Controlled ablation knobs (override using: sbatch --export=ALL,VAR=...):
FRAMEWORK="${FRAMEWORK:-simclr}"            # simclr | augpred
BACKBONE="${BACKBONE:-SFCN}"                # FCN | SFCN | SResNet1D | Transformer | ViT1D
CONTRASTIVE="${CONTRASTIVE:-instance}"      # simclr only: instance | supcon
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-70}"
LINCLS_EPOCHS="${LINCLS_EPOCHS:-70}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
LR_CLS="${LR_CLS:-1e-3}"
FINETUNE_BACKBONE="${FINETUNE_BACKBONE:-0}" # 0|1 (linear probe vs finetune)
FINETUNE_SCOPE="${FINETUNE_SCOPE:-last_block}"  # last_block | all
BACKBONE_LR="${BACKBONE_LR:-1e-4}"

# ViT1D-only knobs (ignored for other backbones):
VIT_PATCH_SIZE="${VIT_PATCH_SIZE:-20}"
VIT_DIM="${VIT_DIM:-128}"
VIT_DEPTH="${VIT_DEPTH:-4}"
VIT_HEADS="${VIT_HEADS:-4}"
VIT_MLP_DIM="${VIT_MLP_DIM:-256}"
VIT_DROPOUT="${VIT_DROPOUT:-0.1}"

echo "==== SSL ABLATION CONFIG ===="
echo "FRAMEWORK=${FRAMEWORK} BACKBONE=${BACKBONE} CONTRASTIVE=${CONTRASTIVE}"
echo "EPOCHS(pretrain/lincls)=${PRETRAIN_EPOCHS}/${LINCLS_EPOCHS} BATCH=${BATCH_SIZE}"
echo "LR=${LR} LR_CLS=${LR_CLS} FINETUNE_BACKBONE=${FINETUNE_BACKBONE} FINETUNE_SCOPE=${FINETUNE_SCOPE}"
echo "============================="

COMMON_ARGS=(
  --dataset wisdm --cases random --device Phones
  --framework "${FRAMEWORK}" --backbone "${BACKBONE}"
  --wisdm_feat fdiff
  --pretrain_epochs "${PRETRAIN_EPOCHS}" --lincls_epochs "${LINCLS_EPOCHS}"
  --early_stop_patience 0
  --batch_size "${BATCH_SIZE}" --lr "${LR}" --lr_cls "${LR_CLS}"
  --lincls_backbone_lr "${BACKBONE_LR}" --lincls_finetune_scope "${FINETUNE_SCOPE}"
)

if [[ "${FINETUNE_BACKBONE}" == "1" ]]; then
  COMMON_ARGS+=(--lincls_finetune_backbone)
fi

if [[ "${FRAMEWORK}" == "simclr" ]]; then
  COMMON_ARGS+=(--criterion NTXent --aug1 jit_scal --aug2 perm_jit --p 128 --phid 128 --simclr_contrastive "${CONTRASTIVE}")
  if [[ "${CONTRASTIVE}" == "supcon" ]]; then
    COMMON_ARGS+=(--supcon_temperature 0.07)
  fi
elif [[ "${FRAMEWORK}" == "augpred" ]]; then
  COMMON_ARGS+=(--augpred_transforms t_flip,perm,t_warp)
else
  echo "Unsupported FRAMEWORK=${FRAMEWORK}. Use simclr or augpred."
  exit 2
fi

if [[ "${BACKBONE}" == "SFCN" || "${BACKBONE}" == "SResNet1D" ]]; then
  COMMON_ARGS+=(--tau 0.75 --thresh 0.5)
elif [[ "${BACKBONE}" == "ViT1D" ]]; then
  COMMON_ARGS+=(--vit_patch_size "${VIT_PATCH_SIZE}" --vit_dim "${VIT_DIM}" --vit_depth "${VIT_DEPTH}" --vit_heads "${VIT_HEADS}" --vit_mlp_dim "${VIT_MLP_DIM}" --vit_dropout "${VIT_DROPOUT}")
fi

python main_ssl.py "${COMMON_ARGS[@]}"
