## SSL Experiment Summary (WISDM)

This note explains the experiment setup, model architectures, training stages, and run-by-run outcomes for the current self-supervised learning (SSL) experiments on WISDM.

## 1) Problem setup

### Task
- Human Activity Recognition (HAR) on WISDM.
- Target is multiclass classification with `n_class=18`.

### Data and input format (as used in these runs)
- Dataset: `wisdm`
- Split mode: `cases=random`
- Feature mode: `wisdm_feat=fdiff` (first temporal difference)
- Sequence length: `len_sw=200`
- Input channels: `n_feature=6` (accel + gyro in the current pipeline)

### Training protocol
Two-stage SSL protocol:
1. **Pretraining** (self-supervised objective)
2. **Linear classifier stage** (`lincls_epochs`) for downstream labels

For these listed runs, backbone finetuning is disabled (`lincls_finetune_backbone=False`), so this is a strict probe-style setup.

## 2) Metric definitions

- **Test Acc (%)**: overall test accuracy.
- **miF (%)**: micro-F1 (for single-label multiclass, usually close to accuracy).
- **maF (%)**: macro-F1 (class-balanced F1 average; sensitive to weak classes).
- **Runtime**: end-to-end wall-clock (pretrain + linear classifier stage).

## 3) Architectures and objectives used

### Backbones
- **FCN**: 1D convolutional temporal encoder.
- **Transformer**: sequence transformer encoder over time tokens.

### SSL frameworks
- **SimCLR**: contrastive objective (`NTXent`) over two augmented views.
  - `simclr_contrastive=supcon` means same-class examples in a batch are treated as additional positives.
- **AugPred**: augmentation-prediction pretext task.

### Common optimizer settings in these runs
- Main SSL LR usually `1e-3`
- Linear head LR mostly `1e-3` (one run uses `1.5e-3`)
- Batch sizes: `64` or `128`

## 4) Results table

| Slurm log | Framework | Backbone | Key settings | Test Acc (%) | miF (%) | maF (%) | Runtime |
|---|---|---|---|---:|---:|---:|---|
| `slurm_56221148_simclr_fcn.out` | SimCLR | FCN | `supcon`, `fdiff`, `70/70`, `bs=64`, `lr=1e-3`, `lr_cls=1e-3` | **35.94** | **35.94** | **34.62** | `0:05:44` |
| `slurm_56221396_augpred_xfm.out` | AugPred | Transformer | `fdiff`, `70/70`, `bs=64`, `lr=1e-3`, `lr_cls=1e-3` | 33.01 | 33.01 | 31.52 | `0:14:00` |
| `slurm_56221498_simclr_xfm.out` | SimCLR | Transformer | `supcon`, `fdiff`, `100/80`, `bs=128`, `lr=1e-3`, `lr_cls=1.5e-3` | 32.14 | 32.14 | 28.91 | `0:13:12` |
| `slurm_56221091_augpred_fcn.out` | AugPred | FCN | `fdiff`, `70/70`, `bs=64`, `lr=1e-3`, `lr_cls=1e-3` | 23.94 | 23.94 | 19.77 | `0:10:25` |

## 5) Run-by-run interpretation

### A) `slurm_56221148_simclr_fcn.out` (best SSL in this set)
- Best overall on both accuracy and macro-F1.
- Indicates FCN + SimCLR/SupCon is currently the strongest representation-learning combination among tested settings.

### B) `slurm_56221396_augpred_xfm.out`
- Reasonably stable and competitive.
- Better than SimCLR-Transformer in this comparison, suggesting objective-backbone interaction matters.

### C) `slurm_56221498_simclr_xfm.out`
- Increased compute (100/80, larger batch, higher `lr_cls`) did not improve over AugPred-Transformer.
- Suggests optimization mismatch or representation mismatch for Transformer under this SimCLR recipe.

### D) `slurm_56221091_augpred_fcn.out` (worst)
- Low acc and macro-F1 indicate poor downstream transfer and/or class-collapse tendency in the classifier stage.

## 6) Cross-run takeaways

- **Framework effect (in FCN family):** SimCLR+SupCon > AugPred.
- **Backbone effect (for these SSL recipes):** FCN > Transformer.
- **Compute scaling effect:** More epochs/bigger batch did not automatically improve Transformer SimCLR.
- **Current best recipe:** **SimCLR + FCN + SupCon + fdiff**.

## 7) Recommended next experiments

To improve SSL further with controlled comparisons:
1. Keep the best baseline (`SimCLR + FCN + SupCon + fdiff`).
2. Test partial backbone finetuning in classifier stage (`last_block`).
3. Compare against `SResNet1D` (spiking) and `ViT1D` using the same train budget.
4. Add preprocessing ablations (`raw`, `fdiff`, `l2`) while keeping model settings fixed.
## SSL Experiment Summary (WISDM)

### Metric Notes
- **Test Acc (%)**: Overall test-set accuracy.
- **miF (%)**: Micro-F1 (for single-label multiclass, often close to accuracy).
- **maF (%)**: Macro-F1 (average across classes; sensitive to minority/weak classes).
- **Runtime**: End-to-end run time (pretrain + linear classifier stage).

### Results Table

| Slurm log | Framework | Backbone | Key settings | Test Acc (%) | miF (%) | maF (%) | Runtime |
|---|---|---|---|---:|---:|---:|---|
| `slurm_56221148_simclr_fcn.out` | SimCLR | FCN | `supcon`, `fdiff`, `70/70`, `bs=64`, `lr=1e-3`, `lr_cls=1e-3` | **35.94** | **35.94** | **34.62** | `0:05:44` |
| `slurm_56221396_augpred_xfm.out` | AugPred | Transformer | `fdiff`, `70/70`, `bs=64`, `lr=1e-3`, `lr_cls=1e-3` | 33.01 | 33.01 | 31.52 | `0:14:00` |
| `slurm_56221498_simclr_xfm.out` | SimCLR | Transformer | `supcon`, `fdiff`, `100/80`, `bs=128`, `lr=1e-3`, `lr_cls=1.5e-3` | 32.14 | 32.14 | 28.91 | `0:13:12` |
| `slurm_56221091_augpred_fcn.out` | AugPred | FCN | `fdiff`, `70/70`, `bs=64`, `lr=1e-3`, `lr_cls=1e-3` | 23.94 | 23.94 | 19.77 | `0:10:25` |

### Model-by-model explanation

#### 1) `slurm_56221148_simclr_fcn.out` — Best SSL so far
- **Framework:** SimCLR (contrastive SSL)
- **Backbone:** FCN (1D conv encoder)
- **Key setup:**
  - `supcon` mode: same-class samples in batch treated as extra positives
  - `fdiff` preprocessing
  - `pretrain_epochs=70`, `lincls_epochs=70`
  - `batch=64`, `lr=1e-3`, `lr_cls=1e-3`
- **Interpretation:** Strongest class separation among your SSL runs (best acc + macro-F1).

#### 2) `slurm_56221396_augpred_xfm.out`
- **Framework:** AugPred (augmentation-prediction SSL)
- **Backbone:** Transformer
- **Key setup:**
  - `fdiff`
  - `70/70` epochs
  - `batch=64`, `lr=1e-3`, `lr_cls=1e-3`
- **Interpretation:** Decent and stable, but below SimCLR+FCN. Transformer worked okay here.

#### 3) `slurm_56221498_simclr_xfm.out`
- **Framework:** SimCLR
- **Backbone:** Transformer
- **Key setup:**
  - `supcon`
  - `fdiff`
  - longer schedule: `100` pretrain / `80` lincls
  - larger batch `128`
  - `lr=1e-3`, `lr_cls=1.5e-3`
- **Interpretation:** More compute did not help; likely optimization/representation mismatch for this backbone+objective combination.

#### 4) `slurm_56221091_augpred_fcn.out` — Worst
- **Framework:** AugPred
- **Backbone:** FCN
- **Key setup:** same nominal `70/70`, `batch=64`, `fdiff`, `lr=1e-3`
- **Interpretation:** Likely poor convergence / class-collapse tendency for this config.
