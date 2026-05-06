# Final-Bio: Research, Code Changes, and Experiment History

Updated: 2026-05-06

## 1. Goal

The goal of this work was to improve downstream F1 on WISDM watch HAR using SNN-oriented preprocessing, spike encodings, and spiking backbones, then port the working ideas into the `final-bio` repository and validate them there.

The project evolved in two stages:

1. research prototyping in the earlier `SNN_HAR` repo
2. migration, cleanup, debugging, and improvement inside `final-bio`

This document records:

- what we changed
- why we changed it
- which papers motivated each change
- what the papers reported
- what our own runs produced

Important note:

- the papers below do not all report the same metric
- HAR SNN papers often report `accuracy`, not `F1`
- some encoding papers report signal-quality conclusions rather than downstream HAR classification metrics

So the right way to read this document is:

- `paper result` = what the source paper itself reported
- `our result` = what happened in our codebase and our WISDM setup

## 2. Starting Point

At the start, the codebase already had:

- SSL frameworks: `SimCLR`, `BYOL`, `SimSiam`, `NNCLR`, `TSTCC`
- standard HAR backbones
- a vendored `iSpikformer`-based `SNN_Transformer`
- WISDM preprocessing paths

But it did **not** yet have:

- research preprocessing copied cleanly for WISDM
- switchable `poisson_rate`, `step_forward`, and `moving_window` encodings in the new path
- a clean repo-local explanation of what had been tried
- sparsity reporting
- F1-oriented downstream loss
- a learnable sensor/channel gate for the SNN transformer path
- configurable iSpikformer membrane dynamics in the research runner

It also had a startup issue in `final-bio`:

- `trainer.py` eagerly imported all dataset preprocessors
- the UCIHAR path imported `torchvision`
- WISDM jobs failed before training with `RuntimeError: operator torchvision::nms does not exist`

That bug was fixed by changing dataset imports to lazy imports inside `setup_dataloaders()`.

## 3. Research Sources and How We Used Them

### 3.1 Temporal spike encoding

Paper:

- Petro, Kasabov, Kiss, "Selection and Optimization of Temporal Spike Encoding Methods for Spiking Neural Networks", IEEE TNNLS 2020
- DOI: https://doi.org/10.1109/TNNLS.2019.2906158
- Accessible summary: https://pubmed.ncbi.nlm.nih.gov/30990446/

What the paper says:

- compares multiple temporal encodings
- studies `threshold-based`, `step-forward (SW)`, and `moving-window (MW)` encodings
- concludes that `SW` was the most effective and robust across tested signal types

Paper-reported outcome:

- this paper is primarily an encoding analysis paper
- it does **not** report WISDM HAR F1 or accuracy
- its core result is methodological: `step-forward` is robust and easy to optimize, while `moving-window` is also a valid temporal-contrast encoder

How we used it:

- implemented `step_forward`
- implemented `moving_window`
- kept them switchable in the research path and sbatches

Code:

- `input_encoding_research.py`

### 3.2 Rate coding baseline

Paper:

- Auge et al., "A Survey of Encoding Techniques for Signal Processing in Spiking Neural Networks", Neurocomputing 2021
- DOI: https://doi.org/10.1007/s11063-021-10562-2
- Link: https://link.springer.com/article/10.1007/s11063-021-10562-2

What the paper says:

- rate coding is a standard baseline
- temporal encodings can preserve timing information better than simple dense rate codes in many signal-processing tasks

Paper-reported outcome:

- this is a survey, so it does not provide one single HAR benchmark number
- its value here was in framing `poisson_rate` as the standard baseline to compare against temporal encoders

How we used it:

- implemented `poisson_rate` as the main rate-coding baseline
- used it as the comparison point for `step_forward` and `moving_window`

Code:

- `input_encoding_research.py`

### 3.3 HAR-specific SNN membrane dynamics

Paper:

- Li et al., "Efficient human activity recognition with spatio-temporal spiking neural networks", Frontiers in Neuroscience 2023
- DOI: https://doi.org/10.3389/fnins.2023.1233037
- Link: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1233037/full

What the paper says:

- SNN performance on HAR is highly sensitive to membrane decay `tau`
- threshold and reset strategy matter too
- for their HAR setup, `tau > 0` is necessary; `tau = 0` behaves like a bad binary-activation network

Paper-reported outcome:

- on UCI-HAR with `SpikeDCL`, `tau = 0.75` reached `98.86%`
- on the same setup, `tau = 0` gave `94.36%`
- for firing threshold, `V_th = 0.5` was best in their ablation
- they also report energy reduction up to `94%`

How we used it:

- exposed iSpikformer/SNN-transformer membrane dynamics in the research path
- added `--ispf_tau`
- added `--ispf_common_thr`
- added `--ispf_detach_reset`

Code:

- `models/seqsnn_ispikformer.py`
- `models/local_ispikformer/core.py`
- `models/local_ispikformer/encoder_sj.py`
- `models/local_ispikformer/spike_attention_block.py`
- `models/spike.py`
- `trainer.py`
- `main_ssl_research.py`

### 3.4 Imbalance-aware learning

Primary implementation paper:

- Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
- DOI: https://doi.org/10.1109/CVPR.2019.00949
- Link: https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html

Related SNN motivation paper:

- Hens et al., "STAL: Spike Threshold Adaptive Learning Encoder for Classification of Pain-Related Biosignal Data", arXiv 2024
- Link: https://arxiv.org/abs/2407.08362

What the papers say:

- Cui et al.: class-balanced weighting using the effective number of samples improves learning on long-tailed data
- STAL-SRNN: imbalance handling matters in biosignal SNN tasks and should not be ignored

Paper-reported outcome:

- Cui et al. report significant gains on long-tailed CIFAR/ImageNet/iNaturalist, but this is not a HAR paper
- STAL-SRNN reports `Accuracy 80.43`, `AUC 67.90`, `F1 52.60`, `MCC 0.437` on the EmoPain task

How we used it:

- added `FocalLoss`
- added `ClassBalancedFocalLoss`
- made `cb_focal` the research default for downstream linear evaluation / classification

Code:

- `models/loss.py`
- `trainer.py`
- `main_ssl_research.py`

Important honesty note:

- this part was not copied from a HAR SNN paper line-by-line
- it is an engineering adaptation: imbalance-aware classification for downstream F1, motivated by both long-tailed learning literature and the need for better class balance in biosignal/SNN tasks

### 3.5 Channel reweighting / attention over biosignals

Paper:

- Li et al., "Accurate ECG Classification Based on Spiking Neural Network and Attentional Mechanism for Real-Time Implementation on Personal Portable Devices", Electronics 2022
- DOI: https://doi.org/10.3390/electronics11121889
- Link: https://www.mdpi.com/2079-9292/11/12/1889

What the paper says:

- a channel-wise attentional module helps SNNs focus on more informative parts of biosignal representations

Paper-reported outcome:

- `Accuracy 98.26%`
- `Sensitivity 94.75%`
- `F1 89.09%`

How we used it:

- added a lightweight squeeze-excitation style sensor/channel gate before the `SNN_Transformer`
- this is not an exact ECG architecture port
- it is a concept adaptation: learn which IMU channels matter more

Code:

- `models/spike.py`

### 3.6 Event-style sparse encoding intuition for time series

Paper:

- Manna et al., "Time Series Forecasting via Derivative Spike Encoding and Bespoke Loss Functions for Spiking Neural Networks", Computers 2024
- DOI: https://doi.org/10.3390/computers13080202
- Link: https://www.mdpi.com/2073-431X/13/8/202

What the paper says:

- thresholded event-style encodings increase sparsity
- they reduce noise propagation
- they omit unchanging redundant elements

Paper-reported outcome:

- this is a time-series forecasting paper, not a HAR benchmark paper
- its main result is that encoded spike-based time series can be learned effectively by SNNs, and their proposed loss outperformed SLAYER SpikeTime loss in that forecasting setting

How we used it:

- as additional support for event-like sparse encodings and sparsity measurement
- not as a direct architecture copy

## 4. What We Implemented in Code

### 4.1 Research preprocessing path for WISDM

We created a copied research path so the original code stayed intact.

Main ideas:

- configurable resampling
- configurable motion/gravity filtering
- fixed windows in seconds
- encoding after preprocessing and before model forward

Modes we added:

- `raw`
- `motion`
- `gravity`
- `motion_gravity`
- later `motion_gravity_accel` for the 6-channel accel-only variant

Key code:

- `data_preprocess/data_preprocess_wisdm_research.py`
- `trainer_research.py`
- `main_ssl_research.py`

### 4.2 Research encodings

Implemented:

- `poisson_rate`
- `step_forward`
- `moving_window`

Key code:

- `input_encoding_research.py`

### 4.3 Final-bio bug fix

Problem:

- WISDM jobs crashed before training due to unrelated eager dataset imports

Fix:

- lazy imports inside `setup_dataloaders()`

Key code:

- `trainer.py`

### 4.4 6-channel accel-only mode

Added:

- `motion_gravity_accel`

Meaning:

- keep `motion_acc (3)` + `gravity_acc (3)`
- drop gyro

Key code:

- `data_preprocess/data_preprocess_wisdm_research.py`

### 4.5 F1-oriented upgrade pass

Added:

- `FocalLoss`
- `ClassBalancedFocalLoss`
- learnable sensor/channel gate for `SNN_Transformer`
- configurable `ispf_tau`, `ispf_common_thr`, `ispf_detach_reset`
- encoded-input and internal-spike sparsity tracking

Key code:

- `models/loss.py`
- `models/spike.py`
- `models/seqsnn_ispikformer.py`
- `models/local_ispikformer/core.py`
- `models/local_ispikformer/encoder_sj.py`
- `models/local_ispikformer/spike_attention_block.py`
- `trainer.py`
- `main_ssl_research.py`

## 5. Experiment History

## 5.1 Prototype-stage experiments in the earlier SNN_HAR repo

These experiments were done before the final-bio port, but they directly guided what we moved over.

### Baseline ANN Transformer in the older repo

Log:

- `slurm_56397794_w_sc_tr.out`

Result:

- Test Acc `62.8494`
- miF `62.8494`
- weighted F1 `62.6134`
- macro F1 `63.1250`

Interpretation:

- the plain Transformer backbone was already a strong baseline in the earlier repo

### Research Transformer encoding sweep in the older repo

Logs:

- `slurm_56575429_tr_pois.out`
- `slurm_56575440_tr_step.out`
- `slurm_56575454_tr_mw.out`

Results:

| Backbone | Encoding | Acc | weighted F1 | macro F1 | Notes |
|---|---:|---:|---:|---:|---|
| Transformer | poisson_rate | 71.3098 | 70.8586 | 70.9598 | best old Transformer run |
| Transformer | step_forward | 62.1782 | 61.4743 | 61.7248 | below Poisson |
| Transformer | moving_window | 59.9232 | 58.1809 | 58.4198 | worst of the 3 |

Interpretation:

- in that earlier plain Transformer path, `poisson_rate` worked best
- `step_forward` and `moving_window` did not help there

### Earlier ResNet-family reference runs in the older repo

Best old ResNet baseline:

- log: `slurm_56388486_w_sc_sr.out`
- Test Acc `52.4996`
- miF `52.4996`
- weighted F1 `50.4701`

Older research ResNet encoding sweep:

| Backbone | Encoding | Acc | weighted F1 | macro F1 |
|---|---:|---:|---:|---:|
| SResNet1D | poisson_rate | 13.5601 | 9.6662 | 9.7207 |
| SResNet1D | step_forward | 40.0256 | 37.6824 | 38.2691 |
| SResNet1D | moving_window | 48.7782 | 46.6216 | 47.2168 |

Interpretation:

- the early copied research path was not yet a good fit for ResNet
- this motivated a cleaner port and later the F1-oriented upgrade work in `final-bio`

## 5.2 Final-bio migration and debugging

### First SNN_Transformer jobs in final-bio

Jobs:

- `56689342`
- `56689344`
- `56689347`

Status:

- failed immediately

Reason:

- import-time `torchvision::nms` crash due to eager dataset imports

Fix:

- lazy imports in `trainer.py`

### Successful 9-channel SNN_Transformer encoding sweep in final-bio

Jobs:

- `56690532`
- `56690535`
- `56690538`

Results:

| Job | Backbone | Encoding | Channels | Acc | weighted F1 | macro F1 |
|---|---|---|---:|---:|---:|---:|
| 56690532 | SNN_Transformer | poisson_rate | 9 | 50.5997 | 49.6329 | 49.7358 |
| 56690535 | SNN_Transformer | step_forward | 9 | 50.2799 | 48.9045 | 49.2071 |
| 56690538 | SNN_Transformer | moving_window | 9 | 46.7136 | 45.3512 | 45.6465 |

Interpretation:

- `poisson_rate` again won among the three encodings
- but F1 was still behind the older plain Transformer result from the earlier repo

### 6-channel accel-only SNN_Transformer

Job:

- `56695186`

Setup:

- `poisson_rate`
- `motion_gravity_accel`
- 6 channels

Result:

- Acc `44.9384`
- weighted F1 `43.9711`
- macro F1 `43.9538`

Interpretation:

- dropping gyro hurt performance in this setup
- the 9-channel motion+gravity+gyro variant was better

## 5.3 Final-bio F1-oriented upgrade experiments

### Upgraded SNN_Transformer

Job:

- `56722824`

Setup:

- backbone `SNN_Transformer`
- encoding `poisson_rate`
- 9-channel `motion_gravity`
- downstream loss `cb_focal`
- sensor/channel gate `se`
- configurable iSpikformer dynamics enabled
- sparsity reporting enabled

Result:

- Acc `60.9947`
- miF `60.9947`
- weighted F1 `60.3976`
- macro F1 `60.5205`
- training time `0:48:42`

Sparsity:

- final test `input_sparsity = 0.6209`
- final test `spike_sparsity = 0.8979`
- final test `spike_density = 0.1021`

Interpretation:

- this was a large improvement over the earlier final-bio SNN_Transformer Poisson run
- weighted F1 improved from `49.6329` to `60.3976`

### Upgraded SResNet1D

Job:

- `56729897`

Setup:

- backbone `SResNet1D`
- encoding `poisson_rate`
- 9-channel `motion_gravity`
- downstream loss `cb_focal`
- sparsity reporting enabled

Result:

- Acc `66.8799`
- miF `66.8799`
- weighted F1 `66.5695`
- macro F1 `66.7052`
- training time `3:24:10`

Sparsity:

- final test `input_sparsity = 0.6209`
- final test `spike_sparsity = 0.6244`
- final test `spike_density = 0.3756`

Interpretation:

- this became the best result among the runs we checked
- it beat:
  - the old best ResNet baseline
  - the earlier final-bio SNN_Transformer runs
  - the upgraded SNN_Transformer run

## 6. Best Results Summary

### Best result in final-bio so far

| Rank | Job | Backbone | Encoding | Acc | weighted F1 | macro F1 |
|---|---|---|---|---:|---:|---:|
| 1 | 56729897 | SResNet1D | poisson_rate | 66.8799 | 66.5695 | 66.7052 |
| 2 | 56722824 | SNN_Transformer | poisson_rate | 60.9947 | 60.3976 | 60.5205 |
| 3 | 56690532 | SNN_Transformer | poisson_rate | 50.5997 | 49.6329 | 49.7358 |

### Biggest validated improvements

#### SNN_Transformer

- old final-bio 9ch Poisson weighted F1: `49.6329`
- upgraded final-bio 9ch Poisson weighted F1: `60.3976`
- gain: `+10.7647`

#### SResNet1D

- old older-repo best weighted F1: `50.4701`
- upgraded final-bio weighted F1: `66.5695`
- gain: `+16.0994`

## 7. SNN Sparsity Interpretation

The research literature does not use one perfectly standardized sparsity metric across all time-series SNN papers.

Typical paper-level measurements include:

- spikes per neuron
- firing rate
- active fraction
- event density

Our repo now logs:

- `input_sparsity`
- `spike_sparsity`
- `spike_density`

Layman meaning:

- `input_sparsity`: how much of the encoded input is zero
- `spike_sparsity`: how often internal SNN activations are zero
- `spike_density`: the opposite of spike sparsity; how often spikes happen

### Our two important final-bio sparsity profiles

#### Upgraded SNN_Transformer

- input sparsity `0.6209`
- spike sparsity `0.8979`
- spike density `0.1021`

Interpretation:

- roughly `90%` of internal neuron-time slots are silent
- this is meaningfully sparse
- it looks like a real spiking regime, not a dense ANN-like regime

#### Upgraded SResNet1D

- input sparsity `0.6209`
- spike sparsity `0.6244`
- spike density `0.3756`

Interpretation:

- much denser internally than the SNN transformer
- but it classified better on this WISDM setup

### Context from research

Relevant papers:

- survey on sparse spike coding: https://link.springer.com/article/10.1007/s11063-021-10562-2
- derivative/event-style time-series encoding and sparsity: https://www.mdpi.com/2073-431X/13/8/202
- very sparse deep SNN regime: Stanojevic et al., Nature Communications 2024, https://www.nature.com/articles/s41467-024-51110-5

What that means for us:

- our upgraded `SNN_Transformer` is sparse in a practically healthy way
- our upgraded `SResNet1D` is less sparse, but currently gives the best F1
- better sparsity does **not** automatically mean better classification
- the right target is the best balance of:
  - F1
  - sparsity
  - training stability

## 8. What We Implemented vs What We Did Not Yet Implement

### Implemented

- research WISDM preprocessing copy
- switchable `poisson_rate`, `step_forward`, `moving_window`
- 6-channel accel-only branch
- fixed final-bio import bug
- class-balanced focal downstream loss
- sensor/channel gate before `SNN_Transformer`
- configurable iSpikformer membrane parameters
- sparsity reporting

### Considered from literature but not yet implemented

- trainable adaptive spike encoder like STAL
- topology-aware IMU mixer / graph prior like PAS-Net
- dynamic threshold neurons / neuromodulated thresholds
- temporal early-exit loss
- explicit mutual-information feature selection

## 9. Main Conclusions

1. Among the three tested research encodings, `poisson_rate` was consistently the strongest in our actual WISDM runs.

2. The copied research preprocessing by itself was not enough; the strong gains came later after:
   - cleaner final-bio port
   - imbalance-aware downstream loss
   - learnable channel emphasis
   - explicit control over spiking dynamics

3. The upgraded `SNN_Transformer` improved a lot:
   - weighted F1 `49.6329 -> 60.3976`

4. The upgraded `SResNet1D` is currently the best model we tested:
   - weighted F1 `66.5695`

5. The 6-channel accel-only variant was worse than the 9-channel motion+gravity+gyro setup in our experiments.

6. Our sparsity reporting showed an important tradeoff:
   - `SNN_Transformer` was much sparser internally
   - `SResNet1D` was denser but currently more accurate

## 10. Recommended Next Experiments

If we continue from here, the cleanest next ablations are:

1. `SNN_Transformer` upgraded setup with `--snn_tf_channel_gate none`
   - to isolate the gain from the channel gate

2. `SNN_Transformer` upgraded setup with a small `ispf_tau` sweep
   - e.g. `1.0`, `1.5`, `2.0`, `2.5`

3. `SResNet1D` upgraded setup with `step_forward`
   - to test whether the new F1-oriented path rescues temporal encoders for ResNet better than the earlier prototype did

4. trainable adaptive encoding
   - this is the biggest literature-backed thing still missing

## 11. Key Files in Final-Bio

- `trainer.py`
- `main_ssl_research.py`
- `trainer_research.py`
- `input_encoding_research.py`
- `data_preprocess/data_preprocess_wisdm_research.py`
- `models/loss.py`
- `models/spike.py`
- `models/seqsnn_ispikformer.py`
- `models/local_ispikformer/core.py`
- `models/local_ispikformer/encoder_sj.py`
- `models/local_ispikformer/spike_attention_block.py`

## 12. Reference Links

- Petro, Kasabov, Kiss 2020 temporal spike encoding:
  https://pubmed.ncbi.nlm.nih.gov/30990446/

- Auge et al. 2021 encoding survey:
  https://link.springer.com/article/10.1007/s11063-021-10562-2

- Li et al. 2023 HAR SNN ablations:
  https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1233037/full

- Cui et al. 2019 class-balanced loss:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html

- Hens et al. 2024 STAL-SRNN:
  https://arxiv.org/abs/2407.08362

- Li et al. 2022 ECG attention SNN:
  https://www.mdpi.com/2079-9292/11/12/1889

- Manna et al. 2024 derivative spike encoding for time series:
  https://www.mdpi.com/2073-431X/13/8/202

- Stanojevic et al. 2024 high-performance sparse deep SNNs:
  https://www.nature.com/articles/s41467-024-51110-5

## 13. Complete Final-Bio Run Ledger (This Workspace)

This section lists all tracked `final-bio` Slurm runs that were part of this project cycle.

### 13.1 Failed or blocked runs

| Job | Log | Stage | Failure type | Root cause | Fix status |
|---|---|---|---|---|---|
| 56689342 | `slurm_56689342_snn_tr_pois.err` | startup | `RuntimeError` | `operator torchvision::nms does not exist` from eager UCIHAR import path | fixed via lazy imports in `trainer.py` |
| 56689344 | `slurm_56689344_snn_tr_step.err` | startup | `RuntimeError` | same as above | fixed |
| 56689347 | `slurm_56689347_snn_tr_mw.err` | startup | `RuntimeError` | same as above | fixed |
| 56741293 | `slurm_56741293_snn_tr_maxf1.err` | lincls start | `ValueError` | `--lincls_finetune_backbone` unsupported for `SNN_Transformer` in `_apply_lincls_finetune_settings` | fixed by adding `SNN_Transformer` branch |

### 13.2 Successful runs with final metrics

| Job | Log | Backbone | Encoding / setup | Test Acc | miF | weighted F1 | macro F1 | Test loss |
|---|---|---|---|---:|---:|---:|---:|---:|
| 56690532 | `slurm_56690532_snn_tr_pois.out` | SNN_Transformer | poisson_rate, 9ch motion+gravity | 50.5997 | 50.5997 | 49.6329 | 49.7358 | 1.4553 |
| 56690535 | `slurm_56690535_snn_tr_step.out` | SNN_Transformer | step_forward, 9ch motion+gravity | 50.2799 | 50.2799 | 48.9045 | 49.2071 | 1.5766 |
| 56690538 | `slurm_56690538_snn_tr_mw.out` | SNN_Transformer | moving_window, 9ch motion+gravity | 46.7136 | 46.7136 | 45.3512 | 45.6465 | 1.6405 |
| 56695186 | `slurm_56695186_snn_tr_pois_acc6.out` | SNN_Transformer | poisson_rate, 6ch motion_gravity_accel | 44.9384 | 44.9384 | 43.9711 | 43.9538 | 1.6571 |
| 56722824 | `slurm_56722824_snn_tr_pois_f1.out` | SNN_Transformer | poisson_rate, F1-oriented upgrade | 60.9947 | 60.9947 | 60.3976 | 60.5205 | 0.7260 |
| 56729897 | `slurm_56729897_sr_pois_f1.out` | SResNet1D | poisson_rate, F1-oriented upgrade | 66.8799 | 66.8799 | 66.5695 | 66.7052 | 0.6212 |
| 56741457 | `slurm_56741457_sr_maxf1.out` | SResNet1D | hybrid_ds_rate + max-F1 finetune settings | **69.5026** | **69.5026** | **69.1953** | **69.3358** | 1.1324 |

### 13.3 Ordering by weighted F1 (best to worst successful run)

| Rank | Job | Backbone | weighted F1 | macro F1 | Note |
|---|---|---|---:|---:|---|
| 1 | 56741457 | SResNet1D | **69.1953** | 69.3358 | current best verified run in `final-bio` |
| 2 | 56729897 | SResNet1D | 66.5695 | 66.7052 | strong F1-oriented Poisson run |
| 3 | 56722824 | SNN_Transformer | 60.3976 | 60.5205 | strongest successful SNN_Transformer run |
| 4 | 56690532 | SNN_Transformer | 49.6329 | 49.7358 | initial successful 9ch Poisson |
| 5 | 56690535 | SNN_Transformer | 48.9045 | 49.2071 | step_forward 9ch |
| 6 | 56690538 | SNN_Transformer | 45.3512 | 45.6465 | moving_window 9ch |
| 7 | 56695186 | SNN_Transformer | 43.9711 | 43.9538 | 6ch accel-only variant |

## 14. Max-F1 Boost Pass (Latest Cycle)

This was the latest focused engineering pass after the earlier F1 improvements.

### 14.1 What we added

- `hybrid_ds_rate` encoding in `input_encoding_research.py` (event-style threshold dynamics blended with Poisson-rate behavior).
- linear-eval tuning knobs in `main_ssl.py`:
  - `--lincls_label_smoothing`
  - `--lincls_scheduler {cosine,onecycle,none}`
  - `--lincls_grad_clip`
  - `--lincls_select_metric {loss,macrof1}`
  - `--lincls_logit_adjust_tau`
  - `--lincls_calibrate_temperature` and sweep controls
- F1-aware training behavior in `trainer.py`:
  - optional class-prior logit adjustment
  - CE label smoothing support
  - macro-F1 checkpoint selection option
  - OneCycle and grad clipping support in linear-eval
  - post-hoc temperature calibration sweep
  - richer linear-head payload save/load path
- `mlp_lincls_from_simclr_ckpt.py` support for richer linear-head payloads.
- two max-F1 sbatches:
  - `sbatch/wisdm_snn_transformer_maxf1_boost.sbatch`
  - `sbatch/wisdm_sresnet_maxf1_boost.sbatch`

### 14.2 Results from this pass

#### SResNet1D max-F1 run (`56741457`)

- Completed successfully.
- Best observed metrics in this repo cycle:
  - Test Acc `69.5026`
  - weighted F1 `69.1953`
  - macro F1 `69.3358`
- Sparsity at test:
  - input `0.4705`
  - spike sparsity `0.6298`
  - spike density `0.3702`

#### SNN_Transformer max-F1 run (`56741293`)

- Did not complete due to linear-eval finetune compatibility gap:
  - `_apply_lincls_finetune_settings` lacked explicit support for `SNN_Transformer`.
- Follow-up fix implemented:
  - support added for `SNN_Transformer` with:
    - `all` scope: unfreeze all parameters
    - `last_block` scope: unfreeze `trained_backbone._core.core.blocks[-1]`

### 14.3 Practical interpretation

- The max-F1 stack was validated as effective immediately on `SResNet1D`.
- The same stack is now unblocked for `SNN_Transformer`; re-run is required for a finalized, apples-to-apples max-F1 result for that backbone.
- At this checkpoint:
  - `SResNet1D` is best for pure classification quality.
  - `SNN_Transformer` remains more internally sparse in upgraded Poisson runs, but still trails SResNet1D on weighted/macro F1.
