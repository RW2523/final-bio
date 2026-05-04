# SimCLR vs AugPred Ablation (WISDM `v1Logs`)

## Goal
These ablations test which SSL objective gives better transferable representations on WISDM while keeping runtime practical.
Because Spiking ResNet1D runs are expensive, we did targeted comparisons instead of exhaustively running all combinations/hyperparameter grids.

## Per-Backbone Results

| Backbone | SSL Method | Test Acc | maF | Train Time | Note |
|---|---:|---:|---:|---:|---|
| FCN | SimCLR | 35.94% | 34.62% | 0:05:44 | Best overall |
| FCN | AugPred | 23.94% | 19.77% | 0:10:25 |  |
| Transformer | SimCLR | 32.14% | 28.91% | 0:13:12 |  |
| Transformer | AugPred | 33.01% | 31.52% | 0:14:00 | Best Transformer |

## Backbone-Wise Comparison

| Backbone | Better Method | Accuracy Delta | Summary |
|---|---|---:|---|
| FCN | SimCLR | +12.00% | Large SimCLR advantage |
| Transformer | AugPred | +0.87% | AugPred slightly better |
| ResNet1D | SimCLR | +14.84% | Large AugPred drop |

## Key Takeaways
- SimCLR is the stronger default across backbones in this set, especially for FCN and ResNet1D.
- AugPred is competitive only on Transformer, where it gives a modest gain (+0.87% acc).
- Best transfer performance here is from FCN + SimCLR.
- For expensive backbones like Spiking ResNet1D, this targeted ablation is a reasonable tradeoff between rigor and compute cost.

## One-Line Conclusion
On WISDM (`v1Logs`), SimCLR generally yields better transferable features than AugPred, with the only exception being a small Transformer gain for AugPred.
