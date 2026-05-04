# SNN_HAR
Pytorch implementation of Spiking Neural Networks for Human Activity Recognition. 
Paper link: [Wearable-based Human Activity Recognition with
Spatio-Temporal Spiking Neural Networks](https://arxiv.org/pdf/2212.02233.pdf).

## Installation and usage

Our implementation is based on Pytorch 1.10 or higher and other libraries. 
Please install the packages in the ``requirements.txt``:
```bash
pip install -r requirements.txt
```

## Datasets download

Please download the dataset and decompress them under the `data` directory before running the code:

- UCIHAR [link](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- SHAR [link](http://www.sal.disco.unimib.it/technologies/unimib-shar/)
- HHAR [link](http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition)
- WISDM [link](https://www.cis.fordham.edu/wisdm/dataset.php)

For WISDM specifically, this repo expects the **AR v1.1** layout with `raw/phone/accel`, `raw/phone/gyro`, etc.
If you cloned this repository with the bundled data, you can point preprocessing at:

- `./wisdm-dataset` (recommended; also the default `--wisdm_data_dir` for `main_ssl.py`)

## Running ANNs

```bash
python main.py --dataset ucihar --backbone FCN --rep 5
python main.py --dataset shar --backbone FCN --rep 5
python main.py --dataset hhar --backbone FCN --rep 5
```

Here, for ANN baselines, we can change our backbone to FCN (CNN in the paper), DCL, LSTM, Transformer.
`--rep` specifies the number of repeats for computing mean and standard deviation of the accuracies. 


## Running SNNs

```bash
python main.py --dataset ucihar --backbone SFCN --lr 1e-3 --tau 0.75 --thresh 0.5
python main.py --dataset shar --backbone SFCN --lr 1e-3 --tau 0.25 --thresh 0.5
python main.py --dataset hhar --backbone SFCN --lr 1e-3 --tau 0.75 --thresh 0.5
python main.py --dataset wisdm --backbone SFCN --lr 1e-3 --tau 0.75 --thresh 0.5 --wisdm_data_dir /path/to/wisdm-dataset
```

For spiking versions of backbone, we offer SFCN adn SDCL.
Note that they need to be trained with slightly larger learning rate.
The `--tau` specifies the decay factor in LIF neurons (Sec 3.3 in paper). 

### WISDM + self-supervised + iSpikformer (`SNN_Transformer` / `--backbone iSpikformer`)

The iSpikformer stack from [SeqSNN](https://github.com/microsoft/SeqSNN) is **vendored** under `models/local_ispikformer/` (SpikingJelly LIF). Install **`spikingjelly`** with the rest of the stack (`pip install -r requirements.txt`); you do **not** need `pip install -e SeqSNN` for SNN_HAR.

Run SSL, for example:

```bash
python main_ssl.py --dataset wisdm --device Phones --framework simclr --backbone iSpikformer \
  --ispf_dim 512 --ispf_depths 2 --ispf_num_steps 4 --wisdm_feat fdiff --input_encoding none \
  --aug1 jit_scal --aug2 perm_jit --batch_size 32 --pretrain_epochs 70 --lincls_epochs 70
```

Checkpoints are named with an `_ispf` suffix so they do not collide with runs that use the same flags but a different backbone. A Slurm example is in `sbatch/wisdm_simclr_ispikformer.sbatch`.

### Experimental spiking transformer: `SpikeFormer` (`--backbone SpikeFormer`)

This is a **new exploratory spiking self-attention backbone** implemented in **pure PyTorch** under `models/exp_spikeformer/`.
It is intentionally separate from the vendored **iSpikformer / SpikingJelly** stack in `models/local_ispikformer/`.

**Input / encoding:** `SpikeFormer` uses the same `--input_encoding` pipeline as the rest of the repo (`none`, `zcsf`, `arima`, `zcsf_arima`) via `trainer.apply_input_encoding()`.

**SSL entrypoint:** use `main_ssl.py` (same SimCLR/BYOL/etc. frameworks as other backbones).

**Model hyperparameters (CLI):**

- `--spk_tf_dim` (embedding size; also the linear-probe input dim)
- `--spk_tf_depth` (number of spiking transformer blocks)
- `--spk_tf_heads` (attention heads; must divide `--spk_tf_dim`)
- `--spk_tf_mlp_ratio` (FFN expansion ratio)
- `--spk_tf_dropout`, `--spk_tf_attn_dropout`
- `--spk_tf_spike_threshold` (threshold for the surrogate LIF-style spike nonlinearity used inside attention/MLP)

**Note:** `--tau` / `--thresh` in `main_ssl.py` are used by *other* SNN backbones (SFCN/SDCL/etc.). They are **not** the primary knobs for `SpikeFormer` (use `--spk_tf_spike_threshold` instead).

Example (WISDM + SimCLR + `SpikeFormer`):

```bash
python main_ssl.py \
  --dataset wisdm \
  --device Phones \
  --framework simclr \
  --backbone SpikeFormer \
  --wisdm_data_dir ./wisdm-dataset \
  --wisdm_feat fdiff \
  --input_encoding none \
  --spk_tf_dim 256 \
  --spk_tf_depth 4 \
  --spk_tf_heads 8 \
  --aug1 jit_scal \
  --aug2 perm_jit \
  --batch_size 32 \
  --pretrain_epochs 70 \
  --lincls_epochs 70
```

Run artifacts are tagged with a `_spktf` suffix in `trainer.setup()` naming to reduce checkpoint collisions vs other backbones.


## Acknowledgement

Our code framework is strongly based on [Tian0426/CL_HAR](https://github.com/Tian0426/CL-HAR).
We'd like to thank the authors of this repo for their efforts. 

If you find our paper interesting, please kindly consider cite the paper. 
```
@article{li2022wearable,
  title={Wearable-based Human Activity Recognition with Spatio-Temporal Spiking Neural Networks},
  author={Li, Yuhang and Yin, Ruokai and Park, Hyoungseob and Kim, Youngeun and Panda, Priyadarshini},
  journal={arXiv preprint arXiv:2212.02233},
  year={2022}
}
```




