# WISDM 12-Channel SSL (Bio-Style Setup)

This folder adds a dedicated pipeline for WISDM with:

- **Sampling rate:** 20 Hz (nominal)
- **Window:** 10.0 s
- **Stride:** 5.0 s
- **Samples/window:** 200
- **Channels:** 12 (`phone_acc_xyz + phone_gyro_xyz + watch_acc_xyz + watch_gyro_xyz`)

## What this uses

- Existing SNN_HAR trainer/framework code for SSL + linear probe
- Backbones supported here:
  - `SFCN`
  - `SNN_Transformer` (spiking transformer)
  - `Transformer`
  - `SResNet1D` (ResNet-1D)
- SSL frameworks supported here:
  - `simclr`
  - `byol`

## New files

- `preprocess_wisdm12.py` - fused 12-channel window builder + dataloaders
- `run_ssl_wisdm12.py` - dedicated entrypoint for this setup
- `run_examples.sh` - ready-to-run command batch

## Run one experiment

```bash
python SNN_HAR/wisdm12_ssl/run_ssl_wisdm12.py \
  --wisdm_data_dir /path/to/wisdm-dataset \
  --framework simclr \
  --backbone SResNet1D \
  --batch_size 64 \
  --pretrain_epochs 70 \
  --lincls_epochs 70
```

## Run all requested combinations

```bash
bash SNN_HAR/wisdm12_ssl/run_examples.sh /path/to/wisdm-dataset
```

## Notes

- Fused stream alignment is on the **watch-accel timeline**.
- Sensor values from phone accel, phone gyro, and watch gyro are linearly interpolated to those timestamps.
- Only windows fully contained in single-activity contiguous segments are used.
- Cached dataset file is written under `./data/wisdm12/`.
