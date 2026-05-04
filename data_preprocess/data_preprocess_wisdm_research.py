from __future__ import annotations

import hashlib
import os
import pickle as cp

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_preprocess.data_preprocess_utils import (
    get_sample_weights,
    normalize,
    opp_sliding_window_w_d,
    train_test_val_split,
)
from data_preprocess.data_preprocess_wisdm import (
    ACTIVITY_TO_ID,
    _apply_wisdm_feat_to_stream_array,
    _apply_wisdm_feat_to_windows,
    _load_sensor_file,
    _resolve_wisdm_root,
    _sensor_dirs,
    data_loader_wisdm,
)


def estimate_n_feature(args) -> int:
    filter_mode = str(getattr(args, "bio_filter_mode", "raw")).lower()
    if filter_mode == "motion_gravity":
        return 9
    return 6


def _linear_resample(arr: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    if arr.shape[0] < 2 or abs(dst_hz - src_hz) < 1e-8:
        return arr.astype(np.float32, copy=False)
    if src_hz <= 0 or dst_hz <= 0:
        raise ValueError(f"resampling rates must be > 0, got src_hz={src_hz}, dst_hz={dst_hz}")

    duration = (arr.shape[0] - 1) / src_hz
    new_len = max(2, int(round(duration * dst_hz)) + 1)
    old_t = np.linspace(0.0, duration, num=arr.shape[0], endpoint=True)
    new_t = np.linspace(0.0, duration, num=new_len, endpoint=True)

    out = np.empty((new_len, arr.shape[1]), dtype=np.float32)
    for c in range(arr.shape[1]):
        out[:, c] = np.interp(new_t, old_t, arr[:, c]).astype(np.float32, copy=False)
    return out


def _fft_filter(arr: np.ndarray, sample_hz: float, low_hz: float | None, high_hz: float | None) -> np.ndarray:
    if arr.shape[0] < 2:
        return arr.astype(np.float32, copy=False)
    nyquist = sample_hz * 0.5
    low = 0.0 if low_hz is None else max(0.0, float(low_hz))
    high = nyquist if high_hz is None else min(float(high_hz), nyquist)
    if high <= low:
        raise ValueError(
            f"Invalid filter band: low_hz={low_hz}, high_hz={high_hz}, sample_hz={sample_hz}",
        )

    spec = np.fft.rfft(arr, axis=0)
    freqs = np.fft.rfftfreq(arr.shape[0], d=1.0 / sample_hz)
    keep = (freqs >= low) & (freqs <= high)
    spec[~keep, :] = 0.0
    filt = np.fft.irfft(spec, n=arr.shape[0], axis=0)
    return filt.astype(np.float32, copy=False)


def _apply_research_preproc(arr: np.ndarray, args) -> np.ndarray:
    src_hz = float(getattr(args, "bio_source_hz", 20.0))
    dst_hz = float(getattr(args, "bio_resample_hz", src_hz))
    filter_mode = str(getattr(args, "bio_filter_mode", "raw")).lower()
    gravity_cutoff_hz = float(getattr(args, "bio_gravity_cutoff_hz", 0.25))
    motion_low_hz = float(getattr(args, "bio_motion_low_hz", 0.25))
    motion_high_hz = float(getattr(args, "bio_motion_high_hz", 15.0))

    arr = _linear_resample(arr, src_hz=src_hz, dst_hz=dst_hz)
    if filter_mode == "raw":
        return arr

    if arr.shape[1] != 6:
        raise ValueError(f"Expected 6-channel WISDM accel+gyro input, got shape {arr.shape}")

    acc = arr[:, :3]
    gyr = arr[:, 3:]
    nyquist = dst_hz * 0.5
    high = min(motion_high_hz, max(motion_low_hz + 1e-4, nyquist - 1e-4))

    gravity = _fft_filter(acc, sample_hz=dst_hz, low_hz=0.0, high_hz=gravity_cutoff_hz)
    motion_acc = _fft_filter(acc, sample_hz=dst_hz, low_hz=motion_low_hz, high_hz=high)
    motion_gyr = _fft_filter(gyr, sample_hz=dst_hz, low_hz=motion_low_hz, high_hz=high)

    if filter_mode == "motion":
        return np.concatenate([motion_acc, motion_gyr], axis=1).astype(np.float32, copy=False)
    if filter_mode == "gravity":
        # Gravity is physically meaningful for acceleration only; gyroscope channels are
        # passed through at the resampled rate so the model still sees the original 6-axis layout.
        return np.concatenate([gravity, gyr], axis=1).astype(np.float32, copy=False)
    if filter_mode == "motion_gravity":
        return np.concatenate([motion_acc, gravity, motion_gyr], axis=1).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported --bio_filter_mode '{filter_mode}'")


def _build_windows_for_subject_research(accel_path, gyro_path, ws, ss, args):
    df_acc = _load_sensor_file(accel_path).rename(columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"})
    df_gyr = _load_sensor_file(gyro_path).rename(columns={"x": "gyr_x", "y": "gyr_y", "z": "gyr_z"})

    key_cols = ["subject", "activity", "timestamp"]
    df_acc["dup_idx"] = df_acc.groupby(key_cols).cumcount()
    df_gyr["dup_idx"] = df_gyr.groupby(key_cols).cumcount()
    df = df_acc.merge(df_gyr, on=key_cols + ["dup_idx"], how="inner").drop(columns=["dup_idx"])
    if df.empty:
        return None, None, None

    feat_mode = str(getattr(args, "wisdm_feat", "raw")).lower()
    stream_feat_mode = "fdiff" if feat_mode == "fdiff" else "raw"

    x_all, y_all, d_all = [], [], []
    subject_id = int(df["subject"].iloc[0])

    for activity_code, group in df.groupby("activity", sort=False):
        arr = group[["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]].to_numpy(dtype=np.float32)
        arr = _apply_research_preproc(arr, args)
        if stream_feat_mode == "fdiff":
            arr = _apply_wisdm_feat_to_stream_array(arr, stream_feat_mode)
        if arr.shape[0] < ws:
            continue

        labels = np.full((arr.shape[0],), ACTIVITY_TO_ID[activity_code], dtype=np.uint8)
        domains = np.full((arr.shape[0],), subject_id, dtype=np.int32)
        x_win, y_win, d_win = opp_sliding_window_w_d(arr, labels, domains, ws, ss)
        x_all.append(x_win)
        y_all.append(y_win)
        d_all.append(d_win)

    if not x_all:
        return None, None, None
    return np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0), np.concatenate(d_all, axis=0)


def prep_wisdm(args, SLIDING_WINDOW_LEN=200, SLIDING_WINDOW_STEP=100, device="Phones"):
    if args.cases != "random":
        raise ValueError("WISDM preprocessing currently supports only '--cases random'.")

    top_k_classes = int(getattr(args, "wisdm_top_k_classes", 0))
    if top_k_classes < 0:
        raise ValueError(f"--wisdm_top_k_classes must be >= 0, got {top_k_classes}")

    root_dir = _resolve_wisdm_root(getattr(args, "wisdm_data_dir", None))
    accel_dir, gyro_dir = _sensor_dirs(root_dir, device)

    cache_dir = "./data/wisdm"
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    cache_cfg = (
        f"device={device}|feat={getattr(args, 'wisdm_feat', 'raw')}|k={top_k_classes}|"
        f"ws={SLIDING_WINDOW_LEN}|ss={SLIDING_WINDOW_STEP}|src={getattr(args, 'bio_source_hz', 20.0)}|"
        f"dst={getattr(args, 'bio_resample_hz', 20.0)}|mode={getattr(args, 'bio_filter_mode', 'raw')}|"
        f"g={getattr(args, 'bio_gravity_cutoff_hz', 0.25)}|ml={getattr(args, 'bio_motion_low_hz', 0.25)}|"
        f"mh={getattr(args, 'bio_motion_high_hz', 15.0)}"
    )
    digest = hashlib.md5(cache_cfg.encode("utf-8")).hexdigest()[:12]
    cache_file = os.path.join(cache_dir, f"wisdm_research_{digest}.data")

    if os.path.isfile(cache_file):
        with open(cache_file, "rb") as f:
            data = cp.load(f)
        x_train, y_train, d_train = data[0]
        x_val, y_val, d_val = data[1]
        x_test, y_test, d_test = data[2]
    else:
        x_all, y_all, d_all = [], [], []
        accel_files = sorted(f for f in os.listdir(accel_dir) if f.startswith("data_") and f.endswith(".txt"))
        for acc_file in accel_files:
            gyro_file = acc_file.replace("accel", "gyro")
            gyro_path = os.path.join(gyro_dir, gyro_file)
            acc_path = os.path.join(accel_dir, acc_file)
            if not os.path.isfile(gyro_path):
                continue
            x, y, d = _build_windows_for_subject_research(
                acc_path,
                gyro_path,
                ws=SLIDING_WINDOW_LEN,
                ss=SLIDING_WINDOW_STEP,
                args=args,
            )
            if x is None:
                continue
            x_all.append(x)
            y_all.append(y)
            d_all.append(d)

        if not x_all:
            raise RuntimeError("No valid WISDM windows were created. Check dataset files and permissions.")

        x_all = np.concatenate(x_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        d_all = np.concatenate(d_all, axis=0)

        if top_k_classes > 0:
            classes, counts = np.unique(y_all, return_counts=True)
            top_k_classes = min(top_k_classes, len(classes))
            keep = classes[np.argsort(counts)[::-1][:top_k_classes]]
            keep_set = set(int(k) for k in keep.tolist())
            mask = np.array([int(y) in keep_set for y in y_all], dtype=bool)
            x_all = x_all[mask]
            y_all = y_all[mask]
            d_all = d_all[mask]

            kept_sorted = sorted(int(k) for k in keep.tolist())
            remap = {old: new for new, old in enumerate(kept_sorted)}
            y_all = np.array([remap[int(y)] for y in y_all], dtype=np.uint8)

        unique_subjects = sorted(np.unique(d_all))
        subject_to_domain = {sid: idx for idx, sid in enumerate(unique_subjects)}
        d_all = np.array([subject_to_domain[sid] for sid in d_all], dtype=np.int32)

        x_all = _apply_wisdm_feat_to_windows(x_all, str(getattr(args, "wisdm_feat", "raw")).lower())
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = train_test_val_split(
            x_all,
            y_all,
            d_all,
            split_ratio=args.split_ratio,
        )
        x_train = normalize(x_train)
        x_val = normalize(x_val)
        x_test = normalize(x_test)

        obj = [(x_train, y_train, d_train), (x_val, y_val, d_val), (x_test, y_test, d_test)]
        with open(cache_file, "wb") as f:
            cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)

    args.n_feature = int(x_train.shape[-1])
    args.n_class = int(len(np.unique(y_train)))

    unique_y, counts_y = np.unique(y_train, return_counts=True)
    weights = 100.0 / torch.Tensor(counts_y)
    sample_weights = get_sample_weights(y_train, weights.double())
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_set = data_loader_wisdm(x_train, y_train, d_train)
    val_set = data_loader_wisdm(x_val, y_val, d_val)
    test_set = data_loader_wisdm(x_test, y_test, d_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return [train_loader], val_loader, test_loader
