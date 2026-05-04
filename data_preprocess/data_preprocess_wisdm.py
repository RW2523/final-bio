"""
Data preprocessing for WISDM smartphone/smartwatch HAR dataset.
"""

import os
import pickle as cp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_preprocess.base_loader import base_loader
from data_preprocess.data_preprocess_utils import (
    get_sample_weights,
    normalize,
    opp_sliding_window_w_d,
    train_test_val_split,
)


ACTIVITY_CODES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S']
ACTIVITY_TO_ID = {code: idx for idx, code in enumerate(ACTIVITY_CODES)}
SHARED_WISDM_ROOT = '/home/sriramkannan_umass_edu/690R-BioMarkers/wisdm-dataset'
LOCAL_WISDM_ROOT = './dataset/WISDM_ar_v1.1'
ALT_WISDM_ROOT = './wisdm-dataset'


class data_loader_wisdm(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_wisdm, self).__init__(samples, labels, domains)


def _fdiff_pad(x_trial, axis=0):
    """
    First difference along time, keeping length by repeating the first sample.
    x_trial: (T, C) float32
    """
    if x_trial.shape[0] < 2:
        return x_trial
    d = np.diff(x_trial, axis=axis)
    return np.vstack([x_trial[:1], d]).astype(np.float32, copy=False)


def _apply_wisdm_feat_to_stream_array(arr, feat_mode):
    """
    arr: (N, C) rows in time order (within a subject/activity group)
    feat_mode:
      - 'raw': no change
      - 'fdiff': per-channel first difference (padded) on the continuous stream
    """
    if feat_mode in (None, 'raw', 'none', ''):
        return arr
    if feat_mode == 'fdiff':
        return _fdiff_pad(arr, axis=0)
    raise ValueError(f"Unknown WISDM feat mode: {feat_mode}")


def _apply_wisdm_feat_to_windows(x_win, feat_mode):
    """
    x_win: (Nwin, T, C)
    feat_mode:
      - 'l2': L2-normalize each window (per sample) across (T,C) dims
    """
    if feat_mode in (None, 'raw', 'none', 'fdiff', ''):
        return x_win
    if feat_mode == 'l2':
        x = x_win.reshape(x_win.shape[0], -1).astype(np.float32, copy=False)
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        x = x / denom
        return x.reshape(x_win.shape)
    raise ValueError(f"Unknown WISDM window feat mode: {feat_mode}")


def _resolve_wisdm_root(wisdm_data_dir=None):
    """
    Resolution order:
    1) explicit --wisdm_data_dir
    2) shared lab root if present: SHARED_WISDM_ROOT
    3) WISDM_DATA_DIR environment variable
    4) ./dataset/WISDM_ar_v1.1 (local clone)
    """
    candidates = []
    if wisdm_data_dir and str(wisdm_data_dir).strip():
        candidates.append(str(wisdm_data_dir).strip())
    candidates.append(SHARED_WISDM_ROOT)
    env_dir = os.environ.get("WISDM_DATA_DIR", "").strip()
    if env_dir:
        candidates.append(env_dir)
    candidates.append(ALT_WISDM_ROOT)
    candidates.append(LOCAL_WISDM_ROOT)
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    raise FileNotFoundError(
        'WISDM dataset not found. Set --wisdm_data_dir or WISDM_DATA_DIR to the AR v1.1 root '
        f'(with raw/phone/accel, raw/phone/gyro), or place data at {LOCAL_WISDM_ROOT}. '
        f'Also checked: {SHARED_WISDM_ROOT}'
    )


def _sensor_dirs(root_dir, device):
    if device == 'Watch':
        rel = 'raw/watch'
    else:
        rel = 'raw/phone'
    return os.path.join(root_dir, rel, 'accel'), os.path.join(root_dir, rel, 'gyro')


def _load_sensor_file(path):
    df = pd.read_csv(
        path,
        header=None,
        names=['subject', 'activity', 'timestamp', 'x', 'y', 'z'],
        sep=',',
        engine='python',
    )
    df['z'] = df['z'].astype(str).str.rstrip(';')
    df = df[df['activity'].isin(ACTIVITY_TO_ID.keys())]
    df = df.dropna()
    df['subject'] = df['subject'].astype(np.int32)
    df['activity'] = df['activity'].astype(str)
    df['timestamp'] = df['timestamp'].astype(np.int64)
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(np.float32)
    return df


def _build_windows_for_subject(accel_path, gyro_path, ws, ss, stream_feat_mode='raw'):
    df_acc = _load_sensor_file(accel_path).rename(
        columns={'x': 'acc_x', 'y': 'acc_y', 'z': 'acc_z'}
    )
    df_gyr = _load_sensor_file(gyro_path).rename(
        columns={'x': 'gyr_x', 'y': 'gyr_y', 'z': 'gyr_z'}
    )

    # Some WISDM files have repeated timestamps in either stream.
    # Pair duplicate keys deterministically by occurrence index, then merge.
    key_cols = ['subject', 'activity', 'timestamp']
    df_acc['dup_idx'] = df_acc.groupby(key_cols).cumcount()
    df_gyr['dup_idx'] = df_gyr.groupby(key_cols).cumcount()
    df = pd.merge(
        df_acc,
        df_gyr,
        on=key_cols + ['dup_idx'],
        how='inner',
    ).drop(columns=['dup_idx'])
    if df.empty:
        return None, None, None

    x_all, y_all, d_all = [], [], []
    subject_id = int(df['subject'].iloc[0])
    for activity_code, group in df.groupby('activity', sort=False):
        arr = group[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']].to_numpy(dtype=np.float32)
        # Optional stream-level transforms (apply BEFORE sliding windowing)
        if stream_feat_mode in ('fdiff',):
            arr = _apply_wisdm_feat_to_stream_array(arr, stream_feat_mode)
        if arr.shape[0] < ws:
            continue
        labels = np.full((arr.shape[0],), ACTIVITY_TO_ID[activity_code], dtype=np.uint8)
        domains = np.full((arr.shape[0],), subject_id, dtype=np.int32)
        x_win, y_win, d_win = opp_sliding_window_w_d(arr, labels, domains, ws, ss)
        x_all.append(x_win)
        y_all.append(y_win)
        d_all.append(d_win)

    if len(x_all) == 0:
        return None, None, None
    return np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0), np.concatenate(d_all, axis=0)


def prep_wisdm(args, SLIDING_WINDOW_LEN=200, SLIDING_WINDOW_STEP=100, device='Phones'):
    if args.cases != 'random':
        raise ValueError("WISDM preprocessing currently supports only '--cases random'.")

    feat_mode = str(getattr(args, 'wisdm_feat', 'raw')).lower()
    top_k_classes = int(getattr(args, 'wisdm_top_k_classes', 0))
    if top_k_classes < 0:
        raise ValueError(f'--wisdm_top_k_classes must be >= 0, got {top_k_classes}')
    # Stream-level and window-level modes are separate (e.g. fdiff is stream; l2 is per-window)
    stream_feat_mode = 'fdiff' if feat_mode == 'fdiff' else 'raw'

    root_dir = _resolve_wisdm_root(getattr(args, 'wisdm_data_dir', None))
    accel_dir, gyro_dir = _sensor_dirs(root_dir, device)

    cache_dir = './data/wisdm'
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    safe_feat = feat_mode.replace(os.sep, '_')
    cache_file = os.path.join(
        cache_dir,
        f'wisdm_{device.lower()}_f{safe_feat}_k{top_k_classes}_ws{SLIDING_WINDOW_LEN}_ss{SLIDING_WINDOW_STEP}.data'
    )

    if os.path.isfile(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        x_train, y_train, d_train = data[0]
        x_val, y_val, d_val = data[1]
        x_test, y_test, d_test = data[2]
        if hasattr(args, 'n_class'):
            args.n_class = int(len(np.unique(y_train)))
    else:
        x_all, y_all, d_all = [], [], []
        accel_files = sorted(
            [f for f in os.listdir(accel_dir) if f.startswith('data_') and f.endswith('.txt')]
        )
        for acc_file in accel_files:
            gyro_file = acc_file.replace('accel', 'gyro')
            gyro_path = os.path.join(gyro_dir, gyro_file)
            acc_path = os.path.join(accel_dir, acc_file)
            if not os.path.isfile(gyro_path):
                continue
            x, y, d = _build_windows_for_subject(
                acc_path,
                gyro_path,
                ws=SLIDING_WINDOW_LEN,
                ss=SLIDING_WINDOW_STEP,
                stream_feat_mode=stream_feat_mode,
            )
            if x is None:
                continue
            x_all.append(x)
            y_all.append(y)
            d_all.append(d)

        if len(x_all) == 0:
            raise RuntimeError('No valid WISDM windows were created. Check dataset files and permissions.')

        x_all = np.concatenate(x_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        d_all = np.concatenate(d_all, axis=0)

        # Keep only top-K most frequent classes globally, then remap to contiguous IDs.
        if top_k_classes > 0:
            classes, counts = np.unique(y_all, return_counts=True)
            if top_k_classes > len(classes):
                top_k_classes = len(classes)
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

        # Optional window-level transforms
        x_all = _apply_wisdm_feat_to_windows(x_all, feat_mode)

        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = train_test_val_split(
            x_all, y_all, d_all, split_ratio=args.split_ratio
        )
        x_train = normalize(x_train)
        x_val = normalize(x_val)
        x_test = normalize(x_test)
        if hasattr(args, 'n_class'):
            args.n_class = int(len(np.unique(y_train)))

        obj = [(x_train, y_train, d_train), (x_val, y_val, d_val), (x_test, y_test, d_test)]
        with open(cache_file, 'wb') as f:
            cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)

    unique_y, counts_y = np.unique(y_train, return_counts=True)
    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()
    sample_weights = get_sample_weights(y_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_set = data_loader_wisdm(x_train, y_train, d_train)
    val_set = data_loader_wisdm(x_val, y_val, d_val)
    test_set = data_loader_wisdm(x_test, y_test, d_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return [train_loader], val_loader, test_loader
