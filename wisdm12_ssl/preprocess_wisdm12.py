"""
WISDM fused 12-channel preprocessing for SSL in SNN_HAR.

Channels per timestep (C=12):
  [phone_acc_xyz, phone_gyro_xyz, watch_acc_xyz, watch_gyro_xyz]

Windowing defaults (bio-style):
  - nominal_sample_rate_hz = 20
  - window_seconds = 10.0  -> 200 samples
  - stride_seconds = 5.0   -> 100 samples
"""

from __future__ import annotations

import glob
import os
import pickle as cp
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_preprocess.base_loader import base_loader
from data_preprocess.data_preprocess_utils import (
    get_sample_weights,
    normalize,
    opp_sliding_window_w_d,
    train_test_val_split,
)


ACTIVITY_ORDER = "ABCDEFGHIJKLMOPQRS"
ACT_TO_IDX = {c: i for i, c in enumerate(ACTIVITY_ORDER)}


class data_loader_wisdm12(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_wisdm12, self).__init__(samples, labels, domains)


@dataclass(frozen=True)
class SensorRow:
    sid: int
    act: str
    ts: float
    xyz: np.ndarray


def _parse_line(line: str) -> Optional[SensorRow]:
    line = line.strip()
    if not line:
        return None
    line = line.rstrip(";").strip()
    parts = line.split(",")
    if len(parts) < 6:
        return None
    try:
        sid = int(parts[0])
        act = parts[1].strip()
        ts = float(parts[2])
        x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
    except ValueError:
        return None
    if act not in ACT_TO_IDX:
        return None
    return SensorRow(sid=sid, act=act, ts=ts, xyz=np.array([x, y, z], dtype=np.float32))


def _read_stream(path: Path) -> List[SensorRow]:
    rows: List[SensorRow] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_line(line)
            if parsed is not None:
                rows.append(parsed)
    rows.sort(key=lambda r: r.ts)
    return rows


def _resolve_wisdm_root(cli_dir: Optional[str]) -> Path:
    candidates: List[str] = []
    if cli_dir:
        candidates.append(str(cli_dir))
    env_dir = os.environ.get("WISDM_DATA_DIR", "").strip()
    if env_dir:
        candidates.append(env_dir)
    candidates.extend(
        [
            "/home/sriramkannan_umass_edu/690R-BioMarkers/wisdm-dataset",
            "./dataset/WISDM_ar_v1.1",
            "./wisdm-dataset",
        ]
    )
    for c in candidates:
        p = Path(c).expanduser().resolve()
        if p.is_dir() and (p / "raw").is_dir():
            return p
    raise FileNotFoundError(
        "WISDM root not found. Set --wisdm_data_dir or WISDM_DATA_DIR to dataset root."
    )


def _discover_subject_ids(root: Path) -> List[int]:
    pat = str(root / "raw" / "watch" / "accel" / "data_*_accel_watch.txt")
    ids: List[int] = []
    for p in sorted(glob.glob(pat)):
        m = re.search(r"data_(\d+)_accel_watch", Path(p).name)
        if m:
            ids.append(int(m.group(1)))
    return ids


def _paths_for_sid(root: Path, sid: int) -> dict:
    return {
        "watch_accel": root / "raw" / "watch" / "accel" / f"data_{sid}_accel_watch.txt",
        "watch_gyro": root / "raw" / "watch" / "gyro" / f"data_{sid}_gyro_watch.txt",
        "phone_accel": root / "raw" / "phone" / "accel" / f"data_{sid}_accel_phone.txt",
        "phone_gyro": root / "raw" / "phone" / "gyro" / f"data_{sid}_gyro_phone.txt",
    }


def _interp_xyz_on_times(
    stream: List[SensorRow], times_master: np.ndarray
) -> Optional[np.ndarray]:
    if len(times_master) == 0:
        return None
    # WISDM timestamps are often in ns-scale integers; a fixed +/-0.5 tolerance
    # is far too tight and can drop nearly all cross-device matches.
    # Use a dynamic pad based on the master's median dt.
    if len(times_master) >= 2:
        dt = np.diff(times_master.astype(np.float64))
        dt = dt[dt > 0]
        pad = float(np.median(dt)) if dt.size > 0 else 0.0
    else:
        pad = 0.0
    if pad <= 0:
        pad = max(abs(float(times_master[-1]) - float(times_master[0])) * 0.01, 1.0)
    t0 = float(times_master[0]) - pad
    t1 = float(times_master[-1]) + pad
    # Do NOT require matching activity labels across devices.
    # In WISDM, phone/watch activity tags can desync around boundaries.
    # We align by timestamp range and interpolate onto the master timeline.
    rows = [r for r in stream if t0 <= r.ts <= t1]
    if len(rows) < 2:
        return None
    ts = np.array([r.ts for r in rows], dtype=np.float64)
    order = np.argsort(ts)
    ts = ts[order]
    xyz = np.stack([rows[i].xyz for i in order], axis=0).astype(np.float64)
    tm = times_master.astype(np.float64)
    out = np.zeros((len(tm), 3), dtype=np.float32)
    for k in range(3):
        out[:, k] = np.interp(tm, ts, xyz[:, k]).astype(np.float32)
    return out


def _fused_segments_to_windows(
    watch_accel: List[SensorRow],
    watch_gyro: List[SensorRow],
    phone_accel: List[SensorRow],
    phone_gyro: List[SensorRow],
    window_len: int,
    stride_len: int,
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    windows: List[np.ndarray] = []
    labels: List[int] = []
    domains: List[int] = []
    if not watch_accel:
        return windows, labels, domains

    segs: List[List[SensorRow]] = []
    cur: List[SensorRow] = []
    last_act: Optional[str] = None
    for row in watch_accel:
        if last_act is not None and row.act != last_act:
            segs.append(cur)
            cur = []
        cur.append(row)
        last_act = row.act
    if cur:
        segs.append(cur)

    for seg in segs:
        if len(seg) < window_len:
            continue
        act = seg[0].act
        if act not in ACT_TO_IDX:
            continue
        sid = seg[0].sid
        label = ACT_TO_IDX[act]
        ts_arr = np.array([r.ts for r in seg], dtype=np.float64)
        wa = np.stack([r.xyz for r in seg], axis=0).astype(np.float32)
        pa = _interp_xyz_on_times(phone_accel, ts_arr)
        pg = _interp_xyz_on_times(phone_gyro, ts_arr)
        wg = _interp_xyz_on_times(watch_gyro, ts_arr)
        if pa is None or pg is None or wg is None:
            continue

        fused = np.concatenate([pa, pg, wa, wg], axis=1)  # [L,12]
        L = fused.shape[0]
        for start in range(0, L - window_len + 1, stride_len):
            sl = slice(start, start + window_len)
            windows.append(fused[sl].astype(np.float32))  # [T,12]
            labels.append(label)
            domains.append(int(sid))

    return windows, labels, domains


def _build_cache(
    root: Path,
    cache_file: Path,
    window_len: int,
    stride_len: int,
    split_ratio: float,
    max_subjects: Optional[int] = None,
    seed: int = 42,
):
    sids = _discover_subject_ids(root)
    rng = random.Random(seed)
    rng.shuffle(sids)
    if max_subjects is not None and max_subjects > 0:
        sids = sids[: int(max_subjects)]

    x_all: List[np.ndarray] = []
    y_all: List[int] = []
    d_all: List[int] = []

    for sid in sids:
        p = _paths_for_sid(root, sid)
        if not all(pp.is_file() for pp in p.values()):
            continue
        wa = _read_stream(p["watch_accel"])
        wg = _read_stream(p["watch_gyro"])
        pa = _read_stream(p["phone_accel"])
        pg = _read_stream(p["phone_gyro"])

        xs, ys, ds = _fused_segments_to_windows(wa, wg, pa, pg, window_len, stride_len)
        x_all.extend(xs)
        y_all.extend(ys)
        d_all.extend(ds)

    if not x_all:
        raise RuntimeError("No fused windows found. Check WISDM raw paths and files.")

    x = np.stack(x_all, axis=0).astype(np.float32)  # [N,T,12]
    y = np.array(y_all, dtype=np.uint8)
    d_orig = np.array(d_all, dtype=np.int32)

    unique_subjects = sorted(np.unique(d_orig))
    sid_to_domain = {sid: i for i, sid in enumerate(unique_subjects)}
    d = np.array([sid_to_domain[int(sid)] for sid in d_orig], dtype=np.int32)

    x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = train_test_val_split(
        x, y, d, split_ratio=split_ratio
    )
    x_train = normalize(x_train)
    x_val = normalize(x_val)
    x_test = normalize(x_test)

    obj = [(x_train, y_train, d_train), (x_val, y_val, d_val), (x_test, y_test, d_test)]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)


def prep_wisdm_fused12(
    args,
    nominal_hz: float = 20.0,
    window_seconds: float = 10.0,
    stride_seconds: float = 5.0,
):
    window_len = int(round(window_seconds * nominal_hz))
    stride_len = int(round(stride_seconds * nominal_hz))
    root = _resolve_wisdm_root(getattr(args, "wisdm_data_dir", None))
    cache_dir = Path("./data/wisdm12")
    cache_file = cache_dir / (
        f"wisdm_fused12_ws{window_len}_ss{stride_len}"
        f"_sr{int(nominal_hz)}_split{getattr(args, 'split_ratio', 0.2)}.data"
    )

    if not cache_file.is_file():
        _build_cache(
            root=root,
            cache_file=cache_file,
            window_len=window_len,
            stride_len=stride_len,
            split_ratio=float(getattr(args, "split_ratio", 0.2)),
            max_subjects=getattr(args, "max_subjects", None),
            seed=int(getattr(args, "seed", 42)),
        )

    data = np.load(str(cache_file), allow_pickle=True)
    x_train, y_train, d_train = data[0]
    x_val, y_val, d_val = data[1]
    x_test, y_test, d_test = data[2]

    args.n_feature = 12
    args.len_sw = window_len
    args.n_class = int(len(np.unique(y_train)))

    unique_y, counts_y = np.unique(y_train, return_counts=True)
    weights = 100.0 / torch.Tensor(counts_y)
    weights = weights.double()
    sample_weights = get_sample_weights(y_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_set = data_loader_wisdm12(x_train, y_train, d_train)
    val_set = data_loader_wisdm12(x_val, y_val, d_val)
    test_set = data_loader_wisdm12(x_test, y_test, d_test)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return [train_loader], val_loader, test_loader
