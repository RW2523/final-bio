"""
WISDM preprocessing with IMU-physics–inspired features and optional SNN-style deltas.

Reuses the same raw file layout and windowing as data_preprocess_wisdm.py:
  <root>/raw/phone|watch/accel|gyro/data_*_{accel,gyro}.txt

IMU model (strapdown, body frame; WISDM order: x, y, z as in the dataset files):
  - Accelerometer measures *specific force* a (m/s^2), including gravity.
  - Gyroscope measures angular rate ω (rad/s) about the same body axes.

Physics-inspired steps (per continuous activity segment; state is reset on activity change):
  1) Gravity proxy: exponential low-pass of a (time constant τ) → g_est.
  2) Linear acceleration: a_lin = a - g_est (removes slow gravity for dynamic motion;
     still an approximation when orientation changes quickly without τ tuning).
  3) Jerk: j = (a[t] - a[t-1]) / Δt (finite difference on raw specific force).
  4) Tilt (slow): pitch, roll from normalized g_est (static / quasi-static direction).
  5) |ω| as rotational intensity.

Default stack "physics12" = [a_lin(3), ω(3), jerk(3), |ω|, pitch, roll] → 12 channels.

Optionally append raw specific-force accelerometer a(3) for ablations (--with-raw) → 15 channels.

Optional --stream-fdiff: first-difference each channel in time (TR-style change coding)
before windowing, like the AbouHassan et al. threshold-encoding motivation.

Output: same pickle as prep_wisdm: list of 3 tuples
  [(x_train,y_train,d_train), (x_val,y_val,d_val), (x_test,y_test,d_test)]
for float32 windows (N, T, C), uint8 labels, uint8 domain indices.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle as cp
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running as `python wisdm_imu_physics_preprocess.py` from this directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from data_preprocess.data_preprocess_utils import (  # noqa: E402
    normalize,
    opp_sliding_window_w_d,
    train_test_val_split,
)
from data_preprocess.data_preprocess_wisdm import (  # noqa: E402
    ACTIVITY_TO_ID,
    _load_sensor_file,
    _resolve_wisdm_root,
    _sensor_dirs,
)


def _fdiff_pad(x: np.ndarray, axis: int = 0) -> np.ndarray:
    if x.shape[0] < 2:
        return x.astype(np.float32, copy=False)
    d = np.diff(x, axis=axis)
    return np.vstack([x[:1], d]).astype(np.float32, copy=False)


def _temporal_laplacian(x: np.ndarray) -> np.ndarray:
    """Second-order finite difference along time with edge padding."""
    if x.shape[0] < 3:
        return np.zeros_like(x, dtype=np.float32)
    lap = np.zeros_like(x, dtype=np.float32)
    lap[1:-1] = x[2:] - 2.0 * x[1:-1] + x[:-2]
    lap[0] = lap[1]
    lap[-1] = lap[-2]
    return lap


def _ema_lowpass(
    x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    IIR exponential moving average along time (axis 0).
    y[t] = (1-a)*y[t-1] + a*x[t], y[0] = x[0].
    """
    if x.shape[0] == 0:
        return x
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    oa = 1.0 - alpha
    for t in range(1, x.shape[0]):
        y[t] = oa * y[t - 1] + alpha * x[t]
    return y


def _kalman_1d(z: np.ndarray, q: float, r: float) -> np.ndarray:
    """
    Minimal 1D Kalman filter for a scalar measurement stream.
    Assumes x_k = x_{k-1} + w, z_k = x_k + v.
    """
    out = np.empty_like(z, dtype=np.float32)
    xhat = np.float32(z[0])
    p = np.float32(1.0)
    qf = np.float32(max(q, 1e-8))
    rf = np.float32(max(r, 1e-8))
    out[0] = xhat
    for i in range(1, z.shape[0]):
        p = p + qf
        k = p / (p + rf)
        xhat = xhat + k * (np.float32(z[i]) - xhat)
        p = (1.0 - k) * p
        out[i] = xhat
    return out


def _kalman_filter_multichannel(x: np.ndarray, q: float = 1e-4, r: float = 1e-2) -> np.ndarray:
    """Apply independent 1D Kalman filter to each channel."""
    y = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[1]):
        y[:, c] = _kalman_1d(x[:, c], q=q, r=r)
    return y


def _infer_dt_seconds(timestamps: np.ndarray, nominal_hz: float) -> float:
    if timestamps.size < 2:
        return 1.0 / max(nominal_hz, 1e-6)
    d = np.diff(timestamps.astype(np.float64))
    d = d[d > 0]
    if d.size == 0:
        return 1.0 / max(nominal_hz, 1e-6)
    # WISDM timestamps are often in nanoseconds
    med = float(np.median(d))
    if med > 1e6:  # ns
        return med * 1e-9
    if med > 1e3:  # µs
        return med * 1e-6
    if med < 1e-3:  # already seconds
        return max(med, 1e-4)
    return 1.0 / max(nominal_hz, 1e-6)


def _tilt_pitch_roll_from_gravity(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    g: (T,3) gravity direction estimate in body frame.
    Uses common phone-axis convention: pitch/roll from components of g_hat.
    """
    eps = 1e-8
    gn = g / (np.linalg.norm(g, axis=1, keepdims=True) + eps)
    gx, gy, gz = gn[:, 0], gn[:, 1], gn[:, 2]
    pitch = np.arctan2(-gx, np.sqrt(gy * gy + gz * gz) + eps)
    roll = np.arctan2(gy, gz + eps)
    return pitch.astype(np.float32), roll.astype(np.float32)


def compute_physics_features(
    acc: np.ndarray,
    gyro: np.ndarray,
    timestamps: np.ndarray,
    gravity_tau: float = 0.5,
    nominal_hz: float = 20.0,
) -> dict[str, np.ndarray]:
    """
    acc, gyro: (T,3) float32
    timestamps: (T,) monotonic within segment
    """
    acc = np.asarray(acc, dtype=np.float32)
    gyro = np.asarray(gyro, dtype=np.float32)
    ts = np.asarray(timestamps, dtype=np.float64)
    dt = _infer_dt_seconds(ts, nominal_hz)
    # EMA: α = dt / (τ + dt)  single-pole low-pass
    tau = max(float(gravity_tau), 1e-4)
    alpha = dt / (tau + dt)
    g_est = _ema_lowpass(acc, alpha)
    a_lin = acc - g_est
    jerk = np.vstack(
        [np.zeros((1, 3), dtype=np.float32), np.diff(acc, axis=0)]
    ).astype(np.float32) / (dt + 1e-12)
    w_norm = np.linalg.norm(gyro, axis=1).astype(np.float32)
    pitch, roll = _tilt_pitch_roll_from_gravity(g_est)
    return {
        "a_lin": a_lin,
        "gyro": gyro,
        "jerk": jerk,
        "g_est": g_est,
        "w_norm": w_norm,
        "pitch": pitch,
        "roll": roll,
        "dt": np.float32(dt),
    }


def _stack_physics12(feat: dict[str, np.ndarray], with_raw: bool) -> np.ndarray:
    a_lin = feat["a_lin"]
    gyro = feat["gyro"]
    jerk = feat["jerk"]
    wn = feat["w_norm"][:, None]
    pitch = feat["pitch"][:, None]
    roll = feat["roll"][:, None]
    parts: list[np.ndarray] = [a_lin, gyro, jerk, wn, pitch, roll]
    if with_raw:
        acc = feat.get("acc_raw")
        if acc is None:
            acc = a_lin + feat["g_est"]
        parts.append(acc)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _adaptive_thresholds(arr: np.ndarray, scale: float) -> np.ndarray:
    std = np.std(arr, axis=0).astype(np.float32)
    return np.maximum(scale * std, 1e-6).astype(np.float32)


def _delta_spike_encode(arr: np.ndarray, theta_scale: float = 0.5) -> np.ndarray:
    """
    Bipolar event encoding:
      +1 if delta > +theta, -1 if delta < -theta, else 0.
    """
    if arr.shape[0] < 2:
        return np.zeros_like(arr, dtype=np.float32)
    delta = np.vstack([np.zeros((1, arr.shape[1]), dtype=np.float32), np.diff(arr, axis=0)])
    theta = _adaptive_thresholds(delta[1:], scale=theta_scale)
    spk = np.zeros_like(delta, dtype=np.float32)
    spk[delta > theta] = 1.0
    spk[delta < -theta] = -1.0
    return spk


def _apply_spike_encoding(arr: np.ndarray, mode: str, theta_scale: float) -> np.ndarray:
    if mode == "none":
        return arr
    if mode == "delta":
        return _delta_spike_encode(arr, theta_scale=theta_scale)
    if mode == "delta_laplacian":
        delta = _delta_spike_encode(arr, theta_scale=theta_scale)
        lap = _delta_spike_encode(_temporal_laplacian(arr), theta_scale=theta_scale)
        return np.concatenate([delta, lap], axis=1).astype(np.float32, copy=False)
    raise ValueError(f"Unknown spike encoding mode: {mode}")


def _merge_accel_gyro(
    acc_path: str, gyro_path: str
) -> Optional[pd.DataFrame]:
    df_acc = _load_sensor_file(acc_path).rename(
        columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"}
    )
    df_gyr = _load_sensor_file(gyro_path).rename(
        columns={"x": "gyr_x", "y": "gyr_y", "z": "gyr_z"}
    )
    key_cols = ["subject", "activity", "timestamp"]
    df_acc["dup_idx"] = df_acc.groupby(key_cols).cumcount()
    df_gyr["dup_idx"] = df_gyr.groupby(key_cols).cumcount()
    df = pd.merge(
        df_acc, df_gyr, on=key_cols + ["dup_idx"], how="inner"
    ).drop(columns=["dup_idx"])
    if df.empty:
        return None
    return df


def _build_windows_for_subject(
    acc_path: str,
    gyro_path: str,
    ws: int,
    ss: int,
    gravity_tau: float,
    nominal_hz: float,
    with_raw: bool,
    stream_fdiff: bool,
    filter_mode: str,
    kalman_q: float,
    kalman_r: float,
    spike_enc: str,
    spike_theta_scale: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    df = _merge_accel_gyro(acc_path, gyro_path)
    if df is None:
        return None, None, None
    x_all, y_all, d_all = [], [], []
    subject_id = int(df["subject"].iloc[0])
    for activity_code, group in df.groupby("activity", sort=False):
        t = group["timestamp"].to_numpy()
        acc = group[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=np.float32)
        gyr = group[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(dtype=np.float32)
        if acc.shape[0] < 2 or acc.shape[0] < ws:
            continue
        if filter_mode == "kalman":
            acc = _kalman_filter_multichannel(acc, q=kalman_q, r=kalman_r)
            gyr = _kalman_filter_multichannel(gyr, q=kalman_q, r=kalman_r)
        feat = compute_physics_features(
            acc, gyr, t, gravity_tau=gravity_tau, nominal_hz=nominal_hz
        )
        # expose raw for optional stacking
        feat["acc_raw"] = acc
        arr = _stack_physics12(feat, with_raw=with_raw)
        if stream_fdiff:
            arr = _fdiff_pad(arr, axis=0)
        arr = _apply_spike_encoding(arr, mode=spike_enc, theta_scale=spike_theta_scale)
        labels = np.full(
            (arr.shape[0],), ACTIVITY_TO_ID[activity_code], dtype=np.uint8
        )
        domains = np.full((arr.shape[0],), subject_id, dtype=np.int32)
        x_win, y_win, d_win = opp_sliding_window_w_d(
            arr, labels, domains, ws, ss
        )
        x_all.append(x_win)
        y_all.append(y_win)
        d_all.append(d_win)
    if len(x_all) == 0:
        return None, None, None
    return (
        np.concatenate(x_all, axis=0),
        np.concatenate(y_all, axis=0),
        np.concatenate(d_all, axis=0),
    )


def _first_subject_activity_paths(accel_dir: str, gyro_dir: str) -> Tuple[str, str]:
    acc_files = sorted(
        f for f in os.listdir(accel_dir) if f.startswith("data_") and f.endswith(".txt")
    )
    if not acc_files:
        raise RuntimeError("No accel files found for plotting.")
    for acc_file in acc_files:
        gyr_name = acc_file.replace("accel", "gyro")
        acc_path = os.path.join(accel_dir, acc_file)
        gyr_path = os.path.join(gyro_dir, gyr_name)
        if os.path.isfile(gyr_path):
            return acc_path, gyr_path
    raise RuntimeError("No matching accel/gyro pair found for plotting.")


def _save_diagnostic_plot(
    acc_path: str,
    gyro_path: str,
    out_png: str,
    max_points: int,
    gravity_tau: float,
    nominal_hz: float,
    filter_mode: str,
    kalman_q: float,
    kalman_r: float,
    with_raw: bool,
    stream_fdiff: bool,
    spike_enc: str,
    spike_theta_scale: float,
) -> None:
    df = _merge_accel_gyro(acc_path, gyro_path)
    if df is None or df.empty:
        return
    activity_code = str(df["activity"].iloc[0])
    g = df[df["activity"] == activity_code]
    if g.empty:
        return
    t = g["timestamp"].to_numpy()
    acc = g[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=np.float32)
    gyr = g[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(dtype=np.float32)
    if filter_mode == "kalman":
        acc_f = _kalman_filter_multichannel(acc, q=kalman_q, r=kalman_r)
        gyr_f = _kalman_filter_multichannel(gyr, q=kalman_q, r=kalman_r)
    else:
        acc_f, gyr_f = acc, gyr
    feat = compute_physics_features(acc_f, gyr_f, t, gravity_tau=gravity_tau, nominal_hz=nominal_hz)
    feat["acc_raw"] = acc_f
    arr = _stack_physics12(feat, with_raw=with_raw)
    if stream_fdiff:
        arr = _fdiff_pad(arr, axis=0)
    arr_enc = _apply_spike_encoding(arr, mode=spike_enc, theta_scale=spike_theta_scale)
    n = min(max_points, acc.shape[0], arr_enc.shape[0])
    xs = np.arange(n)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(xs, acc[:n, 0], label="acc_x raw", alpha=0.7)
    axes[0].plot(xs, acc_f[:n, 0], label="acc_x filtered", alpha=0.9)
    axes[0].plot(xs, feat["g_est"][:n, 0], label="g_est_x", alpha=0.9)
    axes[0].set_title(f"WISDM diagnostics (activity={activity_code})")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, feat["a_lin"][:n, 0], label="a_lin_x")
    axes[1].plot(xs, feat["jerk"][:n, 0], label="jerk_x")
    axes[1].plot(xs, feat["w_norm"][:n], label="|w|")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(xs, feat["pitch"][:n], label="pitch")
    axes[2].plot(xs, feat["roll"][:n], label="roll")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    c0 = min(0, arr_enc.shape[1] - 1)
    c1 = min(1, arr_enc.shape[1] - 1)
    axes[3].plot(xs, arr_enc[:n, c0], label=f"enc_ch{c0}")
    axes[3].plot(xs, arr_enc[:n, c1], label=f"enc_ch{c1}")
    axes[3].set_xlabel("time index")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def channel_names(with_raw: bool, spike_enc: str) -> list[str]:
    base = [
        "a_lin_x",
        "a_lin_y",
        "a_lin_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "jerk_x",
        "jerk_y",
        "jerk_z",
        "|ω|",
        "pitch",
        "roll",
    ]
    if with_raw:
        base = base + ["a_spec_x", "a_spec_y", "a_spec_z"]
    if spike_enc == "delta":
        return [f"{n}_spk" for n in base]
    if spike_enc == "delta_laplacian":
        return [f"{n}_dspk" for n in base] + [f"{n}_lspk" for n in base]
    return base


def main() -> None:
    ap = argparse.ArgumentParser(
        description="WISDM IMU-physics preprocessing to pickle (same format as data_preprocess_wisdm)."
    )
    ap.add_argument(
        "--wisdm_data_dir",
        type=str,
        default="",
        help="Root of WISDM AR v1.1 (raw/phone/accel, ...). Or set WISDM_DATA_DIR.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="Phones",
        choices=("Phones", "Watch"),
    )
    ap.add_argument(
        "--window",
        type=int,
        default=200,
        help="Sliding window length (samples).",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Sliding window step (samples).",
    )
    ap.add_argument(
        "--gravity-tau",
        type=float,
        default=0.5,
        help="EMA time constant (seconds) for gravity estimate. Larger → slower g_est ≈ gravity.",
    )
    ap.add_argument(
        "--nominal-hz",
        type=float,
        default=20.0,
        help="Used only if timestamps are unusable, for Δt fallback (WISDM is often 20 Hz).",
    )
    ap.add_argument(
        "--with-raw",
        action="store_true",
        help="Append raw specific-force a_spec (3) after physics stack → 15 channels (compare to linear acc).",
    )
    ap.add_argument(
        "--stream-fdiff",
        action="store_true",
        help="Apply per-channel first difference in time (with first-row pad) before windowing.",
    )
    ap.add_argument(
        "--filter",
        type=str,
        default="none",
        choices=("none", "kalman"),
        help="Optional denoising on accel/gyro before feature extraction.",
    )
    ap.add_argument(
        "--kalman-q",
        type=float,
        default=1e-4,
        help="Kalman process noise (when --filter kalman).",
    )
    ap.add_argument(
        "--kalman-r",
        type=float,
        default=1e-2,
        help="Kalman measurement noise (when --filter kalman).",
    )
    ap.add_argument(
        "--spike-enc",
        type=str,
        default="none",
        choices=("none", "delta", "delta_laplacian"),
        help="Spike/event encoding mode applied after feature stack.",
    )
    ap.add_argument(
        "--spike-theta-scale",
        type=float,
        default=0.5,
        help="Adaptive threshold scale for spike encoding: theta = scale * channel_std(delta).",
    )
    ap.add_argument(
        "--split-ratio",
        type=float,
        default=0.2,
        help="Test set fraction; validation split uses same ratio on remaining train (matches data_preprocess_utils).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output .data pickle path. Default: ./data/wisdm/physics_....data",
    )
    ap.add_argument(
        "--plot-signals",
        action="store_true",
        help="Save a diagnostic signal plot (raw/filter/physics/encoded) for one activity segment.",
    )
    ap.add_argument(
        "--plot-max-points",
        type=int,
        default=1200,
        help="Max timesteps to draw in diagnostic plot.",
    )
    ap.add_argument(
        "--plot-out",
        type=str,
        default="",
        help="Optional custom path for diagnostic plot PNG.",
    )
    args = ap.parse_args()
    root_dir = _resolve_wisdm_root(
        args.wisdm_data_dir or None
    )
    accel_dir, gyro_dir = _sensor_dirs(root_dir, args.device)
    out_dir = "./data/wisdm"
    os.makedirs(out_dir, exist_ok=True)
    tag = f"physics{'15' if args.with_raw else '12'}"
    if args.filter != "none":
        tag += f"_{args.filter}"
    if args.stream_fdiff:
        tag += "_fdiff"
    if args.spike_enc != "none":
        tag += f"_{args.spike_enc}_th{args.spike_theta_scale}"
    cache_file = args.out or os.path.join(
        out_dir,
        f'wisdm_{args.device.lower()}_{tag}_g{args.gravity_tau}_ws{args.window}_ss{args.stride}.data',
    )
    cache_parent = os.path.dirname(os.path.abspath(cache_file))
    if cache_parent:
        os.makedirs(cache_parent, exist_ok=True)

    x_all, y_all, d_all = [], [], []
    for acc_file in sorted(
        f
        for f in os.listdir(accel_dir)
        if f.startswith("data_") and f.endswith(".txt")
    ):
        gyr_name = acc_file.replace("accel", "gyro")
        acc_path = os.path.join(accel_dir, acc_file)
        gyr_path = os.path.join(gyro_dir, gyr_name)
        if not os.path.isfile(gyr_path):
            continue
        x, y, d = _build_windows_for_subject(
            acc_path,
            gyr_path,
            ws=args.window,
            ss=args.stride,
            gravity_tau=args.gravity_tau,
            nominal_hz=args.nominal_hz,
            with_raw=bool(args.with_raw),
            stream_fdiff=bool(args.stream_fdiff),
            filter_mode=str(args.filter),
            kalman_q=float(args.kalman_q),
            kalman_r=float(args.kalman_r),
            spike_enc=str(args.spike_enc),
            spike_theta_scale=float(args.spike_theta_scale),
        )
        if x is None:
            continue
        x_all.append(x)
        y_all.append(y)
        d_all.append(d)

    if not x_all:
        raise RuntimeError("No WISDM windows; check paths and WISDM AR v1.1 layout.")

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    d_all = np.concatenate(d_all, axis=0)
    unique_subjects = sorted(np.unique(d_all))
    subject_to_domain = {sid: i for i, sid in enumerate(unique_subjects)}
    d_all = np.array([subject_to_domain[sid] for sid in d_all], dtype=np.int32)

    class Args:
        split_ratio = float(args.split_ratio)

    split_args = Args()
    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        d_train,
        d_val,
        d_test,
    ) = train_test_val_split(
        x_all, y_all, d_all, split_ratio=split_args.split_ratio
    )
    x_train = normalize(x_train)
    x_val = normalize(x_val)
    x_test = normalize(x_test)
    n_class = int(len(np.unique(y_train)))
    obj = [
        (x_train, y_train, d_train),
        (x_val, y_val, d_val),
        (x_test, y_test, d_test),
    ]
    with open(cache_file, "wb") as f:
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
    meta = {
        "root": root_dir,
        "device": args.device,
        "n_channels": int(x_train.shape[2]),
        "n_classes": n_class,
        "windows": f"{x_train.shape[0]}/{x_val.shape[0]}/{x_test.shape[0]}",
        "filter": args.filter,
        "spike_enc": args.spike_enc,
        "spike_theta_scale": float(args.spike_theta_scale),
        "channel_order": channel_names(args.with_raw, args.spike_enc),
    }
    with open(cache_file + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if args.plot_signals:
        default_plot = cache_file.replace(".data", "_diagnostic.png")
        plot_out = args.plot_out or default_plot
        acc_path, gyr_path = _first_subject_activity_paths(accel_dir, gyro_dir)
        _save_diagnostic_plot(
            acc_path=acc_path,
            gyro_path=gyr_path,
            out_png=plot_out,
            max_points=int(args.plot_max_points),
            gravity_tau=float(args.gravity_tau),
            nominal_hz=float(args.nominal_hz),
            filter_mode=str(args.filter),
            kalman_q=float(args.kalman_q),
            kalman_r=float(args.kalman_r),
            with_raw=bool(args.with_raw),
            stream_fdiff=bool(args.stream_fdiff),
            spike_enc=str(args.spike_enc),
            spike_theta_scale=float(args.spike_theta_scale),
        )
        print(f"Saved diagnostic plot: {plot_out}")
    print(
        f"Wrote {cache_file} | C={x_train.shape[2]} T={x_train.shape[1]} | classes={n_class}"
    )
    print(f"  Channel names: {meta['channel_order']}")


if __name__ == "__main__":
    main()
