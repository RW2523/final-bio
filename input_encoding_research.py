from __future__ import annotations

import torch


RESEARCH_ENCODINGS = (
    "poisson_rate",
    "step_forward",
    "moving_window",
    "hybrid_ds_rate",
)

LEGACY_ENCODINGS = (
    "none",
    "zcsf",
    "arima",
    "zcsf_arima",
)


def supported_input_encodings():
    return LEGACY_ENCODINGS + RESEARCH_ENCODINGS


def _normalize_maxabs(x: torch.Tensor, eps: float) -> torch.Tensor:
    scale = torch.amax(torch.abs(x), dim=1, keepdim=True).clamp_min(eps)
    return x / scale


def poisson_rate_encode(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Signed Poisson-style rate code.

    Values are max-abs normalized to [-1, 1]. Positive and negative magnitudes are
    then converted into Bernoulli samples and collapsed back to a bipolar spike stream
    in {-1, 0, +1}, which preserves the [B, T, C] input shape expected by this repo.
    """
    x = _normalize_maxabs(x, eps=eps)
    p_pos = x.clamp(min=0.0, max=1.0)
    p_neg = (-x).clamp(min=0.0, max=1.0)
    pos = torch.bernoulli(p_pos)
    neg = torch.bernoulli(p_neg)
    return pos - neg


def step_forward_encode(
    x: torch.Tensor,
    threshold: float = 0.15,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Bipolar step-forward temporal contrast encoding.

    This mirrors the temporal-contrast family described by Petro et al. and keeps a
    running reference per channel. A spike is emitted whenever the current sample
    crosses the reference by more than `threshold`.
    """
    if threshold <= 0:
        raise ValueError(f"step-forward threshold must be > 0, got {threshold}")

    x = _normalize_maxabs(x, eps=eps)
    encoded = torch.zeros_like(x)
    ref = x[:, 0:1, :].clone()

    for t in range(1, x.shape[1]):
        cur = x[:, t : t + 1, :]
        delta = cur - ref
        pos = (delta >= threshold).to(x.dtype)
        neg = (delta <= -threshold).to(x.dtype)
        spike = pos - neg
        encoded[:, t : t + 1, :] = spike
        ref = ref + spike * threshold

    return encoded


def moving_window_encode(
    x: torch.Tensor,
    window: int = 8,
    threshold: float = 0.15,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Bipolar moving-window temporal contrast encoding.

    The current value is compared against the mean of the previous `window` values.
    Positive/negative spikes are emitted if the current sample exceeds that moving
    baseline by more than `threshold`.
    """
    if window < 1:
        raise ValueError(f"moving-window size must be >= 1, got {window}")
    if threshold <= 0:
        raise ValueError(f"moving-window threshold must be > 0, got {threshold}")

    x = _normalize_maxabs(x, eps=eps)
    encoded = torch.zeros_like(x)
    csum = torch.cumsum(x, dim=1)

    for t in range(1, x.shape[1]):
        start = max(0, t - window)
        hist = csum[:, t - 1 : t, :]
        if start > 0:
            hist = hist - csum[:, start - 1 : start, :]
        count = t - start
        base = hist / max(count, 1)
        cur = x[:, t : t + 1, :]
        pos = (cur >= base + threshold).to(x.dtype)
        neg = (cur <= base - threshold).to(x.dtype)
        encoded[:, t : t + 1, :] = pos - neg

    return encoded


def apply_research_input_encoding(sample: torch.Tensor, args) -> torch.Tensor:
    if sample.dim() != 3:
        raise ValueError(f"Expected input [B, T, C], got shape {tuple(sample.shape)}")

    encoding = str(getattr(args, "input_encoding", "none")).lower()
    eps = float(getattr(args, "encoding_norm_eps", 1e-6))
    if eps <= 0:
        raise ValueError(f"--encoding_norm_eps must be > 0, got {eps}")

    if encoding == "poisson_rate":
        return poisson_rate_encode(sample, eps=eps)
    if encoding == "step_forward":
        return step_forward_encode(
            sample,
            threshold=float(getattr(args, "sf_threshold", 0.15)),
            eps=eps,
        )
    if encoding == "moving_window":
        return moving_window_encode(
            sample,
            window=int(getattr(args, "mw_window", 8)),
            threshold=float(getattr(args, "mw_threshold", 0.15)),
            eps=eps,
        )
    if encoding == "hybrid_ds_rate":
        # Fuse change-sensitive (step-forward / delta-sigma like) and rate-sensitive
        # views into one bipolar stream while keeping shape [B, T, C].
        sf = step_forward_encode(
            sample,
            threshold=float(getattr(args, "sf_threshold", 0.15)),
            eps=eps,
        )
        rate = poisson_rate_encode(sample, eps=eps)
        return (sf + rate).clamp(min=-1.0, max=1.0)

    raise ValueError(f"Unsupported research input encoding: {encoding}")
