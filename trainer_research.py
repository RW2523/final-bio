from __future__ import annotations

import trainer as _base

from data_preprocess import data_preprocess_wisdm_research
from input_encoding_research import RESEARCH_ENCODINGS, apply_research_input_encoding


_LEGACY_APPLY_INPUT_ENCODING = _base.apply_input_encoding
_LEGACY_SETUP_DATALOADERS = _base.setup_dataloaders


def apply_input_encoding(sample, args):
    encoding = str(getattr(args, "input_encoding", "none")).lower()
    if encoding in RESEARCH_ENCODINGS:
        return apply_research_input_encoding(sample, args)
    return _LEGACY_APPLY_INPUT_ENCODING(sample, args)


def setup_dataloaders(args):
    if args.dataset != "wisdm" or bool(getattr(args, "wisdm_fused12", False)):
        return _LEGACY_SETUP_DATALOADERS(args)

    args.n_class = 18
    if args.cases not in ["subject", "subject_large"]:
        args.target_domain = "1600"

    target_hz = float(getattr(args, "bio_resample_hz", 20.0))
    window_seconds = float(getattr(args, "bio_window_seconds", 10.0))
    stride_seconds = float(getattr(args, "bio_stride_seconds", 5.0))
    if target_hz <= 0:
        raise ValueError(f"--bio_resample_hz must be > 0, got {target_hz}")
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError(
            f"--bio_window_seconds and --bio_stride_seconds must be > 0, got {window_seconds}, {stride_seconds}",
        )

    args.len_sw = max(1, int(round(target_hz * window_seconds)))
    sliding_window_step = max(1, int(round(target_hz * stride_seconds)))
    args.n_feature = data_preprocess_wisdm_research.estimate_n_feature(args)

    return data_preprocess_wisdm_research.prep_wisdm(
        args,
        SLIDING_WINDOW_LEN=args.len_sw,
        SLIDING_WINDOW_STEP=sliding_window_step,
        device=args.device,
    )


_base.apply_input_encoding = apply_input_encoding
_base.setup_dataloaders = setup_dataloaders
_base.data_preprocess_wisdm = data_preprocess_wisdm_research


delete_files = _base.delete_files
lock_backbone = _base.lock_backbone
setup = _base.setup
test = _base.test
test_lincls = _base.test_lincls
train = _base.train
train_lincls = _base.train_lincls
