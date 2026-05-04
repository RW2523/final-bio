from __future__ import annotations

from datetime import datetime
import os

import numpy as np
import torch

if "MPLCONFIGDIR" not in os.environ:
    mpl_root = "/scratch/login" if os.path.isdir("/scratch/login") else "/tmp"
    os.environ["MPLCONFIGDIR"] = os.path.join(mpl_root, "matplotlib-codex")

import main_ssl as _base_main
from input_encoding_research import supported_input_encodings
from trainer_research import (
    delete_files,
    lock_backbone,
    setup,
    setup_dataloaders,
    test,
    test_lincls,
    train,
    train_lincls,
)


parser = _base_main.parser


def _find_action(dest_name):
    for action in parser._actions:
        if getattr(action, "dest", None) == dest_name:
            return action
    return None


def _has_dest(dest_name):
    return _find_action(dest_name) is not None


def _maybe_add_argument(*args, **kwargs):
    dest = kwargs.get("dest")
    if dest is None:
        for arg in args:
            if arg.startswith("--"):
                dest = arg[2:].replace("-", "_")
                break
    if dest and _has_dest(dest):
        return
    parser.add_argument(*args, **kwargs)


input_encoding_action = _find_action("input_encoding")
if input_encoding_action is not None:
    input_encoding_action.choices = supported_input_encodings()
    input_encoding_action.help = (
        "optional preprocessing before forward: legacy encodings plus research-backed "
        "signed Poisson rate, step-forward temporal contrast, and moving-window temporal contrast"
    )


_maybe_add_argument("--bio_source_hz", type=float, default=20.0, help="nominal WISDM source sampling rate before optional resampling")
_maybe_add_argument("--bio_resample_hz", type=float, default=20.0, help="target sampling rate for research preprocessing; 30.0 matches the resampling shown in the reference flow")
_maybe_add_argument("--bio_window_seconds", type=float, default=10.0, help="window length in seconds for research WISDM preprocessing")
_maybe_add_argument("--bio_stride_seconds", type=float, default=5.0, help="window stride in seconds for research WISDM preprocessing")
_maybe_add_argument(
    "--bio_filter_mode",
    type=str,
    default="raw",
    choices=["raw", "motion", "gravity", "motion_gravity"],
    help="research preprocessing mode: raw stream, motion-focused body signal, gravity-focused acceleration, or concatenated motion+gravity features",
)
_maybe_add_argument("--bio_gravity_cutoff_hz", type=float, default=0.25, help="low-pass cutoff used to estimate gravity from accelerometer channels")
_maybe_add_argument("--bio_motion_low_hz", type=float, default=0.25, help="low cutoff for motion-focused band-pass filtering")
_maybe_add_argument("--bio_motion_high_hz", type=float, default=15.0, help="high cutoff for motion-focused band-pass filtering (clamped to Nyquist)")
_maybe_add_argument("--encoding_norm_eps", type=float, default=1e-6, help="epsilon for max-abs input normalization before research encoders")
_maybe_add_argument("--sf_threshold", type=float, default=0.15, help="threshold for step-forward temporal encoding after per-sample max-abs normalization")
_maybe_add_argument("--mw_window", type=int, default=8, help="history window length for moving-window temporal encoding")
_maybe_add_argument("--mw_threshold", type=float, default=0.15, help="threshold for moving-window temporal encoding after per-sample max-abs normalization")


if __name__ == "__main__":
    args = parser.parse_args()
    pretrain_epochs = args.pretrain_epochs if args.pretrain_epochs is not None else args.n_epoch
    lincls_epochs = args.lincls_epochs if args.lincls_epochs is not None else args.n_epoch
    args.n_epoch = pretrain_epochs
    device = torch.device("cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu")
    print("device:", device, "dataset:", args.dataset, "framework:", args.framework, "backbone:", args.backbone)

    training_start = datetime.now()
    test_acc_list = []
    miF_list = []
    maF_list = []
    macroF_list = []

    for r in range(args.rep):
        _base_main.seed_all(seed=1000 + r)
        train_loaders, val_loader, test_loader = setup_dataloaders(args)
        model, optimizers, schedulers, criterion, logger, fitlog, classifier, criterion_cls, optimizer_cls = setup(args, device)

        if not args.eval:
            ssl_ckpt = str(getattr(args, "ssl_ckpt", "") or "").strip()
            if ssl_ckpt:
                print(f"[ssl_ckpt] Loading SSL weights from: {ssl_ckpt}")
                obj = torch.load(ssl_ckpt, map_location=device, weights_only=False)
                if isinstance(obj, dict) and "model_state_dict" in obj:
                    state = obj["model_state_dict"]
                elif isinstance(obj, dict):
                    state = obj
                else:
                    raise TypeError(f"Unsupported checkpoint object type: {type(obj)}")

                strict = bool(getattr(args, "ssl_ckpt_strict", False))
                missing, unexpected = model.load_state_dict(state, strict=strict)
                if (missing or unexpected) and not strict:
                    print(
                        f"[ssl_ckpt] load_state_dict(strict=False): "
                        f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
                    )

                print("[ssl_ckpt] Skipping SSL pretraining (checkpoint provided).")
                trained_ssl = test(test_loader, model.state_dict(), logger, fitlog, device, criterion, args)
                trained_backbone = lock_backbone(trained_ssl, args)
            elif args.skip_pretrain:
                print("Skipping SSL pretraining; using randomly initialized frozen backbone for linear probe.")
                trained_backbone = lock_backbone(model, args)
            else:
                args.n_epoch = pretrain_epochs
                best_model = train(
                    train_loaders,
                    val_loader,
                    model,
                    logger,
                    fitlog,
                    device,
                    optimizers,
                    schedulers,
                    criterion,
                    args,
                )
                trained_ssl = test(test_loader, best_model, logger, fitlog, device, criterion, args)
                trained_backbone = lock_backbone(trained_ssl, args)
        else:
            raise NotImplementedError("--eval mode is not implemented for main_ssl_research.py yet.")

        args.n_epoch = lincls_epochs
        best_lincls = train_lincls(
            train_loaders,
            val_loader,
            trained_backbone,
            classifier,
            logger,
            fitlog,
            device,
            optimizer_cls,
            criterion_cls,
            args,
        )

        test_acc, miF, maF, macroF = test_lincls(
            test_loader,
            trained_backbone,
            best_lincls,
            logger,
            fitlog,
            device,
            criterion_cls,
            args,
            plt=False,
        )
        test_acc_list.append(test_acc)
        miF_list.append(miF)
        maF_list.append(maF)
        macroF_list.append(macroF)

    training_end = datetime.now()
    print("Training time:", training_end - training_start)
    print("Final Test Acc: {:.4f} +/- {:.4f}".format(np.mean(test_acc_list), np.std(test_acc_list)))
    print("Final miF: {:.4f} +/- {:.4f}".format(np.mean(miF_list), np.std(miF_list)))
    print("Final maF (weighted F1): {:.4f} +/- {:.4f}".format(np.mean(maF_list), np.std(maF_list)))
    print("Final macro F1: {:.4f} +/- {:.4f}".format(np.mean(macroF_list), np.std(macroF_list)))

    if args.cleanup_checkpoints:
        delete_files(args)
