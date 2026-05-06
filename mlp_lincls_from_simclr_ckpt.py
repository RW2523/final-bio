#!/usr/bin/env python3
"""
Train an MLP classification head on top of a frozen SimCLR-pretrained backbone,
starting from a `results/pretrain_*.pt` checkpoint produced by `trainer.py`.

Typical checkpoints are saved as:
  torch.save({'model_state_dict': model.state_dict()}, path)

This script:
  1) rebuilds the SimCLR wrapper + backbone using the same flags as training
  2) loads `model_state_dict` (strict=False by default; projector keys may differ if dims mismatch)
  3) freezes the encoder (`lock_backbone`)
  4) trains `--lincls_head mlp` via existing `train_lincls` / `test_lincls`

Example (WISDM Watch, SFCN, SupCon SimCLR, 60 pretrain epochs ckpt):
  python mlp_lincls_from_simclr_ckpt.py \\
    --checkpoint results/pretrain_try_scheduler_simclr_pretrain_wisdm_eps60_....pt \\
    --backbone SFCN --device Watch --wisdm_feat fdiff \\
    --batch_size 128 --p 128 --phid 128 --aug1 jit_scal --aug2 perm_jit \\
    --simclr_contrastive supcon --supcon_temperature 0.07 \\
    --lincls_epochs 80 --lr_cls 1e-3 \\
    --lincls_head mlp --lincls_hidden_dim 512 --lincls_hidden_dim2 256 --lincls_dropout 0.2
"""

from __future__ import annotations

import argparse
import hashlib
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

import fitlog

from trainer import (
    lock_backbone,
    setup_dataloaders,
    setup_linclf,
    setup_model_optm,
    test_lincls,
    train_lincls,
)
from utils import _logger


def _short_model_name(args: argparse.Namespace) -> str:
    """
    `train_lincls()` saves files like:
      results/lincls_<model_name><epoch>.pt

    If `<model_name>` embeds the full pretrain checkpoint filename, the path can exceed the OS
    filename limit (you'll see: RuntimeError: File name too long). Keep this string short but unique.
    """
    ckpt_tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
    sig = "|".join(
        [
            ckpt_tag,
            str(args.framework),
            str(args.backbone),
            str(args.dataset),
            str(args.device),
            str(args.wisdm_feat),
            str(args.batch_size),
            str(args.p),
            str(args.phid),
            str(args.aug1),
            str(args.aug2),
            str(args.simclr_contrastive),
            str(args.supcon_temperature),
            str(args.lincls_head),
            str(int(args.lincls_epochs)),
            str(args.lr_cls),
            str(int(args.lincls_hidden_dim)),
            str(int(args.lincls_hidden_dim2)),
            str(float(args.lincls_dropout)),
            str(bool(getattr(args, "lincls_finetune_backbone", False))),
            str(getattr(args, "lincls_finetune_scope", "last_block")),
            str(float(getattr(args, "lincls_backbone_lr", 0.0))),
            str(int(getattr(args, "early_stop_patience", 0))),
            str(float(getattr(args, "early_stop_min_delta", 0.0))),
        ]
    )
    digest = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]
    head = str(getattr(args, "lincls_head", "linear"))
    return f"mlpprobe_{digest}_h{head}_e{int(args.lincls_epochs)}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLP linear-eval from SimCLR pretrain checkpoint")

    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--checkpoint", type=str, required=True, help="Path to results/pretrain_*.pt")

    # Must match the pretraining run enough to rebuild identical modules.
    p.add_argument("--framework", type=str, default="simclr", choices=["simclr"])
    p.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["SFCN", "SResNet1D", "FCN", "ViT1D", "iSpikformer", "SDCL", "SLSTM", "SNN_AE", "SNN_CNN_AE", "SNN_Transformer"],
    )

    p.add_argument("--dataset", type=str, default="wisdm", choices=["wisdm"])
    p.add_argument("--cases", type=str, default="random", choices=["random", "subject", "subject_large", "cross_device", "joint_device"])
    p.add_argument("--device", type=str, default="Phones", choices=["Phones", "Watch"])
    p.add_argument("--wisdm_feat", type=str, default="fdiff", choices=["raw", "fdiff", "l2"])
    p.add_argument(
        "--wisdm_data_dir",
        type=str,
        default="/home/sriramkannan_umass_edu/690R-BioMarkers/wisdm-dataset",
    )
    p.add_argument("--wisdm_fused12", action="store_true")
    p.add_argument("--wisdm_top_k_classes", type=int, default=0)

    # SSL model geometry (must match pretraining for clean loading)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3, help="Must match pretraining if you rely on optimizer state (not used here)")
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--EMA", type=float, default=0.996)
    p.add_argument("--lr_mul", type=float, default=1.0)
    p.add_argument("--temp_unit", type=str, default="tsfm")
    p.add_argument("--mmb_size", type=int, default=1024)
    p.add_argument("--lambda1", type=float, default=1.0)
    p.add_argument("--lambda2", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.75)
    p.add_argument("--thresh", type=float, default=0.5)
    p.add_argument("--p", type=int, default=128, help="SimCLR projection dim (must match checkpoint)")
    p.add_argument("--phid", type=int, default=128, help="unused for SimCLR projector here, kept for parity with training CLI")
    p.add_argument("--aug1", type=str, default="jit_scal")
    p.add_argument("--aug2", type=str, default="perm_jit")
    p.add_argument("--criterion", type=str, default="NTXent")
    p.add_argument("--simclr_contrastive", type=str, default="instance", choices=["instance", "supcon"])
    p.add_argument("--supcon_temperature", type=float, default=0.1)

    # ViT knobs (only used if backbone == ViT1D)
    p.add_argument("--vit_patch_size", type=int, default=20)
    p.add_argument("--vit_dim", type=int, default=128)
    p.add_argument("--vit_depth", type=int, default=4)
    p.add_argument("--vit_heads", type=int, default=4)
    p.add_argument("--vit_mlp_dim", type=int, default=256)
    p.add_argument("--vit_dropout", type=float, default=0.1)

    # Downstream (MLP) training
    p.add_argument("--lincls_epochs", type=int, default=80)
    p.add_argument("--lr_cls", type=float, default=1e-3)
    p.add_argument("--lincls_head", type=str, default="mlp", choices=["mlp", "linear"])
    p.add_argument("--lincls_hidden_dim", type=int, default=512)
    p.add_argument("--lincls_hidden_dim2", type=int, default=256)
    p.add_argument("--lincls_dropout", type=float, default=0.2)
    p.add_argument("--lincls_finetune_backbone", action="store_true")
    p.add_argument("--lincls_finetune_scope", type=str, default="last_block", choices=["last_block", "all"])
    p.add_argument("--lincls_backbone_lr", type=float, default=1e-4)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--batch_log_every", type=int, default=20)
    # Must mirror main_ssl.py / trainer.train_lincls expectations.
    p.add_argument("--scheduler", action="store_true", default=True, help="use cosine LR scheduler during lincls (matches main_ssl.py default)")

    p.add_argument("--input_encoding", type=str, default="none", choices=["none", "zcsf", "arima", "zcsf_arima"])
    p.add_argument("--logdir", type=str, default="log/")

    p.add_argument("--strict_load", action="store_true", help="torch.load strict=True for model.load_state_dict")
    p.add_argument(
        "--save_classifier",
        type=str,
        default="",
        help="Optional path to save best MLP head state_dict (.pt). Default: results/lincls_mlp_from_ckpt_<ts>.pt",
    )
    return p.parse_args()


def _load_checkpoint_state_dict(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and any(k.endswith("weight") for k in obj.keys()):
        # Heuristic: raw state_dict
        return obj
    raise ValueError(f"Unrecognized checkpoint format at {path!r}; expected dict with key 'model_state_dict'.")


def main() -> None:
    args = _parse_args()
    # Be defensive: `train_lincls()` always references args.scheduler.
    if not hasattr(args, "scheduler"):
        args.scheduler = True

    device = torch.device("cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu")
    print("device:", device, "dataset:", args.dataset, "framework:", args.framework, "backbone:", args.backbone)

    # Dataloaders + label space
    train_loaders, val_loader, test_loader = setup_dataloaders(args)

    # Parity with trainer.setup() for SimCLR defaults that affect model_name / losses.
    if args.framework in ["simclr", "nnclr"]:
        args.criterion = "NTXent"
        args.weight_decay = 1e-6

    # `train_lincls()` saves per-epoch heads to results/lincls_<model_name><epoch>.pt
    # and expects `args.model_name` to exist (normally set in trainer.setup()).
    args.model_name = _short_model_name(args)
    print("[run] model_name=", args.model_name)

    # Build SimCLR wrapper without the unused SSL classifier head from setup()
    model, optimizers = setup_model_optm(args, device, classifier=False)

    ckpt_sd = _load_checkpoint_state_dict(args.checkpoint)
    missing, unexpected = model.load_state_dict(ckpt_sd, strict=bool(args.strict_load))
    if not args.strict_load:
        print(f"[load] strict=False missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        if missing:
            print("[load] missing (first 20):", missing[:20])
        if unexpected:
            print("[load] unexpected (first 20):", unexpected[:20])

    trained_backbone = lock_backbone(model, args)

    classifier = setup_linclf(args, device, trained_backbone.out_dim)
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr_cls)
    criterion_cls = nn.CrossEntropyLoss()

    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = os.path.join(args.logdir, f"mlp_from_simclr_ckpt_{ts}.log")
    logger = _logger(log_name)
    logger.debug(args)

    fitlog.set_log_dir(args.logdir)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)

    args.n_epoch = int(args.lincls_epochs)
    best_head = train_lincls(
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

    # Reload best head weights for reporting (matches main_ssl.py flow)
    if isinstance(best_head, dict) and "state_dict" in best_head:
        classifier.load_state_dict(best_head["state_dict"])
    else:
        classifier.load_state_dict(best_head)
    test_acc, miF, maF, macroF = test_lincls(
        test_loader,
        trained_backbone,
        best_head,
        logger,
        fitlog,
        device,
        criterion_cls,
        args,
        plt=False,
    )

    out_path = args.save_classifier.strip()
    if not out_path:
        os.makedirs("results", exist_ok=True)
        out_path = os.path.join("results", f"lincls_mlp_from_ckpt_{ts}.pt")
    to_save = best_head["state_dict"] if isinstance(best_head, dict) and "state_dict" in best_head else best_head
    torch.save({"classifier_state_dict": to_save, "lincls_payload": best_head, "args": vars(args)}, out_path)
    print(f"Final Test Acc: {test_acc:.4f}")
    print(f"Final miF: {miF:.4f}")
    print(f"Final maF (weighted F1): {maF:.4f}")
    print(f"Final macro F1: {macroF:.4f}")
    print(f"Saved MLP head to: {out_path}")


if __name__ == "__main__":
    # trainer.py mutates np.str for older fitlog/numpy combos; keep same guard for subprocess imports.
    if not hasattr(np, "str"):
        np.str = str  # type: ignore[attr-defined]
    main()
