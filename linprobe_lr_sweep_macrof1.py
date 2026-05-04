#!/usr/bin/env python3
"""
Sweep linear-probe learning rates for one or more SimCLR checkpoints (backbones),
report test macro-F1 (and accuracy), save CSV + line plot + radar/spider chart.

Example (WISDM Watch, fdiff — match your pretrain flags):

  python linprobe_lr_sweep_macrof1.py \\
    --pair SFCN results/pretrain_..._sfcn.pt \\
    --pair SResNet1D results/pretrain_..._sr.pt \\
    --device Watch --wisdm_feat fdiff --batch_size 64 \\
    --p 128 --aug1 jit_scal --aug2 perm_jit \\
    --simclr_contrastive supcon --supcon_temperature 0.07 \\
    --lr_grid 1e-4,3e-4,1e-3,3e-3 \\
    --lincls_epochs 80 --lincls_head mlp

Outputs under results/linprobe_lr_sweep_<timestamp>/:
  sweep_metrics.csv, lr_vs_macrof1_lines.png, lr_vs_macrof1_radar.png
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

import fitlog

from mlp_lincls_from_simclr_ckpt import _load_checkpoint_state_dict, _short_model_name
from trainer import (
    lock_backbone,
    setup_dataloaders,
    setup_linclf,
    setup_model_optm,
    test_lincls,
    train_lincls,
)
from utils import _logger


def _parse_lr_grid(s: str) -> list[float]:
    out = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("empty --lr_grid")
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LR sweep for linear probe macro-F1 (multi-backbone)")
    p.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("BACKBONE", "CKPT"),
        required=True,
        help="Repeat per backbone, e.g. --pair SFCN path.pt --pair SResNet1D path.pt",
    )
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--lr_grid", type=str, default="1e-4,3e-4,1e-3,3e-3")

    p.add_argument("--framework", type=str, default="simclr", choices=["simclr"])
    p.add_argument("--dataset", type=str, default="wisdm", choices=["wisdm"])
    p.add_argument("--cases", type=str, default="random")
    p.add_argument("--device", type=str, default="Watch", choices=["Phones", "Watch"])
    p.add_argument("--wisdm_feat", type=str, default="fdiff", choices=["raw", "fdiff", "l2"])
    p.add_argument(
        "--wisdm_data_dir",
        type=str,
        default="/home/sriramkannan_umass_edu/690R-BioMarkers/wisdm-dataset",
    )
    p.add_argument("--wisdm_fused12", action="store_true")
    p.add_argument("--wisdm_top_k_classes", type=int, default=0)
    p.add_argument("--split_ratio", type=float, default=0.2)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--EMA", type=float, default=0.996)
    p.add_argument("--lr_mul", type=float, default=1.0)
    p.add_argument("--temp_unit", type=str, default="tsfm")
    p.add_argument("--mmb_size", type=int, default=1024)
    p.add_argument("--lambda1", type=float, default=1.0)
    p.add_argument("--lambda2", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.75)
    p.add_argument("--thresh", type=float, default=0.5)
    p.add_argument("--p", type=int, default=128)
    p.add_argument("--phid", type=int, default=128)
    p.add_argument("--aug1", type=str, default="jit_scal")
    p.add_argument("--aug2", type=str, default="perm_jit")
    p.add_argument("--criterion", type=str, default="NTXent")
    p.add_argument("--simclr_contrastive", type=str, default="supcon", choices=["instance", "supcon"])
    p.add_argument("--supcon_temperature", type=float, default=0.07)

    p.add_argument("--vit_patch_size", type=int, default=20)
    p.add_argument("--vit_dim", type=int, default=128)
    p.add_argument("--vit_depth", type=int, default=4)
    p.add_argument("--vit_heads", type=int, default=4)
    p.add_argument("--vit_mlp_dim", type=int, default=256)
    p.add_argument("--vit_dropout", type=float, default=0.1)

    p.add_argument("--lincls_epochs", type=int, default=80)
    p.add_argument("--lincls_head", type=str, default="mlp", choices=["mlp", "linear"])
    p.add_argument("--lincls_hidden_dim", type=int, default=512)
    p.add_argument("--lincls_hidden_dim2", type=int, default=256)
    p.add_argument("--lincls_dropout", type=float, default=0.2)
    p.add_argument("--lincls_finetune_backbone", action="store_true")
    p.add_argument("--lincls_finetune_scope", type=str, default="last_block", choices=["last_block", "all"])
    p.add_argument("--lincls_backbone_lr", type=float, default=1e-4)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--batch_log_every", type=int, default=50)
    p.add_argument("--scheduler", action="store_true", default=True)
    p.add_argument("--input_encoding", type=str, default="none")

    p.add_argument("--strict_load", action="store_true")

    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Directory for CSV/plots (default: results/linprobe_lr_sweep_<timestamp>)",
    )
    return p


def main() -> None:
    if not hasattr(np, "str"):
        np.str = str  # type: ignore[attr-defined]

    parser = _build_parser()
    cli = parser.parse_args()
    lrs = _parse_lr_grid(cli.lr_grid)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cli.out_dir.strip() or os.path.join("results", f"linprobe_lr_sweep_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logdir = os.path.join(out_dir, "log")
    os.makedirs(logdir, exist_ok=True)

    device = torch.device("cuda:" + str(cli.cuda) if torch.cuda.is_available() else "cpu")
    print("[sweep] device:", device, "lr_grid:", lrs)

    base = argparse.Namespace(**vars(cli))
    base.lr = 1e-3
    base.lr_cls = 1e-3  # set per sweep step
    base.lincls_skip_epoch_save = True
    base.save_classifier = ""
    base.logdir = logdir
    base.criterion = "NTXent"
    base.weight_decay = 1e-6

    # Dataloaders depend only on dataset/WISDM settings (not backbone).
    base.backbone = cli.pair[0][0]
    base.checkpoint = cli.pair[0][1]
    train_loaders, val_loader, test_loader = setup_dataloaders(base)

    rows: list[dict[str, object]] = []
    finetune = bool(getattr(cli, "lincls_finetune_backbone", False))

    for backbone_name, ckpt_path in cli.pair:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        base.backbone = backbone_name
        base.checkpoint = ckpt_path

        print(f"\n[sweep] backbone={backbone_name} checkpoint={ckpt_path}")

        trained_backbone = None
        criterion_cls = nn.CrossEntropyLoss()

        for lr in lrs:
            if finetune or trained_backbone is None:
                model, _ = setup_model_optm(base, device, classifier=False)
                ckpt_sd = _load_checkpoint_state_dict(ckpt_path)
                missing, unexpected = model.load_state_dict(ckpt_sd, strict=bool(cli.strict_load))
                if not cli.strict_load:
                    print(f"[load] strict=False missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
                trained_backbone = lock_backbone(model, base)

            base.lr_cls = lr
            base.lincls_skip_epoch_save = True
            base.model_name = _short_model_name(base)
            base.n_epoch = int(base.lincls_epochs)

            classifier = setup_linclf(base, device, trained_backbone.out_dim)
            optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=lr)

            lr_tag = str(lr).replace(".", "p")
            log_path = os.path.join(logdir, f"{backbone_name}_lr{lr_tag}.log")
            logger = _logger(log_path)

            fitlog.set_log_dir(logdir)

            print(f"  [lr] {lr:g} model_name={base.model_name}")
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
                base,
            )

            test_acc, _miF, _maF, macroF = test_lincls(
                test_loader,
                trained_backbone,
                best_head,
                logger,
                fitlog,
                device,
                criterion_cls,
                base,
                plt=False,
            )

            rows.append(
                {
                    "backbone": backbone_name,
                    "lr_cls": lr,
                    "test_acc": float(test_acc),
                    "macro_f1": float(macroF),
                    "checkpoint": ckpt_path,
                    "model_name": base.model_name,
                }
            )
            print(f"    test_acc={test_acc:.2f} macro_f1={macroF:.2f}")

    csv_path = os.path.join(out_dir, "sweep_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["backbone", "lr_cls", "test_acc", "macro_f1"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[sweep] wrote {csv_path}")

    _plot_line_and_radar(rows, out_dir)


def _plot_line_and_radar(rows: list[dict[str, object]], out_dir: str) -> None:
    if not rows:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("[sweep] matplotlib not installed; skipping plots. CSV still saved.", e)
        return

    backbones = sorted({str(r["backbone"]) for r in rows})
    # Line plot: log-scaled LR vs macro F1
    fig, ax = plt.subplots(figsize=(8, 5))
    for bb in backbones:
        sub = [r for r in rows if str(r["backbone"]) == bb]
        sub.sort(key=lambda x: float(x["lr_cls"]))  # type: ignore[arg-type, return-value]
        xs = [float(r["lr_cls"]) for r in sub]  # type: ignore[arg-type]
        ys = [float(r["macro_f1"]) for r in sub]  # type: ignore[arg-type]
        ax.plot(xs, ys, marker="o", label=bb)
    ax.set_xscale("log")
    ax.set_xlabel("Linear probe lr (lr_cls)")
    ax.set_ylabel("Test macro F1 (%)")
    ax.set_title("LR vs macro F1 (linear / MLP probe)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    line_path = os.path.join(out_dir, "lr_vs_macrof1_lines.png")
    fig.savefig(line_path, dpi=160)
    plt.close(fig)
    print(f"[sweep] wrote {line_path}")

    # Radar / spider: one axis per LR (same grid for every backbone)
    grid = sorted({float(r["lr_cls"]) for r in rows})  # type: ignore[arg-type]
    if len(grid) < 2:
        print("[sweep] skip radar (need at least 2 LR points)")
        return

    N = len(grid)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"))
    for bb in backbones:
        by_lr = {float(r["lr_cls"]): float(r["macro_f1"]) for r in rows if str(r["backbone"]) == bb}  # type: ignore[arg-type]
        vals = [by_lr[g] for g in grid]
        if len(vals) != N:
            continue
        vals_closed = vals + vals[:1]
        ax2.plot(angles_closed, vals_closed, linewidth=2, label=bb, marker="o")
        ax2.fill(angles_closed, vals_closed, alpha=0.08)

    ax2.set_xticks(angles)
    ax2.set_xticklabels([f"{g:g}" for g in grid])
    ax2.set_title("Macro F1 across LR (spider / radar)")
    ymin = min(float(r["macro_f1"]) for r in rows)
    ymax = max(float(r["macro_f1"]) for r in rows)
    pad = max(2.0, (ymax - ymin) * 0.1)
    ax2.set_ylim(max(0.0, ymin - pad), min(100.0, ymax + pad))
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig2.tight_layout()
    radar_path = os.path.join(out_dir, "lr_vs_macrof1_radar.png")
    fig2.savefig(radar_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"[sweep] wrote {radar_path}")


if __name__ == "__main__":
    main()
