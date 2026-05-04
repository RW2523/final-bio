#!/usr/bin/env python3
"""SSL pretrain + linear eval for fused 12-channel WISDM (phone+watch, accel+gyro)."""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer import (
    delete_files,
    lock_backbone,
    setup,
    test,
    test_lincls,
    train,
    train_lincls,
)
from preprocess_wisdm12 import prep_wisdm_fused12


def seed_all(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_dataloaders_wisdm12(args):
    # Fixed bio-style setup:
    # 20 Hz, 10 s window, 5 s stride => 200 samples/window, 100 stride.
    args.dataset = "wisdm_fused12"
    args.n_feature = 12
    args.len_sw = 200
    args.n_class = 18
    return prep_wisdm_fused12(
        args=args,
        nominal_hz=float(args.nominal_sample_rate_hz),
        window_seconds=float(args.window_seconds),
        stride_seconds=float(args.stride_seconds),
    )


parser = argparse.ArgumentParser(description="SSL pretrain + linear eval for fused 12-channel WISDM")

# runtime
parser.add_argument("--cuda", default=0, type=int)
parser.add_argument("--rep", default=1, type=int)
parser.add_argument("--eval", action="store_true")
parser.add_argument("--cleanup_checkpoints", action="store_true")
parser.add_argument("--seed", type=int, default=42)

# fixed WISDM fused settings (override if needed)
parser.add_argument("--nominal_sample_rate_hz", type=float, default=20.0)
parser.add_argument("--window_seconds", type=float, default=10.0)
parser.add_argument("--stride_seconds", type=float, default=5.0)
parser.add_argument("--wisdm_data_dir", type=str, default="")
parser.add_argument("--max_subjects", type=int, default=None)
parser.add_argument("--split_ratio", type=float, default=0.2)

# optimization
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_epoch", type=int, default=50)
parser.add_argument("--pretrain_epochs", type=int, default=None)
parser.add_argument("--lincls_epochs", type=int, default=None)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_cls", type=float, default=1e-3)
parser.add_argument("--lincls_head", type=str, default="linear", choices=["linear", "mlp"])
parser.add_argument("--lincls_hidden_dim", type=int, default=512)
parser.add_argument("--lincls_hidden_dim2", type=int, default=256)
parser.add_argument("--lincls_dropout", type=float, default=0.2)
parser.add_argument("--lincls_finetune_backbone", action="store_true")
parser.add_argument("--lincls_finetune_scope", type=str, default="last_block", choices=["last_block", "all"])
parser.add_argument("--lincls_backbone_lr", type=float, default=1e-4)
parser.add_argument("--early_stop_patience", type=int, default=0)
parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
parser.add_argument("--scheduler", action="store_true", default=True)
parser.add_argument("--logdir", type=str, default="log/")
parser.add_argument("--batch_log_every", type=int, default=20)

# SSL
parser.add_argument("--framework", type=str, default="simclr", choices=["simclr", "byol"])
parser.add_argument(
    "--backbone",
    type=str,
    default="SResNet1D",
    choices=["SFCN", "SNN_Transformer", "Transformer", "SResNet1D"],
)
parser.add_argument("--criterion", type=str, default="NTXent", choices=["cos_sim", "NTXent"])
parser.add_argument("--aug1", type=str, default="jit_scal")
parser.add_argument("--aug2", type=str, default="perm_jit")
parser.add_argument("--p", type=int, default=128)
parser.add_argument("--phid", type=int, default=128)
parser.add_argument("--EMA", type=float, default=0.996)
parser.add_argument("--lr_mul", type=float, default=1.0)
parser.add_argument("--weight_decay", type=float, default=1e-6)
parser.add_argument("--lambda1", type=float, default=1.0)
parser.add_argument("--lambda2", type=float, default=1.0)
parser.add_argument("--tau", type=float, default=0.75)
parser.add_argument("--thresh", type=float, default=0.5)
parser.add_argument("--input_encoding", type=str, default="none", choices=["none", "zcsf", "arima", "zcsf_arima"])
parser.add_argument("--zcsf_step", type=float, default=0.1)
parser.add_argument("--arima_diff_order", type=int, default=1)
parser.add_argument("--arima_eps", type=float, default=1e-6)
parser.add_argument("--skip_pretrain", action="store_true")

# required by trainer.setup naming/logging paths
parser.add_argument("--dataset", type=str, default="wisdm_fused12")
parser.add_argument("--cases", type=str, default="random")
parser.add_argument("--target_domain", type=str, default="0")
parser.add_argument("--device", type=str, default="Phones")
parser.add_argument("--n_feature", type=int, default=12)
parser.add_argument("--len_sw", type=int, default=200)
parser.add_argument("--n_class", type=int, default=18)
parser.add_argument("--wisdm_feat", type=str, default="raw")
parser.add_argument("--wisdm_top_k_classes", type=int, default=0)
parser.add_argument("--temp_unit", type=str, default="tsfm")
parser.add_argument("--mmb_size", type=int, default=1024)
parser.add_argument("--simclr_contrastive", type=str, default="instance", choices=["instance", "supcon"])
parser.add_argument("--supcon_temperature", type=float, default=0.1)


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
        seed_all(seed=int(args.seed) + r)
        train_loaders, val_loader, test_loader = setup_dataloaders_wisdm12(args)
        model, optimizers, schedulers, criterion, logger, fitlog, classifier, criterion_cls, optimizer_cls = setup(
            args, device
        )

        if not args.eval:
            if args.skip_pretrain:
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
            raise NotImplementedError("--eval mode is not implemented for run_ssl_wisdm12.py yet.")

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
