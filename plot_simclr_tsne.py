# encoding=utf-8
"""
t-SNE of SimCLR **backbone** features (not the projection head) for WISDM HAR.

Example (SResNet1D + instance SimCLR, epoch 69):
  cd SNN_HAR && source .venv/bin/activate
  python plot_simclr_tsne.py --backbone SResNet1D --simclr_contrastive instance \\
    --checkpoint results/pretrain_try_scheduler_simclr_pretrain_wisdm_eps70_lr0.001_bs64_aug1jit_scal_aug2perm_jit_dim-pdim128-128_EMA0.996_criterion_NTXent_lambda1_1.0_lambda2_1.0_tempunit_tsfm_sclr_instance69.pt \\
    --out plot/tsne_simclr_sresnet1d_wisdm.png

Example (FCN + SupCon from v1Logs, epoch 69):
  python plot_simclr_tsne.py --backbone FCN --simclr_contrastive supcon \\
    --checkpoint results/pretrain_try_scheduler_simclr_pretrain_wisdm_eps70_lr0.001_bs64_aug1jit_scal_aug2perm_jit_dim-pdim128-128_EMA0.996_criterion_NTXent_lambda1_1.0_lambda2_1.0_tempunit_tsfm_sclr_supcon69.pt \\
    --out plot/tsne_simclr_fcn_wisdm.png
"""
from __future__ import annotations

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from data_preprocess.data_preprocess_wisdm import ACTIVITY_TO_ID
from trainer import apply_input_encoding, lock_backbone, setup_dataloaders, setup_model_optm


def _build_model_name(args) -> str:
    _sclr = str(getattr(args, "simclr_contrastive", "instance")) if args.framework == "simclr" else "na"
    return (
        "try_scheduler_"
        + args.framework
        + "_pretrain_"
        + args.dataset
        + "_eps"
        + str(args.n_epoch)
        + "_lr"
        + str(args.lr)
        + "_bs"
        + str(args.batch_size)
        + "_aug1"
        + args.aug1
        + "_aug2"
        + args.aug2
        + "_dim-pdim"
        + str(args.p)
        + "-"
        + str(args.phid)
        + "_EMA"
        + str(args.EMA)
        + "_criterion_"
        + args.criterion
        + "_lambda1_"
        + str(args.lambda1)
        + "_lambda2_"
        + str(args.lambda2)
        + "_tempunit_"
        + args.temp_unit
        + ("_sclr_" + _sclr if args.framework == "simclr" else "")
        + ("_ispf" if str(getattr(args, "backbone", "")) == "iSpikformer" else "")
    )


def _id_to_activity_name(y: int) -> str:
    inv = {v: k for k, v in ACTIVITY_TO_ID.items()}
    return inv.get(int(y), f"id_{y}")


def _collect_backbone_features(args, trained_backbone, loader, device: torch.device):
    feats: list[torch.Tensor] = []
    labels: list[int] = []
    trained_backbone.eval()
    with torch.no_grad():
        for sample, target, _domain in loader:
            sample = sample.to(device).float()
            target = target.to(device).long()
            sample = apply_input_encoding(sample, args)
            _, feat = trained_backbone(sample)
            if feat.dim() == 3:
                feat = feat.reshape(feat.shape[0], -1)
            feats.append(feat.cpu())
            labels.append(target.cpu().numpy())
    return torch.cat(feats, 0), np.concatenate(labels, 0)


def _load_checkpoint_state_dict(path: str, device: torch.device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def _tsne_and_save(
    z: np.ndarray,
    y: np.ndarray,
    out_path: str,
    max_points: int,
    seed: int,
    perplexity: float,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rng = np.random.default_rng(seed)
    n = z.shape[0]
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        z = z[idx]
        y = y[idx]
    n = z.shape[0]
    perp = min(float(perplexity), max(2.0, (n - 1) * 0.9))
    ts = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perp, max_iter=1000, random_state=seed)
    z2 = ts.fit_transform(z.astype(np.float64))
    hue = np.array([_id_to_activity_name(int(t)) for t in y])
    plt.figure(figsize=(14, 10))
    n_class = len(np.unique(hue))
    sns.scatterplot(
        x=z2[:, 0],
        y=z2[:, 1],
        hue=hue,
        palette=sns.color_palette("hls", n_class),
        alpha=0.45,
        s=12,
        linewidth=0,
    )
    plt.title("t-SNE of SimCLR backbone features (WISDM test windows)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path} ({n} points, perplexity={perp:.1f})")


def main():
    p = argparse.ArgumentParser(description="t-SNE for SimCLR backbone embeddings on WISDM")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--checkpoint", type=str, required=True, help="Path to pretrain_*.pt SimCLR full model state dict")
    p.add_argument("--out", type=str, default="plot/tsne_simclr_wisdm.png", help="Output image path")
    p.add_argument("--max_points", type=int, default=8000, help="Subsample for t-SNE speed (None = all)")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=1029)
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Which split to embed (default: test)",
    )
    # Pass-through: same as main_ssl for WISDM SimCLR v1 runs
    p.add_argument("--dataset", type=str, default="wisdm", choices=["wisdm"])
    p.add_argument("--framework", type=str, default="simclr", choices=["simclr"])
    p.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["FCN", "SResNet1D", "Transformer", "ViT1D", "SFCN", "iSpikformer"],
    )
    p.add_argument("--cases", type=str, default="random")
    p.add_argument("--split_ratio", type=float, default=0.2)
    p.add_argument("--target_domain", type=str, default="1600")
    p.add_argument("--device", type=str, default="Phones", choices=["Phones", "Watch"])
    p.add_argument("--wisdm_feat", type=str, default="fdiff", choices=["raw", "fdiff", "l2"])
    p.add_argument("--wisdm_top_k_classes", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epoch", type=int, default=70, help="Pretrain epochs used in model_name (must match checkpoint run)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--p", type=int, default=128)
    p.add_argument("--phid", type=int, default=128)
    p.add_argument("--EMA", type=float, default=0.996)
    p.add_argument("--criterion", type=str, default="NTXent")
    p.add_argument("--aug1", type=str, default="jit_scal")
    p.add_argument("--aug2", type=str, default="perm_jit")
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--temp_unit", type=str, default="tsfm")
    p.add_argument("--lambda1", type=float, default=1.0)
    p.add_argument("--lambda2", type=float, default=1.0)
    p.add_argument("--mmb_size", type=int, default=1024)
    p.add_argument(
        "--simclr_contrastive",
        type=str,
        default="instance",
        choices=["instance", "supcon"],
    )
    p.add_argument("--supcon_temperature", type=float, default=0.1)
    p.add_argument("--use_augpred", action="store_true")
    p.add_argument("--augpred_transforms", type=str, default="na,t_flip,perm,t_warp,jit_scal,noise,scale,rotation")
    p.add_argument("--input_encoding", type=str, default="none")
    p.add_argument("--vit_patch_size", type=int, default=20)
    p.add_argument("--vit_dim", type=int, default=128)
    p.add_argument("--vit_depth", type=int, default=4)
    p.add_argument("--vit_heads", type=int, default=4)
    p.add_argument("--vit_mlp_dim", type=int, default=256)
    p.add_argument("--vit_dropout", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.75)
    p.add_argument("--thresh", type=float, default=0.5)
    p.add_argument("--ispf_dim", type=int, default=512)
    p.add_argument("--ispf_d_ff", type=int, default=0)
    p.add_argument("--ispf_depths", type=int, default=2)
    p.add_argument("--ispf_num_steps", type=int, default=4)
    p.add_argument("--ispf_heads", type=int, default=8)
    p.add_argument("--ispf_encoder", type=str, default="conv", choices=["conv", "repeat", "delta"])
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    args.n_class = 18
    _train_loaders, val_loader, test_loader = setup_dataloaders(args)
    # n_class / cache must match the checkpoint run (set in prep_wisdm)
    args.model_name = _build_model_name(args)
    if args.split == "test":
        loader = test_loader
    elif args.split == "val":
        loader = val_loader
    else:
        loader = _train_loaders[0] if isinstance(_train_loaders, list) else _train_loaders

    model, _ = setup_model_optm(args, device, classifier=False)
    state = _load_checkpoint_state_dict(args.checkpoint, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_state_dict] missing keys: {len(missing)} (showing up to 8) -> {missing[:8]}")
    if unexpected:
        print(f"[load_state_dict] unexpected keys: {len(unexpected)} (showing up to 8) -> {unexpected[:8]}")
    bb = lock_backbone(model, args)

    z, y = _collect_backbone_features(args, bb, loader, device)
    max_pts = args.max_points if args.max_points > 0 else z.shape[0]
    _tsne_and_save(
        z.numpy(),
        y,
        args.out,
        max_points=max_pts,
        seed=args.seed,
        perplexity=args.perplexity,
    )


if __name__ == "__main__":
    main()
