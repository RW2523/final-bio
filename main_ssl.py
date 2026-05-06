# encoding=utf-8
import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch

from trainer import (
    delete_files,
    lock_backbone,
    setup,
    setup_dataloaders,
    test,
    test_lincls,
    train,
    train_lincls,
)


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='SSL pretrain + linear eval for HAR')

# runtime
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID, 0/1')
parser.add_argument('--rep', default=1, type=int, help='repeats for multiple runs')
parser.add_argument('--eval', action='store_true', help='run linear eval only from saved checkpoints')
parser.add_argument(
    '--cleanup_checkpoints',
    action='store_true',
    help='delete per-epoch checkpoints after run (default: keep them)',
)

# dataset
parser.add_argument('--dataset', type=str, default='wisdm', choices=['ucihar', 'shar', 'hhar', 'wisdm'])
parser.add_argument('--cases', type=str, default='random', choices=['random', 'subject', 'subject_large', 'cross_device', 'joint_device'])
parser.add_argument('--split_ratio', type=float, default=0.2, help='split ratio for train_test_val_split')
parser.add_argument('--target_domain', type=str, default='0')
parser.add_argument('--device', type=str, default='Phones', choices=['Phones', 'Watch'])
parser.add_argument('--n_feature', type=int, default=6)
parser.add_argument('--len_sw', type=int, default=200)
parser.add_argument('--n_class', type=int, default=18)
parser.add_argument(
    '--wisdm_feat',
    type=str,
    default='fdiff',
    choices=['raw', 'fdiff', 'l2'],
    help="WISDM feature mode: 'raw', 'fdiff' (time-differenced streams, default), 'l2' (per-window L2 unit norm)",
)
parser.add_argument(
    '--wisdm_top_k_classes',
    type=int,
    default=0,
    help='If >0, keep only the K most frequent WISDM classes (global), then remap labels to 0..K-1.',
)
parser.add_argument(
    '--wisdm_data_dir',
    type=str,
    default='./wisdm-dataset',
    help='WISDM AR v1.1 root (contains raw/phone/accel, .../gyro). Default: ./wisdm-dataset',
)
parser.add_argument(
    '--wisdm_fused12',
    action='store_true',
    help='Use fused 12-channel WISDM (phone+watch accel+gyro) with fixed 20Hz, 10s window, 5s stride.',
)

# optimization
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epoch', type=int, default=50, help='default epochs for pretrain+linear when pretrain_epochs/lincls_epochs are unset')
parser.add_argument('--pretrain_epochs', type=int, default=None, help='epochs for SSL backbone pretraining')
parser.add_argument('--lincls_epochs', type=int, default=None, help='epochs for linear classifier training')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_cls', type=float, default=1e-3)
parser.add_argument('--lincls_head', type=str, default='linear', choices=['linear', 'mlp'], help='head used in downstream evaluation: linear probe or frozen-backbone MLP probe')
parser.add_argument('--lincls_hidden_dim', type=int, default=512, help='MLP probe hidden dim (used when --lincls_head mlp)')
parser.add_argument('--lincls_hidden_dim2', type=int, default=256, help='second MLP probe hidden dim (used when --lincls_head mlp)')
parser.add_argument('--lincls_dropout', type=float, default=0.2, help='dropout for MLP probe (used when --lincls_head mlp)')
parser.add_argument(
    '--lincls_finetune_backbone',
    action='store_true',
    help='unfreeze (part of) the backbone during linear eval training (often helps if frozen features are weak)',
)
parser.add_argument(
    '--lincls_finetune_scope',
    type=str,
    default='last_block',
    choices=['last_block', 'all'],
    help='what to unfreeze when --lincls_finetune_backbone is set (SFCN/FCN: conv_block3, SResNet1D: layer4, ViT1D: last transformer block)',
)
parser.add_argument(
    '--lincls_backbone_lr',
    type=float,
    default=1e-4,
    help='lr for backbone params when finetuning during linear eval (head uses --lr_cls)',
)
parser.add_argument(
    '--lincls_label_smoothing',
    type=float,
    default=0.0,
    help='label smoothing for CE-based linear-head training (ignored by focal losses)',
)
parser.add_argument(
    '--lincls_scheduler',
    type=str,
    default='cosine',
    choices=['cosine', 'onecycle', 'none'],
    help='scheduler for linear-head stage',
)
parser.add_argument(
    '--lincls_grad_clip',
    type=float,
    default=0.0,
    help='if >0, clip global grad norm during linear-head training',
)
parser.add_argument(
    '--lincls_select_metric',
    type=str,
    default='loss',
    choices=['loss', 'macrof1'],
    help='best-checkpoint criterion for linear-head stage',
)
parser.add_argument(
    '--lincls_logit_adjust_tau',
    type=float,
    default=0.0,
    help='if >0, apply logit adjustment with class priors during linear-head train/eval',
)
parser.add_argument(
    '--lincls_calibrate_temperature',
    action='store_true',
    help='sweep temperature on validation set after linear-head training to maximize macro-F1',
)
parser.add_argument('--lincls_temp_min', type=float, default=0.7, help='min temperature for calibration sweep')
parser.add_argument('--lincls_temp_max', type=float, default=1.6, help='max temperature for calibration sweep')
parser.add_argument('--lincls_temp_steps', type=int, default=10, help='number of temperature points in calibration sweep')
parser.add_argument(
    '--early_stop_patience',
    type=int,
    default=0,
    help='validation-loss early stopping patience (pretrain + linear in trainer); 0 = disabled, run all epochs',
)
parser.add_argument('--early_stop_min_delta', type=float, default=0.0, help='minimum validation loss improvement to reset patience')
parser.add_argument('--scheduler', action='store_true', default=True, help='use cosine scheduler for linear eval')
parser.add_argument('--logdir', type=str, default='log/')
parser.add_argument(
    '--batch_log_every',
    type=int,
    default=20,
    help='print in-epoch progress every N training batches (pretrain + linear eval)',
)

# SSL setup (defaults: WISDM + SimCLR + spiking ResNet-1D)
parser.add_argument('--framework', type=str, default='simclr', choices=['augpred', 'byol', 'simsiam', 'simclr', 'nnclr', 'tstcc'])
parser.add_argument(
    '--backbone',
    type=str,
    default='SResNet1D',
    choices=[
        'SFCN', 'SDCL', 'SLSTM', 'SNN_AE', 'SNN_CNN_AE', 'SNN_Transformer',
        'FCN', 'DCL', 'LSTM', 'AE', 'CNN_AE', 'Transformer',
        'SResNet1D', 'ViT1D', 'iSpikformer', 'SpikeFormer'
    ],
)
parser.add_argument(
    '--dcl-conv-kernels',
    type=int,
    default=64,
    help='DCL/SDCL: number of conv feature maps per conv block (smaller = lighter/faster).',
)
parser.add_argument(
    '--dcl-kernel-size',
    type=int,
    default=5,
    help='DCL/SDCL: temporal kernel size for conv blocks.',
)
parser.add_argument(
    '--dcl-lstm-units',
    type=int,
    default=128,
    help='DCL/SDCL: LSTM hidden size (smaller = lighter/faster).',
)
# iSpikformer (SpikingJelly; vendored in models/local_ispikformer — pip install spikingjelly)
parser.add_argument('--ispf_dim', type=int, default=512, help='iSpikformer d_model (backbone / probe dim)')
parser.add_argument(
    '--ispf_d_ff',
    type=int,
    default=0,
    help='iSpikformer MLP hidden; 0 = 4*ispf_dim (default)',
)
parser.add_argument('--ispf_depths', type=int, default=2, help='iSpikformer number of spiking attention blocks')
parser.add_argument('--ispf_num_steps', type=int, default=4, help='iSpikformer SNN time steps (encoding repeats)')
parser.add_argument('--ispf_heads', type=int, default=8, help='iSpikformer attention heads')
parser.add_argument(
    '--ispf_encoder',
    type=str,
    default='conv',
    choices=['conv', 'repeat', 'delta'],
    help='iSpikformer input spike encoder (SpikingJelly)',
)

# SpikeFormer (experimental, pure-torch spiking transformer)
parser.add_argument('--spk_tf_dim', type=int, default=256, help='SpikeFormer embedding dim')
parser.add_argument('--spk_tf_depth', type=int, default=4, help='SpikeFormer transformer depth')
parser.add_argument('--spk_tf_heads', type=int, default=8, help='SpikeFormer attention heads')
parser.add_argument('--spk_tf_mlp_ratio', type=float, default=4.0, help='SpikeFormer MLP expansion ratio')
parser.add_argument('--spk_tf_dropout', type=float, default=0.1, help='SpikeFormer dropout')
parser.add_argument('--spk_tf_attn_dropout', type=float, default=0.1, help='SpikeFormer attention dropout')
parser.add_argument('--spk_tf_spike_threshold', type=float, default=0.5, help='SpikeFormer spike threshold')
parser.add_argument(
    '--vit_patch_size',
    type=int,
    default=20,
    help='ViT1D: time steps per non-overlapping patch; len_sw must be divisible by this',
)
parser.add_argument('--vit_dim', type=int, default=128, help='ViT1D embedding / CLS dimension')
parser.add_argument('--vit_depth', type=int, default=4, help='ViT1D transformer depth')
parser.add_argument('--vit_heads', type=int, default=4, help='ViT1D attention heads')
parser.add_argument('--vit_mlp_dim', type=int, default=256, help='ViT1D FFN hidden size')
parser.add_argument('--vit_dropout', type=float, default=0.1, help='ViT1D dropout after patch+pos')
parser.add_argument('--criterion', type=str, default='NTXent', choices=['cos_sim', 'NTXent'])
parser.add_argument('--aug1', type=str, default='jit_scal')
parser.add_argument('--aug2', type=str, default='perm_jit')
parser.add_argument('--p', type=int, default=128, help='projection dimension')
parser.add_argument('--phid', type=int, default=128, help='predictor hidden dimension')
parser.add_argument('--EMA', type=float, default=0.996, help='EMA momentum for BYOL')
parser.add_argument('--lr_mul', type=float, default=1.0, help='lr multiplier for predictor (BYOL/SimSiam)')
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--temp_unit', type=str, default='tsfm')
parser.add_argument('--mmb_size', type=int, default=1024, help='memory bank size for NNCLR')
parser.add_argument('--lambda1', type=float, default=1.0)
parser.add_argument('--lambda2', type=float, default=1.0)
parser.add_argument('--tau', type=float, default=0.75, help='LIF membrane decay for SNN backbones')
parser.add_argument('--thresh', type=float, default=0.5, help='LIF firing threshold for SNN backbones')
parser.add_argument(
    '--input_encoding',
    type=str,
    default='none',
    choices=['none', 'zcsf', 'arima', 'zcsf_arima'],
    help='optional input preprocessing before backbone forward pass',
)
parser.add_argument(
    '--zcsf_step',
    type=float,
    default=0.1,
    help='step size for Zero-Crossing Step-Forward (ZCSF) input encoding',
)
parser.add_argument(
    '--arima_diff_order',
    type=int,
    default=1,
    help='d in ARIMA-like preprocessing: number of differencing steps',
)
parser.add_argument(
    '--arima_eps',
    type=float,
    default=1e-6,
    help='numerical epsilon for ARIMA-like residual extraction',
)
parser.add_argument(
    '--skip_pretrain',
    action='store_true',
    help='skip SSL pretraining and evaluate a randomly initialized frozen backbone with linear probe',
)
parser.add_argument(
    '--ssl_ckpt',
    type=str,
    default='',
    help=(
        "Path to a pretrained SSL checkpoint from trainer.train() "
        "(expects {'model_state_dict': ...}, or a raw state_dict). "
        "When set, SSL pretraining is skipped and weights are loaded into the SSL model before lincls."
    ),
)
parser.add_argument(
    '--ssl_ckpt_strict',
    action='store_true',
    help='If set, load SSL checkpoint with strict=True (default: strict=False for partial matches).',
)
parser.add_argument('--use_augpred', action='store_true', help='enable augmentation prediction auxiliary loss')
parser.add_argument('--augpred_lambda', type=float, default=0.5, help='weight for augmentation prediction loss')
parser.add_argument(
    '--augpred_transforms',
    type=str,
    default='na,t_flip,perm,t_warp,jit_scal,noise,scale,rotation',
    help='comma-separated augmentations for augmentation prediction head',
)
parser.add_argument(
    '--simclr_contrastive',
    type=str,
    default='instance',
    choices=['instance', 'supcon'],
    help="SimCLR only: 'instance' = standard two-view NTXent; 'supcon' = all same-class pairs in batch are positives (uses activity labels; supervised contrastive, Khosla et al. 2020).",
)
parser.add_argument(
    '--supcon_temperature',
    type=float,
    default=0.1,
    help='Temperature for SupCon (when --simclr_contrastive supcon).',
)


if __name__ == '__main__':
    args = parser.parse_args()
    pretrain_epochs = args.pretrain_epochs if args.pretrain_epochs is not None else args.n_epoch
    lincls_epochs = args.lincls_epochs if args.lincls_epochs is not None else args.n_epoch
    args.n_epoch = pretrain_epochs
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset, 'framework:', args.framework, 'backbone:', args.backbone)

    training_start = datetime.now()
    test_acc_list = []
    miF_list = []
    maF_list = []
    macroF_list = []

    for r in range(args.rep):
        seed_all(seed=1000 + r)
        train_loaders, val_loader, test_loader = setup_dataloaders(args)
        model, optimizers, schedulers, criterion, logger, fitlog, classifier, criterion_cls, optimizer_cls = setup(args, DEVICE)

        if not args.eval:
            ssl_ckpt = str(getattr(args, "ssl_ckpt", "") or "").strip()
            if ssl_ckpt:
                print(f"[ssl_ckpt] Loading SSL weights from: {ssl_ckpt}")
                obj = torch.load(ssl_ckpt, map_location=DEVICE, weights_only=False)
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
                # Optional lightweight sanity pass (rebuilds a backbone-only module internally).
                trained_ssl = test(test_loader, model.state_dict(), logger, fitlog, DEVICE, criterion, args)
                trained_backbone = lock_backbone(trained_ssl, args)
            elif args.skip_pretrain:
                print('Skipping SSL pretraining; using randomly initialized frozen backbone for linear probe.')
                trained_backbone = lock_backbone(model, args)
            else:
                args.n_epoch = pretrain_epochs
                best_model = train(
                    train_loaders,
                    val_loader,
                    model,
                    logger,
                    fitlog,
                    DEVICE,
                    optimizers,
                    schedulers,
                    criterion,
                    args,
                )
                trained_ssl = test(test_loader, best_model, logger, fitlog, DEVICE, criterion, args)
                trained_backbone = lock_backbone(trained_ssl, args)
        else:
            raise NotImplementedError('--eval mode is not implemented for main_ssl.py yet.')

        args.n_epoch = lincls_epochs
        best_lincls = train_lincls(
            train_loaders,
            val_loader,
            trained_backbone,
            classifier,
            logger,
            fitlog,
            DEVICE,
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
            DEVICE,
            criterion_cls,
            args,
            plt=False,
        )
        test_acc_list.append(test_acc)
        miF_list.append(miF)
        maF_list.append(maF)
        macroF_list.append(macroF)

    training_end = datetime.now()
    print('Training time:', training_end - training_start)
    print('Final Test Acc: {:.4f} +/- {:.4f}'.format(np.mean(test_acc_list), np.std(test_acc_list)))
    print('Final miF: {:.4f} +/- {:.4f}'.format(np.mean(miF_list), np.std(miF_list)))
    print('Final maF (weighted F1): {:.4f} +/- {:.4f}'.format(np.mean(maF_list), np.std(maF_list)))
    print('Final macro F1: {:.4f} +/- {:.4f}'.format(np.mean(macroF_list), np.std(macroF_list)))

    if args.cleanup_checkpoints:
        delete_files(args)
