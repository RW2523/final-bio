import torch
import torch.nn as nn
import numpy as np
import os
import pickle as cp
import random
from augmentations import gen_aug
from utils import tsne, mds, _logger
import time
from models.frameworks import *
from models.backbones import *
from models.resnet1d import SResNet1D
from models.vit1d import ViT1D
from models.loss import *
from models.spike import *
from data_preprocess import data_preprocess_ucihar
from data_preprocess import data_preprocess_shar
from data_preprocess import data_preprocess_hhar
from data_preprocess import data_preprocess_wisdm
from wisdm12_ssl.preprocess_wisdm12 import prep_wisdm_fused12

from sklearn.metrics import f1_score
import seaborn as sns
import fitlog
from copy import deepcopy

# fitlog uses deprecated NumPy aliases (e.g., np.str) on newer NumPy.
if not hasattr(np, 'str'):
    np.str = str

# create directory for saving models and plots
global model_dir_name
model_dir_name = 'results'
if not os.path.exists(model_dir_name):
    os.makedirs(model_dir_name)
global plot_dir_name
plot_dir_name = 'plot'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)
global augpred_classifier, augpred_optimizer, augpred_criterion, augpred_transforms
augpred_classifier = None
augpred_optimizer = None
augpred_criterion = None
augpred_transforms = []


def apply_input_encoding(sample, args):
    """
    Optional preprocessing on model inputs.
    Supported:
      - none: pass-through
      - zcsf: Zero-Crossing Step-Forward event encoding
      - arima: ARIMA-like residual extraction (differencing + AR(1) residual)
      - zcsf_arima: zcsf followed by arima
    """
    encoding = str(getattr(args, 'input_encoding', 'none')).lower()
    if encoding == 'none':
        return sample
    if encoding not in ['zcsf', 'arima', 'zcsf_arima']:
        raise ValueError(f"Unsupported --input_encoding '{encoding}'.")

    # Expected shape: [batch, time, channels]
    if sample.dim() != 3:
        raise ValueError(f'ZCSF expects input [B,T,C], got shape {tuple(sample.shape)}')

    def _zcsf_encode(x):
        step = float(getattr(args, 'zcsf_step', 0.1))
        if step <= 0:
            raise ValueError(f'--zcsf_step must be > 0, got {step}')

        encoded = torch.zeros_like(x)
        ref = x[:, 0:1, :].clone()
        for t in range(1, x.shape[1]):
            cur = x[:, t:t + 1, :]
            delta = cur - ref
            pos = (delta >= step).to(x.dtype)
            neg = (delta <= -step).to(x.dtype)
            spike = pos - neg
            encoded[:, t:t + 1, :] = spike
            ref = ref + spike * step
        return encoded

    def _arima_like_residual(x):
        # Lightweight ARIMA-style extractor:
        # 1) differencing (order d), 2) AR(1) fit per sample/channel, 3) residual stream.
        d = int(getattr(args, 'arima_diff_order', 1))
        eps = float(getattr(args, 'arima_eps', 1e-6))
        if d < 0:
            raise ValueError(f'--arima_diff_order must be >= 0, got {d}')
        if eps <= 0:
            raise ValueError(f'--arima_eps must be > 0, got {eps}')

        y = x
        for _ in range(d):
            if y.shape[1] < 2:
                return torch.zeros_like(x)
            y = y[:, 1:, :] - y[:, :-1, :]

        if y.shape[1] < 2:
            return torch.zeros_like(x)

        y_prev = y[:, :-1, :]
        y_next = y[:, 1:, :]
        num = (y_prev * y_next).sum(dim=1, keepdim=True)
        den = (y_prev * y_prev).sum(dim=1, keepdim=True) + eps
        phi = num / den
        resid = y_next - phi * y_prev

        out = torch.zeros_like(x)
        start = x.shape[1] - resid.shape[1]
        out[:, start:, :] = resid
        return out

    if encoding == 'zcsf':
        return _zcsf_encode(sample)
    if encoding == 'arima':
        return _arima_like_residual(sample)
    # zcsf_arima
    return _arima_like_residual(_zcsf_encode(sample))


def setup_dataloaders(args):
    if args.dataset == 'ucihar':
        args.n_feature = 9
        args.len_sw = 128
        args.n_class = 6
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain == '0'
        train_loaders, val_loader, test_loader = data_preprocess_ucihar.prep_ucihar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int( args.len_sw * 0.5))
    if args.dataset == 'shar':
        args.n_feature = 3
        args.len_sw = 151
        args.n_class = 17
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain == '1'
        train_loaders, val_loader, test_loader = data_preprocess_shar.prep_shar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5))
    if args.dataset == 'hhar':
        args.n_feature = 6
        args.len_sw = 100
        args.n_class = 6
        source_domain = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        # source_domain.remove(args.target_domain)
        train_loaders, val_loader, test_loader = data_preprocess_hhar.prep_hhar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5),
                                                                                device=args.device,
                                                                                train_user=source_domain,
                                                                                test_user=args.target_domain)
    if args.dataset == 'wisdm':
        if bool(getattr(args, 'wisdm_fused12', False)):
            # Fused setup mirrors requested bio-style defaults:
            #   sample rate=20Hz, window=10s (200), stride=5s (100), channels=12.
            args.n_feature = 12
            args.len_sw = 200
            args.n_class = 18
            train_loaders, val_loader, test_loader = prep_wisdm_fused12(
                args=args,
                nominal_hz=20.0,
                window_seconds=10.0,
                stride_seconds=5.0,
            )
        else:
            args.n_feature = 6
            # WISDM raw stream is sampled at 20Hz; 10 seconds => 200 samples/window.
            args.len_sw = 200
            args.n_class = 18
            if args.cases not in ['subject', 'subject_large']:
                args.target_domain = '1600'
            train_loaders, val_loader, test_loader = data_preprocess_wisdm.prep_wisdm(
                args,
                SLIDING_WINDOW_LEN=args.len_sw,
                SLIDING_WINDOW_STEP=int(args.len_sw * 0.5),
                device=args.device,
            )

    return train_loaders, val_loader, test_loader


def setup_linclf(args, DEVICE, bb_dim):
    '''
    @param bb_dim: output dimension of the backbone network
    @return: a downstream classifier head (linear by default, optional MLP probe)
    '''
    classifier = Classifier(
        bb_dim=bb_dim,
        n_classes=args.n_class,
        head_type=str(getattr(args, 'lincls_head', 'linear')),
        hidden_dim=int(getattr(args, 'lincls_hidden_dim', 512)),
        hidden_dim2=int(getattr(args, 'lincls_hidden_dim2', 256)),
        dropout=float(getattr(args, 'lincls_dropout', 0.2)),
    )
    if str(getattr(args, 'lincls_head', 'linear')) == 'linear':
        classifier.classifier.weight.data.normal_(mean=0.0, std=0.01)
        classifier.classifier.bias.data.zero_()
    classifier = classifier.to(DEVICE)
    return classifier


def setup_model_optm(args, DEVICE, classifier=True):
    snn_params = {"tau": getattr(args, 'tau', 0.5), "thresh": getattr(args, 'thresh', 0.5)}
    # set up backbone network
    if args.backbone in ('FCN', 'SFCN'):
        backbone = SFCN(
            n_channels=args.n_feature,
            n_classes=args.n_class,
            backbone=True,
            len_sw=args.len_sw,
            **snn_params,
        )
    elif args.backbone == 'SResNet1D':
        backbone = SResNet1D(
            n_channels=args.n_feature,
            n_classes=args.n_class,
            len_sw=args.len_sw,
            backbone=True,
            **snn_params,
        )
    elif args.backbone == 'ViT1D':
        vps = int(getattr(args, "vit_patch_size", 20))
        if int(args.len_sw) % vps != 0:
            raise ValueError(
                f"--len_sw {args.len_sw} must be divisible by --vit_patch_size {vps} for ViT1D (non-overlapping time patches).",
            )
        vit_dim = int(getattr(args, "vit_dim", 128))
        vit_depth = int(getattr(args, "vit_depth", 4))
        vit_heads = int(getattr(args, "vit_heads", 4))
        vit_mlp_dim = int(getattr(args, "vit_mlp_dim", 256))
        vit_dropout = float(getattr(args, "vit_dropout", 0.1))
        backbone = ViT1D(
            n_channels=args.n_feature,
            n_classes=args.n_class,
            len_sw=int(args.len_sw),
            patch_size=vps,
            dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            dropout=vit_dropout,
            backbone=True,
        )
    elif args.backbone in ('DCL', 'SDCL'):
        backbone = SDCL(
            n_channels=args.n_feature,
            n_classes=args.n_class,
            conv_kernels=int(getattr(args, 'dcl_conv_kernels', 64)),
            kernel_size=int(getattr(args, 'dcl_kernel_size', 5)),
            LSTM_units=int(getattr(args, 'dcl_lstm_units', 128)),
            backbone=True,
            **snn_params,
        )
    elif args.backbone in ('LSTM', 'SLSTM'):
        backbone = SLSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=True, **snn_params)
    elif args.backbone in ('AE', 'SNN_AE'):
        backbone = SNN_AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=True, **snn_params)
    elif args.backbone in ('CNN_AE', 'SNN_CNN_AE'):
        backbone = SNN_CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=True, **snn_params)
    elif args.backbone in ('Transformer', 'SNN_Transformer'):
        backbone = SNN_Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True, **snn_params)
    elif args.backbone == 'iSpikformer':
        from models.seqsnn_ispikformer import SeqSNNiSpikformerBackbone

        ispf_d_ff = int(getattr(args, 'ispf_d_ff', 0)) or None
        backbone = SeqSNNiSpikformerBackbone(
            n_channels=args.n_feature,
            n_classes=args.n_class,
            len_sw=int(args.len_sw),
            dim=int(getattr(args, 'ispf_dim', 512)),
            d_ff=ispf_d_ff,
            depths=int(getattr(args, 'ispf_depths', 2)),
            num_steps=int(getattr(args, 'ispf_num_steps', 4)),
            heads=int(getattr(args, 'ispf_heads', 8)),
            encoder_type=str(getattr(args, 'ispf_encoder', 'conv')),
        )
    elif args.backbone == 'SpikeFormer':
        from models.exp_spikeformer import SpikeFormerBackbone

        backbone = SpikeFormerBackbone(
            n_channels=int(args.n_feature),
            len_sw=int(args.len_sw),
            dim=int(getattr(args, 'spk_tf_dim', 256)),
            depth=int(getattr(args, 'spk_tf_depth', 4)),
            heads=int(getattr(args, 'spk_tf_heads', 8)),
            mlp_ratio=float(getattr(args, 'spk_tf_mlp_ratio', 4.0)),
            dropout=float(getattr(args, 'spk_tf_dropout', 0.1)),
            attn_dropout=float(getattr(args, 'spk_tf_attn_dropout', 0.1)),
            spike_threshold=float(getattr(args, 'spk_tf_spike_threshold', 0.5)),
            backbone=True,
        )
    else:
        raise NotImplementedError

    # set up model and optimizers
    if args.framework in ['byol', 'simsiam']:
        model = BYOL(DEVICE, backbone, window_size=args.len_sw, n_channels=args.n_feature, projection_size=args.p,
                     projection_hidden_size=args.phid, moving_average=args.EMA)
        optimizer1 = torch.optim.Adam(model.online_encoder.parameters(),
                                      args.lr,
                                      weight_decay=args.weight_decay)
        optimizer2 = torch.optim.Adam(model.online_predictor.parameters(),
                                      args.lr * args.lr_mul,
                                      weight_decay=args.weight_decay)
        optimizers = [optimizer1, optimizer2]
    elif args.framework == 'simclr':
        model = SimCLR(backbone=backbone, dim=args.p)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizers = [optimizer]
    elif args.framework == 'nnclr':
        model = NNCLR(backbone=backbone, dim=args.p, pred_dim=args.phid)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework == 'tstcc':
        model = TSTCC(backbone=backbone, DEVICE=DEVICE, temp_unit=args.temp_unit, tc_hidden=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework == 'augpred':
        # Pure pretext training (no contrastive/bootstrapping objective).
        model = backbone
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizers = [optimizer]

    else:
        raise NotImplementedError

    model = model.to(DEVICE)

    # set up linear classfier
    if classifier:
        bb_dim = backbone.out_dim
        classifier = setup_linclf(args, DEVICE, bb_dim)
        return model, classifier, optimizers

    else:
        return model, optimizers


def delete_files(args):
    for epoch in range(args.n_epoch):
        model_dir = model_dir_name + '/pretrain_' + args.model_name + str(epoch) + '.pt'
        if os.path.isfile(model_dir):
            os.remove(model_dir)

        cls_dir = model_dir_name + '/lincls_' + args.model_name + str(epoch) + '.pt'
        if os.path.isfile(cls_dir):
            os.remove(cls_dir)


def setup(args, DEVICE):
    # set up default hyper-parameters
    if args.framework == 'byol':
        args.weight_decay = 1.5e-6
    if args.framework == 'simsiam':
        args.weight_decay = 1e-4
        args.EMA = 1.0
        args.lr_mul = 1.0
    if args.framework in ['simclr', 'nnclr']:
        args.criterion = 'NTXent'
        args.weight_decay = 1e-6
    if args.framework == 'tstcc':
        args.criterion = 'NTXent'
        # TSTCC temporal contrast expects (B, C, T) conv feature maps; FCN and SFCN only.
        if str(args.backbone) not in ('FCN', 'SFCN'):
            raise ValueError(
                f"TSTCC needs a conv backbone with 3D features; use --backbone FCN or SFCN, got {args.backbone!r}."
            )
        args.weight_decay = 3e-4
    if args.framework == 'augpred':
        args.criterion = 'NTXent'
        args.use_augpred = True

    model, classifier, optimizers = setup_model_optm(args, DEVICE, classifier=True)

    # loss fn
    if args.framework == 'augpred':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'cos_sim':
        criterion = nn.CosineSimilarity(dim=1)
    elif args.criterion == 'NTXent':
        if args.framework == 'simclr' and str(getattr(args, 'simclr_contrastive', 'instance')) == 'supcon':
            criterion = SupConTwoViewLoss(
                temperature=float(getattr(args, 'supcon_temperature', 0.1)),
            )
        elif args.framework == 'tstcc':
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.2)
        else:
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.1)

    _sclr = str(getattr(args, 'simclr_contrastive', 'instance')) if args.framework == 'simclr' else 'na'
    args.model_name = 'try_scheduler_' + args.framework + '_pretrain_' + args.dataset + '_eps' + str(args.n_epoch) + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) \
                      + '_aug1' + args.aug1 + '_aug2' + args.aug2 + '_dim-pdim' + str(args.p) + '-' + str(args.phid) \
                      + '_EMA' + str(args.EMA) + '_criterion_' + args.criterion + '_lambda1_' + str(args.lambda1) + '_lambda2_' + str(args.lambda2) + '_tempunit_' + args.temp_unit \
                      + ('_sclr_' + _sclr if args.framework == 'simclr' else '') \
                      + ('_ispf' if str(getattr(args, 'backbone', '')) == 'iSpikformer' else '') \
                      + ('_spktf' if str(getattr(args, 'backbone', '')) == 'SpikeFormer' else '') \
                      + ('_lincls_' + str(getattr(args, 'lincls_head', 'linear')) if str(getattr(args, 'lincls_head', 'linear')) != 'linear' else '')

    # log
    if os.path.isdir(args.logdir) == False:
        os.makedirs(args.logdir)
    log_file_name = os.path.join(args.logdir, args.model_name + f".log")
    logger = _logger(log_file_name)
    logger.debug(args)

    # fitlog
    fitlog.set_log_dir(args.logdir)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)

    criterion_cls = nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr_cls)

    schedulers = []
    for optimizer in optimizers:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=0)
        schedulers.append(scheduler)

    global nn_replacer
    nn_replacer = None
    if args.framework == 'nnclr':
        nn_replacer = NNMemoryBankModule(size=args.mmb_size)

    global recon
    recon = None
    if args.backbone in ['AE', 'CNN_AE', 'SNN_AE', 'SNN_CNN_AE']:
        recon = nn.MSELoss()

    global augpred_classifier, augpred_optimizer, augpred_criterion, augpred_transforms
    augpred_classifier = None
    augpred_optimizer = None
    augpred_criterion = None
    augpred_transforms = []
    if hasattr(args, 'use_augpred') and args.use_augpred:
        augpred_transforms = [x.strip() for x in args.augpred_transforms.split(',') if len(x.strip()) > 0]
        if len(augpred_transforms) < 2:
            raise ValueError('AugPred needs at least two transforms in --augpred_transforms.')
        if args.framework in ['simsiam', 'byol']:
            augpred_bb_dim = model.online_encoder.net.out_dim
        elif args.framework in ['simclr', 'nnclr', 'tstcc']:
            augpred_bb_dim = model.encoder.out_dim
        elif args.framework == 'augpred':
            augpred_bb_dim = model.out_dim
        else:
            raise NotImplementedError
        augpred_classifier = Classifier(bb_dim=augpred_bb_dim, n_classes=len(augpred_transforms)).to(DEVICE)
        augpred_optimizer = torch.optim.Adam(augpred_classifier.parameters(), lr=args.lr)
        augpred_criterion = nn.CrossEntropyLoss()

    return model, optimizers, schedulers, criterion, logger, fitlog, classifier, criterion_cls, optimizer_cls


def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def _get_backbone_features(args, model, sample):
    if args.framework in ['simsiam', 'byol']:
        backbone = model.online_encoder.net
    elif args.framework in ['simclr', 'nnclr', 'tstcc']:
        backbone = model.encoder
    elif args.framework == 'augpred':
        backbone = model
    else:
        raise NotImplementedError

    if backbone.__class__.__name__ in ['AE', 'CNN_AE', 'SNN_AE', 'SNN_CNN_AE']:
        _, feat = backbone(sample)
    else:
        _, feat = backbone(sample)
    if len(feat.shape) == 3:
        feat = feat.reshape(feat.shape[0], -1)
    return feat


def calculate_augpred_loss(args, sample, model, DEVICE):
    if augpred_classifier is None:
        return None

    aug_idx = random.randint(0, len(augpred_transforms) - 1)
    aug_name = augpred_transforms[aug_idx]
    aug_sample = _to_tensor(gen_aug(sample, aug_name)).to(DEVICE).float()
    aug_sample = apply_input_encoding(aug_sample, args)
    labels = torch.full((aug_sample.shape[0],), aug_idx, dtype=torch.long, device=DEVICE)
    feat = _get_backbone_features(args, model, aug_sample)
    logits = augpred_classifier(feat)
    return augpred_criterion(logits, labels)


def calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=None, nn_replacer=None):
    if args.framework == 'augpred':
        if augpred_classifier is None:
            raise ValueError('AugPred framework requires augmentation head. Check --use_augpred/--augpred_transforms.')
        return calculate_augpred_loss(args, sample, model, DEVICE)

    aug_sample1 = _to_tensor(gen_aug(sample, args.aug1))
    aug_sample2 = _to_tensor(gen_aug(sample, args.aug2))
    aug_sample1, aug_sample2, target = aug_sample1.to(DEVICE).float(), aug_sample2.to(DEVICE).float(), target.to(
        DEVICE).long()
    aug_sample1 = apply_input_encoding(aug_sample1, args)
    aug_sample2 = apply_input_encoding(aug_sample2, args)
    if args.framework in ['byol', 'simsiam']:
        assert args.criterion == 'cos_sim'
    if args.framework in ['tstcc', 'simclr', 'nnclr']:
        assert args.criterion == 'NTXent'
    if args.framework in ['byol', 'simsiam', 'nnclr']:
        if args.backbone in ['AE', 'CNN_AE', 'SNN_AE', 'SNN_CNN_AE']:
            x1_encoded, x2_encoded, p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
        else:
            p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
        if args.framework == 'nnclr':
            z1 = nn_replacer(z1, update=False)
            z2 = nn_replacer(z2, update=True)
        if args.criterion == 'cos_sim':
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        elif args.criterion == 'NTXent':
            loss = (criterion(p1, z2) + criterion(p2, z1)) * 0.5
        if args.backbone in ['AE', 'CNN_AE', 'SNN_AE', 'SNN_CNN_AE']:
            loss = loss * args.lambda1 + recon_loss * args.lambda2
    if args.framework == 'simclr':
        if args.backbone in ['AE', 'CNN_AE', 'SNN_AE', 'SNN_CNN_AE']:
            x1_encoded, x2_encoded, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
        else:
            z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
        if str(getattr(args, 'simclr_contrastive', 'instance')) == 'supcon':
            loss = criterion(z1, z2, target)
        else:
            loss = criterion(z1, z2)
        if args.backbone in ['AE', 'CNN_AE', 'SNN_AE', 'SNN_CNN_AE']:
            loss = loss * args.lambda1 + recon_loss * args.lambda2
    if args.framework == 'tstcc':
        nce1, nce2, p1, p2 = model(x1=aug_sample1, x2=aug_sample2)
        tmp_loss = nce1 + nce2
        ctx_loss = criterion(p1, p2)
        loss = tmp_loss * args.lambda1 + ctx_loss * args.lambda2
    if hasattr(args, 'use_augpred') and args.use_augpred and augpred_classifier is not None:
        augpred_loss = calculate_augpred_loss(args, sample, model, DEVICE)
        loss = loss + args.augpred_lambda * augpred_loss
    return loss


def train(train_loaders, val_loader, model, logger, fitlog, DEVICE, optimizers, schedulers, criterion, args):
    best_model = None
    min_val_loss = 1e8
    no_improve_epochs = 0
    wall_start = time.time()

    log_every = int(getattr(args, "batch_log_every", 20))
    if log_every <= 0:
        log_every = 20

    for epoch in range(args.n_epoch):
        epoch_start = time.time()
        print(f"[pretrain] epoch {epoch + 1}/{args.n_epoch} start")
        logger.debug(f'\nEpoch : {epoch}')
        total_loss = 0
        n_batches = 0
        model.train()
        for i, train_loader in enumerate(train_loaders):
            try:
                loader_len = len(train_loader)
            except TypeError:
                loader_len = None
            for idx, (sample, target, domain) in enumerate(train_loader):
                for optimizer in optimizers:
                    optimizer.zero_grad()
                if augpred_optimizer is not None:
                    augpred_optimizer.zero_grad()
                if sample.size(0) != args.batch_size:
                    continue
                n_batches += 1
                loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                total_loss += loss.item()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                if augpred_optimizer is not None:
                    augpred_optimizer.step()
                if args.framework == 'byol':
                    model.update_moving_average()
                if (n_batches % log_every) == 0:
                    avg_loss = total_loss / max(n_batches, 1)
                    if loader_len is not None:
                        print(
                            f"[pretrain] epoch {epoch + 1}/{args.n_epoch} "
                            f"loader={i + 1}/{len(train_loaders)} batch={idx + 1}/{loader_len} "
                            f"avg_loss={avg_loss:.4f}"
                        )
                    else:
                        print(
                            f"[pretrain] epoch {epoch + 1}/{args.n_epoch} "
                            f"loader={i + 1}/{len(train_loaders)} batch={idx + 1} "
                            f"avg_loss={avg_loss:.4f}"
                        )
        fitlog.add_loss(optimizers[0].param_groups[0]['lr'], name="learning rate", step=epoch)
        for scheduler in schedulers:
            scheduler.step()

        # save model
        model_dir = model_dir_name + '/pretrain_' + args.model_name + str(epoch) + '.pt'
        print('Saving model at {} epoch to {}'.format(epoch, model_dir))
        torch.save({'model_state_dict': model.state_dict()}, model_dir)

        train_epoch_loss = total_loss / n_batches
        logger.debug(f'Train Loss     : {train_epoch_loss:.4f}')
        print(f"[pretrain] epoch {epoch + 1}/{args.n_epoch} train_loss={train_epoch_loss:.4f}")
        fitlog.add_loss(train_epoch_loss, name="pretrain training loss", step=epoch)

        if args.cases in ['subject', 'subject_large']:
            with torch.no_grad():
                best_model = deepcopy(model.state_dict())
                break
        else:
            with torch.no_grad():
                model.eval()
                total_loss = 0
                n_batches = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    if sample.size(0) != args.batch_size:
                        continue
                    n_batches += 1
                    loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon,
                                                nn_replacer=nn_replacer)
                    total_loss += loss.item()
                val_epoch_loss = total_loss / n_batches
                if val_epoch_loss <= (min_val_loss - args.early_stop_min_delta):
                    min_val_loss = val_epoch_loss
                    best_model = deepcopy(model.state_dict())
                    print('update')
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                logger.debug(f'Val Loss     : {total_loss / n_batches:.4f}')
                fitlog.add_loss(total_loss / n_batches, name="pretrain validation loss", step=epoch)
                elapsed = time.time() - epoch_start
                total_elapsed = time.time() - wall_start
                print(
                    f"[pretrain] epoch {epoch + 1}/{args.n_epoch} val_loss={val_epoch_loss:.4f} "
                    f"epoch_time={elapsed:.1f}s total={total_elapsed/60:.1f}m"
                )
                if (
                    getattr(args, "early_stop_patience", 0) > 0
                    and no_improve_epochs >= args.early_stop_patience
                ):
                    logger.debug(
                        f'Early stopping pretrain at epoch {epoch} '
                        f'(patience={args.early_stop_patience}, best_val_loss={min_val_loss:.4f})'
                    )
                    break
    return best_model


def test(test_loader, best_model, logger, fitlog, DEVICE, criterion, args):
    model, _ = setup_model_optm(args, DEVICE, classifier=False)
    model.load_state_dict(best_model)
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 0
        for idx, (sample, target, domain) in enumerate(test_loader):
            if sample.size(0) != args.batch_size:
                continue
            n_batches += 1
            loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
            total_loss += loss.item()
        logger.debug(f'Test Loss     : {total_loss / n_batches:.4f}')
        fitlog.add_best_metric({"dev": {"pretrain test loss": total_loss / n_batches}})

    return model


def lock_backbone(model, args):
    for name, param in model.named_parameters():
        param.requires_grad = False

    if args.framework in ['simsiam', 'byol']:
        trained_backbone = model.online_encoder.net
    elif args.framework in ['simclr', 'nnclr', 'tstcc']:
        trained_backbone = model.encoder
    elif args.framework == 'augpred':
        trained_backbone = model
    else:
        raise NotImplementedError

    return trained_backbone


def calculate_lincls_output(sample, target, trained_backbone, classifier, criterion, args):
    sample = apply_input_encoding(sample, args)
    _, feat = trained_backbone(sample)
    if len(feat.shape) == 3:
        feat = feat.reshape(feat.shape[0], -1)
    output = classifier(feat)
    loss = criterion(output, target)
    _, predicted = torch.max(output.data, 1)
    return loss, predicted, feat


def _apply_lincls_finetune_settings(trained_backbone, args):
    """
    Optionally unfreeze part/all of the backbone for linear eval training.
    This is the main knob when linear-probe performance is much worse than supervised.
    """
    for p in trained_backbone.parameters():
        p.requires_grad = False

    bb_name = trained_backbone.__class__.__name__
    if bb_name in ['SFCN', 'FCN']:
        if args.lincls_finetune_scope == 'all':
            for p in trained_backbone.parameters():
                p.requires_grad = True
        else:
            # 'last_block' => last conv block (matches variable names in FCN/SFCN)
            for p in trained_backbone.conv_block3.parameters():
                p.requires_grad = True
    elif bb_name in ['SResNet1D']:
        if args.lincls_finetune_scope == 'all':
            for p in trained_backbone.parameters():
                p.requires_grad = True
        else:
            for p in trained_backbone.layer4.parameters():
                p.requires_grad = True
    elif bb_name == 'ViT1D':
        if args.lincls_finetune_scope == 'all':
            for p in trained_backbone.parameters():
                p.requires_grad = True
        else:
            for p in trained_backbone.last_block_parameters():
                p.requires_grad = True
    elif bb_name == 'SeqSNNiSpikformerBackbone':
        if args.lincls_finetune_scope == 'all':
            for p in trained_backbone.parameters():
                p.requires_grad = True
        else:
            for p in trained_backbone.core.blocks[-1].parameters():
                p.requires_grad = True
    else:
        raise ValueError(
            f"--lincls_finetune_backbone is not supported for backbone={bb_name} yet. "
            f"Use SFCN/FCN/SResNet1D/ViT1D/SeqSNNiSpikformerBackbone, or extend _apply_lincls_finetune_settings()."
        )


def train_lincls(train_loaders, val_loader, trained_backbone, classifier, logger, fitlog, DEVICE, optimizer, criterion, args):
    best_lincls = None
    min_val_loss = 1e8
    no_improve_epochs = 0

    # Default: only train the linear classifier; optionally finetune part of the backbone too.
    finetune = bool(getattr(args, 'lincls_finetune_backbone', False))
    if finetune:
        _apply_lincls_finetune_settings(trained_backbone, args)
        head_params = list(classifier.parameters())
        bb_params = [p for p in trained_backbone.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            [
                {'params': head_params, 'lr': args.lr_cls},
                {'params': bb_params, 'lr': args.lincls_backbone_lr},
            ],
            # keep weight decay on head only (backbone often does better without extra L2 in finetune)
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=0)

    wall_start = time.time()
    log_every = int(getattr(args, "batch_log_every", 20))
    if log_every <= 0:
        log_every = 20

    for epoch in range(args.n_epoch):
        epoch_start = time.time()
        print(f"[lincls] epoch {epoch + 1}/{args.n_epoch} start")
        if finetune and any(p.requires_grad for p in trained_backbone.parameters()):
            trained_backbone.train()
        else:
            trained_backbone.eval()
        classifier.train()
        logger.debug(f'\nEpoch : {epoch}')
        total_loss = 0
        total = 0
        correct = 0
        for i, train_loader in enumerate(train_loaders):
            try:
                loader_len = len(train_loader)
            except TypeError:
                loader_len = None
            for idx, (sample, target, domain) in enumerate(train_loader):
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                loss, predicted, _ = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion, args)
                bsz = target.size(0)
                total_loss += loss.item() * bsz
                total += bsz
                correct += (predicted == target).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ((idx + 1) % log_every) == 0:
                    cur_loss = total_loss / max(total, 1)
                    cur_acc = float(correct) * 100.0 / max(total, 1)
                    if loader_len is not None:
                        print(
                            f"[lincls] epoch {epoch + 1}/{args.n_epoch} "
                            f"loader={i + 1}/{len(train_loaders)} batch={idx + 1}/{loader_len} "
                            f"avg_loss={cur_loss:.4f} avg_acc={cur_acc:.2f}"
                        )
                    else:
                        print(
                            f"[lincls] epoch {epoch + 1}/{args.n_epoch} "
                            f"loader={i + 1}/{len(train_loaders)} batch={idx + 1} "
                            f"avg_loss={cur_loss:.4f} avg_acc={cur_acc:.2f}"
                        )

        # save model (optional skip during LR sweeps to avoid thousands of files)
        if not getattr(args, 'lincls_skip_epoch_save', False):
            model_dir = model_dir_name + '/lincls_' + args.model_name + str(epoch) + '.pt'
            print('Saving model at {} epoch to {}'.format(epoch, model_dir))
            torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': classifier.state_dict()}, model_dir)

        acc_train = float(correct) * 100.0 / total
        mean_train_loss = total_loss / total
        logger.debug(f'epoch train loss     : {mean_train_loss:.4f}, train acc     : {acc_train:.4f}')
        print(f"[lincls] epoch {epoch + 1}/{args.n_epoch} train_loss={mean_train_loss:.4f} train_acc={acc_train:.2f}")
        fitlog.add_loss(mean_train_loss, name="Train Loss", step=epoch)
        fitlog.add_metric({"dev": {"Train Acc": acc_train}}, step=epoch)

        if args.scheduler:
            scheduler.step()

        if args.cases in ['subject', 'subject_large']:
            with torch.no_grad():
                best_lincls = deepcopy(classifier.state_dict())
        else:
            with torch.no_grad():
                # Keep backbone in eval for validation to get stable BN / dropout behavior
                # even if we finetune it during the training steps.
                trained_backbone.eval()
                classifier.eval()
                total_loss = 0
                total = 0
                correct = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                    loss, predicted, _ = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion, args)
                    bsz = target.size(0)
                    total_loss += loss.item() * bsz
                    total += bsz
                    correct += (predicted == target).sum()
                acc_val = float(correct) * 100.0 / total
                mean_val_loss = total_loss / total
                if mean_val_loss <= (min_val_loss - args.early_stop_min_delta):
                    min_val_loss = mean_val_loss
                    best_lincls = deepcopy(classifier.state_dict())
                    print('update')
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                logger.debug(f'epoch val loss     : {mean_val_loss:.4f}, val acc     : {acc_val:.4f}')
                elapsed = time.time() - epoch_start
                total_elapsed = time.time() - wall_start
                print(
                    f"[lincls] epoch {epoch + 1}/{args.n_epoch} val_loss={mean_val_loss:.4f} "
                    f"val_acc={acc_val:.2f} epoch_time={elapsed:.1f}s total={total_elapsed/60:.1f}m"
                )
                fitlog.add_loss(mean_val_loss, name="Val Loss", step=epoch)
                fitlog.add_metric({"dev": {"Val Acc": acc_val}}, step=epoch)
                if (
                    getattr(args, "early_stop_patience", 0) > 0
                    and no_improve_epochs >= args.early_stop_patience
                ):
                    logger.debug(
                        f'Early stopping linear eval at epoch {epoch} '
                        f'(patience={args.early_stop_patience}, best_val_loss={min_val_loss:.4f})'
                    )
                    break
    return best_lincls


def test_lincls(test_loader, trained_backbone, best_lincls, logger, fitlog, DEVICE, criterion, args, plt=False):
    classifier = setup_linclf(args, DEVICE, trained_backbone.out_dim)
    classifier.load_state_dict(best_lincls)
    total_loss = 0
    total = 0
    correct = 0
    confusion_matrix = torch.zeros(args.n_class, args.n_class)
    feats = None
    trgs = np.array([])
    preds = np.array([])
    with torch.no_grad():
        trained_backbone.eval()
        classifier.eval()
        for idx, (sample, target, domain) in enumerate(test_loader):
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            loss, predicted, feat = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion, args)
            bsz = target.size(0)
            total_loss += loss.item() * bsz
            if feats is None:
                feats = feat
            else:
                feats = torch.cat((feats, feat), 0)
            trgs = np.append(trgs, target.data.cpu().numpy())
            preds = np.append(preds, predicted.data.cpu().numpy())
            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            total += bsz
            correct += (predicted == target).sum()
        acc_test = float(correct) * 100.0 / total
        mean_test_loss = total_loss / total

        miF = f1_score(trgs, preds, average='micro') * 100
        maF = f1_score(trgs, preds, average='weighted') * 100
        macroF = f1_score(trgs, preds, average='macro', zero_division=0) * 100

        logger.debug(
            f'epoch test loss     : {mean_test_loss:.4f}, test acc     : {acc_test:.4f}, '
            f'miF     : {miF:.4f}, maF(w)     : {maF:.4f}, macroF     : {macroF:.4f}'
        )

        fitlog.add_best_metric({"dev": {"Test Loss": mean_test_loss}})
        fitlog.add_best_metric({"dev": {"Test Acc": acc_test}})
        fitlog.add_best_metric({"dev": {"miF": miF}})
        fitlog.add_best_metric({"dev": {"maF": maF}})
        fitlog.add_best_metric({"dev": {"macroF": macroF}})

        logger.debug(confusion_matrix)
        logger.debug(confusion_matrix.diag() / confusion_matrix.sum(1))

    if plt == True:
        tsne(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_tsne.png')
        mds(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_mds.png')
        sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(plot_dir_name + '/' + args.model_name + '_confmatrix.png')
        print('plots saved to ', plot_dir_name)
    return acc_test, miF, maF, macroF
