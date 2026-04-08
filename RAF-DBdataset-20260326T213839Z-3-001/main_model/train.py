"""
v6 training: Center Loss + SCE Loss + Focal Loss

Three loss components work together:
1. Focal Loss: classify correctly, focus on hard samples
2. Center Loss: pull same-class features together in embedding space
3. SCE Loss: robust to RAF-DB's noisy labels

Total loss = Focal + λ_center * Center + λ_sce * SCE
"""

import os
import csv
import time
import random
import argparse
import importlib

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

try:
    from torch.amp import autocast, GradScaler
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _NEW_AMP = False

from dataset import create_dataloaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Loss Functions ────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1, weight=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class CenterLoss(nn.Module):
    """Pull features toward their class center in embedding space."""
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features, labels):
        centers_batch = self.centers[labels]
        return ((features - centers_batch) ** 2).sum(dim=1).mean()


class SCELoss(nn.Module):
    """Symmetric Cross Entropy: robust to noisy labels."""
    def __init__(self, alpha=1.0, beta=0.5, num_classes=7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, targets):
        # Forward CE
        ce = nn.functional.cross_entropy(logits, targets, reduction='mean')
        # Reverse CE
        pred = torch.softmax(logits, dim=1).clamp(1e-7, 1.0)
        onehot = torch.zeros_like(pred).scatter_(1, targets.unsqueeze(1), 1.0)
        onehot = onehot.clamp(1e-4, 1.0)
        rce = -(pred * torch.log(onehot)).sum(dim=1).mean()
        return self.alpha * ce + self.beta * rce


# ─── Class-aware Mixup/CutMix ────────────────────────────────────────────────

def _build_class_aware_perm(labels, target_classes=(1, 2), neighbor_map=None):
    """
    Build a permutation for class-aware augmentation.
    Only target classes are remapped; others keep identity mapping.
    Labels are 0-based: Fear=1, Disgust=2.
    """
    if neighbor_map is None:
        # Fear(1) close to Surprise(0), Sadness(4), Neutral(6)
        # Disgust(2) close to Anger(5), Neutral(6), Sadness(4)
        neighbor_map = {1: [0, 4, 6], 2: [5, 6, 4]}

    bsz = labels.size(0)
    device = labels.device
    perm = torch.arange(bsz, device=device)
    target_set = set(target_classes)

    for i in range(bsz):
        cls = int(labels[i].item())
        if cls not in target_set:
            continue
        allowed = [cls] + neighbor_map.get(cls, [])
        allowed_t = torch.tensor(allowed, device=device)
        cand = torch.where(torch.isin(labels, allowed_t))[0]
        cand = cand[cand != i]
        if cand.numel() == 0:
            continue
        j = cand[torch.randint(0, cand.numel(), (1,), device=device)].item()
        perm[i] = j
    return perm


def _rand_bbox(h, w, lam):
    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_rat)
    cut_w = int(w * cut_rat)
    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    return int(x1), int(y1), int(x2), int(y2)


def class_aware_mixcut(images, labels, mixup_prob=0.4, cutmix_prob=0.3,
                       mixup_alpha=0.2, cutmix_alpha=1.0,
                       target_classes=(1, 2)):
    """
    Apply class-aware mixup/cutmix only to target classes.
    Returns images, y_a, y_b, lam.
    """
    op_p = np.random.rand()
    if op_p >= (mixup_prob + cutmix_prob):
        return images, labels, labels, 1.0

    perm = _build_class_aware_perm(labels, target_classes=target_classes)
    y_a, y_b = labels, labels[perm]
    target_mask = torch.isin(labels, torch.tensor(target_classes,
                                                  device=labels.device))
    if target_mask.sum().item() == 0:
        return images, labels, labels, 1.0

    if op_p < mixup_prob:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        lam = max(lam, 1.0 - lam)
        mixed = images.clone()
        mixed[target_mask] = (
            lam * images[target_mask] +
            (1.0 - lam) * images[perm][target_mask]
        )
        return mixed, y_a, y_b, lam

    lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
    _, _, h, w = images.shape
    x1, y1, x2, y2 = _rand_bbox(h, w, lam)
    mixed = images.clone()
    mixed[target_mask, :, y1:y2, x1:x2] = images[perm][target_mask, :, y1:y2, x1:x2]
    area = (x2 - x1) * (y2 - y1)
    lam_adj = 1.0 - area / float(h * w)
    return mixed, y_a, y_b, lam_adj


# ─── Training / Evaluation ─────────────────────────────────────────────────────

def _amp_ctx(device):
    if _NEW_AMP:
        return autocast(device_type='cuda', enabled=(device.type == 'cuda'))
    return autocast(enabled=(device.type == 'cuda'))


def train_one_epoch(model, loader, focal, center_loss, sce,
                    optimizer, center_optimizer, scaler, device,
                    lam_center, lam_sce,
                    mixup_prob=0.0, cutmix_prob=0.0,
                    mixup_alpha=0.2, cutmix_alpha=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        if len(batch) == 3:
            images, labels, landmarks = batch
            landmarks = landmarks.to(device)
        else:
            images, labels = batch
            landmarks = None
        images = images.to(device)
        labels = labels.to(device)

        images, labels_a, labels_b, lam = class_aware_mixcut(
            images, labels,
            mixup_prob=mixup_prob,
            cutmix_prob=cutmix_prob,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            target_classes=(1, 2),
        )

        with _amp_ctx(device):
            logits, features = model(images, landmarks=landmarks,
                                     return_features=True)
            if lam < 1.0:
                loss_focal = lam * focal(logits, labels_a) + (1.0 - lam) * focal(logits, labels_b)
                loss_sce = lam * sce(logits, labels_a) + (1.0 - lam) * sce(logits, labels_b)
            else:
                loss_focal = focal(logits, labels)
                loss_sce = sce(logits, labels)
            loss_center = center_loss(features, labels)
            loss = loss_focal + lam_center * loss_center + lam_sce * loss_sce

        optimizer.zero_grad()
        center_optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.step(center_optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_pred, all_label = [], []
    for batch in loader:
        if len(batch) == 3:
            images, labels, landmarks = batch
            landmarks = landmarks.to(device)
        else:
            images, labels = batch
            landmarks = None
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images, landmarks=landmarks)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += images.size(0)
        all_pred.append(pred.cpu().numpy())
        all_label.append(labels.cpu().numpy())
    acc = correct / total
    macro_f1 = f1_score(np.concatenate(all_label), np.concatenate(all_pred),
                        average='macro')
    return acc, macro_f1


def run_seed(args, seed):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr, va, te, class_counts, class_weights, meta = create_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_ratio=0.1,
        seed=seed,
        landmarks_dir=args.landmarks_dir,
        strong_fd_aug=(not args.disable_strong_fd_aug),
        fd_aug_prob=args.fd_aug_prob,
    )

    model_module = importlib.import_module(
        'model' if args.model_type == 'main' else 'baseline_model'
    )
    build_model = model_module.build_model

    model = build_model(
        num_classes=7, pretrained=not args.no_pretrained, dropout=args.dropout,
    ).to(device)

    # Load Stage 1 checkpoint if provided
    if args.stage1_ckpt and os.path.exists(args.stage1_ckpt):
        ckpt = torch.load(args.stage1_ckpt, map_location=device,
                          weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f'Loaded Stage 1: {args.stage1_ckpt}')

    # Losses
    focal = FocalLoss(gamma=args.focal_gamma,
                      label_smoothing=args.label_smoothing,
                      weight=class_weights.to(device))
    base_weight = class_weights.to(device)
    final_weight = base_weight.clone()
    final_weight[1] *= args.final_fear_boost
    final_weight[2] *= args.final_disgust_boost
    final_weight = final_weight / final_weight.mean()
    center_loss = CenterLoss(num_classes=7, feat_dim=model.feat_dim).to(device)
    sce = SCELoss(alpha=1.0, beta=args.sce_beta, num_classes=7)

    # Two optimizers: one for model, one for center loss
    param_groups = model.get_param_groups(
        backbone_lr=args.backbone_lr, head_lr=args.head_lr)
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    center_optimizer = torch.optim.SGD(center_loss.parameters(), lr=args.center_lr)

    # T_max controls when cosine reaches minimum LR.
    # Set independently from epochs so LR fully anneals at the right time.
    t_max = args.cosine_t_max if args.cosine_t_max > 0 else args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-7)

    if _NEW_AMP:
        scaler = GradScaler(device='cuda', enabled=(device.type == 'cuda'))
    else:
        scaler = GradScaler(enabled=(device.type == 'cuda'))

    best_val, best_epoch, patience_counter = 0.0, 0, 0
    best_state = None

    model_label = (
        'IR-50 + CBAM + SA + CenterLoss + SCE'
        if args.model_type == 'main'
        else 'ResNet18 Baseline + CenterLoss + SCE'
    )
    print(f'\n{"="*72}')
    print(f'Seed {seed} | {model_label}')
    print(f'LR: backbone={args.backbone_lr} head={args.head_lr} center={args.center_lr}')
    print(f'Loss: Focal(γ={args.focal_gamma}) + {args.lam_center}*Center + {args.lam_sce}*SCE')
    print(f'Counts: {meta.get("expanded_counts", meta.get("original_counts", "?"))}')
    print(f'{"="*72}')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        final_phase = epoch > (args.epochs - args.final_tune_epochs)
        if final_phase:
            focal.weight = final_weight
            lam_center_cur = args.lam_center * args.final_lam_center_mult
            lam_sce_cur = args.lam_sce * args.final_lam_sce_mult
        else:
            focal.weight = base_weight
            lam_center_cur = args.lam_center
            lam_sce_cur = args.lam_sce

        tr_loss, tr_acc = train_one_epoch(
            model, tr, focal, center_loss, sce,
            optimizer, center_optimizer, scaler, device,
            lam_center_cur, lam_sce_cur,
            mixup_prob=args.mixup_prob,
            cutmix_prob=args.cutmix_prob,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha)
        va_acc, va_f1 = evaluate(model, va, device)
        scheduler.step()

        lr_info = f"{optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e}"
        print(f'Ep {epoch}/{args.epochs} ({time.time()-t0:.1f}s) | '
              f'lr={lr_info} | Train {tr_loss:.4f}/{tr_acc:.4f} | '
              f'Val acc/f1 {va_acc:.4f}/{va_f1:.4f}')

        monitor = va_f1 if args.early_stop_metric == 'macro_f1' else va_acc
        if monitor > best_val:
            best_val = monitor
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print('Early stopping.')
                break

    model.load_state_dict(best_state)
    te_acc, te_f1 = evaluate(model, te, device)

    ckpt_path = os.path.join(args.save_dir, f'best_seed{seed}.pth')
    torch.save({
        'seed': seed,
        'best_val_acc': best_val,
        'best_epoch': best_epoch,
        'test_acc': te_acc,
        'test_f1_macro': te_f1,
        'model_state_dict': model.state_dict(),
        'args': vars(args),
    }, ckpt_path)

    print(f'Seed {seed} → val={best_val:.4f}@ep{best_epoch} test={te_acc:.4f}')
    return best_val, te_acc, ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../RAF-DBdataset')
    parser.add_argument('--save_dir', type=str, default='checkpoints_v7b')
    parser.add_argument('--model_type', type=str, default='main',
                        choices=['main', 'baseline'])

    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--cosine_t_max', type=int, default=80,
                        help='Cosine annealing cycle length (0=same as epochs)')
    parser.add_argument('--early_stop_metric', type=str, default='acc',
                        choices=['acc', 'macro_f1'])

    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--backbone_lr', type=float, default=2e-6)
    parser.add_argument('--head_lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.4)

    parser.add_argument('--focal_gamma', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--lam_center', type=float, default=0.01,
                        help='Center Loss weight')
    parser.add_argument('--center_lr', type=float, default=0.5,
                        help='Learning rate for center vectors')
    parser.add_argument('--lam_sce', type=float, default=0.5,
                        help='SCE Loss weight')
    parser.add_argument('--sce_beta', type=float, default=0.5,
                        help='SCE reverse CE weight')
    parser.add_argument('--mixup_prob', type=float, default=0.4)
    parser.add_argument('--cutmix_prob', type=float, default=0.3)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    parser.add_argument('--final_tune_epochs', type=int, default=30)
    parser.add_argument('--final_fear_boost', type=float, default=1.25)
    parser.add_argument('--final_disgust_boost', type=float, default=1.35)
    parser.add_argument('--final_lam_center_mult', type=float, default=1.15)
    parser.add_argument('--final_lam_sce_mult', type=float, default=0.9)

    parser.add_argument('--stage1_ckpt', type=str, default=None)
    parser.add_argument('--landmarks_dir', type=str, default=None,
                        help='Directory with landmarks JSON files (None=no landmarks)')
    parser.add_argument('--disable_strong_fd_aug', action='store_true',
                        help='Disable extra Fear/Disgust-only augmentation branch')
    parser.add_argument('--fd_aug_prob', type=float, default=0.6)
    parser.add_argument('--seeds', type=str, default='0')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

    all_val, all_test, paths = [], [], []
    for seed in seeds:
        v, t, p = run_seed(args, seed)
        all_val.append(v)
        all_test.append(t)
        paths.append(p)

    all_val, all_test = np.array(all_val), np.array(all_test)

    csv_path = os.path.join(args.save_dir, 'seed_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'val_acc', 'test_acc', 'checkpoint'])
        for s, v, t, p in zip(seeds, all_val, all_test, paths):
            w.writerow([s, f'{v:.6f}', f'{t:.6f}', p])
        w.writerow(['mean', f'{all_val.mean():.6f}', f'{all_test.mean():.6f}', ''])
        w.writerow(['std', f'{all_val.std():.6f}', f'{all_test.std():.6f}', ''])

    print(f'\n{"="*60}')
    print(f'Val:  {all_val.mean():.4f} +/- {all_val.std():.4f}')
    print(f'Test: {all_test.mean():.4f} +/- {all_test.std():.4f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
