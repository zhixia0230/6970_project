"""
Evaluation script: confusion matrix, F1-score, Accuracy, MAE, per-class metrics.
Supports Test-Time Augmentation (TTA) for extra accuracy.
"""

import os
import argparse
import importlib
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    accuracy_score, ConfusionMatrixDisplay
)
from dataset import create_dataloaders, EMOTION_NAMES


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        images, labels, landmarks = batch
        images = images.to(device)
        landmarks = landmarks.to(device)
        outputs = model(images, landmarks=landmarks)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())
    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs))


@torch.no_grad()
def get_predictions_tta(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        images, labels, landmarks = batch
        images = images.to(device)
        landmarks = landmarks.to(device)
        out1 = torch.softmax(model(images, landmarks=landmarks), dim=1)
        out2 = torch.softmax(model(torch.flip(images, dims=[3]),
                                   landmarks=landmarks), dim=1)
        probs = (out1 + out2) / 2.0
        preds = probs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())
    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs))


@torch.no_grad()
def get_predictions_tta10(model, loader, device, crop_ratio=0.9):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        images, labels, landmarks = batch
        images = images.to(device)
        landmarks = landmarks.to(device)
        _, _, h, w = images.shape
        ch = max(16, int(h * crop_ratio))
        cw = max(16, int(w * crop_ratio))
        y1 = h - ch
        x1 = w - cw

        v_orig = images
        v_tl = F.interpolate(images[:, :, 0:ch, 0:cw], size=(h, w),
                              mode='bilinear', align_corners=False)
        v_tr = F.interpolate(images[:, :, 0:ch, x1:w], size=(h, w),
                              mode='bilinear', align_corners=False)
        v_bl = F.interpolate(images[:, :, y1:h, 0:cw], size=(h, w),
                              mode='bilinear', align_corners=False)
        v_br = F.interpolate(images[:, :, y1:h, x1:w], size=(h, w),
                              mode='bilinear', align_corners=False)
        views = [v_orig, v_tl, v_tr, v_bl, v_br]

        probs_sum = 0.0
        for v in views:
            probs_sum += torch.softmax(model(v, landmarks=landmarks), dim=1)
            probs_sum += torch.softmax(
                model(torch.flip(v, dims=[3]), landmarks=landmarks), dim=1)
        probs = probs_sum / 10.0

        preds = probs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())
    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs))


def compute_mae(preds, labels):
    return np.mean(np.abs(preds.astype(float) - labels.astype(float)))


def main():
    parser = argparse.ArgumentParser(description='Evaluate Dual-EmoNet-style Main Model')
    parser.add_argument('--data_root', type=str, default='../RAF-DBdataset')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='main',
                        choices=['main', 'baseline'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tta', action='store_true', help='Enable Test-Time Augmentation')
    parser.add_argument('--tta_mode', type=str, default='hflip', choices=['hflip', 'tencrop'])
    parser.add_argument('--tta_crop_ratio', type=float, default=0.9)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--landmarks_dir', type=str, default='.',
                        help='Directory with landmarks JSON files')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    _, _, test_loader, _, _, _ = create_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_ratio=0.1,
        seed=args.seed,
        landmarks_dir=args.landmarks_dir,
    )

    # Model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    ckpt_args = ckpt.get('args', {})
    model_type = ckpt_args.get('model_type', args.model_type)
    model_module = importlib.import_module(
        'model' if model_type == 'main' else 'baseline_model'
    )
    build_model = model_module.build_model
    model = build_model(
        num_classes=7,
        pretrained=not bool(ckpt_args.get('no_pretrained', False)),
        dropout=float(ckpt_args.get('dropout', args.dropout)),
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f'Missing keys: {len(missing)} (showing first 5) {missing[:5]}')
    if unexpected:
        print(f'Unexpected keys: {len(unexpected)} (showing first 5) {unexpected[:5]}')
    print(
        f"Loaded checkpoint seed={ckpt.get('seed', 'NA')} "
        f"best_epoch={ckpt.get('best_epoch', 'NA')} "
        f"best_val={ckpt.get('best_val_acc', 'NA')}"
    )

    # Predictions
    if args.tta:
        if args.tta_mode == 'tencrop':
            print(f'Using 10-view TTA (orig+4 corners+flips), crop_ratio={args.tta_crop_ratio}')
            preds, labels, probs = get_predictions_tta10(
                model, test_loader, device, crop_ratio=args.tta_crop_ratio
            )
        else:
            print("Using 2-view TTA (original + hflip)")
            preds, labels, probs = get_predictions_tta(model, test_loader, device)
    else:
        preds, labels, probs = get_predictions(model, test_loader, device)

    # Metrics
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    mae = compute_mae(preds, labels)

    print(f"\n{'='*50}")
    print(f"Test Accuracy:      {acc:.4f}")
    print(f"F1 (macro):         {f1_macro:.4f}")
    print(f"F1 (weighted):      {f1_weighted:.4f}")
    print(f"MAE:                {mae:.4f}")
    print(f"{'='*50}\n")

    report = classification_report(labels, preds, target_names=EMOTION_NAMES, digits=4)
    print(report)

    # Save metrics
    with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 macro: {f1_macro:.4f}\n")
        f.write(f"F1 weighted: {f1_weighted:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"TTA: {args.tta}\n")
        f.write(f"TTA mode: {args.tta_mode if args.tta else 'none'}\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=EMOTION_NAMES)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix - Main Model (CNN+Attention)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'confusion_matrix.png'), dpi=150)
    print(f"Saved: {args.save_dir}/confusion_matrix.png")

    # Confidence distribution
    max_probs = probs.max(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(max_probs[preds == labels], bins=50, alpha=0.7, label='Correct', color='green')
    ax.hist(max_probs[preds != labels], bins=50, alpha=0.7, label='Wrong', color='red')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'confidence_dist.png'), dpi=150)
    print(f"Saved: {args.save_dir}/confidence_dist.png")


if __name__ == '__main__':
    main()
