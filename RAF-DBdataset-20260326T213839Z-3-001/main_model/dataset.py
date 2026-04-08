"""
RAF-DB data pipeline with landmark support.
- Returns (image, label, landmarks) per sample
- Landmarks: 5 points (left_eye, right_eye, nose, left_mouth, right_mouth)
- Loaded from precomputed JSON files
"""

import os
import csv
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


EMOTION_NAMES = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

# Default landmarks (center of face) for images where MTCNN fails
DEFAULT_LANDMARKS = [
    [33.6, 33.6], [78.4, 33.6],   # eyes (30%, 70% of 112)
    [56.0, 56.0],                   # nose (50%)
    [33.6, 84.0], [78.4, 84.0],   # mouth (30%, 70%, 75%)
]


class RAFDBDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, indices=None,
                 landmarks_json=None, strong_fd_aug=False, fd_aug_prob=0.6):
        self.samples = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row['image'], int(row['label'])))
        self.img_dir = img_dir
        self.transform = transform
        self.indices = indices
        self.strong_fd_aug = strong_fd_aug
        self.fd_aug_prob = fd_aug_prob

        # Load landmarks
        self.landmarks = {}
        if landmarks_json and os.path.exists(landmarks_json):
            with open(landmarks_json) as f:
                self.landmarks = json.load(f)

        # Extra augmentation only for minority hard classes:
        # Fear(1) and Disgust(2) in 1-based RAF-DB labels.
        self.fd_extra_aug = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=18, translate=(0.08, 0.08),
                    scale=(0.92, 1.08), shear=8
                )
            ], p=0.8),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.35, contrast=0.35,
                    saturation=0.25, hue=0.08
                )
            ], p=0.8),
        ])

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.samples)

    def _resolve_path(self, image_name, label_1b):
        p1 = os.path.join(self.img_dir, image_name)
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(self.img_dir, str(label_1b), image_name)
        if os.path.exists(p2):
            return p2
        raise FileNotFoundError(f'Cannot find {image_name} under {self.img_dir}')

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        image_name, label_1b = self.samples[real_idx]
        img_path = self._resolve_path(image_name, label_1b)
        img = Image.open(img_path).convert('RGB')

        if self.strong_fd_aug and label_1b in (2, 3) and np.random.rand() < self.fd_aug_prob:
            img = self.fd_extra_aug(img)

        # Get original image size before transform
        orig_w, orig_h = img.size

        # Get landmarks in original image coordinates
        lm = self.landmarks.get(image_name, DEFAULT_LANDMARKS)
        lm = torch.tensor(lm, dtype=torch.float32)  # (5, 2)

        # Scale landmarks to 112×112 (model input size)
        # After transform, image becomes 112×112
        lm[:, 0] = lm[:, 0] / orig_w * 112.0
        lm[:, 1] = lm[:, 1] / orig_h * 112.0

        if self.transform is not None:
            img = self.transform(img)

        return img, label_1b - 1, lm

    def get_all_labels(self):
        return [label - 1 for _, label in self.samples]


def get_transforms(split='train', image_size=112):
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0),
                                         ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.18)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def stratified_split_indices(labels, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    tr, va = [], []
    for cls in range(7):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_va = max(1, int(round(len(idx) * val_ratio)))
        va.extend(idx[:n_va].tolist())
        tr.extend(idx[n_va:].tolist())
    rng.shuffle(tr)
    rng.shuffle(va)
    return np.array(tr, dtype=np.int64), np.array(va, dtype=np.int64)


def moderate_oversample(indices, labels, cap=2000, seed=42):
    rng = np.random.default_rng(seed)
    by_class = {}
    for idx in indices:
        c = int(labels[idx])
        by_class.setdefault(c, []).append(int(idx))

    expanded = []
    for cls in range(7):
        cls_idx = by_class.get(cls, [])
        if not cls_idx:
            continue
        if len(cls_idx) >= cap:
            expanded.extend(cls_idx)
        else:
            extra = rng.choice(cls_idx, size=cap - len(cls_idx),
                               replace=True).tolist()
            expanded.extend(cls_idx + extra)
    rng.shuffle(expanded)
    return np.array(expanded, dtype=np.int64)


def get_class_weights(labels, num_classes=7, boost_classes=None):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = 1.0 / (counts + 1e-8)
    if boost_classes:
        for cls, mult in boost_classes.items():
            w[cls] *= mult
    w = w / w.mean()
    return torch.FloatTensor(w)


def create_dataloaders(data_root, batch_size=32, image_size=112,
                       num_workers=4, val_ratio=0.1, seed=42,
                       oversample_cap=2000,
                       landmarks_dir=None,
                       strong_fd_aug=True,
                       fd_aug_prob=0.6,
                       **kwargs):
    """
    landmarks_dir: directory containing landmarks_train.json and landmarks_test.json
    """
    train_csv = os.path.join(data_root, 'train_labels.csv')
    test_csv = os.path.join(data_root, 'test_labels.csv')
    train_dir = os.path.join(data_root, 'DATASET', 'train')
    test_dir = os.path.join(data_root, 'DATASET', 'test')

    # Landmark files
    lm_train = os.path.join(landmarks_dir, 'landmarks_train.json') \
        if landmarks_dir else None
    lm_test = os.path.join(landmarks_dir, 'landmarks_test.json') \
        if landmarks_dir else None

    full_train = RAFDBDataset(train_csv, train_dir)
    all_labels = np.array(full_train.get_all_labels(), dtype=np.int64)
    tr_idx, va_idx = stratified_split_indices(all_labels, val_ratio, seed)

    tr_expanded = moderate_oversample(tr_idx, all_labels, cap=oversample_cap,
                                      seed=seed)

    train_tf = get_transforms('train', image_size)
    val_tf = get_transforms('val', image_size)
    test_tf = get_transforms('test', image_size)

    train_ds = RAFDBDataset(train_csv, train_dir, train_tf,
                            tr_expanded.tolist(), lm_train,
                            strong_fd_aug=strong_fd_aug,
                            fd_aug_prob=fd_aug_prob)
    val_ds = RAFDBDataset(train_csv, train_dir, val_tf,
                          va_idx.tolist(), lm_train)
    test_ds = RAFDBDataset(test_csv, test_dir, test_tf,
                           None, lm_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    exp_labels = all_labels[tr_expanded]
    class_counts = np.bincount(exp_labels, minlength=7).tolist()
    class_weights = get_class_weights(exp_labels,
                                      boost_classes={1: 1.5, 2: 1.8})

    meta = {
        'original_counts': np.bincount(all_labels[tr_idx], minlength=7).tolist(),
        'expanded_counts': class_counts,
    }
    return train_loader, val_loader, test_loader, class_counts, class_weights, meta
