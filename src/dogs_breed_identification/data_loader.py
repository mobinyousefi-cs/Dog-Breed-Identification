#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Dog's Breed Identification
File: data_loader.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
PyTorch Dataset and DataLoader utilities for the Kaggle Dog Breed Identification dataset.

Usage:
    from dogs_breed_identification.config import Config
    from dogs_breed_identification.data_loader import create_dataloaders

    cfg = Config.from_yaml("configs/training_config.yaml")
    train_loader, val_loader, label_encoder = create_dataloaders(cfg)

Notes:
- Expects the Kaggle train images under cfg.paths.train_dir and labels in cfg.paths.labels_csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from .config import Config


class DogBreedDataset(Dataset):
    """Custom Dataset for Dog Breed images."""

    def __init__(
        self,
        image_paths: list[Path],
        labels: Optional[np.ndarray],
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.image_paths)

    def __getitem__(self, idx: int):  # type: ignore[override]
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is None:
            return img, img_path.stem

        label = self.labels[idx]
        return img, int(label)


def _build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_tf, val_tf


def create_dataloaders(
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, LabelEncoder]:
    """Create train and validation dataloaders and return the fitted LabelEncoder."""

    labels_df = pd.read_csv(cfg.paths.labels_csv)
    labels_df["image_path"] = labels_df["id"].apply(
        lambda x: str(cfg.paths.train_dir / f"{x}.jpg"),
    )

    le = LabelEncoder()
    labels_df["label_id"] = le.fit_transform(labels_df["breed"].values)

    train_df, val_df = train_test_split(
        labels_df,
        test_size=cfg.training.val_split,
        stratify=labels_df["label_id"].values,
        random_state=cfg.training.seed,
    )

    train_paths = [Path(p) for p in train_df["image_path"].tolist()]
    val_paths = [Path(p) for p in val_df["image_path"].tolist()]
    train_labels = train_df["label_id"].values.astype(np.int64)
    val_labels = val_df["label_id"].values.astype(np.int64)

    train_tf, val_tf = _build_transforms(cfg.augmentations.image_size)

    train_dataset = DogBreedDataset(train_paths, train_labels, transform=train_tf)
    val_dataset = DogBreedDataset(val_paths, val_labels, transform=val_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, le