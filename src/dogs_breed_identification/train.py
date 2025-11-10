#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Dog's Breed Identification
File: train.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Main training entry point for Dog's Breed Identification.

Usage:
    python -m dogs_breed_identification.train \
        --config configs/training_config.yaml

Notes:
- Expects Kaggle dataset downloaded and extracted under the configured paths.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

from .config import Config
from .data_loader import create_dataloaders
from .models import build_model
from .utils import accuracy, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dog's Breed Identification model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_acc = accuracy(outputs, labels)

        running_loss += batch_loss * images.size(0)
        running_acc += batch_acc * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)

    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_loss = loss.item()
            batch_acc = accuracy(outputs, labels)

            running_loss += batch_loss * images.size(0)
            running_acc += batch_acc * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)

    return epoch_loss, epoch_acc


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.ensure_dirs()

    set_seed(cfg.training.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(cfg.paths.logs_dir) / f"{cfg.experiment_name}_{timestamp}.log"
    logger = setup_logging(log_file)

    logger.info("Configuration loaded: %s", cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, val_loader, label_encoder = create_dataloaders(cfg)
    num_classes = len(label_encoder.classes_)
    logger.info("Number of classes: %d", num_classes)

    model = build_model(
        backbone=cfg.model.name,  # type: ignore[arg-type]
        num_classes=num_classes,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = StepLR(
        optimizer,
        step_size=cfg.training.step_size,
        gamma=cfg.training.gamma,
    )

    best_val_acc = 0.0
    best_model_path = Path(cfg.paths.model_dir) / f"{cfg.experiment_name}_best.pt"

    for epoch in range(1, cfg.training.num_epochs + 1):
        logger.info("Epoch %d/%d", epoch, cfg.training.num_epochs)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )
        scheduler.step()

        logger.info(
            "Epoch %d | Train Loss: %.4f | Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_classes": label_encoder.classes_.tolist(),
                    "config": cfg.__dict__,
                },
                best_model_path,
            )
            logger.info("New best model saved to %s", best_model_path)

    logger.info("Training finished. Best Val Acc: %.4f", best_val_acc)


if __name__ == "__main__":
    main()