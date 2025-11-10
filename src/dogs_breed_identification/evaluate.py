#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Dog's Breed Identification
File: evaluate.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Evaluate a trained model on the validation set (or any labeled set using the same format).

Usage:
    python -m dogs_breed_identification.evaluate \
        --config configs/training_config.yaml \
        --checkpoint models/resnet50_dogs_breed_baseline_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import Config
from .data_loader import create_dataloaders
from .models import build_model
from .utils import accuracy, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Dog's Breed Identification model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    logger = setup_logging(Path(cfg.paths.logs_dir) / "evaluate.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    _, val_loader, label_encoder = create_dataloaders(cfg)
    num_classes = len(label_encoder.classes_)

    model = build_model(
        backbone=cfg.model.name,  # type: ignore[arg-type]
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            batch_acc = accuracy(outputs, labels)

            total_correct += batch_acc * images.size(0)
            total_samples += images.size(0)

    final_acc = total_correct / total_samples if total_samples > 0 else 0.0
    logger.info("Validation Accuracy: %.4f", final_acc)


if __name__ == "__main__":
    main()