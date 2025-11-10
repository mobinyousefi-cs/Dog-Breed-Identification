#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Dog's Breed Identification
File: config.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Configuration dataclasses and helpers for the Dog's Breed Identification project.

Usage:
    from dogs_breed_identification.config import Config
    cfg = Config.from_yaml("configs/training_config.yaml")

Notes:
- Keeps all hyperparameters and paths in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PathsConfig:
    train_dir: Path
    test_dir: Path
    labels_csv: Path
    model_dir: Path
    logs_dir: Path


@dataclass
class ModelConfig:
    name: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_workers: int = 4
    num_epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    step_size: int = 7
    gamma: float = 0.1
    seed: int = 42
    val_split: float = 0.15


@dataclass
class AugmentationsConfig:
    image_size: int = 224


@dataclass
class Config:
    experiment_name: str
    paths: PathsConfig
    model: ModelConfig
    training: TrainingConfig
    augmentations: AugmentationsConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        paths_cfg = PathsConfig(
            train_dir=Path(raw["paths"]["train_dir"]),
            test_dir=Path(raw["paths"]["test_dir"]),
            labels_csv=Path(raw["paths"]["labels_csv"]),
            model_dir=Path(raw["paths"]["model_dir"]),
            logs_dir=Path(raw["paths"]["logs_dir"]),
        )

        model_cfg = ModelConfig(**raw.get("model", {}))
        training_cfg = TrainingConfig(**raw.get("training", {}))
        aug_cfg = AugmentationsConfig(**raw.get("augmentations", {}))

        return cls(
            experiment_name=raw.get("experiment_name", "experiment"),
            paths=paths_cfg,
            model=model_cfg,
            training=training_cfg,
            augmentations=aug_cfg,
        )

    def ensure_dirs(self) -> None:
        """Create required directories if they do not exist."""

        self.paths.model_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)