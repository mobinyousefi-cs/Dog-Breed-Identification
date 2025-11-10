#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Dog's Breed Identification
File: test_data_loader.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Lightweight tests for the data loader.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dogs_breed_identification.config import Config
from dogs_breed_identification.data_loader import create_dataloaders


@pytest.mark.skipif(
    not Path("configs/training_config.yaml").exists(),
    reason="Config or data not available in CI.",
)
def test_create_dataloaders_runs():
    cfg = Config.from_yaml("configs/training_config.yaml")
    assert cfg.paths.labels_csv.exists(), "labels.csv must exist for this test."

    train_loader, val_loader, le = create_dataloaders(cfg)

    assert len(le.classes_) > 0
    assert len(train_loader) > 0
    assert len(val_loader) > 0