#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Dog's Breed Identification
File: metrics.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Evaluation metrics utilities.
"""

from __future__ import annotations

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute top-1 accuracy for a batch."""

    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = (preds == target).sum().item()
        return correct / target.size(0)