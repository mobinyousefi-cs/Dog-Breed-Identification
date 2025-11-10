#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Dog's Breed Identification
File: models.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Model factory for Dog's Breed Identification using torchvision backbones.

Usage:
    from dogs_breed_identification.models import build_model
    model = build_model("resnet50", num_classes=120)

Notes:
- Uses pretrained ImageNet weights for transfer learning by default.
"""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision import models


BackboneName = Literal["resnet18", "resnet34", "resnet50"]


def build_model(
    backbone: BackboneName,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a classification model with a ResNet backbone."""

    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
    elif backbone == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(in_features, num_classes)

    return model