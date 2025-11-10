#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Dog's Breed Identification
File: test_models.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Simple tests for model construction and forward pass.
"""

from __future__ import annotations

import torch

from dogs_breed_identification.models import build_model


def test_build_model_forward():
    num_classes = 5
    model = build_model("resnet18", num_classes=num_classes, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    assert y.shape == (2, num_classes)