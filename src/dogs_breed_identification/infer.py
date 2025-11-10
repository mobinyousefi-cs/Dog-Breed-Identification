#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Dog's Breed Identification
File: infer.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Run inference with a trained model on one or more images.

Usage:
    python -m dogs_breed_identification.infer \
        --checkpoint models/resnet50_dogs_breed_baseline_best.pt \
        --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms

from .models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer dog breed from image(s)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to input image(s).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size to which inputs will be resized.",
    )
    return parser.parse_args()


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes: List[str] = checkpoint["label_classes"]

    num_classes = len(classes)
    model = build_model("resnet50", num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tf = _build_transform(args.image_size)

    for img_path_str in args.image:
        img_path = Path(img_path_str)
        img = Image.open(img_path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            pred_class = classes[idx]
            pred_prob = float(probs[idx].item())

        print(f"{img_path}: {pred_class} ({pred_prob:.4f})")


if __name__ == "__main__: 

    main()
