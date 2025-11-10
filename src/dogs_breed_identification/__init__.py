#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Dog's Breed Identification
File: __init__.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-10
Updated: 2025-11-10
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Package initializer for the Dog's Breed Identification project.

Usage:
Import from this package, for example:

    from dogs_breed_identification.config import TrainingConfig

Notes:
- This file exposes the main public API if needed.
"""

from .config import Config

__all__ = ["Config"]