"""
prfe — Pontryagin Random Feature Embedding

Top-level exports:
    layers    — embedding and feature-map modules
    models    — end-to-end model classes
    training  — trainers and loss utilities
    utils     — kernel helpers and random sampling
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("prfe")
except PackageNotFoundError:
    __version__ = "0.1.0.dev"

from prfe import layers, losses, models, training, utils

__all__ = ["layers", "losses", "models", "training", "utils", "__version__"]
