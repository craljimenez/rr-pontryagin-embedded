"""Config for UNet-backbone segmentation experiments.

Four models share the same UNetBackbone and dataset:
    vanilla     — standard UNet + CE (no class weights)
    euclidean   — UNet + class-weighted CE (linear head)
    hyperbolic  — UNet + Poincaré-ball MLR
    pontryagin  — UNet + PontryaginEmbedding + PontryaginMLR (no cone penalty)
"""

from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(__file__).parents[3] / "002_rff_ss" / "DATASETS" / "UAV_segmantation"
RESULTS_DIR  = Path(__file__).parents[1] / "results" / "seg_uav_unet"

# ── dataset ───────────────────────────────────────────────────────────────────
IMG_SIZE             = 256
BINARY_SUGARCANE     = True
BINARY_ONLY_SUGARCANE = True
N_CLASSES            = 2
CLASS_NAMES          = ["background", "Sugarcane"]
CLASS_WEIGHTS        = [0.20, 1.00]

# ── UNet backbone ─────────────────────────────────────────────────────────────
BACKBONE_OUT_CH = 64    # output channels of UNetBackbone (fed to embedding)
UNET_BASE_CH    = 64    # first encoder stage channels; doubles each stage
UNET_DEPTH      = 4     # encoder/decoder stages (3–5)

# ── Pontryagin-specific defaults ─────────────────────────────────────────────
KAPPA          = 4
D_POLY         = 2
RFF_MULTIPLIER = 1
N_RFF          = RFF_MULTIPLIER * BACKBONE_OUT_CH
SIGMA          = 1.0
LAMBDA_BALANCE = 0.1
LAMBDA_TOPO    = 0.05
TOPO_KWARGS    = {"lambda_lc": 1.0, "lambda_cc": 0.5, "lambda_sb": 0.5, "lc_epsilon": 0.1}

# ── Hyperbolic-specific ───────────────────────────────────────────────────────
HYPERBOLIC_C = 1.0

# ── training ──────────────────────────────────────────────────────────────────
DEVICE               = "cuda"
BATCH_SIZE           = 2
NUM_WORKERS          = 4
EPOCHS               = 60
LR                   = 1e-3
LR_MIN               = 1e-5
WEIGHT_DECAY         = 1e-4
EARLY_STOP_PATIENCE  = 15

AVAILABLE_MODELS = ("vanilla", "euclidean", "hyperbolic", "pontryagin")
