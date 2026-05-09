"""Config for ConvNeXtV2-Tiny classification on Sugarcane Leaf Disease Dataset.

Three models share the same ConvNeXtV2-Tiny backbone and training protocol:
    euclidean          — ConvNeXtV2-Tiny + fine-tuned linear head (CE + label smoothing)
    pontryagin         — ConvNeXtV2-Tiny + PontryaginEmbedding + PontryaginMLR
                         (CE + L_balance_W + L_topo)
    pontryagin_margin  — Same architecture as pontryagin but with
                         PontryaginMarginCLS head
                         (CE + L_balance_W + L_margin_proto + L_orth_W)

Kaggle dataset: nirmalsankalana/sugarcane-leaf-disease-dataset
Expected classes (resolved at runtime from folder names):
    Healthy, Mosaic, RedRot, Rust, Yellow  → 5 classes
"""

from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parents[1] / "results" / "cls_sugarcane"

# DATASET_ROOT is set at runtime by run_cls_sugarcane.py after kagglehub download.
# Override via --data-root CLI arg or SUGARCANE_DATA_ROOT env var.
DATASET_ROOT = None

# ── dataset ───────────────────────────────────────────────────────────────────
IMG_SIZE   = 224   # ConvNeXtV2-Tiny default resolution
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.15
SEED       = 42

# N_CLASSES and CLASS_NAMES resolved at runtime from folder structure.
N_CLASSES   = None
CLASS_NAMES = None

# ── ConvNeXtV2-Tiny backbone ──────────────────────────────────────────────────
BACKBONE_NAME   = "convnextv2_tiny.fcmae_ft_in22k_in1k"
BACKBONE_OUT_CH = 768   # channels of last ConvNeXt stage before global pool
PRETRAINED      = True

# ── Pontryagin-specific defaults ──────────────────────────────────────────────
# srf_multiplier: fraction of BACKBONE_OUT_CH used as negative-subspace dimension.
#   n_srf = kappa = max(1, round(srf_multiplier * BACKBONE_OUT_CH))
#   e.g. 0.05 → 38 dims,  0.25 → 192 dims,  0.5 → 384 dims,  0.8 → 614 dims
SRF_MULTIPLIER = 0.05
RFF_MULTIPLIER = 1
N_RFF          = RFF_MULTIPLIER * BACKBONE_OUT_CH
D_POLY         = 2
SIGMA          = 1.0
LAMBDA_BALANCE = 0.1
LAMBDA_TOPO    = 0.05
TOPO_KWARGS    = {"lambda_lc": 1.0, "lambda_cc": 0.5, "lambda_sb": 0.5, "lc_epsilon": 0.1}

# ── Pontryagin margin head defaults (pontryagin_margin only) ──────────────────
LAMBDA_MARGIN  = 0.5    # weight for batch-prototype margin penalty
LAMBDA_ORTH_W  = 0.1    # weight for J-orthogonality on classifier W
MARGIN         = 1.0    # J-pseudo-distance margin m

# ── derived (not tuned directly) ──────────────────────────────────────────────
def n_srf_from_multiplier(srf_multiplier: float) -> int:
    return max(1, round(srf_multiplier * BACKBONE_OUT_CH))

# ── training ──────────────────────────────────────────────────────────────────
DEVICE               = "cuda"
BATCH_SIZE           = 32
NUM_WORKERS          = 4
EPOCHS               = 30
LR                   = 3e-4    # head / embedding learning rate
LR_BACKBONE          = 3e-5   # discriminative lr for pretrained backbone
LR_MIN               = 1e-6
WEIGHT_DECAY         = 1e-4
EARLY_STOP_PATIENCE  = 10
LABEL_SMOOTHING      = 0.1    # for euclidean CE loss

AVAILABLE_MODELS = ("euclidean", "pontryagin", "pontryagin_margin")
