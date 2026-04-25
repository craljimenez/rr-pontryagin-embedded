"""Hyperparameter config for UAV segmentation experiment.

Three heads share the same backbone, dataset, and training schedule:
    • euclidean  — classical FCN (linear 1×1 conv classifier)
    • hyperbolic — Ganea/Atigh Poincaré-ball MLR
    • pontryagin — this work
"""

from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(__file__).parents[3] / "002_rff_ss" / "DATASETS" / "UAV_segmantation"
RESULTS_DIR  = Path(__file__).parents[1] / "results" / "seg_uav"

# ── dataset ───────────────────────────────────────────────────────────────────
IMG_SIZE    = 256
BINARY_SUGARCANE = True      # collapse the problem to sugarcane vs background
BINARY_ONLY_SUGARCANE = True # drop images without sugarcane from the dataset
N_CLASSES   = 2              # 0=background, 1=Sugarcane
CLASS_NAMES = [
    "background",
    "Sugarcane",
]

# ── class weights for imbalanced CE ──────────────────────────────────────────
# Background is common; sugarcane is the positive minority class.
CLASS_WEIGHTS = [
    0.20,
    1.00,
]

# ── model ─────────────────────────────────────────────────────────────────────
BACKBONE_OUT_CH = 64      # channels produced by the backbone decoder
BACKBONE_DEPTH  = 4       # number of strided conv blocks in the backbone

# Pontryagin-specific
#   KAPPA          = κ = signature = N_SRF (number of negative dims in J)
#   D_POLY         = polynomial degree of K_-(x,y)=(xᵀy)^{d_poly}, independent of κ
#   N_RFF          = RFF_MULTIPLIER × BACKBONE_OUT_CH  (positive subspace width)
KAPPA           = 2       # Pontryagin signature κ = N_SRF
D_POLY          = 2       # SRF polynomial degree (tuned independently of κ)
RFF_MULTIPLIER  = 1       # N_RFF = RFF_MULTIPLIER * BACKBONE_OUT_CH
N_RFF           = RFF_MULTIPLIER * BACKBONE_OUT_CH  # RFF half-dim → p = 2*N_RFF
SIGMA           = 1.0     # RBF bandwidth for K_+
LAMBDA_CONE_W   = 0.0     # disabled — cone penalty kills the negative subspace
LAMBDA_BALANCE  = 0.1     # weight RMS balance: (rms(W_+) − rms(W_−))²
LAMBDA_TOPO     = 0.05
TOPO_KWARGS     = {"lambda_lc": 1.0, "lambda_cc": 0.5, "lambda_sb": 0.5, "lc_epsilon": 0.1}

# Hyperbolic-specific
HYPERBOLIC_C    = 1.0     # curvature of the Poincaré ball

# ── training ──────────────────────────────────────────────────────────────────
DEVICE      = "cuda"
BATCH_SIZE  = 8
NUM_WORKERS = 4
EPOCHS      = 60
LR          = 1e-3
LR_MIN      = 1e-5        # cosine annealing floor
WEIGHT_DECAY = 1e-4

# Early stopping on val mIoU (macro over non-empty classes incl. background)
EARLY_STOP_PATIENCE = 15  # epochs without improvement before aborting

# Models available through the CLI
AVAILABLE_MODELS = ("euclidean", "hyperbolic", "pontryagin")
