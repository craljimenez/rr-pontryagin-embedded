"""Config for 1-way K-shot few-shot segmentation on the UAV Sugarcane dataset.

Two models share the same backbone, dataset, and training protocol:
    euclidean   — UNetBackbone + cosine similarity (PANet baseline)
    pontryagin  — UNetBackbone + PontryaginEmbedding + J-inner product

Dimensioned for Colab Pro (T4 16 GB) or local GPU.
At BATCH_EPISODES=4 and K_SHOT=1 the peak memory is ~4–6 GB.
K_SHOT=5 roughly triples support memory, still fits a T4 at BATCH_EPISODES=4.
"""

from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(__file__).parents[3] / "002_rff_ss" / "DATASETS" / "UAV_segmantation"
RESULTS_DIR  = Path(__file__).parents[1] / "results" / "fss_sugarcane"

# ── dataset ───────────────────────────────────────────────────────────────────
IMG_SIZE        = 256
TARGET_CLS      = 5            # YOLO class index: Sugarcane
N_EPISODES_TRAIN = 400         # virtual epoch length for train split
N_EPISODES_VAL   = 100         # fixed episodes for val (seed-reproducible)
N_EPISODES_TEST  = 200         # fixed episodes for test

# K-shot variants evaluated; training always uses K_SHOT
K_SHOT          = 1            # change to 5 for 5-shot run

# ── backbone (UNet) ───────────────────────────────────────────────────────────
BASE_CH         = 32           # first encoder block channels
DEPTH           = 3            # number of encoder / decoder stages

# ── Pontryagin-specific ───────────────────────────────────────────────────────
N_RFF           = 16           # RFF half-dim  → positive subspace p = 2*N_RFF = 32
N_SRF           = 4            # SRF dim = kappa (negative subspace q = 4)
KAPPA           = 2            # Pontryagin signature κ
SIGMA           = 1.0          # RBF bandwidth for K_+
TRAINABLE_RFF   = False        # set True to make RFF/SRF weights learnable

# ── Pontryagin loss weights ───────────────────────────────────────────────────
LAMBDA_CONE     = 0.1
LAMBDA_ORTH     = 0.05
CONE_EPSILON    = 0.1

# ── shared loss ───────────────────────────────────────────────────────────────
BCE_WEIGHT      = 0.5          # weight on BCE; Dice gets (1 - BCE_WEIGHT)
POS_WEIGHT      = 5.0          # class-imbalance correction for BCE

# ── training ──────────────────────────────────────────────────────────────────
DEVICE          = "cuda"
BATCH_EPISODES  = 4            # episodes per gradient step (≤8 for T4 safety)
NUM_WORKERS     = 2
EPOCHS          = 50
LR              = 1e-3
LR_MIN          = 1e-5         # cosine annealing floor
WEIGHT_DECAY    = 1e-4
EARLY_STOP_PATIENCE = 10       # epochs without val IoU improvement

AVAILABLE_MODELS = ("euclidean", "pontryagin")
