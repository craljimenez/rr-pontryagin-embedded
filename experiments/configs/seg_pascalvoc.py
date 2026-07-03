"""Config for UNet-backbone segmentation on PASCAL VOC 2012 (21 classes).

Public multiclass benchmark used to test PRFE generalisation beyond the
binary sugarcane-vs-background UAV segmentation task. Four models share the
same backbone and dataset (PretrainedUNetBackbone by default — see
PRETRAINED_BACKBONE below):
    vanilla     — standard UNet + CE (no class weights)
    euclidean   — UNet + class-weighted CE (linear head)
    hyperbolic  — UNet + Poincaré-ball MLR
    pontryagin  — UNet + PontryaginEmbedding + PontryaginMLR (no cone penalty)

Dataset: torchvision.datasets.VOCSegmentation, year=2012.
Official "train" split (1464 images) is used for training; the official
"val" split (1449 images) has no disjoint public test set, so it is split
deterministically in half into val/test here.
VOID_LABEL=255 marks unlabeled border pixels in the official annotations
and is excluded from both the loss (remapped to -100, PyTorch's default
ignore_index) and the evaluation metrics.
"""

from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
# DATASET_ROOT is where torchvision will look for / download VOCdevkit/.
# Override via --data-root CLI arg or PASCALVOC_DATA_ROOT env var.
DATASET_ROOT = Path(__file__).parents[1] / "data" / "pascal_voc"
RESULTS_DIR  = Path(__file__).parents[1] / "results" / "seg_pascalvoc"

# ── dataset ───────────────────────────────────────────────────────────────────
VOC_YEAR   = "2012"
IMG_SIZE   = 256
VOID_LABEL = 255     # unlabeled border pixels — excluded from loss + metrics
VAL_TEST_SPLIT = 0.5 # fraction of the official "val" set held out as "test"
SEED       = 42

N_CLASSES   = 21
CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# ── class weights for CE ─────────────────────────────────────────────────────
# Uniform by default — VOC's imbalance is heavy (background dominates every
# image) but no per-class weighting has been tuned yet for this dataset.
CLASS_WEIGHTS = [1.0] * N_CLASSES

# ── UNet backbone ─────────────────────────────────────────────────────────────
BACKBONE_OUT_CH = 64    # output channels of the backbone (fed to embedding)

# PRETRAINED_BACKBONE=True swaps UNetBackbone (from-scratch) for
# PretrainedUNetBackbone (timm ImageNet-pretrained encoder + from-scratch
# decoder). Matters a lot here: VOC's official train split is only 1464
# images, too little to learn good low-level filters from scratch — unlike
# the from-scratch UNetBackbone this pipeline was designed for on the UAV
# sugarcane dataset. Set False to fall back to the from-scratch encoder.
PRETRAINED_BACKBONE = True
BACKBONE_VARIANT    = "resnet34"   # any timm model name valid for features_only=True

# Only used when PRETRAINED_BACKBONE=False (from-scratch UNetBackbone).
UNET_BASE_CH    = 64    # first encoder stage channels; doubles each stage
UNET_DEPTH      = 4     # encoder/decoder stages (3–5)

# ── Pontryagin-specific defaults ─────────────────────────────────────────────
# Inherited from seg_uav_unet as a starting point — re-tune for 21 classes.
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
BATCH_SIZE           = 8
NUM_WORKERS          = 4
EPOCHS               = 60
LR                   = 1e-3    # decoder + head learning rate
# Discriminative LR for the pretrained encoder — a fresh decoder/head next to
# frozen-scale ImageNet features needs the encoder to move much more slowly,
# or early large gradients wreck the pretrained filters (catastrophic
# forgetting). Ignored when PRETRAINED_BACKBONE=False (backbone uses LR).
LR_BACKBONE          = 1e-4
LR_MIN               = 1e-5
WEIGHT_DECAY         = 1e-4
EARLY_STOP_PATIENCE  = 15

AVAILABLE_MODELS = ("vanilla", "euclidean", "hyperbolic", "pontryagin")
