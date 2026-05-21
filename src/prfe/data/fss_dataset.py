"""Episodic dataset for 1-way K-shot binary segmentation on the UAV dataset.

Episode format:
    support_imgs:  (K, 3, H, W)  float32 — normalised RGB
    support_masks: (K, H, W)     float32 — binary foreground (1=sugarcane)
    query_img:     (3, H, W)     float32 — normalised RGB
    query_mask:    (H, W)        float32 — binary foreground for evaluation

The foreground class is set by ``target_cls`` (YOLO 0-indexed, default 5=Sugarcane).
Only images that contain at least one foreground polygon are kept in the pool.

Sampling strategy:
    - Train split: each __getitem__ draws a fresh random episode (no seed).
    - Val / test splits: pass ``seed`` to get reproducible episodes;
      __getitem__(i) is deterministic via seed+i.
"""

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms as T


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_normalise     = T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)


def _rasterise_binary_mask(
    label_path: Path, w: int, h: int, target_cls: int
) -> np.ndarray:
    """YOLO polygon labels → binary float32 mask (H, W), 1=target class."""
    mask = np.zeros((h, w), dtype=np.float32)
    if not label_path.exists():
        return mask
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:          # need cls + at least 3 (x,y) pairs
                continue
            if int(parts[0]) != target_cls:
                continue
            coords = list(map(float, parts[1:]))
            polygon = [
                (coords[i] * w, coords[i + 1] * h)
                for i in range(0, len(coords) - 1, 2)
            ]
            if len(polygon) < 3:
                continue
            tmp = Image.new("L", (w, h), 0)
            ImageDraw.Draw(tmp).polygon(polygon, fill=1)
            mask = np.maximum(mask, np.asarray(tmp, dtype=np.float32))
    return mask


class EpisodicUAVDataset(Dataset):
    """1-way K-shot episodic dataset over the UAV segmentation dataset.

    Args:
        root:        Path to UAV_segmantation root (contains train/valid/test)
        split:       'train', 'valid', or 'test'
        k_shot:      number of support images per episode
        n_episodes:  virtual length of the dataset (episodes per epoch)
        img_size:    spatial resolution H=W after resize
        target_cls:  YOLO class index of the foreground (5 = Sugarcane)
        augment:     random horizontal + vertical flip on support images
        seed:        fixed seed for reproducible val/test episodes;
                     None = random (train)
    """

    SUGARCANE_CLS = 5

    def __init__(
        self,
        root: Path,
        split: str,
        k_shot: int = 1,
        n_episodes: int = 200,
        img_size: int = 256,
        target_cls: int = SUGARCANE_CLS,
        augment: bool = False,
        seed: int | None = None,
    ) -> None:
        self.img_dir    = Path(root) / split / "images"
        self.lbl_dir    = Path(root) / split / "labels"
        self.k_shot     = k_shot
        self.n_episodes = n_episodes
        self.img_size   = img_size
        self.target_cls = target_cls
        self.augment    = augment
        self.seed       = seed

        all_stems = sorted(p.stem for p in self.img_dir.glob("*.jpg"))
        self.samples = [s for s in all_stems if self._has_target(s)]

        if len(self.samples) < k_shot + 1:
            raise RuntimeError(
                f"Split '{split}' has only {len(self.samples)} images with "
                f"YOLO class {target_cls}; need at least k_shot+1={k_shot+1}."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_target(self, stem: str) -> bool:
        lbl = self.lbl_dir / f"{stem}.txt"
        if not lbl.exists():
            return False
        with open(lbl) as f:
            return any(
                line.strip() and int(line.split()[0]) == self.target_cls
                for line in f
            )

    def _load(self, stem: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (img (3,H,W), binary_mask (H,W)) for one image."""
        img_path = self.img_dir / f"{stem}.jpg"
        lbl_path = self.lbl_dir / f"{stem}.txt"

        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_t = _normalise(T.functional.to_tensor(img))        # (3, H, W)

        mask = _rasterise_binary_mask(
            lbl_path, self.img_size, self.img_size, self.target_cls
        )
        mask_t = torch.from_numpy(mask)                        # (H, W)

        return img_t, mask_t

    def _flip(
        self, img: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            img  = T.functional.hflip(img)
            mask = T.functional.hflip(mask.unsqueeze(0)).squeeze(0)
        if random.random() < 0.5:
            img  = T.functional.vflip(img)
            mask = T.functional.vflip(mask.unsqueeze(0)).squeeze(0)
        return img, mask

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx if self.seed is not None else None)

        chosen        = rng.sample(self.samples, k=self.k_shot + 1)
        support_stems = chosen[: self.k_shot]
        query_stem    = chosen[self.k_shot]

        sup_imgs, sup_masks = [], []
        for stem in support_stems:
            img, mask = self._load(stem)
            if self.augment:
                img, mask = self._flip(img, mask)
            sup_imgs.append(img)
            sup_masks.append(mask)

        q_img, q_mask = self._load(query_stem)

        return {
            "support_imgs":  torch.stack(sup_imgs),    # (K, 3, H, W)
            "support_masks": torch.stack(sup_masks),   # (K, H, W)
            "query_img":     q_img,                    # (3, H, W)
            "query_mask":    q_mask,                   # (H, W)
        }
