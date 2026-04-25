"""Standard U-Net backbone for semantic segmentation.

Architecture (Ronneberger et al., 2015 — adapted for variable depth):
  Encoder : N double-conv blocks, each followed by MaxPool2d(2)
  Bottleneck : one double-conv block (no pooling)
  Decoder : N upsampling blocks (bilinear + skip-concat + double-conv)
  Projection : 1×1 conv to out_channels

Output shape: (B, out_channels, H, W) — same spatial resolution as the input.
This makes it a drop-in backbone for any downstream head (Euclidean, Hyperbolic,
or Pontryagin) that expects (B, C, H, W) feature maps.

Typical channel schedule (base_ch=64, depth=4):
  enc1 : 3  → 64    enc2 : 64  → 128
  enc3 : 128 → 256  enc4 : 256 → 512
  bottleneck : 512 → 1024
  dec4 : 1024+512 → 512   dec3 : 512+256 → 256
  dec2 : 256+128  → 128   dec1 : 128+64  → 64
  proj : 64 → out_channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetBackbone(nn.Module):
    """U-Net encoder-decoder producing (B, out_channels, H, W) feature maps.

    Args:
        in_channels:  input image channels (default 3 — RGB)
        out_channels: number of output feature channels (default 64)
        base_ch:      channel count at the first encoder block (default 64);
                      subsequent blocks double the channels up to 8×base_ch.
        depth:        number of encoder/decoder stages (default 4, range 3–5)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        base_ch: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if not (3 <= depth <= 5):
            raise ValueError(f"depth must be 3-5, got {depth}")
        self.depth = depth

        # Channel schedule: base_ch * [1, 2, 4, 8, 8]
        chs = [base_ch * min(2 ** i, 8) for i in range(depth + 1)]

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        in_c = in_channels
        for ch in chs[:depth]:
            self.enc_blocks.append(_double_conv(in_c, ch))
            in_c = ch

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = _double_conv(chs[depth - 1], chs[depth])

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec_blocks = nn.ModuleList()
        dec_in = chs[depth]
        for ch in reversed(chs[:depth]):
            # after concat with skip: dec_in + ch → ch
            self.dec_blocks.append(_double_conv(dec_in + ch, ch))
            dec_in = ch

        # ── Final projection ─────────────────────────────────────────────────
        self.proj = nn.Conv2d(chs[0], out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            (B, out_channels, H, W)
        """
        # Encoder — collect skip tensors
        skips = []
        h = x
        for enc in self.enc_blocks:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder — upsample + concat skip + double-conv
        for dec, skip in zip(self.dec_blocks, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        return self.proj(h)
