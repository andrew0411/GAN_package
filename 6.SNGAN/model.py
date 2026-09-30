"""SNGAN (Miyato 2018) CIFAR-10 ResNet Generator / Discriminator.

- Generator: z(128) → Linear → (256, 4, 4) → ResBlock-up × 3 → BN, ReLU, conv3x3 → Tanh → (3, 32, 32)
- Discriminator: OptimizedDisBlock(down) → DisBlock(down) → DisBlock → DisBlock
  → ReLU → global sum pooling → Linear → logit
- spectral normalization(SN)은 D에만 건다: W_SN = W / σ(W), σ(W)는 W의 최대 특이값(spectral norm).
  layer마다 Lipschitz 상수를 1로 묶어 D의 gradient 크기를 제한한다. G는 평범한 BatchNorm ResNet이다.

가중치 초기화는 원 구현(pfnet-research/sngan_projection, chainer)을 따른다:
residual conv는 xavier_uniform(gain √2), shortcut 1x1·Linear·출력 conv는 xavier_uniform(gain 1), bias 0.
D는 "초기화 → spectral_norm" 순서로 만든다. SN이 걸리는 순간 power iteration 벡터 u가 그 weight 기준으로 잡히기 때문이다.
논문 구조는 32px용이다. `image_size`를 바꾸면 G의 up-block 수만 log2(image_size / 4)로 조정된다 (D는 그대로).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gan_common.layers import spectral_norm

BOTTOM_WIDTH = 4  # G가 시작하는 feature map 해상도


def _xavier(module: nn.Module, gain: float) -> nn.Module:
    """xavier_uniform(gain) weight + 0 bias (chainer GlorotUniform(gain)과 같은 분포)."""
    nn.init.xavier_uniform_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


def _upsample(x: Tensor) -> Tensor:
    return F.interpolate(x, scale_factor=2.0, mode="nearest")  # (N, C, 2H, 2W)


def _downsample(x: Tensor) -> Tensor:
    return F.avg_pool2d(x, 2)  # (N, C, H/2, W/2)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class GenBlock(nn.Module):
    """G residual block (해상도 2배).

    residual: BN → ReLU → upsample(nearest) → conv3x3 → BN → ReLU → conv3x3
    shortcut: upsample → conv1x1
    원 구현 규칙상 in≠out이거나 upsample이면 shortcut이 학습되는 1x1 conv다. G 블록은 모두 upsample하므로 항상 1x1 conv.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.b1 = nn.BatchNorm2d(in_ch)
        self.c1 = _xavier(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), math.sqrt(2))
        self.b2 = nn.BatchNorm2d(out_ch)
        self.c2 = _xavier(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), math.sqrt(2))
        self.c_sc = _xavier(nn.Conv2d(in_ch, out_ch, kernel_size=1), 1.0)

    def forward(self, x: Tensor) -> Tensor:
        h = self.c1(_upsample(F.relu(self.b1(x))))  # (N, out, 2H, 2W)
        h = self.c2(F.relu(self.b2(h)))
        return h + self.c_sc(_upsample(x))  # (N, out, 2H, 2W)


class Generator(nn.Module):
    """z (N, z_dim) → 이미지 (N, channels, image_size, image_size) in [-1, 1]."""

    def __init__(self, z_dim: int = 128, channels: int = 3, ch: int = 256, image_size: int = 32) -> None:
        super().__init__()
        n_up = int(round(math.log2(image_size / BOTTOM_WIDTH))) if image_size >= 2 * BOTTOM_WIDTH else 0
        if n_up < 1 or BOTTOM_WIDTH * 2**n_up != image_size:
            raise ValueError(f"image_size는 4·2^k (k ≥ 1) 이어야 합니다 (예: 32): {image_size}")
        self.z_dim = z_dim
        self.ch = ch
        self.l1 = _xavier(nn.Linear(z_dim, BOTTOM_WIDTH * BOTTOM_WIDTH * ch), 1.0)
        self.blocks = nn.Sequential(*[GenBlock(ch, ch) for _ in range(n_up)])
        self.bn = nn.BatchNorm2d(ch)
        self.conv_out = _xavier(nn.Conv2d(ch, channels, kernel_size=3, padding=1), 1.0)

    def forward(self, z: Tensor) -> Tensor:
        h = self.l1(z).reshape(z.size(0), self.ch, BOTTOM_WIDTH, BOTTOM_WIDTH)  # (N, ch, 4, 4)
        h = self.blocks(h)  # (N, ch, S, S), S = image_size
        h = F.relu(self.bn(h))
        return torch.tanh(self.conv_out(h))  # (N, channels, S, S)


# ---------------------------------------------------------------------------
# Discriminator (모든 conv/linear에 spectral_norm)
# ---------------------------------------------------------------------------


def _sn_conv(in_ch: int, out_ch: int, kernel_size: int, gain: float) -> nn.Module:
    """초기화한 conv에 spectral_norm을 씌운다 (순서 중요: init → SN)."""
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
    return spectral_norm(_xavier(conv, gain))


class OptimizedDisBlock(nn.Module):
    """D의 첫 블록. 입력이 RGB 이미지이므로 앞에 ReLU를 두지 않는다.

    residual: conv3x3 → ReLU → conv3x3 → avgpool
    shortcut: avgpool → conv1x1 (pool을 먼저 해서 1x1 conv 연산량을 줄인다)
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.c1 = _sn_conv(in_ch, out_ch, 3, math.sqrt(2))
        self.c2 = _sn_conv(out_ch, out_ch, 3, math.sqrt(2))
        self.c_sc = _sn_conv(in_ch, out_ch, 1, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        h = _downsample(self.c2(F.relu(self.c1(x))))  # (N, out, H/2, W/2)
        return h + self.c_sc(_downsample(x))


class DisBlock(nn.Module):
    """D residual block.

    residual: ReLU → conv3x3(in→in) → ReLU → conv3x3(in→out) → (avgpool)
    shortcut: in≠out이거나 downsample이면 conv1x1 → (avgpool), 아니면 identity
    원 구현처럼 hidden 채널 = in_ch다 (G block은 hidden = out_ch). 이 모델은 모든 DisBlock이 ch→ch라 차이가 없다.
    """

    def __init__(self, in_ch: int, out_ch: int, downsample: bool) -> None:
        super().__init__()
        self.downsample = downsample
        self.c1 = _sn_conv(in_ch, in_ch, 3, math.sqrt(2))
        self.c2 = _sn_conv(in_ch, out_ch, 3, math.sqrt(2))
        self.c_sc = _sn_conv(in_ch, out_ch, 1, 1.0) if (in_ch != out_ch or downsample) else None

    def forward(self, x: Tensor) -> Tensor:
        # F.relu는 in-place가 아니다: shortcut이 원래 x를 써야 하므로 x를 덮어쓰면 안 된다
        h = self.c2(F.relu(self.c1(F.relu(x))))
        sc = x if self.c_sc is None else self.c_sc(x)
        if self.downsample:
            h, sc = _downsample(h), _downsample(sc)
        return h + sc  # (N, out, H or H/2, W or W/2)


class Discriminator(nn.Module):
    """이미지 (N, channels, S, S) → logits (N,). Sigmoid 없음 (hinge loss에 logit을 그대로 넣는다).

    S = image_size (논문 32). global sum pooling이라 입력 해상도에 상관없이 동작한다.
    """

    def __init__(self, channels: int = 3, ch: int = 128) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            OptimizedDisBlock(channels, ch),  # S → S/2
            DisBlock(ch, ch, downsample=True),  # S/2 → S/4
            DisBlock(ch, ch, downsample=False),  # S/4
            DisBlock(ch, ch, downsample=False),  # S/4
        )
        self.linear = spectral_norm(_xavier(nn.Linear(ch, 1), 1.0))

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.blocks(x))  # (N, ch, S/4, S/4)
        h = h.sum(dim=(2, 3))  # global sum pooling: (N, ch). 원 구현은 평균이 아니라 합을 쓴다
        return self.linear(h).squeeze(1)  # (N,) logits
