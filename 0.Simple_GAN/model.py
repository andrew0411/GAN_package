"""Simple GAN (Goodfellow 2014)의 MLP Generator / Discriminator.

이미지를 1-D 벡터(C·H·W)로 펴서 fully-connected layer만으로 다룬다.
Discriminator는 Sigmoid 없이 logits를 반환한다 (loss는 BCEWithLogits 계열인 GANLoss("vanilla")).
"""

from __future__ import annotations

import math

import torch.nn as nn
from torch import Tensor


class Generator(nn.Module):
    """z → 이미지. Linear(z_dim → hidden) → LeakyReLU(0.01) → Linear(hidden → C·H·W) → Tanh."""

    def __init__(self, z_dim: int = 64, img_shape: tuple[int, int, int] = (1, 28, 28), hidden: int = 256) -> None:
        super().__init__()
        self.img_shape = tuple(img_shape)
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden, math.prod(self.img_shape)),
            nn.Tanh(),  # 입력 이미지를 [-1, 1]로 정규화했으므로 출력도 [-1, 1]
        )

    def forward(self, z: Tensor) -> Tensor:
        x = self.net(z)  # (N, C·H·W)
        return x.reshape(z.size(0), *self.img_shape)  # (N, C, H, W)


class Discriminator(nn.Module):
    """이미지 → logit. Linear(C·H·W → hidden) → LeakyReLU(0.01) → Linear(hidden → 1)."""

    def __init__(self, img_shape: tuple[int, int, int] = (1, 28, 28), hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(math.prod(img_shape), hidden),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden, 1),  # Sigmoid 없음: logits
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.flatten(1)).squeeze(1)  # (N, C, H, W) → (N,) logits
