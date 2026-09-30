"""DCGAN (Radford 2016) Generator / Discriminator. 0~3, 10, 11번 폴더가 공유한다.

채널 공식 (image_size = S, features = f):
- Generator: 해상도 r(4, 8, …, S/2)의 채널 = f · (S / r)
- Discriminator: 해상도 r(S/2, …, 4)의 채널 = f · (S / (2r))
S=64, f=64이면 G: 1024@4 → 512@8 → 256@16 → 128@32 → C@64, D: 64@32 → 128@16 → 256@8 → 512@4 → logit.
가중치 초기화는 호출자가 `gan_common.weights.init_weights(net)`로 한다 (DCGAN 관례 N(0, 0.02)).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

_IMAGE_SIZES = (32, 64, 128)


def _check_image_size(image_size: int) -> None:
    if image_size not in _IMAGE_SIZES:
        raise ValueError(f"image_size는 {_IMAGE_SIZES} 중 하나: {image_size}")


class DCGANGenerator(nn.Module):
    """z → 이미지. ConvTranspose(4, 2, 1)로 해상도를 2배씩 키운다. 출력은 Tanh로 [-1, 1]."""

    def __init__(self, z_dim: int = 100, channels: int = 1, features: int = 64, image_size: int = 64) -> None:
        super().__init__()
        _check_image_size(image_size)
        self.z_dim = z_dim

        def ch(r: int) -> int:
            return features * (image_size // r)

        # z (N, z_dim, 1, 1) → (N, ch(4), 4, 4)
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(z_dim, ch(4), kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch(4)),
            nn.ReLU(True),
        ]
        r = 4
        while r < image_size // 2:  # (N, ch(r), r, r) → (N, ch(2r), 2r, 2r)
            layers += [
                nn.ConvTranspose2d(ch(r), ch(2 * r), kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch(2 * r)),
                nn.ReLU(True),
            ]
            r *= 2
        # (N, ch(S/2), S/2, S/2) → (N, channels, S, S)
        layers += [nn.ConvTranspose2d(ch(r), channels, kernel_size=4, stride=2, padding=1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        if z.dim() == 2:
            z = z[:, :, None, None]  # (N, z_dim) → (N, z_dim, 1, 1)
        return self.net(z)  # (N, channels, S, S)


def _d_norm(norm: str, num_ch: int) -> nn.Module:
    """Discriminator용 norm layer.

    instance는 affine=True다 (원래 3.WGAN-GP critic 구현). image2image.get_norm_layer("instance")가
    junyanz 관례대로 affine=False인 것과 의도적으로 다르다. layer는 GroupNorm(1, C)로
    (C, H, W) 전체를 정규화하는 LayerNorm 대용이다 (채널별 affine).
    WGAN-GP처럼 샘플별 gradient를 쓰는 경우 batch 통계가 샘플을 섞으므로 batch 대신 instance/layer를 쓴다.
    """
    if norm == "batch":
        return nn.BatchNorm2d(num_ch)
    if norm == "instance":
        return nn.InstanceNorm2d(num_ch, affine=True)
    if norm == "layer":
        return nn.GroupNorm(1, num_ch)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"norm은 batch | instance | layer | none 중 하나: {norm!r}")


class DCGANDiscriminator(nn.Module):
    """이미지 → logit (N,). Conv(4, 2, 1)로 해상도를 절반씩 줄이고, 4x4에서 Conv(4, 1, 0)로 1x1."""

    def __init__(self, channels: int = 1, features: int = 64, image_size: int = 64, norm: str = "batch") -> None:
        super().__init__()
        _check_image_size(image_size)
        use_bias = norm == "none"  # 뒤에 norm이 오면 conv bias는 중복이다

        def ch(r: int) -> int:
            return features * (image_size // (2 * r))

        # 첫 conv는 norm 없음: (N, channels, S, S) → (N, ch(S/2), S/2, S/2)
        r = image_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(channels, ch(r), kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]
        while r > 4:  # (N, ch(r), r, r) → (N, ch(r/2), r/2, r/2)
            layers += [
                nn.Conv2d(ch(r), ch(r // 2), kernel_size=4, stride=2, padding=1, bias=use_bias),
                _d_norm(norm, ch(r // 2)),
                nn.LeakyReLU(0.2, True),
            ]
            r //= 2
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(ch(4), 1, kernel_size=4, stride=1, padding=0)  # (N, ch(4), 4, 4) → (N, 1, 1, 1)

    def features(self, x: Tensor) -> Tensor:
        """마지막 conv 직전 feature map (N, ch(4), 4, 4). AnoGAN feature matching, GANomaly에서 쓴다."""
        return self.body(x)

    def forward(self, x: Tensor) -> Tensor:
        h = self.body(x)
        return self.head(h).reshape(x.size(0))  # (N,) logits
