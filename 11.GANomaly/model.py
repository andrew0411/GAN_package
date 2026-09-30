"""GANomaly 네트워크 (Akcay et al., ACCV 2018; 공식 구현 samet-akcay/ganomaly의 networks.py 구조).

    NetG:  x ──E1──▶ z ──Decoder──▶ x̂ ──E2──▶ ẑ        encoder-decoder-encoder
    NetD:  x ──features──▶ f(x) ──classifier──▶ logit     (Encoder(nz=1)을 둘로 나눈 것)

- Encoder: DCGAN D식 Conv(4, 2, 1)(+BN)+LeakyReLU(0.2)로 4x4까지 줄인 뒤 Conv(C, nz, 4, 1, 0) → (N, nz, 1, 1).
  공식 구현처럼 모든 conv에 bias가 없다. 채널: 해상도 r에서 ndf·S/(2r) (S=32, ndf=64면 64@16 → 128@8 → 256@4)
- Decoder: DCGAN G와 같은 구조라 gan_common `DCGANGenerator`를 재사용한다. 공식 Decoder 채널(ngf·S/(2r),
  S=32면 256@4 → 128@8 → 64@16)에 맞추려고 features = ngf / 2로 만든다
- NetD: 마지막 conv가 classifier, 나머지가 features. 공식은 classifier 뒤 Sigmoid였고 여기서는 logits를 낸다
anomaly score: A(x) = mean_j (z_j − ẑ_j)²  (`latent_score`)
가중치 초기화는 호출자가 `gan_common.weights.init_weights(net, "normal", 0.02)`로 한다 (공식 weights_init과 같음).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from gan_common.networks.dcgan import DCGANGenerator


class Encoder(nn.Module):
    """이미지 (N, C, S, S) → (N, nz, 1, 1). S는 16 이상의 2의 거듭제곱.

    공식 구현은 16의 배수만 검사하지만, 48처럼 2의 거듭제곱이 아니면 반씩 줄이다 4x4를 지나쳐(48→24→12→6→3)
    마지막 Conv(4, 1, 0)이 실패한다.
    """

    def __init__(self, image_size: int, nz: int, channels: int = 1, ndf: int = 64) -> None:
        super().__init__()
        if image_size < 16 or image_size & (image_size - 1) != 0:
            raise ValueError(f"image_size는 16 이상의 2의 거듭제곱이어야 합니다: {image_size}")
        # 첫 conv는 BN 없음: (N, C, S, S) → (N, ndf, S/2, S/2)
        layers: list[nn.Module] = [
            nn.Conv2d(channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
        ]
        size, ch = image_size // 2, ndf
        while size > 4:  # (N, ch, r, r) → (N, 2ch, r/2, r/2)
            layers += [
                nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch * 2),
                nn.LeakyReLU(0.2, True),
            ]
            ch, size = ch * 2, size // 2
        # (N, ch, 4, 4) → (N, nz, 1, 1)
        layers.append(nn.Conv2d(ch, nz, kernel_size=4, stride=1, padding=0, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # (N, nz, 1, 1)


class Decoder(DCGANGenerator):
    """z (N, nz) → x̂ (N, C, S, S) in [-1, 1]. ConvT(4, 2, 1)+BN+ReLU, 마지막 ConvT+Tanh.

    `DCGANGenerator(features=ngf // 2)`: 해상도 r의 채널이 (ngf/2)·S/r = ngf·S/(2r)이 되어 공식 Decoder와 같다.
    차이: 마지막 ConvTranspose에 bias가 있다 (공식은 bias 없음).
    """

    def __init__(self, image_size: int, nz: int, channels: int = 1, ngf: int = 64) -> None:
        if ngf % 2 != 0:
            raise ValueError(f"ngf는 짝수여야 합니다 (features = ngf / 2): {ngf}")
        super().__init__(z_dim=nz, channels=channels, features=ngf // 2, image_size=image_size)


class NetG(nn.Module):
    """E1 → Decoder → E2. 반환 (x̂, z, ẑ): x̂ (N, C, S, S), z = E1(x) (N, nz), ẑ = E2(x̂) (N, nz).

    공식 구현처럼 E1·Decoder·E2 모두 ngf를 기본 채널로 쓴다. image_size ∈ {32, 64, 128} (Decoder 제약).
    """

    def __init__(self, image_size: int = 32, nz: int = 100, channels: int = 1, ngf: int = 64) -> None:
        super().__init__()
        self.encoder1 = Encoder(image_size, nz, channels, ngf)
        self.decoder = Decoder(image_size, nz, channels, ngf)
        self.encoder2 = Encoder(image_size, nz, channels, ngf)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z = self.encoder1(x).flatten(1)  # (N, nz)
        x_hat = self.decoder(z)  # (N, C, S, S)
        z_hat = self.encoder2(x_hat).flatten(1)  # (N, nz)
        return x_hat, z, z_hat


class NetD(nn.Module):
    """Encoder(nz=1)의 마지막 conv를 classifier로 떼어 낸 판별기. 반환 (logits (N,), features (N, C', 4, 4)).

    features는 G의 adversarial loss(feature matching L2)에 쓴다. S=32, ndf=64면 C' = 256.
    """

    def __init__(self, image_size: int = 32, channels: int = 1, ndf: int = 64) -> None:
        super().__init__()
        layers = list(Encoder(image_size, 1, channels, ndf).net.children())
        self.features = nn.Sequential(*layers[:-1])  # ... → LeakyReLU, (N, C', 4, 4)
        self.classifier = layers[-1]  # Conv(C', 1, 4, 1, 0) → (N, 1, 1, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.features(x)  # (N, C', 4, 4)
        logits = self.classifier(feat).reshape(x.size(0))  # (N,) Sigmoid 없음
        return logits, feat


def latent_score(z: Tensor, z_hat: Tensor) -> Tensor:
    """GANomaly anomaly score A(x) = mean_j (z_j − ẑ_j)². 입력 (N, nz) → (N,)."""
    return (z - z_hat).pow(2).mean(dim=1)
