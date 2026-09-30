"""ProGAN (Karras 2018) Generator / Discriminator: progressive growing + equalized lr + PixelNorm + minibatch stddev.

depth d ↔ 해상도 4·2^d (depth 0 = 4x4). 모든 해상도의 layer를 처음부터 만들어 두고, forward의 `depth`로
현재 해상도까지만 쓴다. 아직 쓰지 않는 layer는 gradient가 None이라 Adam이 건드리지 않는다.
`alpha`(0 → 1)는 fade-in 계수다. 새 해상도 layer를 갑자기 끼우면 이미 학습된 저해상도 layer가 충격을 받으므로,
새 경로 출력과 이전 해상도 경로 출력을 alpha로 섞어 새 layer의 비중을 서서히 올린다 (residual처럼 동작).

- G: PixelNorm(z) → 4x4 block → [upsample(nearest) → (conv3x3 → LReLU(0.2) → PixelNorm) × 2] … → toRGB(1x1)
- D: fromRGB(1x1) → [(conv3x3 → LReLU) × 2 → avgpool] … → 4x4: MinibatchStdDev → conv3x3 → conv4x4 → linear → logit
- 채널: 4–32px 512, 64px 256, 128px 128, 256px 64 (공식 구현 nf(res) = min(16384 / res, 512))
- equalized lr (gan_common.layers): weight를 N(0, 1)로 두고 forward에서 He 상수 gain/√fan_in을 곱한다.
  hidden layer는 기본 gain √2, 뒤에 activation이 없는 toRGB·D 마지막 linear는 gain 1.
  Equalized layer가 스스로 초기화하므로 init_weights를 쓰지 않는다. BatchNorm도 없다
- G 출력에 Tanh가 없다 (논문·공식 구현과 같은 linear 출력). 샘플 저장 시 [-1, 1]로 clamp된다
- 층 순서는 공식 구현을 따른다: conv → LeakyReLU → PixelNorm
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gan_common.layers import EqualizedConv2d, EqualizedLinear, MinibatchStdDev, PixelNorm

CHANNELS: dict[int, int] = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64}
RESOLUTIONS: tuple[int, ...] = tuple(CHANNELS)


def res_to_depth(res: int) -> int:
    """해상도 → depth (4 → 0, 8 → 1, …)."""
    if res not in CHANNELS:
        raise ValueError(f"해상도는 {RESOLUTIONS} 중 하나: {res}")
    return int(math.log2(res)) - 2


def depth_to_res(depth: int) -> int:
    """depth → 해상도 (0 → 4, 1 → 8, …)."""
    return 4 * 2**depth


def _check_depth(depth: int, max_depth: int) -> None:
    if not 0 <= depth <= max_depth:
        raise ValueError(f"depth는 0…{max_depth} 범위여야 합니다: {depth}")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class GInitialBlock(nn.Module):
    """z (N, z_dim) → (N, ch, 4, 4).

    latent PixelNorm → dense(4x4) → LReLU → PixelNorm → conv3x3 → LReLU → PixelNorm.
    논문 표의 "Conv 4x4"(1x1 latent에 4x4 transposed conv)와 같은 연산을 dense로 쓴다.
    He 상수를 conv 기준 fan_in(z_dim·16)에 맞추려고 gain을 √2/4로 준다: √2/4/√z_dim = √2/√(z_dim·16) (공식 구현과 같음).
    """

    def __init__(self, z_dim: int, ch: int) -> None:
        super().__init__()
        self.ch = ch
        self.norm = PixelNorm()
        self.dense = EqualizedLinear(z_dim, ch * 16, gain=math.sqrt(2) / 4)
        self.conv = EqualizedConv2d(ch, ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, z: Tensor) -> Tensor:
        x = self.norm(z)  # latent를 단위 RMS로 정규화 (N, z_dim)
        x = self.dense(x).reshape(z.size(0), self.ch, 4, 4)  # (N, ch, 4, 4)
        x = self.norm(self.act(x))
        return self.norm(self.act(self.conv(x)))  # (N, ch, 4, 4)


class GBlock(nn.Module):
    """(N, in_ch, H, W) → (N, out_ch, 2H, 2W): upsample(nearest) → (conv3x3 → LReLU → PixelNorm) × 2.

    PixelNorm: 픽셀마다 채널 벡터를 단위 RMS로 맞춰 G와 D의 경쟁으로 feature 크기가 폭주하는 것을 막는다
    (학습 파라미터 없는 BatchNorm 대체).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = EqualizedConv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = EqualizedConv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.norm = PixelNorm()

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")  # (N, in_ch, 2H, 2W)
        x = self.norm(self.act(self.conv1(x)))
        return self.norm(self.act(self.conv2(x)))  # (N, out_ch, 2H, 2W)


class Generator(nn.Module):
    """`G(z, alpha, depth)`: z (N, z_dim) → 이미지 (N, channels, 4·2^depth, 4·2^depth).

    blocks[d]와 to_rgb[d]가 해상도 4·2^d를 담당한다 (blocks[0]은 4x4 초기 block).
    fade-in (depth ≥ 1, alpha < 1):
        rgb = alpha · toRGB_d(block_d(h)) + (1 - alpha) · upsample(toRGB_{d-1}(h)),  h = depth d-1까지의 feature
    """

    def __init__(self, z_dim: int = 512, channels: int = 3, max_res: int = 128) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.max_depth = res_to_depth(max_res)
        blocks: list[nn.Module] = [GInitialBlock(z_dim, CHANNELS[4])]
        to_rgb: list[nn.Module] = [EqualizedConv2d(CHANNELS[4], channels, 1, gain=1.0)]
        for d in range(1, self.max_depth + 1):
            res = depth_to_res(d)
            blocks.append(GBlock(CHANNELS[res // 2], CHANNELS[res]))
            to_rgb.append(EqualizedConv2d(CHANNELS[res], channels, 1, gain=1.0))  # activation 없는 출력층: gain 1
        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.ModuleList(to_rgb)

    def forward(self, z: Tensor, alpha: float, depth: int) -> Tensor:
        _check_depth(depth, self.max_depth)
        x = self.blocks[0](z)  # (N, 512, 4, 4)
        if depth == 0:
            return self.to_rgb[0](x)  # (N, C, 4, 4)
        for d in range(1, depth):
            x = self.blocks[d](x)
        prev = x  # (N, ch, res/2, res/2)
        rgb = self.to_rgb[depth](self.blocks[depth](prev))  # 새 해상도 경로 (N, C, res, res)
        if alpha < 1.0:
            # 이전 해상도의 RGB 출력을 nearest upsample해서 새 경로와 선형 보간한다
            skip = F.interpolate(self.to_rgb[depth - 1](prev), scale_factor=2.0, mode="nearest")
            rgb = torch.lerp(skip, rgb, alpha)  # (1 - alpha)·skip + alpha·rgb
        return rgb  # (N, C, res, res). linear 출력 (Tanh 없음)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------


def _from_rgb(channels: int, ch: int) -> nn.Module:
    """fromRGB: 1x1 conv → LReLU. 해상도마다 하나씩 있다."""
    return nn.Sequential(EqualizedConv2d(channels, ch, 1), nn.LeakyReLU(0.2))


class DBlock(nn.Module):
    """(N, in_ch, H, W) → (N, out_ch, H/2, W/2): conv3x3(in→in) → LReLU → conv3x3(in→out) → LReLU → avgpool."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = EqualizedConv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = EqualizedConv2d(in_ch, out_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return F.avg_pool2d(x, 2)  # (N, out_ch, H/2, W/2)


class DFinalBlock(nn.Module):
    """(N, ch, 4, 4) → logits (N,): MinibatchStdDev → conv3x3 → LReLU → conv4x4(valid) → LReLU → linear.

    MinibatchStdDev: batch 안 샘플들의 표준편차(다양성 통계)를 채널 하나로 붙인다. G가 비슷한 샘플만 내면(mode collapse)
    이 값이 real과 달라지므로 D가 알아채고, G는 다양성을 늘리도록 밀린다.
    """

    def __init__(self, ch: int, mbstd_group_size: int = 4) -> None:
        super().__init__()
        self.mbstd = MinibatchStdDev(group_size=mbstd_group_size, num_channels=1)
        self.conv = EqualizedConv2d(ch + 1, ch, 3, padding=1)
        self.conv4 = EqualizedConv2d(ch, ch, 4, padding=0)  # 4x4 → 1x1 (flatten 후 dense와 같다)
        self.linear = EqualizedLinear(ch, 1, gain=1.0)  # activation 없는 출력층: gain 1
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mbstd(x)  # (N, ch + 1, 4, 4)
        x = self.act(self.conv(x))  # (N, ch, 4, 4)
        x = self.act(self.conv4(x)).flatten(1)  # (N, ch)
        return self.linear(x).squeeze(1)  # (N,) logits


class Discriminator(nn.Module):
    """`D(x, alpha, depth)`: 이미지 (N, channels, 4·2^depth, 4·2^depth) → logits (N,). Sigmoid 없음 (WGAN critic).

    from_rgb[d]와 blocks[d]가 해상도 4·2^d를 담당한다 (blocks[0]은 4x4 최종 block).
    fade-in (depth ≥ 1, alpha < 1):
        h = alpha · block_d(fromRGB_d(x)) + (1 - alpha) · fromRGB_{d-1}(avgpool(x)),  이후 depth d-1 … 0 block
    """

    def __init__(self, channels: int = 3, max_res: int = 128, mbstd_group_size: int = 4) -> None:
        super().__init__()
        self.max_depth = res_to_depth(max_res)
        from_rgb: list[nn.Module] = [_from_rgb(channels, CHANNELS[4])]
        blocks: list[nn.Module] = [DFinalBlock(CHANNELS[4], mbstd_group_size)]
        for d in range(1, self.max_depth + 1):
            res = depth_to_res(d)
            from_rgb.append(_from_rgb(channels, CHANNELS[res]))
            blocks.append(DBlock(CHANNELS[res], CHANNELS[res // 2]))
        self.from_rgb = nn.ModuleList(from_rgb)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, alpha: float, depth: int) -> Tensor:
        _check_depth(depth, self.max_depth)
        if x.size(-1) != depth_to_res(depth):
            raise ValueError(f"depth {depth}의 입력 해상도는 {depth_to_res(depth)}여야 합니다: {tuple(x.shape)}")
        h = self.from_rgb[depth](x)  # (N, ch(res), res, res)
        if depth > 0:
            h = self.blocks[depth](h)  # 새 해상도 경로 (N, ch(res/2), res/2, res/2)
            if alpha < 1.0:
                # 입력을 avgpool로 줄여 이전 해상도의 fromRGB에 넣은 경로와 선형 보간한다
                skip = self.from_rgb[depth - 1](F.avg_pool2d(x, 2))
                h = torch.lerp(skip, h, alpha)  # (1 - alpha)·skip + alpha·h
            for d in range(depth - 1, 0, -1):
                h = self.blocks[d](h)
        return self.blocks[0](h)  # (N,) logits
