"""GAN용 레이어: spectral norm, equalized learning rate(ProGAN/StyleGAN), PixelNorm, minibatch stddev."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm


def spectral_norm(module: nn.Module) -> nn.Module:
    """Spectral normalization (Miyato 2018): W / σ(W).

    σ(W)는 W의 최대 특이값으로 power iteration(train 모드 forward마다 1회)으로 추정한다.
    레이어의 Lipschitz 상수를 1로 묶어 D를 안정화한다. `parametrizations` API 래퍼.
    """
    return _spectral_norm(module)


class EqualizedConv2d(nn.Module):
    """Equalized learning rate conv (Karras 2018, ProGAN).

    weight를 N(0, 1)/lr_mul로 초기화하고, forward에서 He 상수 c = gain / sqrt(fan_in)과 lr_mul을 곱한다.
    Adam은 파라미터 scale에 불변인 update를 하므로, 스케일을 runtime에 곱하면 모든 레이어의
    실효 학습률이 같아진다. `lr_mul`은 이 레이어만 학습률을 낮추고 싶을 때 쓴다.

    gain 규약 (He gain을 어디서 곱하느냐):
    - 기본 `gain=sqrt(2)`: ProGAN 방식. 뒤따르는 activation(LeakyReLU 등)은 gain을 곱하지 않는다
    - activation이 이미 sqrt(2)를 곱하면(StyleGAN2 fused_lrelu) `gain=1.0`을 준다 (이중 적용 방지)
    - 뒤에 activation이 없는 toRGB·최종 출력 레이어는 `gain=1.0`
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        gain: float = math.sqrt(2),
        lr_mul: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.scale = gain / math.sqrt(in_ch * kernel_size * kernel_size) * lr_mul
        self.lr_mul = lr_mul
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias * self.lr_mul if self.bias is not None else None
        return F.conv2d(x, self.weight * self.scale, bias, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        out_ch, in_ch, k, _ = self.weight.shape
        return (
            f"{in_ch}, {out_ch}, kernel_size={k}, stride={self.stride}, padding={self.padding}, "
            f"bias={self.bias is not None}, lr_mul={self.lr_mul}"
        )


class EqualizedLinear(nn.Module):
    """Equalized learning rate linear. 원리는 `EqualizedConv2d`와 같다.

    StyleGAN2 mapping network는 `lr_mul=0.01`, style affine은 `bias_init=1.0`을 쓴다.
    실효 bias = bias * lr_mul.

    gain 규약은 `EqualizedConv2d`와 같다: 기본 `gain=sqrt(2)`는 ProGAN 방식(activation에 gain 없음),
    activation이 sqrt(2)를 곱하면(StyleGAN2 fused_lrelu) `gain=1.0`, 뒤에 activation이 없는
    최종 레이어(style affine, D의 마지막 linear 등)도 `gain=1.0`.
    """

    def __init__(
        self,
        in_f: int,
        out_f: int,
        bias: bool = True,
        gain: float = math.sqrt(2),
        lr_mul: float = 1.0,
        bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) / lr_mul)
        self.bias = nn.Parameter(torch.full((out_f,), float(bias_init))) if bias else None
        self.scale = gain / math.sqrt(in_f) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, self.weight * self.scale, bias)

    def extra_repr(self) -> str:
        out_f, in_f = self.weight.shape
        return f"in_features={in_f}, out_features={out_f}, bias={self.bias is not None}, lr_mul={self.lr_mul}"


class PixelNorm(nn.Module):
    """Pixelwise feature normalization (ProGAN): 픽셀마다 채널 벡터를 RMS 1로 맞춘다.

        y = x / sqrt(mean_c(x^2) + eps)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)


class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation (ProGAN/StyleGAN2): batch 다양성 통계를 D에 채널로 붙인다.

    batch를 크기 G의 그룹으로 나눠 그룹 안 표준편차를 구하고, 채널·픽셀 평균을 낸 값을
    `num_channels`개 feature map으로 이어 붙인다. G가 mode collapse(다양성 부족)를 감지하게 해 준다.
    출력 채널 = C + num_channels.

    batch 크기 N이 group_size로 나눠지지 않으면(마지막 batch 등) 전체 batch를 한 그룹(G = N)으로 쓴다.
    """

    def __init__(self, group_size: int = 4, num_channels: int = 1) -> None:
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        g = min(self.group_size, n)
        if n % g != 0:
            g = n
        f = self.num_channels
        if c % f != 0:
            raise ValueError(f"채널 수 {c}가 num_channels={f}로 나눠지지 않습니다.")
        y = x.reshape(g, -1, f, c // f, h, w)  # (G, N/G, F, C/F, H, W), 샘플 i는 그룹 i % (N/G)
        y = y - y.mean(dim=0)  # 그룹 평균 제거
        y = (y.pow(2).mean(dim=0) + 1e-8).sqrt()  # (N/G, F, C/F, H, W) 그룹 내 표준편차
        y = y.mean(dim=(2, 3, 4))  # (N/G, F) 채널·픽셀 평균
        y = y.reshape(-1, f, 1, 1).repeat(g, 1, h, w)  # (N, F, H, W) 원래 샘플 순서로 복제
        return torch.cat([x, y], dim=1)  # (N, C + F, H, W)

    def extra_repr(self) -> str:
        return f"group_size={self.group_size}, num_channels={self.num_channels}"
