"""StyleGAN2용 저수준 연산 (순수 PyTorch): upfirdn2d, FIR 커널, Blur/Upsample, fused bias + LeakyReLU.

공식 StyleGAN2(NVlabs)와 rosinality/stylegan2-pytorch는 upfirdn2d·fused_leaky_relu를 custom CUDA kernel로
구현한다. 여기서는 컴파일 없이 어디서나 돌도록 PyTorch 기본 연산(pad, reshape, conv2d, slicing)만으로
같은 수학을 구현한다. CUDA kernel보다 느리고 메모리를 조금 더 쓰지만 결과는 같다.

모든 연산은 out-of-place이고 PyTorch 기본 연산의 조합이므로 2차 미분이 된다.
R1 penalty·path length regularization은 `autograd.grad(..., create_graph=True)`로 gradient의 gradient를
구하므로, D·G 안의 모든 연산이 double backward를 지원해야 한다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def make_kernel(k: Sequence[float] | Tensor) -> Tensor:
    """1D FIR 계수 → 합이 1인 2D separable 커널.

    StyleGAN2는 resampling 때 [1, 3, 3, 1] (이항 계수, binomial) low-pass filter를 쓴다.
    외적 k^T k로 4x4 커널을 만들고 합으로 나눠 DC gain(평균 밝기 보존)을 1로 맞춘다.
    up/down-sampling 전후에 이 필터를 거치면 aliasing(계단·격자 artifact)이 줄어든다
    (Zhang 2019 "Making Convolutional Networks Shift-Invariant Again", StyleGAN1부터 사용).
    """
    k = torch.as_tensor(k, dtype=torch.float32)
    if k.dim() == 1:
        k = k[None, :] * k[:, None]  # (K,) → (K, K) 외적
    return k / k.sum()


def upfirdn2d(x: Tensor, kernel: Tensor, up: int = 1, down: int = 1, pad: tuple[int, int] = (0, 0)) -> Tensor:
    """Upsample → FIR filter → Downsample (신호처리의 upfirdn을 2D로).

    단계 (x, y 두 축에 같은 up/down/pad를 적용한다):
      1. upsample: 픽셀 사이에 (up - 1)개의 0을 끼워 넣는다 (zero insertion). 해상도 ×up
      2. pad: 앞쪽 pad[0], 뒤쪽 pad[1]만큼 0을 덧댄다. 음수면 그만큼 잘라낸다(crop)
      3. FIR filter: 채널마다 독립인 depthwise conv. `F.conv2d`는 cross-correlation이므로
         커널을 상하좌우로 뒤집어 넣어야 수학적 convolution이 된다 ([1,3,3,1]처럼 대칭이면 결과는 같다)
      4. downsample: `down` 픽셀마다 하나만 남긴다 (stride slicing)

    출력 크기: (H·up + pad0 + pad1 - kH) // down + 1   (W도 같다)

    Args:
        x: (N, C, H, W)
        kernel: (kH, kW) FIR 커널. upsample에 쓸 때는 zero insertion으로 줄어든 평균 밝기를
            보상하도록 호출자가 up² 배를 곱해 둔다 (`Upsample`, `Blur(upsample_factor=...)`)
        up, down: 정수 배율
        pad: (앞, 뒤) padding
    """
    return _upfirdn2d_flipped(x, kernel.flip([0, 1]), up, down, pad)


def _upfirdn2d_flipped(x: Tensor, flipped_kernel: Tensor, up: int, down: int, pad: tuple[int, int]) -> Tensor:
    """`upfirdn2d` 본체. `flipped_kernel`은 이미 상하좌우로 뒤집힌 커널(= conv2d에 그대로 넣을 weight)이다.

    `Blur`·`Upsample`은 뒤집힌 커널을 buffer로 미리 만들어 두고 이 함수를 직접 부른다 (매 forward flip 생략).
    """
    n, c, h, w = x.shape
    kh, kw = flipped_kernel.shape
    pad0, pad1 = pad

    # 1) zero insertion: (N, C, H, 1, W, 1) → 각 원소 뒤에 up-1개의 0 → (N, C, H·up, W·up)
    if up > 1:
        x = x.reshape(n, c, h, 1, w, 1)
        x = F.pad(x, [0, up - 1, 0, 0, 0, up - 1])  # F.pad는 마지막 축부터: (W 뒤 1축), (W), (H 뒤 1축)
        x = x.reshape(n, c, h * up, w * up)

    # 2) pad(양수) / crop(음수)
    x = F.pad(x, [max(pad0, 0), max(pad1, 0), max(pad0, 0), max(pad1, 0)])
    x = x[
        :,
        :,
        max(-pad0, 0) : x.shape[2] - max(-pad1, 0),
        max(-pad0, 0) : x.shape[3] - max(-pad1, 0),
    ]

    # 3) depthwise FIR: 채널을 batch 축으로 펴서 (N·C, 1, H', W')에 (1, 1, kH, kW) 커널 하나를 적용
    hp, wp = x.shape[2], x.shape[3]
    weight = flipped_kernel.to(dtype=x.dtype)[None, None]  # (1, 1, kH, kW)
    x = F.conv2d(x.reshape(n * c, 1, hp, wp), weight)  # (N·C, 1, H'-kH+1, W'-kW+1)
    x = x.reshape(n, c, hp - kh + 1, wp - kw + 1)

    # 4) downsample
    if down > 1:
        x = x[:, :, ::down, ::down]
    return x


class Blur(nn.Module):
    """FIR low-pass filter (up/down 없이 pad + filter만). stride-2 conv 앞(D)이나 transposed conv 뒤(G)에 둔다.

    `upsample_factor > 1`: transposed conv(stride 2)는 zero insertion과 같아 평균 밝기가 1/factor²로 줄므로
    커널에 factor²를 곱해 보상한다.
    """

    def __init__(self, kernel: Sequence[float], pad: tuple[int, int], upsample_factor: int = 1) -> None:
        super().__init__()
        k = make_kernel(kernel)
        if upsample_factor > 1:
            k = k * (upsample_factor**2)
        # 고정 상수이므로 non-persistent buffer: .to(device)로 따라가지만 state_dict에는 들어가지 않는다.
        # conv2d에 바로 넣을 수 있게 미리 뒤집어 둔다 (upfirdn2d 3단계 참고)
        self.register_buffer("flipped_kernel", k.flip([0, 1]), persistent=False)
        self.pad = pad

    def forward(self, x: Tensor) -> Tensor:
        return _upfirdn2d_flipped(x, self.flipped_kernel, 1, 1, self.pad)  # (N, C, H + pad0 + pad1 - kH + 1, ...)

    def extra_repr(self) -> str:
        return f"pad={self.pad}, kernel_size={tuple(self.flipped_kernel.shape)}"


class Upsample(nn.Module):
    """×factor FIR upsampling (zero insertion + [1,3,3,1] filter). G의 ToRGB skip 경로에서 쓴다.

    pad는 출력이 정확히 (H·factor, W·factor)가 되도록 고른다 (kernel 4, factor 2 → pad (2, 1)).
    """

    def __init__(self, kernel: Sequence[float], factor: int = 2) -> None:
        super().__init__()
        self.factor = factor
        k = make_kernel(kernel) * (factor**2)  # zero insertion으로 줄어든 밝기 보상
        self.register_buffer("flipped_kernel", k.flip([0, 1]), persistent=False)  # Blur와 같은 이유로 미리 뒤집는다
        p = k.shape[0] - factor
        self.pad = ((p + 1) // 2 + factor - 1, p // 2)

    def forward(self, x: Tensor) -> Tensor:
        return _upfirdn2d_flipped(x, self.flipped_kernel, self.factor, 1, self.pad)  # (N, C, H·f, W·f)

    def extra_repr(self) -> str:
        return f"factor={self.factor}, pad={self.pad}"


def fused_leaky_relu(
    x: Tensor,
    bias: Tensor | None = None,
    negative_slope: float = 0.2,
    scale: float = math.sqrt(2),
) -> Tensor:
    """(x + bias) → LeakyReLU(negative_slope) → ×scale.

    - bias는 채널 축(dim 1)에 더한다: (N, C) linear 출력, (N, C, H, W) conv 출력 모두 된다
    - ×√2: activation 뒤에서 He gain을 곱해 출력 분산을 입력과 비슷하게 유지한다. 그래서 StyleGAN2의
      equalized 층은 `gain=1.0`을 쓴다 (weight 쪽에서 √2를 또 곱하면 층마다 2배씩 분산이 커진다)
    - "fused"는 CUDA 구현에서 세 연산을 한 kernel로 합친 데서 온 이름이다. 여기서는 단순 합성이다
    """
    if bias is not None:
        x = x + bias.reshape(1, -1, *([1] * (x.dim() - 2)))
    return F.leaky_relu(x, negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    """학습 가능한 채널별 bias(0 초기화) + `fused_leaky_relu`. `bias=False`면 activation·scale만 한다.

    앞 층(EqualizedConv2d/Linear)이 이미 bias를 가지면 `bias=False`로 써서 중복을 피한다.
    """

    def __init__(
        self,
        channel: int,
        bias: bool = True,
        negative_slope: float = 0.2,
        scale: float = math.sqrt(2),
    ) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel)) if bias else None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        return fused_leaky_relu(x, self.bias, self.negative_slope, self.scale)

    def extra_repr(self) -> str:
        return f"bias={self.bias is not None}, negative_slope={self.negative_slope}, scale={self.scale:.4f}"
