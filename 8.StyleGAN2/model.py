"""StyleGAN2 (Karras et al., CVPR 2020) Generator / Discriminator. rosinality/stylegan2-pytorch 구조를 따른다.

Generator (style-based, skip 구조, config-f):
    z ─ PixelNorm ─ 8 × [EqualizedLinear(lr_mul 0.01) + lrelu] ─→ w   (mapping network, Z → W)
    ConstantInput 4×4 ─ StyledConv ─┬─ ToRGB ─────────────────────────────→ rgb(4)
                                    └─ [StyledConv↑, StyledConv] ─ ToRGB + Upsample(rgb) → rgb(8) … rgb(size)
    - 각 층은 w에서 affine으로 얻은 style s로 conv weight를 modulate(×s)하고 demodulate(÷‖w'‖)한다.
      StyleGAN1의 AdaIN(feature 정규화)이 만들던 물방울(droplet) artifact를 없애려고 정규화를 weight로 옮긴 것이다
    - 층마다 noise(학습되는 per-layer 세기)를 더해 머리카락·모공 같은 확률적 세부를 만든다
    - progressive growing 대신 해상도별 ToRGB 출력을 upsample해 누적하는 skip 구조
Discriminator (residual, config-f):
    fromRGB(1×1) → ResBlock(↓) × log2(size/4) → MinibatchStdDev → conv3×3 → linear → lrelu → linear → logit

gain 규약: activation(`fused_leaky_relu`)이 √2를 곱하므로 모든 equalized 층은 `gain=1.0`이다.
gan_common.layers의 기본 gain=√2(ProGAN 방식)를 그대로 쓰면 층마다 √2가 두 번 곱해진다.

채널 수 (channel_multiplier = cm, 상한 512): 4–32px 512, 64px 256·cm, 128px 128·cm, 256px 64·cm.
cm=2가 공식 config-f 채널표다 (64px: 512, 128px: 256, 256px: 128).

upsample conv weight 방향: 미확인: 공식 구현(upsample_conv_2d)은 transposed conv에 넣기 전 weight를 공간적으로
뒤집는(w[::-1, ::-1]) 것으로 알고 있다. 이 구현은 rosinality처럼 뒤집지 않는다. 처음부터 학습할 때는 같은
parametrization의 다른 표현일 뿐이라 영향이 없고, 공식 가중치를 옮겨 올 때만 flip이 필요하다
(공식 networks_stylegan2.py의 upsample_conv_2d를 확인하면 확정된다).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gan_common.layers import EqualizedConv2d, EqualizedLinear, MinibatchStdDev, PixelNorm
from ops import Blur, FusedLeakyReLU, Upsample

BLUR_KERNEL = (1, 3, 3, 1)
SIZES = (8, 16, 32, 64, 128, 256)


def channel_table(channel_multiplier: int = 2) -> dict[int, int]:
    """해상도 → 채널 수. 공식 config-f(cm=2)를 기준으로 하고 512를 넘지 않게 자른다."""
    base = {
        4: 512,
        8: 512,
        16: 512,
        32: 512,
        64: 256 * channel_multiplier,
        128: 128 * channel_multiplier,
        256: 64 * channel_multiplier,
    }
    return {res: min(512, ch) for res, ch in base.items()}


def _check_size(size: int) -> int:
    if size not in SIZES:
        raise ValueError(f"size는 {SIZES} 중 하나여야 합니다: {size}")
    return int(math.log2(size))


# ---------------------------------------------------------------------------
# Generator 구성 요소
# ---------------------------------------------------------------------------


class ModulatedConv2d(nn.Module):
    """Weight modulation / demodulation conv (StyleGAN2의 핵심).

        s   = A(w)                                  style affine, (N, in_ch). bias 1로 초기화해 처음엔 s ≈ 1
        w'  = scale · W · s                         modulation: 입력 채널 i의 weight를 s_i배
        w'' = w' / sqrt(Σ_{i,k,l} w'^2 + ε)          demodulation: 출력 채널마다 weight norm을 1로 → 출력 분산 ≈ 1

    샘플마다 weight가 다르므로 batch를 group 축으로 펴서 grouped conv 한 번으로 계산한다
    (입력 (1, N·in_ch, H, W), weight (N·out_ch, in_ch, k, k), groups=N).

    upsample=True: transposed conv(stride 2)로 해상도를 2배 올린 뒤 Blur(FIR, gain 4)로 aliasing을 줄인다.
    downsample은 G에 필요 없어 구현하지 않는다 (D는 `ConvLayer`의 blur + stride-2 conv를 쓴다).
    weight는 equalized lr 방식(N(0, 1) 초기화 + runtime scale 1/sqrt(fan_in), gain 1)이다.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        style_dim: int,
        demodulate: bool = True,
        upsample: bool = False,
        blur_kernel: Sequence[int] = BLUR_KERNEL,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.eps = eps
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel_size, kernel_size))
        self.scale = 1.0 / math.sqrt(in_ch * kernel_size * kernel_size)  # gain 1 (activation이 √2 담당)
        # style affine: w → s. 뒤에 activation이 없으므로 gain 1, bias 1로 시작 (modulation 초기값 ≈ 1)
        self.modulation = EqualizedLinear(style_dim, in_ch, gain=1.0, bias_init=1.0)

        if upsample:
            # conv_transpose(k, stride 2) 출력 2H+k-2 → blur 후 정확히 2H가 되도록 pad를 고른다 (k=3: (1, 1))
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            self.blur = Blur(blur_kernel, pad=((p + 1) // 2 + factor - 1, p // 2 + 1), upsample_factor=factor)

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        n, _, h, w = x.shape  # x: (N, in_ch, H, W), style: (N, style_dim)
        s = self.modulation(style).reshape(n, 1, self.in_ch, 1, 1)  # (N, 1, in_ch, 1, 1)
        weight = self.scale * self.weight * s  # (N, out_ch, in_ch, k, k) modulation
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum(dim=(2, 3, 4)) + self.eps)  # (N, out_ch)
            weight = weight * demod.reshape(n, self.out_ch, 1, 1, 1)

        k = self.kernel_size
        x = x.reshape(1, n * self.in_ch, h, w)  # batch를 채널 축으로: (1, N·in_ch, H, W)
        if self.upsample:
            # conv_transpose2d weight 형식은 (in, out/groups, k, k) → 샘플별 (in_ch, out_ch)로 전치
            weight = weight.transpose(1, 2).reshape(n * self.in_ch, self.out_ch, k, k)
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=n)  # (1, N·out_ch, 2H+1, 2W+1)
            out = out.reshape(n, self.out_ch, out.shape[2], out.shape[3])
            return self.blur(out)  # (N, out_ch, 2H, 2W)
        weight = weight.reshape(n * self.out_ch, self.in_ch, k, k)
        out = F.conv2d(x, weight, padding=self.padding, groups=n)  # (1, N·out_ch, H, W)
        return out.reshape(n, self.out_ch, h, w)

    def extra_repr(self) -> str:
        return (
            f"{self.in_ch}, {self.out_ch}, kernel_size={self.kernel_size}, "
            f"demodulate={self.demodulate}, upsample={self.upsample}"
        )


class NoiseInjection(nn.Module):
    """x + strength · noise. noise는 (N, 1, H, W) 단일 채널을 모든 채널에 broadcast한다.

    strength는 층마다 스칼라 하나(0 초기화). noise가 주어지지 않으면 매번 새로 뽑는다.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            n, _, h, w = x.shape
            noise = torch.randn(n, 1, h, w, device=x.device, dtype=x.dtype)
        return x + self.weight * noise  # (N, C, H, W)


class ConstantInput(nn.Module):
    """synthesis network의 시작점: 학습되는 상수 (1, C, 4, 4). 입력 z가 아니라 style만으로 이미지를 조절한다."""

    def __init__(self, channel: int, size: int = 4) -> None:
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch_size: int) -> Tensor:
        return self.input.repeat(batch_size, 1, 1, 1)  # (N, C, 4, 4)


class StyledConv(nn.Module):
    """ModulatedConv2d(3×3, demodulate) → NoiseInjection → bias + lrelu×√2."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        style_dim: int,
        upsample: bool = False,
        blur_kernel: Sequence[int] = BLUR_KERNEL,
    ) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(
            in_ch, out_ch, kernel_size, style_dim, demodulate=True, upsample=upsample, blur_kernel=blur_kernel
        )
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_ch)  # modconv에는 bias가 없으므로 여기서 채널 bias

    def forward(self, x: Tensor, style: Tensor, noise: Tensor | None = None) -> Tensor:
        out = self.conv(x, style)  # (N, out_ch, H', W')
        out = self.noise(out, noise)
        return self.activate(out)


class ToRGB(nn.Module):
    """현재 해상도 feature → RGB (modulated 1×1 conv, demodulate 없음) + 이전 해상도 RGB를 upsample해 더한다.

    demodulate하지 않는 이유: 출력 색의 크기 자체가 의미 있는 신호이므로 정규화하지 않는다.
    """

    def __init__(
        self,
        in_ch: int,
        style_dim: int,
        img_channels: int = 3,
        upsample: bool = True,
        blur_kernel: Sequence[int] = BLUR_KERNEL,
    ) -> None:
        super().__init__()
        self.upsample = Upsample(blur_kernel) if upsample else None
        self.conv = ModulatedConv2d(in_ch, img_channels, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, img_channels, 1, 1))

    def forward(self, x: Tensor, style: Tensor, skip: Tensor | None = None) -> Tensor:
        out = self.conv(x, style) + self.bias  # (N, img_ch, H, W)
        if skip is not None:
            if self.upsample is None:
                raise RuntimeError("upsample=False인 ToRGB에는 skip을 줄 수 없습니다.")
            out = out + self.upsample(skip)  # skip (N, img_ch, H/2, W/2) → (N, img_ch, H, W)
        return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator(nn.Module):
    """StyleGAN2 generator (mapping + synthesis, skip 구조).

    latent 개수 n_latent = 2·log2(size) - 2 (64px: 10). 인덱스 배치:
        4×4: conv1 → w[0], to_rgb1 → w[1]
        해상도 r(8…size)의 i번째 블록(i=1,3,5…): conv_up → w[i], conv → w[i+1], to_rgb → w[i+2]
    style mixing은 w 두 개를 inject_index에서 이어 붙여 앞 층(거친 구조)과 뒤 층(세부)에 다른 style을 준다.
    """

    def __init__(
        self,
        size: int,
        style_dim: int = 512,
        n_mlp: int = 8,
        channel_multiplier: int = 2,
        img_channels: int = 3,
        blur_kernel: Sequence[int] = BLUR_KERNEL,
        lr_mlp: float = 0.01,
    ) -> None:
        super().__init__()
        self.size = size
        self.style_dim = style_dim
        self.log_size = _check_size(size)
        self.channels = channel_table(channel_multiplier)

        # mapping network: lr_mul 0.01로 이 부분만 학습률을 100배 낮춘다 (W 공간이 천천히 변해 학습 안정)
        mapping: list[nn.Module] = [PixelNorm()]
        for _ in range(n_mlp):
            mapping += [
                EqualizedLinear(style_dim, style_dim, gain=1.0, lr_mul=lr_mlp),
                FusedLeakyReLU(style_dim, bias=False),  # bias는 EqualizedLinear 쪽(×lr_mul)에 있다
            ]
        self.style = nn.Sequential(*mapping)

        ch4 = self.channels[4]
        self.input = ConstantInput(ch4)
        self.conv1 = StyledConv(ch4, ch4, 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(ch4, style_dim, img_channels, upsample=False)

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        in_ch = ch4
        for i in range(3, self.log_size + 1):
            out_ch = self.channels[2**i]
            self.convs.append(StyledConv(in_ch, out_ch, 3, style_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs.append(StyledConv(out_ch, out_ch, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_ch, style_dim, img_channels, blur_kernel=blur_kernel))
            in_ch = out_ch

        # 고정 noise (randomize_noise=False일 때 사용). layer_idx 0은 4×4, 이후 해상도마다 2개씩
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.noises = nn.Module()
        for layer_idx in range(self.num_layers):
            res = 2 ** ((layer_idx + 5) // 2)
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(1, 1, res, res))

        self.n_latent = self.log_size * 2 - 2

    @torch.no_grad()
    def mean_latent(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        """z n개를 mapping한 w의 평균 (1, style_dim). truncation trick의 중심으로 쓴다.

        `generator`(G와 같은 device의 torch.Generator)를 주면 그 난수로 z를 뽑는다. 고정 seed generator를 주면
        호출마다 같은 z 집합을 쓰므로 w̄의 추정 잡음이 사라지고, 전역 RNG 상태도 소비하지 않는다.
        """
        z = torch.randn(n, self.style_dim, device=self.input.input.device, generator=generator)
        return self.style(z).mean(dim=0, keepdim=True)

    def get_latent(self, z: Tensor) -> Tensor:
        """z (N, style_dim) → w (N, style_dim)."""
        return self.style(z)

    def forward(
        self,
        styles: list[Tensor] | Tensor,
        return_latents: bool = False,
        inject_index: int | None = None,
        truncation: float = 1.0,
        truncation_latent: Tensor | None = None,
        input_is_latent: bool = False,
        noise: list[Tensor | None] | None = None,
        randomize_noise: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        """styles: z(또는 `input_is_latent=True`면 w) 텐서 1개 또는 2개의 list. 각 (N, style_dim).

        2개면 style mixing: w[:inject_index] = 첫째, 나머지 = 둘째 (inject_index 없으면 [1, n_latent-1]에서 무작위).
        truncation < 1: w ← w̄ + ψ(w - w̄) (w̄ = truncation_latent). 다양성을 줄이고 품질을 올린다.
        반환: (이미지 (N, img_ch, size, size), `return_latents`면 w+ (N, n_latent, style_dim) 아니면 None).
        """
        if isinstance(styles, Tensor):
            styles = [styles]
        if not 1 <= len(styles) <= 2:
            raise ValueError(f"styles는 1개 또는 2개여야 합니다: {len(styles)}")
        if not input_is_latent:
            styles = [self.style(s) for s in styles]  # z → w, 각 (N, style_dim)

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]

        if truncation < 1:
            if truncation_latent is None:
                raise ValueError("truncation < 1이면 truncation_latent(mean_latent 결과)가 필요합니다.")
            styles = [truncation_latent + truncation * (s - truncation_latent) for s in styles]

        # w+ : 층별 latent (N, n_latent, style_dim)
        if len(styles) == 1:
            latent = styles[0] if styles[0].dim() == 3 else styles[0].unsqueeze(1).repeat(1, self.n_latent, 1)
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], dim=1)

        out = self.input(latent.size(0))  # (N, ch4, 4, 4)
        out = self.conv1(out, latent[:, 0], noise=noise[0])  # (N, ch4, 4, 4)
        skip = self.to_rgb1(out, latent[:, 1])  # (N, img_ch, 4, 4)

        i = 1
        for conv_up, conv, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs, strict=True
        ):
            out = conv_up(out, latent[:, i], noise=noise1)  # (N, ch_r, r, r) 해상도 2배
            out = conv(out, latent[:, i + 1], noise=noise2)  # (N, ch_r, r, r)
            skip = to_rgb(out, latent[:, i + 2], skip)  # (N, img_ch, r, r)
            i += 2

        return skip, (latent if return_latents else None)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------


class ConvLayer(nn.Sequential):
    """[Blur] → EqualizedConv2d → [bias + lrelu×√2].

    downsample=True: FIR blur 후 stride-2 conv (padding 0). blur pad는 출력이 정확히 H/2가 되게 고른다
    (3×3: (2, 2), 1×1: (1, 1)). activate=True면 bias는 FusedLeakyReLU 쪽에 둔다.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        downsample: bool = False,
        blur_kernel: Sequence[int] = BLUR_KERNEL,
        bias: bool = True,
        activate: bool = True,
    ) -> None:
        layers: list[nn.Module] = []
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            layers.append(Blur(blur_kernel, pad=((p + 1) // 2, p // 2)))
            stride, padding = 2, 0
        else:
            stride, padding = 1, kernel_size // 2
        layers.append(
            EqualizedConv2d(
                in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias and not activate, gain=1.0
            )
        )
        if activate:
            layers.append(FusedLeakyReLU(out_ch, bias=bias))
        super().__init__(*layers)


class ResBlock(nn.Module):
    """out = (conv3×3 → conv3×3↓  +  1×1↓ skip) / √2. 합의 분산이 2배가 되므로 √2로 나눈다."""

    def __init__(self, in_ch: int, out_ch: int, blur_kernel: Sequence[int] = BLUR_KERNEL) -> None:
        super().__init__()
        self.conv1 = ConvLayer(in_ch, in_ch, 3, blur_kernel=blur_kernel)
        self.conv2 = ConvLayer(in_ch, out_ch, 3, downsample=True, blur_kernel=blur_kernel)
        self.skip = ConvLayer(in_ch, out_ch, 1, downsample=True, blur_kernel=blur_kernel, bias=False, activate=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2(self.conv1(x))  # (N, out_ch, H/2, W/2)
        return (out + self.skip(x)) / math.sqrt(2)


class Discriminator(nn.Module):
    """StyleGAN2 residual discriminator. 입력 (N, img_ch, size, size) → logits (N,)."""

    def __init__(
        self,
        size: int,
        channel_multiplier: int = 2,
        img_channels: int = 3,
        blur_kernel: Sequence[int] = BLUR_KERNEL,
    ) -> None:
        super().__init__()
        log_size = _check_size(size)
        channels = channel_table(channel_multiplier)

        convs: list[nn.Module] = [ConvLayer(img_channels, channels[size], 1)]  # fromRGB
        in_ch = channels[size]
        for i in range(log_size, 2, -1):  # size → size/2 → … → 4
            out_ch = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_ch, out_ch, blur_kernel))
            in_ch = out_ch
        self.convs = nn.Sequential(*convs)

        self.stddev = MinibatchStdDev(group_size=4, num_channels=1)
        ch4 = channels[4]
        self.final_conv = ConvLayer(in_ch + 1, ch4, 3)
        self.final_linear = nn.Sequential(
            EqualizedLinear(ch4 * 4 * 4, ch4, gain=1.0),
            FusedLeakyReLU(ch4, bias=False),  # bias는 EqualizedLinear 쪽에 있다
            EqualizedLinear(ch4, 1, gain=1.0),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.convs(x)  # (N, ch4, 4, 4)
        out = self.stddev(out)  # (N, ch4 + 1, 4, 4)
        out = self.final_conv(out)  # (N, ch4, 4, 4)
        out = self.final_linear(out.flatten(1))  # (N, ch4·16) → (N, 1)
        return out.squeeze(1)  # (N,) logits
