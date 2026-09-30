"""pix2pix · CycleGAN 네트워크 (junyanz/pytorch-CycleGAN-and-pix2pix 구조 기준).

- `UnetGenerator`: pix2pix 기본 G. encoder-decoder에 같은 해상도끼리 skip connection
- `ResnetGenerator`: CycleGAN 기본 G. downsample 2회 → ResNet block n개 → upsample 2회
- `NLayerDiscriminator`: PatchGAN. 이미지 전체가 아니라 receptive field(70x70) 단위 patch마다 real/fake logit
- `PixelDiscriminator`: 1x1 conv만 쓰는 1x1 PatchGAN (픽셀 단위)
모든 D는 Sigmoid 없이 logits map을 반환한다. 가중치 초기화는 호출자가 `init_weights(net, "normal", 0.02)`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

NormLayer = Callable[[int], nn.Module]


def _no_norm(num_features: int) -> nn.Module:
    return nn.Identity()


def get_norm_layer(norm: str) -> NormLayer:
    """채널 수를 받아 norm 모듈을 만드는 callable.

    - batch: BatchNorm2d(affine=True, running stats 사용)
    - instance: InstanceNorm2d(affine=False, running stats 없음) — junyanz 관례
    - none: Identity
    """
    if norm == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm == "none":
        return _no_norm
    raise ValueError(f"norm은 batch | instance | none 중 하나: {norm!r}")


def _use_bias(norm: str) -> bool:
    """BatchNorm은 자체 shift(beta)가 있어 conv bias가 중복이다. affine 없는 InstanceNorm과 none은 bias를 쓴다.

    junyanz는 none일 때도 bias=False였지만, norm도 bias도 없는 conv가 되므로 여기서는 True로 둔다.
    """
    return norm != "batch"


# ---------------------------------------------------------------------------
# U-Net generator
# ---------------------------------------------------------------------------


class UnetSkipConnectionBlock(nn.Module):
    """U-Net의 한 층: down → submodule(더 안쪽 층) → up, 그리고 입력 x와 출력을 채널 방향으로 concat.

        x ─ downsample ─ [submodule] ─ upsample ─┐
        └────────────────── skip ─────────────────┴→ cat
    outermost는 skip 없이 Tanh 출력, innermost는 submodule 없이 bottleneck이다.
    """

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer: NormLayer = nn.BatchNorm2d,
        use_bias: bool = False,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        # junyanz와 같이 in-place LeakyReLU다. 블록 입력 x를 직접 바꾸므로 forward의 skip(cat)에는
        # 활성화가 적용된 x가 들어간다 (원 구현과 같은 동작).
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model = [downconv, submodule, uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if use_dropout:
                model.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)  # (N, outer_nc * 2, H, W)


class UnetGenerator(nn.Module):
    """U-Net generator. `num_downs`번 절반으로 줄이므로 입력 크기는 2^num_downs의 배수여야 한다.

    256x256 → num_downs=8 (bottleneck 1x1), 128x128 → num_downs=7.
    채널: ngf → 2ngf → 4ngf → 8ngf (이후 8ngf 유지). pix2pix는 bottleneck 바로 바깥
    `num_downs - 5`개 블록(256px면 3개)에 Dropout(0.5)을 넣어 noise z 대신 확률성을 준다.
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        num_downs: int = 8,
        ngf: int = 64,
        norm: str = "batch",
        use_dropout: bool = True,
    ) -> None:
        super().__init__()
        if num_downs < 5:
            raise ValueError(f"num_downs는 5 이상이어야 합니다: {num_downs}")
        norm_layer = get_norm_layer(norm)
        kw = {"norm_layer": norm_layer, "use_bias": _use_bias(norm)}
        # 안쪽(bottleneck)부터 바깥으로 재귀적으로 감싼다
        block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True, **kw)
        for _ in range(num_downs - 5):
            block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=block, use_dropout=use_dropout, **kw)
        block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=block, **kw)
        block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=block, **kw)
        block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=block, **kw)
        self.model = UnetSkipConnectionBlock(out_ch, ngf, input_nc=in_ch, submodule=block, outermost=True, **kw)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)  # (N, out_ch, H, W) in [-1, 1]


# ---------------------------------------------------------------------------
# ResNet generator
# ---------------------------------------------------------------------------


def _padding(padding_type: str) -> tuple[list[nn.Module], int]:
    """padding layer 목록과 conv에 줄 padding 값. reflect/replicate는 별도 layer, zero는 conv padding=1."""
    if padding_type == "reflect":
        return [nn.ReflectionPad2d(1)], 0
    if padding_type == "replicate":
        return [nn.ReplicationPad2d(1)], 0
    if padding_type == "zero":
        return [], 1
    raise ValueError(f"padding_type은 reflect | replicate | zero 중 하나: {padding_type!r}")


class ResnetBlock(nn.Module):
    """y = x + F(x), F = [pad → conv3x3 → norm → ReLU → (dropout) → pad → conv3x3 → norm]."""

    def __init__(self, dim: int, padding_type: str, norm_layer: NormLayer, use_dropout: bool, use_bias: bool) -> None:
        super().__init__()
        pad, p = _padding(padding_type)
        block: list[nn.Module] = [
            *pad,
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            block.append(nn.Dropout(0.5))
        pad, p = _padding(padding_type)
        block += [*pad, nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        self.conv_block = nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv_block(x)  # (N, dim, H, W)


class ResnetGenerator(nn.Module):
    """CycleGAN generator (Johnson 2016 style transfer 네트워크 기반).

    c7s1-ngf → d(2ngf) → d(4ngf) → R(4ngf) × n_blocks → u(2ngf) → u(ngf) → c7s1-out_ch → Tanh.
    256px는 n_blocks=9, 128px는 6이 관례다.
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        ngf: int = 64,
        norm: str = "instance",
        use_dropout: bool = False,
        n_blocks: int = 9,
        padding_type: str = "reflect",
    ) -> None:
        super().__init__()
        if n_blocks < 0:
            raise ValueError(f"n_blocks는 0 이상: {n_blocks}")
        norm_layer = get_norm_layer(norm)
        use_bias = _use_bias(norm)

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        n_down = 2
        for i in range(n_down):  # (N, ngf·2^i, H, W) → (N, ngf·2^(i+1), H/2, W/2)
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
        mult = 2**n_down
        for _ in range(n_blocks):
            model.append(ResnetBlock(ngf * mult, padding_type, norm_layer, use_dropout, use_bias))
        for i in range(n_down):  # (N, ngf·2^(2-i), H, W) → (N, ngf·2^(1-i), 2H, 2W)
            mult = 2 ** (n_down - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias
                ),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)  # (N, out_ch, H, W) in [-1, 1]


# ---------------------------------------------------------------------------
# Discriminators
# ---------------------------------------------------------------------------


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator. n_layers=3이면 각 출력 logit의 receptive field가 70x70이다.

    C64(stride 2, norm 없음) → C128 → C256 (stride 2) → C512 (stride 1) → conv(1, stride 1).
    256x256 입력 → (N, 1, 30, 30) logits map. pix2pix는 입력으로 [condition, image]를 채널 concat해 넣는다.
    """

    def __init__(self, in_ch: int, ndf: int = 64, n_layers: int = 3, norm: str = "batch") -> None:
        super().__init__()
        norm_layer = get_norm_layer(norm)
        use_bias = _use_bias(norm)
        kw, padw = 4, 1

        seq: list[nn.Module] = [nn.Conv2d(in_ch, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):  # stride 2 layer, 채널은 최대 8ndf
            prev, mult = mult, min(2**n, 8)
            seq += [
                nn.Conv2d(ndf * prev, ndf * mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * mult),
                nn.LeakyReLU(0.2, True),
            ]
        prev, mult = mult, min(2**n_layers, 8)
        seq += [
            nn.Conv2d(ndf * prev, ndf * mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * mult, 1, kernel_size=kw, stride=1, padding=padw),  # 1채널 logits map
        ]
        self.model = nn.Sequential(*seq)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)  # (N, 1, H', W') logits


class PixelDiscriminator(nn.Module):
    """1x1 PatchGAN: 1x1 conv만 써서 픽셀마다 real/fake logit (색 분포만 보고 공간 구조는 보지 않는다)."""

    def __init__(self, in_ch: int, ndf: int = 64, norm: str = "batch") -> None:
        super().__init__()
        norm_layer = get_norm_layer(norm)
        use_bias = _use_bias(norm)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # 최종 logits conv도 bias=use_bias (BatchNorm이면 bias 없음): junyanz 원본을 의도적으로 재현
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # (N, 1, H, W) logits
