"""가중치 초기화. DCGAN·pix2pix·CycleGAN 관례: Conv/Linear N(0, 0.02), norm weight N(1, 0.02)."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch.nn.utils import parametrize

_CONV_LINEAR = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
)
_NORMS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)
_INIT_TYPES = ("normal", "xavier", "kaiming", "orthogonal")


def _raw_weight(m: nn.Module) -> Tensor:
    """spectral_norm 등 parametrization이 걸려 있으면 원본 weight를 돌려준다 (계산된 W/σ가 아니라)."""
    if parametrize.is_parametrized(m, "weight"):
        return m.parametrizations.weight.original
    return m.weight


def init_weights(net: nn.Module, init_type: str = "normal", gain: float = 0.02) -> None:
    """`net`의 모든 하위 모듈을 초기화한다.

    - Conv*/ConvTranspose*/Linear: normal N(0, gain) | xavier_normal(gain) | kaiming_normal(fan_in) | orthogonal(gain), bias 0
    - BatchNorm·affine InstanceNorm: weight N(1, gain), bias 0
    클래스 이름이 아니라 isinstance로 판별하므로 `EqualizedConv2d`·`EqualizedLinear`는 건드리지 않는다
    (equalized lr는 N(0, 1) 초기화가 전제다).
    spectral_norm은 가능하면 이 함수 다음에 적용한다. 이미 적용된 레이어도 원본 weight를 초기화하지만,
    power iteration의 u/v 벡터는 이전 weight 기준이라 몇 step 동안 σ 추정이 부정확하다.
    """
    if init_type not in _INIT_TYPES:
        raise ValueError(f"init_type은 {' | '.join(_INIT_TYPES)} 중 하나: {init_type!r}")

    def init_fn(m: nn.Module) -> None:
        if isinstance(m, _CONV_LINEAR):
            w = _raw_weight(m)
            if init_type == "normal":
                nn.init.normal_(w, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(w, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(w, a=0, mode="fan_in")
            else:
                nn.init.orthogonal_(w, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, _NORMS) and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, gain)
            nn.init.zeros_(m.bias)

    net.apply(init_fn)
