"""Generator weight의 exponential moving average (ProGAN·StyleGAN 계열 관례)."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn


class EMA:
    """θ_ema ← decay·θ_ema + (1 - decay)·θ.

    샘플링·평가는 `ema.module`로 한다. 학습 중 weight 진동이 평균되어 샘플 품질이 안정된다.
    buffer(BatchNorm running stats 등)는 평균하지 않고 그대로 복사한다.
    `model.to(device)` 다음에 생성한다 (deepcopy가 model과 같은 device에 만들어지도록).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """optimizer step 직후 호출한다."""
        for ema_p, p in zip(self.module.parameters(), model.parameters(), strict=True):
            ema_p.lerp_(p.detach(), 1.0 - self.decay)
        for ema_b, b in zip(self.module.buffers(), model.buffers(), strict=True):
            ema_b.copy_(b)

    def state_dict(self) -> dict[str, Any]:
        """EMA 모델의 state_dict. 그대로 generator에 `load_state_dict` 할 수 있다."""
        return self.module.state_dict()

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """`state_dict()`로 저장한 값을 복원한다."""
        self.module.load_state_dict(sd)
