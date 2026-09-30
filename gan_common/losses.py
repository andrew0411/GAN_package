"""GAN adversarial loss. Discriminator는 항상 logits(Sigmoid 이전 값)를 낸다는 전제다.

D(x)를 logit, σ를 sigmoid라 할 때 mode별 목적함수:

- vanilla (Goodfellow 2014, non-saturating G)
    L_D = -E[log σ(D(x))] - E[log(1 - σ(D(G(z))))]   → BCEWithLogits(real, 1) + BCEWithLogits(fake, 0)
    L_G = -E[log σ(D(G(z)))]                          → BCEWithLogits(fake, 1)
    minimax 원형 log(1 - σ(D(G(z))))는 학습 초반 gradient가 사라지므로 G는 non-saturating 형태를 쓴다.
- lsgan (Mao 2017): 0/1 target에 대한 least squares
    L_D = E[(D(x) - 1)^2] + E[D(G(z))^2],   L_G = E[(D(G(z)) - 1)^2]
- hinge (Lim 2017, SNGAN·SAGAN·BigGAN)
    L_D = E[relu(1 - D(x))] + E[relu(1 + D(G(z)))],   L_G = -E[D(G(z))]
- wgan (Arjovsky 2017): critic이 Wasserstein-1 거리를 추정한다 (Lipschitz 제약은 clip 또는 GP로 따로)
    L_D = E[D(G(z))] - E[D(x)],   L_G = -E[D(G(z))]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_MODES = ("vanilla", "lsgan", "hinge", "wgan")


class GANLoss(nn.Module):
    """mode별 D/G loss. 입력 logits는 (N,), (N, 1), PatchGAN (N, 1, H, W) 등 임의 shape이면 된다.

    호출은 `d_loss` / `g_loss` 메서드로 한다 (`forward`는 없다).
    `d_loss`는 real 항과 fake 항의 합이다. pix2pix·CycleGAN처럼 0.5를 곱하려면 호출자가 곱한다.
    """

    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in _MODES:
            raise ValueError(f"GANLoss mode는 {' | '.join(_MODES)} 중 하나: {mode!r}")
        self.mode = mode

    def d_loss(self, real_logits: Tensor, fake_logits: Tensor) -> Tensor:
        """Discriminator(critic) loss = real 항 + fake 항."""
        if self.mode == "vanilla":
            return (
                F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            )
        if self.mode == "lsgan":
            return F.mse_loss(real_logits, torch.ones_like(real_logits)) + F.mse_loss(
                fake_logits, torch.zeros_like(fake_logits)
            )
        if self.mode == "hinge":
            return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
        return fake_logits.mean() - real_logits.mean()  # wgan

    def g_loss(self, fake_logits: Tensor) -> Tensor:
        """Generator loss. D가 fake를 real로 판단하도록 민다."""
        if self.mode == "vanilla":
            return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        if self.mode == "lsgan":
            return F.mse_loss(fake_logits, torch.ones_like(fake_logits))
        return -fake_logits.mean()  # hinge, wgan

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}"
