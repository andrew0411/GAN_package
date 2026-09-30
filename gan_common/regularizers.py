"""Discriminator 정규화: WGAN-GP gradient penalty, R1 penalty, WGAN weight clipping."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor


def gradient_penalty(
    critic_fn: Callable[[Tensor], Tensor],
    real: Tensor,
    fake: Tensor,
    center: float = 1.0,
) -> Tensor:
    """WGAN-GP (Gulrajani 2017) gradient penalty.

        x̂ = α·x + (1 - α)·G(z),  α ~ U[0, 1] (샘플별)
        GP = E[(||∇_x̂ D(x̂)||_2 - center)^2]

    최적 critic은 real·fake 사이 직선 위에서 gradient norm이 1이므로 그 지점에서 norm을 1로 당긴다.
    `critic_fn`은 임의 callable이다 (예: `lambda x: D(x, alpha, step)`). `lambda_gp` 곱은 호출자가 한다.
    """
    n = real.size(0)
    alpha = torch.rand(n, *([1] * (real.dim() - 1)), device=real.device, dtype=real.dtype)  # (N, 1, 1, 1)
    interp = (alpha * real.detach() + (1.0 - alpha) * fake.detach()).requires_grad_(True)
    scores = critic_fn(interp)
    (grad,) = torch.autograd.grad(
        outputs=scores,
        inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,  # penalty 자체를 D 파라미터로 역전파해야 하므로 2차 미분 그래프를 남긴다
    )
    grad_norm = grad.reshape(n, -1).norm(2, dim=1)  # (N,)
    return ((grad_norm - center) ** 2).mean()


def r1_penalty(real_logits: Tensor, real_images: Tensor) -> Tensor:
    """R1 penalty (Mescheder 2018, StyleGAN2): E[||∇_x D(x)||^2], x는 real 이미지만.

    전제: `real_images.requires_grad_(True)` 상태로 D에 넣어 `real_logits`를 얻었다.
    최종 loss에는 `gamma / 2`를 호출자가 곱한다.
    """
    if not real_images.requires_grad:
        raise ValueError("r1_penalty: real_images.requires_grad_(True) 후 D에 넣어야 합니다.")
    (grad,) = torch.autograd.grad(outputs=real_logits.sum(), inputs=real_images, create_graph=True)
    return grad.pow(2).reshape(grad.size(0), -1).sum(1).mean()


def clip_weights_(module: nn.Module, clip: float) -> None:
    """WGAN(Arjovsky 2017) weight clipping: 모든 파라미터를 [-clip, clip]로 in-place clamp."""
    with torch.no_grad():
        for p in module.parameters():
            p.clamp_(-clip, clip)
