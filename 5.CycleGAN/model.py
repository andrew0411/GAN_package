"""CycleGAN 학습 로직: ImagePool과 CycleGAN (네트워크 4개, loss, optimizer, lr scheduler).

이름 규칙 (junyanz 구현과 이름이 다르니 주의):
- G_AB: A → B 번역,  G_BA: B → A 번역
- D_A: domain A 판별 (real a vs G_BA(b)),  D_B: domain B 판별 (real b vs G_AB(a))
  junyanz의 netG_A / netG_B는 여기의 G_AB / G_BA, netD_A / netD_B는 여기의 D_B / D_A에 해당한다.

G objective (λ_A = λ_B = 10, λ_idt = 0.5):
    L_G = L_GAN(G_AB, D_B) + L_GAN(G_BA, D_A)                       adversarial (LSGAN)
        + λ_A·|G_BA(G_AB(a)) - a|₁ + λ_B·|G_AB(G_BA(b)) - b|₁      cycle consistency
        + λ_idt·(λ_B·|G_AB(b) - b|₁ + λ_A·|G_BA(a) - a|₁)          identity
D objective (각 D마다): 0.5·[(D(real) - 1)² + D(pool(fake))²]
"""

from __future__ import annotations

import contextlib
import itertools
import random
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gan_common.losses import GANLoss
from gan_common.networks.image2image import NLayerDiscriminator, ResnetGenerator
from gan_common.weights import init_weights

# 시각화 순서: 한 행 = [a | G_AB(a) | G_BA(G_AB(a)) | b | G_BA(b) | G_AB(G_BA(b))]
VISUAL_KEYS = ("real_A", "fake_B", "rec_A", "real_B", "fake_A", "rec_B")


class ImagePool:
    """G가 과거에 만든 이미지를 최대 `pool_size`장 저장하는 buffer (Shrivastava et al. 2017).

    D를 "지금 G"의 결과만이 아니라 과거 G들의 결과 history로도 학습시켜,
    G와 D가 서로를 쫓아다니며 진동하는 것을 줄인다. query 규칙 (junyanz ImagePool과 같음):
    - pool이 다 차기 전: 들어온 이미지를 저장하고 그대로 돌려준다
    - 다 찬 뒤: 50% 확률로 pool의 무작위 한 장을 돌려주고 그 자리에 새 이미지를 넣는다.
      나머지 50%는 새 이미지를 그대로 돌려준다
    `pool_size=0`이면 buffer 없이 입력을 그대로 돌려준다. pool 내용은 checkpoint에 넣지 않는다 (junyanz와 같음).
    """

    def __init__(self, pool_size: int = 50) -> None:
        self.pool_size = pool_size
        self.images: list[Tensor] = []

    def query(self, images: Tensor) -> Tensor:
        """(N, C, H, W) fake batch → D 학습에 쓸 (N, C, H, W) batch. 반환값은 gradient가 끊겨 있다."""
        images = images.detach()
        if self.pool_size == 0:
            return images
        out = []
        for image in images:
            image = image.unsqueeze(0)  # (1, C, H, W)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                out.append(image)
            elif random.random() > 0.5:
                idx = random.randint(0, self.pool_size - 1)
                out.append(self.images[idx])  # 과거 이미지를 돌려주고
                self.images[idx] = image  # 그 자리를 새 이미지로 교체
            else:
                out.append(image)
        return torch.cat(out, dim=0)


def linear_decay_rule(n_epochs: int, n_epochs_decay: int) -> Callable[[int], float]:
    """LambdaLR용 lr 배율. epoch e(1부터)의 배율 = 1 - max(0, e - n_epochs) / (n_epochs_decay + 1).

    처음 n_epochs 동안 초기 lr을 유지하고, 이후 n_epochs_decay 동안 0을 향해 선형으로 줄인다
    (junyanz lr_policy="linear", epoch_count=1의 공식). LambdaLR의 last_epoch는 0부터 세므로 e = last_epoch + 1.
    scheduler.step()은 PyTorch 권장 순서(optimizer.step() 뒤)대로 epoch이 끝날 때 부른다
    (`CycleGAN.update_learning_rate`). junyanz train.py는 update_learning_rate()를 epoch 시작에서 부르는
    버전이 있어, 그 경우 schedule이 한 epoch 앞당겨지고 마지막 epoch의 lr이 0이 된다.
    여기서는 마지막 epoch lr = lr / (n_epochs_decay + 1).
    """

    def rule(epoch_idx: int) -> float:
        return 1.0 - max(0, epoch_idx + 1 - n_epochs) / float(n_epochs_decay + 1)

    return rule


def check_crop_size(crop_size: int) -> None:
    """ResNet G는 stride-2 downsample 2회 뒤 upsample 2회로 되돌리므로 crop_size가 4의 배수여야
    출력 크기가 입력과 같다 (다르면 cycle·identity L1에서 shape이 맞지 않는다)."""
    if crop_size % 4 != 0:
        raise ValueError(f"crop_size({crop_size})는 4의 배수여야 합니다 (ResNet G downsample 2회 → upsample 2회).")


@contextlib.contextmanager
def preserved_buffers(*nets: nn.Module) -> Iterator[None]:
    """블록 안의 forward가 buffer(BN running stats 등)를 바꿔도, 끝나면 원래 값으로 되돌린다."""
    saved = [[b.clone() for b in net.buffers()] for net in nets]
    try:
        yield
    finally:
        with torch.no_grad():
            for net, bufs in zip(nets, saved):
                for b, s in zip(net.buffers(), bufs):
                    b.copy_(s)


def build_generator(
    ngf: int = 64, norm: str = "instance", use_dropout: bool = False, n_blocks: int = 9
) -> ResnetGenerator:
    """ResNet G (256px는 9 blocks). test.py도 이 함수로 학습 때와 같은 구조를 만든다."""
    return ResnetGenerator(3, 3, ngf=ngf, norm=norm, use_dropout=use_dropout, n_blocks=n_blocks)


class CycleGAN:
    """G_AB, G_BA, D_A, D_B와 optimizer·scheduler를 묶은 학습 객체. `train_step(batch)`가 1 iteration이다."""

    def __init__(
        self,
        device: torch.device,
        *,
        ngf: int = 64,
        ndf: int = 64,
        n_blocks: int = 9,
        n_layers_D: int = 3,
        norm: str = "instance",
        use_dropout: bool = False,
        init_type: str = "normal",
        init_gain: float = 0.02,
        gan_mode: str = "lsgan",
        lambda_A: float = 10.0,
        lambda_B: float = 10.0,
        lambda_identity: float = 0.5,
        pool_size: int = 50,
        lr: float = 2e-4,
        betas: tuple[float, float] = (0.5, 0.999),
        n_epochs: int = 100,
        n_epochs_decay: int = 100,
    ) -> None:
        self.device = device
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_identity = lambda_identity

        self.G_AB = build_generator(ngf, norm, use_dropout, n_blocks)  # A → B
        self.G_BA = build_generator(ngf, norm, use_dropout, n_blocks)  # B → A
        self.D_A = NLayerDiscriminator(3, ndf=ndf, n_layers=n_layers_D, norm=norm)  # real a vs G_BA(b)
        self.D_B = NLayerDiscriminator(3, ndf=ndf, n_layers=n_layers_D, norm=norm)  # real b vs G_AB(a)
        for net in self.nets.values():
            init_weights(net, init_type, init_gain)
            net.to(device)

        self.gan_loss = GANLoss(gan_mode)
        self.pool_A = ImagePool(pool_size)  # G_BA가 만든 fake a의 history → D_A 학습용
        self.pool_B = ImagePool(pool_size)  # G_AB가 만든 fake b의 history → D_B 학습용

        # G 둘, D 둘을 각각 하나의 optimizer로 묶는다 (junyanz 관행)
        g_params = itertools.chain(self.G_AB.parameters(), self.G_BA.parameters())
        d_params = itertools.chain(self.D_A.parameters(), self.D_B.parameters())
        self.opt_G = torch.optim.Adam(g_params, lr=lr, betas=betas)
        self.opt_D = torch.optim.Adam(d_params, lr=lr, betas=betas)
        rule = linear_decay_rule(n_epochs, n_epochs_decay)
        self.sched_G = torch.optim.lr_scheduler.LambdaLR(self.opt_G, rule)
        self.sched_D = torch.optim.lr_scheduler.LambdaLR(self.opt_D, rule)

        self.visuals: dict[str, Tensor] = {}  # 마지막 train_step의 이미지 (VISUAL_KEYS, detach됨)

    @property
    def nets(self) -> dict[str, nn.Module]:
        return {"G_AB": self.G_AB, "G_BA": self.G_BA, "D_A": self.D_A, "D_B": self.D_B}

    @property
    def lr(self) -> float:
        """현재 G optimizer의 lr (D도 같은 schedule)."""
        return self.opt_G.param_groups[0]["lr"]

    @staticmethod
    def _set_requires_grad(*nets: nn.Module, requires_grad: bool) -> None:
        for net in nets:
            net.requires_grad_(requires_grad)

    def _backward_d(self, D: nn.Module, real: Tensor, fake: Tensor) -> Tensor:
        """0.5·[L(D(real), 1) + L(D(fake), 0)]을 계산하고 backward한다.

        ×0.5: D objective를 절반으로 줄여 D가 G보다 빨리 학습하는 것을 늦춘다 (junyanz 관행).
        """
        loss = 0.5 * self.gan_loss.d_loss(D(real), D(fake.detach()))  # D(·): (N, 1, 30, 30) patch logits
        loss.backward()
        return loss

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """1 iteration: G 둘을 한 번 update한 뒤 D 둘을 update한다. 반환값은 로깅용 scalar loss."""
        real_a = batch["A"].to(self.device, non_blocking=True)  # (N, 3, H, W)
        real_b = batch["B"].to(self.device, non_blocking=True)

        # forward: 두 방향 번역과 되돌리기(cycle)
        fake_b = self.G_AB(real_a)  # G_AB(a)
        rec_a = self.G_BA(fake_b)  # G_BA(G_AB(a)) ≈ a
        fake_a = self.G_BA(real_b)  # G_BA(b)
        rec_b = self.G_AB(fake_a)  # G_AB(G_BA(b)) ≈ b

        # (1) G update ---------------------------------------------------------------
        # D는 고정한다: G loss의 gradient는 D를 거쳐 G로만 흐르면 되므로 D weight의 gradient는 계산하지 않는다.
        self._set_requires_grad(self.D_A, self.D_B, requires_grad=False)
        self.opt_G.zero_grad()
        # adversarial (LSGAN): (D_B(G_AB(a)) - 1)², (D_A(G_BA(b)) - 1)²
        #   D가 fake를 real(1)로 보게 만든다. LSGAN은 결정 경계에서 먼 sample에도 gradient를 주어 안정적이다.
        loss_gan_ab = self.gan_loss.g_loss(self.D_B(fake_b))
        loss_gan_ba = self.gan_loss.g_loss(self.D_A(fake_a))
        # cycle consistency: A → B → A, B → A → B로 돌아오면 원래 이미지여야 한다.
        #   짝 데이터 없이도 번역이 입력의 내용(구조)을 보존하게 강제한다. 이 항이 없으면 G는
        #   target domain처럼 보이기만 하면 입력과 무관한 이미지를 내도 adversarial loss를 만족할 수 있다.
        loss_cycle_a = F.l1_loss(rec_a, real_a) * self.lambda_A
        loss_cycle_b = F.l1_loss(rec_b, real_b) * self.lambda_B
        # identity: 이미 target domain인 이미지는 그대로 통과해야 한다 (G_AB(b) ≈ b, G_BA(a) ≈ a).
        #   색감(tint)을 불필요하게 바꾸지 않게 한다. 가중치 = λ_idt × 해당 domain의 cycle λ.
        if self.lambda_identity > 0:
            loss_idt_b = F.l1_loss(self.G_AB(real_b), real_b) * self.lambda_B * self.lambda_identity
            loss_idt_a = F.l1_loss(self.G_BA(real_a), real_a) * self.lambda_A * self.lambda_identity
        else:
            loss_idt_a = loss_idt_b = torch.zeros((), device=self.device)
        loss_g = loss_gan_ab + loss_gan_ba + loss_cycle_a + loss_cycle_b + loss_idt_a + loss_idt_b
        loss_g.backward()
        self.opt_G.step()

        # (2) D update ---------------------------------------------------------------
        # D_A: real a → 1, fake a → 0 / D_B: real b → 1, fake b → 0  (LSGAN: (D(x) - 1)² + D(fake)²)
        # fake는 방금 만든 이미지 대신 ImagePool을 거친 이미지(과거 G의 결과가 섞임)를 쓴다.
        self._set_requires_grad(self.D_A, self.D_B, requires_grad=True)
        self.opt_D.zero_grad()
        loss_d_a = self._backward_d(self.D_A, real_a, self.pool_A.query(fake_a))
        loss_d_b = self._backward_d(self.D_B, real_b, self.pool_B.query(fake_b))
        self.opt_D.step()

        self.visuals = {
            "real_A": real_a,
            "fake_B": fake_b.detach(),
            "rec_A": rec_a.detach(),
            "real_B": real_b,
            "fake_A": fake_a.detach(),
            "rec_B": rec_b.detach(),
        }
        losses = {
            "loss/D_A": loss_d_a,
            "loss/D_B": loss_d_b,
            "loss/G_AB_gan": loss_gan_ab,
            "loss/G_BA_gan": loss_gan_ba,
            "loss/cycle_A": loss_cycle_a,
            "loss/cycle_B": loss_cycle_b,
            "loss/idt_A": loss_idt_a,
            "loss/idt_B": loss_idt_b,
            "loss/G": loss_g,
        }
        values = torch.stack([v.detach() for v in losses.values()]).tolist()  # GPU → CPU 동기화 1회
        return dict(zip(losses, values))

    @torch.no_grad()
    def translate(self, real_a: Tensor, real_b: Tensor) -> dict[str, Tensor]:
        """A → B → A, B → A → B 번역 결과 (VISUAL_KEYS). gradient 없이 고정 샘플 시각화에 쓴다.

        G는 train 모드 그대로 쓴다. 기본 instance norm은 running stats가 없지만, --norm batch로 학습한 경우
        이 forward가 BN running stats를 샘플 이미지로 바꾸지 않도록 G_AB·G_BA의 buffer를 되돌린다.
        """
        with preserved_buffers(self.G_AB, self.G_BA):
            fake_b = self.G_AB(real_a)
            fake_a = self.G_BA(real_b)
            return {
                "real_A": real_a,
                "fake_B": fake_b,
                "rec_A": self.G_BA(fake_b),
                "real_B": real_b,
                "fake_A": fake_a,
                "rec_B": self.G_AB(fake_a),
            }

    def update_learning_rate(self) -> float:
        """epoch이 끝날 때 부른다 (optimizer.step() 뒤, linear_decay_rule 참조).
        선형 감쇠 schedule을 한 칸 진행하고 다음 epoch의 lr을 돌려준다."""
        self.sched_G.step()
        self.sched_D.step()
        return self.lr

    def state_dict(self) -> dict[str, Any]:
        """네트워크·optimizer·scheduler state (`torch.load(weights_only=True)` 호환). ImagePool은 넣지 않는다."""
        sd: dict[str, Any] = {name: net.state_dict() for name, net in self.nets.items()}
        sd["opt_G"] = self.opt_G.state_dict()
        sd["opt_D"] = self.opt_D.state_dict()
        sd["sched_G"] = self.sched_G.state_dict()
        sd["sched_D"] = self.sched_D.state_dict()
        return sd

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """`state_dict()`로 저장한 값을 복원한다 (optimizer state는 각 파라미터의 device로 옮겨진다)."""
        for name, net in self.nets.items():
            net.load_state_dict(sd[name])
        self.opt_G.load_state_dict(sd["opt_G"])
        self.opt_D.load_state_dict(sd["opt_D"])
        self.sched_G.load_state_dict(sd["sched_G"])
        self.sched_D.load_state_dict(sd["sched_D"])
