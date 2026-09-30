"""TadGAN 네트워크 (Geiger et al., IEEE BigData 2020). Orion `tadgan` Keras 기본 구조를 PyTorch로 옮겼다.

구성 (W = window_size 100, d = latent_dim 20, N = batch)
- Encoder E : x (N, W, 1) → z (N, d, 1)           BiLSTM(100) → flatten → Linear(d)
- Generator G: z (N, d, 1) → x̂ (N, W, 1)          flatten → Linear(W/2) → (W/2, 1) → BiLSTM(64)
                                                   → Upsample×2 → BiLSTM(64) → 시점별 Linear(1) → Tanh
- Critic_x C_x: x (N, W, 1) → score (N,)          [Conv1d(64, 5) → LeakyReLU(0.2) → Dropout(0.25)] × 4 → flatten → Linear(1)
- Critic_z C_z: z (N, d, 1) → score (N,)          [Linear(h) → LeakyReLU(0.2) → Dropout(0.2)] × 2 → Linear(1)
                                                   h = critic_z_hidden 100 (Orion v0.1.x·논문 시기 값. 현재 Orion master는 20)
E·G는 x → z → x 순환을 만들고, C_x는 x 공간에서 real x vs G(z)를, C_z는 latent 공간에서 N(0, I) 샘플 vs E(x)를
Wasserstein critic으로 구분한다 (두 critic 모두 Sigmoid 없는 실수 score).

Keras → PyTorch 대응
- LSTM은 모두 `batch_first=True`, 양방향 출력은 concat (Keras `merge_mode="concat"`) → 채널 2·hidden
- Keras `UpSampling1D(2)`(각 시점을 두 번 반복) = `repeat_interleave(2, dim=1)`
- Keras `TimeDistributed(Dense(1))` = 마지막 축에 대한 `nn.Linear` (시점마다 같은 weight)
- Keras `Conv1D`는 channels-last·padding "valid" → (N, C, T)로 transpose, padding 0. 길이 W → W − 16
- Keras `LSTM(dropout=0.2, recurrent_dropout=0.2)`(Generator): PyTorch LSTM의 `dropout`은 층 사이에만 걸리므로
  LSTM 입력 앞에 `nn.Dropout(0.2)`를 둔다 (Keras는 시점 간 같은 mask, 여기서는 시점마다 다른 mask).
  recurrent dropout은 cuDNN LSTM이 지원하지 않아 생략한다
- 초기화: `keras_init_`로 Keras 기본값을 재현한다 (PyTorch 기본 uniform(±1/√fan_in)과 다르다)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def keras_init_(module: nn.Module) -> None:
    """Keras 기본 초기화: Dense/Conv1D kernel glorot_uniform, bias 0,
    LSTM kernel glorot_uniform, recurrent kernel orthogonal, bias 0 + forget gate bias 1 (`unit_forget_bias=True`).

    PyTorch LSTM gate 순서는 (i, f, g, o)라 forget gate는 두 번째 hidden 크기 구간이다.
    PyTorch는 bias가 b_ih, b_hh 두 개이므로 forget 1은 b_ih에만 준다 (합이 Keras의 단일 bias와 같다).
    """
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            h = m.hidden_size
            for name, p in m.named_parameters():
                if name.startswith("weight_ih"):
                    nn.init.xavier_uniform_(p)  # (4h, in): fan_in = in, fan_out = 4h (Keras kernel (in, 4h)과 같다)
                elif name.startswith("weight_hh"):
                    nn.init.orthogonal_(p)  # (4h, h): Keras recurrent kernel (h, 4h)의 transpose
                elif name.startswith("bias"):
                    nn.init.zeros_(p)
                    if name.startswith("bias_ih"):
                        with torch.no_grad():
                            p[h : 2 * h].fill_(1.0)


class Encoder(nn.Module):
    """x (N, W, C) → z (N, latent_dim, 1). BiLSTM의 모든 시점 출력을 펼쳐 한 번에 Linear로 압축한다."""

    def __init__(self, window_size: int = 100, in_channels: int = 1, hidden: int = 100, latent_dim: int = 20) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(window_size * 2 * hidden, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.lstm(x)  # (N, W, 2·hidden)
        z = self.fc(h.flatten(1))  # (N, W·2·hidden) → (N, latent_dim)
        return z.unsqueeze(-1)  # (N, latent_dim, 1) Keras Reshape((20, 1))


class Generator(nn.Module):
    """z (N, latent_dim, 1) → x̂ (N, W, C) in [-1, 1]. 길이 W/2 시퀀스를 만든 뒤 2배로 늘려 W를 맞춘다."""

    def __init__(
        self,
        window_size: int = 100,
        latent_dim: int = 20,
        hidden: int = 64,
        out_channels: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if window_size % 2 != 0:
            raise ValueError(f"Generator는 W/2 → Upsample×2 구조라 window_size가 짝수여야 합니다: {window_size}")
        self.half = window_size // 2
        self.fc = nn.Linear(latent_dim, self.half)
        self.drop1 = nn.Dropout(dropout)
        self.lstm1 = nn.LSTM(1, hidden, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, out_channels)

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z.flatten(1))  # (N, latent_dim) → (N, W/2)
        h = h.unsqueeze(-1)  # (N, W/2, 1) Keras Reshape((50, 1))
        h, _ = self.lstm1(self.drop1(h))  # (N, W/2, 2·hidden)
        h = h.repeat_interleave(2, dim=1)  # (N, W, 2·hidden) UpSampling1D(2): [a, b] → [a, a, b, b]
        h, _ = self.lstm2(self.drop2(h))  # (N, W, 2·hidden)
        return torch.tanh(self.out(h))  # (N, W, C) TimeDistributed(Dense(C)) → tanh


class CriticX(nn.Module):
    """x (N, W, C) → Wasserstein score (N,). 큰 값일수록 real 쪽이다 (Sigmoid 없음)."""

    def __init__(
        self,
        window_size: int = 100,
        in_channels: int = 1,
        channels: int = 64,
        kernel_size: int = 5,
        n_layers: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        length = window_size - n_layers * (kernel_size - 1)  # padding "valid": conv마다 kernel_size − 1씩 줄어든다
        if length < 1:
            raise ValueError(f"window_size({window_size})가 너무 짧습니다: conv {n_layers}층 뒤 길이 {length}")
        layers: list[nn.Module] = []
        c = in_channels
        for _ in range(n_layers):
            layers += [nn.Conv1d(c, channels, kernel_size), nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            c = channels
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(channels * length, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.body(x.transpose(1, 2))  # (N, C, W) → (N, 64, W − 16)
        return self.fc(h.flatten(1)).squeeze(1)  # (N, 64·(W − 16)) → (N,)


class CriticZ(nn.Module):
    """z (N, latent_dim, 1) → Wasserstein score (N,). 큰 값일수록 prior N(0, I) 쪽이다."""

    def __init__(self, latent_dim: int = 20, hidden: int = 100, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # (N, latent_dim, 1) → (N, latent_dim)
            nn.Linear(latent_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z).squeeze(1)  # (N,)


class TadGAN(nn.Module):
    """E, G, C_x, C_z 묶음. checkpoint는 이 모듈 하나의 state_dict로 저장한다.

    학습 optimizer는 부분별로 따로 만든다: (E + G) 하나, C_x 하나, C_z 하나 (Orion과 같다).
    """

    def __init__(self, window_size: int = 100, latent_dim: int = 20, critic_z_hidden: int = 100) -> None:
        super().__init__()
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.encoder = Encoder(window_size, 1, 100, latent_dim)
        self.generator = Generator(window_size, latent_dim, 64, 1, 0.2)
        self.critic_x = CriticX(window_size, 1, 64, 5, 4, 0.25)
        self.critic_z = CriticZ(latent_dim, critic_z_hidden, 0.2)
        keras_init_(self)

    def sample_z(self, n: int, device: torch.device | str) -> Tensor:
        """prior z ~ N(0, I), shape (n, latent_dim, 1)."""
        return torch.randn(n, self.latent_dim, 1, device=device)

    def reconstruct(self, x: Tensor) -> Tensor:
        """x̂ = G(E(x)): (N, W, 1) → (N, W, 1). 이상 탐지 reconstruction error의 기준."""
        return self.generator(self.encoder(x))
