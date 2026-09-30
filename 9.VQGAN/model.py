"""VQGAN first stage autoencoder (Esser et al., "Taming Transformers for High-Resolution Image Synthesis", CVPR 2021).

이미지를 f배 작은 격자의 discrete code(codebook index)로 압축했다가 복원하는 VQ autoencoder다.
    x (N, 3, R, R) → Encoder → quant_conv(1x1) → VectorQuantizer → post_quant_conv(1x1) → Decoder → x̂ (N, 3, R, R)
구조는 taming-transformers의 `taming/modules/diffusionmodules/model.py`(Encoder/Decoder),
`taming/modules/vqvae/quantize.py`(VectorQuantizer), `taming/models/vqgan.py`(VQModel)를 따른다.
모듈 이름(conv_in, down.{i}.block.{j}, mid.attn_1, up.{i}.upsample, quantize.embedding ...)도 taming과 같게 두었다.
stage 2(code 격자 위의 autoregressive transformer prior)는 이 폴더의 범위 밖이다.

기본값(12GB GPU용 축소): image 128, ch 64, ch_mult (1, 2, 2, 4) → downsample factor f = 8,
128px 이미지 → 16x16 = 256개 code, codebook 1024개 × 64차원. taming 공개 설정(ImageNet·FFHQ 등)은
256px, ch 128, ch_mult (1, 1, 2, 2, 4)(f = 16), z_channels 256, embed_dim 256이다.

Decoder 출력에는 Tanh가 없다 (taming과 같음). target이 [-1, 1]이라 출력도 대략 그 범위로 학습되며,
이미지로 저장할 때 gan_common.viz.denorm이 clamp한다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def group_norm(channels: int) -> nn.GroupNorm:
    """taming `Normalize`: GroupNorm(32 groups, eps 1e-6). batch 통계를 쓰지 않아 작은 batch(8)에서도 안정적이다."""
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    """pre-activation residual block: y = shortcut(x) + conv2(dropout(swish(norm2(conv1(swish(norm1(x))))))).

    swish(x) = x·σ(x) = F.silu. 채널이 바뀌면 shortcut에 1x1 conv(nin_shortcut)를 둔다.
    taming의 timestep embedding(temb) 입력은 VQGAN에서 쓰지 않으므로(temb_channels=0) 뺐다.
    """

    def __init__(self, in_channels: int, out_channels: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))  # (N, C_out, H, W)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))  # (N, C_out, H, W)
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    """single-head spatial self-attention. H·W개 위치가 각각 token이 되어 서로를 본다. q/k/v/proj_out은 1x1 conv.

        y = x + proj_out(V · softmax(QᵀK / √C)ᵀ)
    conv는 receptive field 밖(예: 얼굴의 좌우 눈)의 관계를 잡기 어렵다. attention은 비용이 token 수의 제곱이라
    가장 낮은 해상도(기본 16x16 = 256 token)와 mid block에만 넣는다.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = group_norm(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        n, c, height, width = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(n, c, height * width).transpose(1, 2)  # (N, HW, C)
        k = self.k(h).reshape(n, c, height * width)  # (N, C, HW)
        v = self.v(h).reshape(n, c, height * width)  # (N, C, HW)
        attn = torch.softmax(torch.bmm(q, k) * c**-0.5, dim=-1)  # (N, HW_q, HW_k), 행마다 합 1
        out = torch.bmm(v, attn.transpose(1, 2))  # (N, C, HW_q): out[:, :, i] = Σ_j attn[i, j] · v[:, :, j]
        return x + self.proj_out(out.reshape(n, c, height, width))


class Downsample(nn.Module):
    """해상도 1/2: 오른쪽·아래에만 1픽셀 zero pad → 3x3 conv stride 2.

    TensorFlow "SAME" padding(stride 2, 짝수 입력이면 오른쪽·아래에만 pad)을 재현한 것으로,
    DDPM(TF) 구현에서 taming으로 이어졌다. 출력 크기는 padding=1 conv와 같은 H/2다.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0)  # (N, C, H + 1, W + 1)
        return self.conv(x)  # (N, C, H/2, W/2)


class Upsample(nn.Module):
    """해상도 ×2: nearest upsample → 3x3 conv. transposed conv의 checkerboard artifact를 피한다."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")  # (N, C, 2H, 2W)
        return self.conv(x)


class Encoder(nn.Module):
    """x (N, in_channels, R, R) → z_e (N, z_channels, R/f, R/f), f = 2^(len(ch_mult) - 1).

    conv_in → level i마다 [ResnetBlock × num_res_blocks (해상도가 attn_resolutions에 있으면 block마다 AttnBlock)
    → Downsample(마지막 level 제외)] → mid(res-attn-res) → norm_out → swish → conv_out.
    level i의 출력 채널 = ch · ch_mult[i].
    """

    def __init__(
        self,
        *,
        in_channels: int,
        ch: int,
        ch_mult: Sequence[int],
        num_res_blocks: int,
        attn_resolutions: Sequence[int],
        resolution: int,
        z_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1, *ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block, attn = nn.ModuleList(), nn.ModuleList()
            block_in, block_out = ch * in_ch_mult[i_level], ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out, dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()  # taming과 같은 key 구조(down.{i}.block / attn / downsample)를 위한 빈 컨테이너
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res //= 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout)
        self.norm_out = group_norm(block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)  # (N, ch, R, R)
        for i_level, down in enumerate(self.down):
            for i_block in range(self.num_res_blocks):
                h = down.block[i_block](h)  # (N, ch·ch_mult[i], r, r)
                if len(down.attn) > 0:
                    h = down.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = down.downsample(h)  # (N, ch·ch_mult[i], r/2, r/2)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))  # (N, ch·ch_mult[-1], R/f, R/f)
        return self.conv_out(F.silu(self.norm_out(h)))  # (N, z_channels, R/f, R/f)


class Decoder(nn.Module):
    """z (N, z_channels, R/f, R/f) → x̂ (N, out_channels, R, R). Encoder를 거꾸로 따라간다.

    conv_in → mid(res-attn-res) → level i(낮은 해상도부터)마다 [ResnetBlock × (num_res_blocks + 1) (+AttnBlock)
    → Upsample(level 0 제외)] → norm_out → swish → conv_out. 출력에 Tanh는 없다.
    block 수 num_res_blocks + 1은 DDPM U-Net up path(skip 개수에 맞춘 +1)에서 물려받은 값이다 (여기엔 skip이 없다).
    """

    def __init__(
        self,
        *,
        out_channels: int,
        ch: int,
        ch_mult: Sequence[int],
        num_res_blocks: int,
        attn_resolutions: Sequence[int],
        resolution: int,
        z_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[-1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout)

        ups: list[nn.Module] = []
        for i_level in reversed(range(self.num_resolutions)):
            block, attn = nn.ModuleList(), nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out, dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res *= 2
            ups.insert(0, up)  # up[i] = level i (taming과 같은 index 순서)
        self.up = nn.ModuleList(ups)
        self.norm_out = group_norm(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)  # (N, ch·ch_mult[-1], R/f, R/f)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        for i_level in reversed(range(self.num_resolutions)):
            up = self.up[i_level]
            for i_block in range(self.num_res_blocks + 1):
                h = up.block[i_block](h)  # (N, ch·ch_mult[i], r, r)
                if len(up.attn) > 0:
                    h = up.attn[i_block](h)
            if i_level != 0:
                h = up.upsample(h)  # (N, ch·ch_mult[i], 2r, 2r)
        return self.conv_out(F.silu(self.norm_out(h)))  # (N, out_channels, R, R)


class VectorQuantizer(nn.Module):
    """Vector quantization (van den Oord 2017, VQ-VAE). 위치마다 벡터 z_e를 L2로 가장 가까운 codebook 벡터로 바꾼다.

        k* = argmin_k ‖z_e − e_k‖²,   z_q = e_{k*}
        loss = ‖sg[z_e] − z_q‖² + β·‖z_e − sg[z_q]‖²   (sg = stop-gradient, 각 항은 원소 평균)
               codebook 항: code를 encoder 출력 쪽으로 당긴다 / commitment 항: encoder가 고른 code에서 멀어지지 않게 한다
    straight-through estimator: forward 값은 z_q, backward는 ∂L/∂z_q를 그대로 z_e로 넘긴다 (argmin은 미분 불가).

    taming 공개 VQGAN(VQModel)은 VectorQuantizer2를 기본값 `legacy=True`로 쓴다. 이 모드는 β를 codebook 항에 곱해
    실제로는 codebook 항 가중치 β, commitment 항 가중치 1.0이 된다 (원 VectorQuantizer의 버그를 호환용으로 남긴 것).
    여기서는 그 반대, 즉 VQ-VAE 원래 식대로 β를 commitment 항에 곱한다 (VectorQuantizer2 `legacy=False`와 같음).
    codebook 초기화는 taming과 같은 uniform(−1/n_embed, 1/n_embed).
    """

    def __init__(self, n_embed: int, embed_dim: int, beta: float = 0.25) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_embed, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / n_embed, 1.0 / n_embed)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """z (N, D, h, w) → (z_q (N, D, h, w), loss scalar, {"indices": (N, h, w) long, "perplexity": scalar})."""
        n, d, h, w = z.shape
        z = z.permute(0, 2, 3, 1)  # (N, h, w, D)
        z_flat = z.reshape(-1, d)  # (M, D), M = N·h·w
        with torch.no_grad():  # argmin은 미분하지 않으므로 거리 행렬의 그래프를 만들지 않는다
            e = self.embedding.weight  # (K, D)
            dist = z_flat.pow(2).sum(1, keepdim=True) + e.pow(2).sum(1) - 2.0 * z_flat @ e.t()  # (M, K) ‖z − e‖²
            indices = dist.argmin(dim=1)  # (M,)
        z_q = self.embedding(indices).view(n, h, w, d)  # (N, h, w, D)

        codebook_loss = F.mse_loss(z_q, z.detach())  # ‖sg[z_e] − e‖²: codebook만 갱신
        commitment_loss = F.mse_loss(z, z_q.detach())  # ‖z_e − sg[e]‖²: encoder만 갱신
        loss = codebook_loss + self.beta * commitment_loss

        z_q = z + (z_q - z).detach()  # straight-through: 값은 z_q, gradient는 z로
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # (N, D, h, w)

        # perplexity = exp(entropy of code 사용 분포). 1이면 한 code만, n_embed면 모든 code를 고르게 쓴다
        with torch.no_grad():
            probs = torch.bincount(indices, minlength=self.n_embed).float() / indices.numel()  # (K,)
            perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())
        return z_q, loss, {"indices": indices.view(n, h, w), "perplexity": perplexity}

    def embed_code(self, indices: Tensor) -> Tensor:
        """code index (N, h, w) → z_q (N, D, h, w). stage 2가 만든 index를 decode할 때 쓴다."""
        return self.embedding(indices).permute(0, 3, 1, 2).contiguous()


class VQModel(nn.Module):
    """Encoder → quant_conv(1x1) → VectorQuantizer → post_quant_conv(1x1) → Decoder.

    quant_conv / post_quant_conv는 encoder 출력 채널(z_channels)과 codebook 차원(embed_dim)을 잇는다.
    학습 가능한 파라미터 전체(`parameters()`)가 autoencoder optimizer 대상이다 (encoder, decoder, codebook, 두 1x1 conv).
    Encoder/Decoder는 PyTorch 기본 초기화를 쓴다 (taming과 같음).
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        resolution: int = 128,
        ch: int = 64,
        ch_mult: Sequence[int] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16,),
        dropout: float = 0.0,
        z_channels: int = 64,
        embed_dim: int = 64,
        n_embed: int = 1024,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        factor = 2 ** (len(ch_mult) - 1)
        if len(ch_mult) < 1 or resolution % factor != 0:
            raise ValueError(f"resolution({resolution})은 downsample factor 2^(len(ch_mult)-1) = {factor}의 배수여야 합니다.")
        bad = [ch * m for m in (1, *ch_mult) if (ch * m) % 32 != 0]  # 1: conv_in 출력(ch)도 GroupNorm을 거친다
        if bad:
            raise ValueError(f"GroupNorm(32): ch와 ch × ch_mult 채널 {bad}이 32의 배수가 아닙니다.")
        common = {
            "ch": ch,
            "ch_mult": ch_mult,
            "num_res_blocks": num_res_blocks,
            "attn_resolutions": attn_resolutions,
            "resolution": resolution,
            "z_channels": z_channels,
            "dropout": dropout,
        }
        self.encoder = Encoder(in_channels=in_channels, **common)
        self.decoder = Decoder(out_channels=in_channels, **common)
        self.quant_conv = nn.Conv2d(z_channels, embed_dim, kernel_size=1)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=beta)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, kernel_size=1)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """x (N, C, R, R) → (z_q (N, embed_dim, R/f, R/f), q_loss, info)."""
        h = self.quant_conv(self.encoder(x))  # (N, embed_dim, R/f, R/f) = z_e
        return self.quantize(h)

    def decode(self, z_q: Tensor) -> Tensor:
        """z_q (N, embed_dim, R/f, R/f) → x̂ (N, C, R, R)."""
        return self.decoder(self.post_quant_conv(z_q))  # post_quant_conv: (N, z_channels, R/f, R/f)

    def decode_code(self, indices: Tensor) -> Tensor:
        """code index (N, R/f, R/f) → x̂ (N, C, R, R)."""
        return self.decode(self.quantize.embed_code(indices))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """x (N, C, R, R) → (x̂ (N, C, R, R), q_loss scalar, info{"indices", "perplexity"})."""
        z_q, q_loss, info = self.encode(x)
        return self.decode(z_q), q_loss, info

    @property
    def last_layer(self) -> Tensor:
        """decoder 마지막 conv weight. adaptive weight λ의 gradient norm을 이 층에서 잰다."""
        return self.decoder.conv_out.weight


def build_vqmodel(cfg: Mapping[str, Any]) -> VQModel:
    """학습 config(`vars(args)` dict)로 VQModel을 만든다. reconstruct.py도 이 함수로 학습 때와 같은 구조를 만든다."""
    return VQModel(
        in_channels=cfg["channels"],
        resolution=cfg["image_size"],
        ch=cfg["ch"],
        ch_mult=cfg["ch_mult"],
        num_res_blocks=cfg["num_res_blocks"],
        attn_resolutions=cfg["attn_resolutions"],
        dropout=cfg["dropout"],
        z_channels=cfg["z_channels"],
        embed_dim=cfg["embed_dim"],
        n_embed=cfg["n_embed"],
        beta=cfg["beta"],
    )
