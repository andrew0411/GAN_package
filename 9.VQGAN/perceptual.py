"""VGG16 perceptual loss ("LPIPS-lite") — VQGAN 복원 loss의 perceptual 항.

    P(x, y) = Σ_l mean_{h,w} Σ_c ( φ̂_l(x) − φ̂_l(y) )²       → 샘플별 값 (N, 1, 1, 1)
    φ̂_l = VGG16 l번째 tap feature를 픽셀마다 채널 방향 단위 길이로 정규화한 것
tap: relu1_2, relu2_2, relu3_3, relu4_3, relu5_3 (LPIPS와 같은 층). 입력 [-1, 1]은 [0, 1]로 옮긴 뒤
ImageNet mean/std로 정규화한다 (LPIPS ScalingLayer의 shift/scale과 같은 변환).
픽셀 L1은 픽셀 위치가 조금만 어긋나도 크게 벌을 주고 흐린 평균 이미지를 선호하지만, VGG feature 거리는
질감·경계 같은 지각적 차이에 더 민감하다.

실제 LPIPS(Zhang et al. 2018, taming이 쓰는 것)와 다른 점:
    LPIPS는 층마다 제곱차에 사람의 지각 판단 데이터(BAPPS)로 학습한 1x1 linear 가중치(lin 층, 채널별 비음수)를
    곱한 뒤 공간 평균한다. 여기에는 그 학습된 lin 층이 없고 채널을 단순 합한다
    (LPIPS 공식 코드의 `LPIPS(net='vgg', lpips=False)` baseline과 같은 계산. 기본 net은 alex이므로 vgg를 명시해야 같다).
    층마다 값이 [0, 4] 범위(단위 벡터 차의 제곱 norm)이고 학습된 작은 가중치로 줄여지지 않으므로, 값이 실제 LPIPS의
    몇 배가 될 수 있다. taming 기본값 --perceptual_weight 1.0에서는 이 항이 L1 항을 압도할 수 있다.
    train.py의 `loss/perceptual_over_l1` 로그(rec 안에서 perceptual 항 / L1 항)를 보고 가중치를 조정한다.

VGG 가중치: torchvision `vgg16(weights=VGG16_Weights.IMAGENET1K_V1)`. 첫 생성 때 약 528MB를
    $TORCH_HOME/hub/checkpoints (기본 ~/.cache/torch/hub/checkpoints)로 내려받는다 (용량 대부분은 버리는 FC 분류기).
    train.py는 --perceptual_weight > 0일 때만 이 클래스를 만들므로 0이면 다운로드가 일어나지 않는다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import VGG16_Weights, vgg16

# vgg16().features 안에서 각 tap의 index: relu1_2, relu2_2, relu3_3, relu4_3, relu5_3 (채널 64, 128, 256, 512, 512)
_TAPS = (3, 8, 15, 22, 29)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _unit_normalize(x: Tensor, eps: float = 1e-10) -> Tensor:
    """픽셀마다 채널 벡터를 L2 norm 1로 나눈다 (LPIPS normalize_tensor). 층마다 feature 크기 차이를 없앤다."""
    return x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)


class VGGPerceptualLoss(nn.Module):
    """LPIPS-lite: 학습된 lin 층 없이 VGG16 feature 거리를 잰다. `forward(x, y)`는 샘플별 (N, 1, 1, 1).

    VGG는 freeze(requires_grad False)하고 eval로 둔다 (features에는 BatchNorm·Dropout이 없어 mode와 결과는 무관).
    x̂ 쪽 입력으로는 gradient가 흐르므로 autoencoder가 이 loss로 학습된다.
    """

    def __init__(self) -> None:
        super().__init__()
        features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        # tap마다 끊은 구간. 다음 구간은 MaxPool로 시작해 이전 tap 출력을 in-place ReLU가 덮어쓰지 않는다
        self.slices = nn.ModuleList()
        start = 0
        for tap in _TAPS:
            self.slices.append(features[start : tap + 1])
            start = tap + 1
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.requires_grad_(False)
        self.eval()

    def _features(self, x: Tensor) -> list[Tensor]:
        """x (N, 3, H, W) in [-1, 1] → tap feature 5개 [(N, 64, H, W), (N, 128, H/2, W/2), ..., (N, 512, H/16, W/16)]."""
        h = ((x + 1.0) / 2.0 - self.mean) / self.std  # ImageNet 정규화
        feats = []
        for s in self.slices:
            h = s(h)
            feats.append(h)
        return feats

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """x, y (N, 3, H, W) in [-1, 1] → (N, 1, 1, 1). 픽셀별 L1 (N, 3, H, W)에 그대로 broadcast해 더할 수 있다."""
        loss = torch.zeros(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype)
        for fx, fy in zip(self._features(x), self._features(y), strict=True):
            diff = (_unit_normalize(fx) - _unit_normalize(fy)).pow(2)  # (N, C_l, H_l, W_l)
            loss = loss + diff.sum(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)  # 채널 합 → 공간 평균
        return loss
