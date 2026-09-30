"""TadGAN 이상 탐지: 학습된 E·G·C_x로 시점별 이상 점수를 만들고 이상 구간을 찾는다.
기준: Orion `tadgan.score_anomalies` + `timeseries_anomalies.find_anomalies(fixed_threshold=True)`.

입력: train.py checkpoint (`model`, 전처리 통계 `data_stats`, `config`).
신호는 기본적으로 학습 때와 같은 것을 다시 읽는다 (Orion 관례: 학습 신호 자체에서 이상을 찾는다).
`--dataset/--data_path`(+ `--labels_path`)로 다른 신호(예: test split)를 줄 수 있고, 이때도 전처리는
학습 신호로 fit한 대치 평균·min·max를 그대로 쓴다. `--synthetic_seed`를 바꾸면 같은 구조의 새 합성 신호로 평가한다.

파이프라인 (L = 신호 길이, W = window 100, N = L − W + 1개 window, step 1)
 1. 복원: 창마다 x̂_i = G(E(x_i)) (N, W). 시점 t를 덮는 창 i의 값 x̂_i[t − i]들의 median → 복원 신호 x̂(t) (L,)
 2. 복원 오차 RE(t) (`--rec_error`, score_window s = 10)
      point : |x(t) − x̂(t)|
      area  : |A_x(t) − A_x̂(t)|, A(t) = t 중심 길이 s window의 사다리꼴 적분 (가장자리는 있는 점만)
      dtw   : DTW(x[t−h … t+h], x̂[t−h … t+h]), h = ⌊s/2⌋ (길이 11), 신호 밖은 0으로 padding.
              고전 DTW: 점 비용 (a_i − b_j)², 누적 비용의 √ (Orion이 쓰는 pyts `dtw` 기본값과 같은 정의)
    → 평활: centered rolling mean, window = ⌊N·smooth_portion⌋ (N의 1%), min_periods = window // 2
    → RE_score = max(z(RE), 0) + 1        (z = (v − mean) / std)
 3. critic score: 창마다 C_x(x_i) → 시점 t를 덮는 창들의 평균 → |v − μ_IQR| / σ + 1 → 2와 같은 평활
    (μ_IQR = 25–75 분위 안 값들의 평균, σ = 전체 std. 분위 평균은 이상 값에 끌려가지 않는 중심이다)
 4. 결합 (`--comb`)
      mult : S = RE_score × C_score
      sum  : S = α·(RE_score − 1) + (1 − α)·(C_score − 1),  α = `--alpha` 0.5
 5. 임계값 (fixed): 길이 ⌈L·0.33⌉ 창을 ⌈창 길이·0.1⌉ 간격으로 민다. 창마다
      thr = mean + 4·std 초과 점 → 앞뒤 `--anomaly_padding` 50점 확장 → 연속 구간 → 구간 max 점수 내림차순으로
      다음 값 대비 감소율이 `--min_percent` 0.1 이상인 마지막 구간까지만 남김(pruning) → 구간 점수 (max − thr)/(mean + std)
    창들에서 나온 구간 중 겹치거나 맞닿는 것을 병합한다 (점수는 구간 길이 가중 평균).
 6. 라벨이 있으면: overlapping segment P/R/F1 (예측 구간이 참 구간과 한 점이라도 겹치면 검출)
    + point-wise P/R/F1 (시점 단위, gan_common.metrics)
 7. 저장: anomalies.csv (start, end, score; 신호 timestamp 단위), scores.csv (시점별 중간 점수),
    detection.png (신호·복원·점수·검출/참 구간), summary.json (인자·지표)

부호 처리 (4의 곱이 의미를 갖도록)
- C_x는 WGAN 관례대로 real에 큰 값을 준다. 이상 window는 보통 낮은 값을 받지만, 학습 신호에 이상이 섞여 있고
  critic 절대 수준이 run마다 달라 방향을 가정하지 않는다 → |z| (양방향 모두 이상)
- 복원 오차는 클수록 이상이므로 z를 0에서 자른다 (평균 이하 = 중립)
- 둘 다 +1을 더해 1 이상으로 만든다. 곱은 "둘 다 높을 때 크게", 한쪽이 중립(1)이면 다른 쪽 값을 그대로 남긴다

Orion 대비 단순화·차이
 1. critic의 시점 집계: Orion은 시점 t를 덮는 창 critic 값들의 KDE 최빈값(scipy gaussian_kde)을 쓴다.
    여기서는 평균이다 (= 창 score에 길이 W box filter). scipy 의존을 없애고 계산을 단순화했다
 2. DTW 정의는 Orion과 같다 (pyts classic DTW: 제곱 비용 + √, 정확 DTW). 다른 것은 정렬이다:
    Orion은 결과를 h칸 밀어 붙이고 양 끝을 0으로 채운다. 여기서는 모든 시점 t에서 t 중심 window로 계산한다
 3. window: Orion rolling_window_sequences는 예측 target(target_size=1) 때문에 마지막 창을 버린다.
    여기서는 모든 창을 써서 복원·점수 길이가 신호 길이 L과 같다
 4. 구간 병합 가중치: Orion은 stop − start(단일 점 구간이면 0)를 쓴다. 여기서는 길이 stop − start + 1
 5. P/R/F1의 분모가 0이면 (예: 검출 구간 0개의 precision) Orion은 NaN, 여기서는 0.0
 6. 임계값은 fixed(mean + 4·std)만 구현한다. Orion의 z_range 탐색형 dynamic threshold, lower_threshold,
    `comb="rec"`는 구현하지 않았다
 7. area error 적분: Orion은 pandas rolling apply(scipy `trapz`)로 점마다 적분한다. 여기서는 같은 centered
    window 경계에서 누적합으로 한꺼번에 계산한다 (사다리꼴 = 합 − (첫 값 + 끝 값)/2, 같은 값)
 8. point-wise P/R/F1은 Orion 기본 평가(overlapping segment)에 더한 보조 지표다
 9. `--interval`로 집계했다면 시점 t는 [t, t + interval) 구간을 대표한다. 라벨 mask와 segment 겹침 판정은
    이 구간 폭을 반영하고 (정수 timestamp면 검출 구간 끝 = 마지막 구간 시작 + interval − 1, 닫힌 구간),
    anomalies.csv의 start·end는 Orion처럼 구간 시작 timestamp로 적는다

메모리: 창별 복원 x̂ (N, W) float32를 한 번에 들고 있으므로 O(N·W) (N = 100만, W = 100이면 약 400 MB).
시점별 집계·DTW는 chunk 단위로 계산해 추가 메모리를 제한한다. 더 긴 신호는 구간을 나눠 detect를 돌린다.

실행 (repo 루트에서 `pip install -e .` 후):
    cd 12.TadGAN
    python train.py --dataset synthetic
    python detect.py --checkpoint runs/TadGAN/<run>/checkpoints/last.pt
    python detect.py --checkpoint runs/TadGAN/<run>/checkpoints/last.pt --rec_error point --comb sum
    python detect.py --checkpoint runs/TadGAN/<run>/checkpoints/last.pt --synthetic_seed 1   # 새 합성 신호로 평가
    python detect.py --checkpoint <csv로 학습한 run>/checkpoints/last.pt \
        --data_path tadgan/S-1_test.csv --labels_path tadgan/S-1_anomalies.csv                # $DATA_ROOT 기준 상대 경로
    python detect.py --checkpoint runs/TadGAN/<run>/checkpoints/last.pt --output_dir runs/TadGAN/<run>/detect_area --rec_error area
산출물: 기본 <run_dir>/detect/<신호>_<rec_error>_<comb>/ 아래 anomalies.csv, scores.csv, detection.png, summary.json
    (--output_dir로 바꿀 수 있다)
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.lib.stride_tricks import sliding_window_view

from gan_common.checkpoint import load_checkpoint
from gan_common.config import nonneg_int, positive_int
from gan_common.metrics import precision_recall_f1
from gan_common.utils import get_device

from data import DATASETS, load_series, preprocess, rolling_windows  # 같은 폴더의 data.py
from model import TadGAN  # 같은 폴더의 model.py

FIXED_THRESHOLD_K = 4.0  # Orion `_fixed_threshold`: mean + 4·std
Interval = tuple[int, int, float]  # (start index, end index, score), 양 끝 포함


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TadGAN anomaly scoring and detection (Orion score_anomalies + find_anomalies)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="train.py checkpoint (runs/TadGAN/<run>/checkpoints/last.pt)")

    g = p.add_argument_group("data (주지 않으면 checkpoint config 값을 쓴다)")
    g.add_argument("--dataset", type=str, default=None, choices=DATASETS)
    g.add_argument("--data_path", type=str, default=None, help="`timestamp,value` CSV. 상대 경로는 없으면 DATA_ROOT 기준")
    g.add_argument(
        "--labels_path",
        type=str,
        default=None,
        help="이상 구간 CSV(`start,end`). --data_path를 새로 주면 checkpoint의 라벨 경로는 이어받지 않는다",
    )
    g.add_argument("--data_root", type=str, default=None, help="없으면 checkpoint config → DATA_ROOT env → ~/data")
    g.add_argument("--synthetic_n", type=positive_int, default=None, help="합성 신호 길이")
    g.add_argument("--synthetic_seed", type=int, default=None, help="합성 신호 seed (학습과 다르면 새 신호로 평가)")

    g = p.add_argument_group("scoring (Orion score_anomalies)")
    g.add_argument("--rec_error", type=str, default="dtw", choices=("dtw", "point", "area"), help="복원 오차 종류")
    g.add_argument("--score_window", type=positive_int, default=10, help="area 적분 / DTW 비교 window 길이 (2 이상)")
    g.add_argument(
        "--smooth_portion",
        type=float,
        default=0.01,
        help="평활 window = ⌊N·portion⌋ (N = window 수). 2 미만이면 평활하지 않는다",
    )
    g.add_argument("--comb", type=str, default="mult", choices=("mult", "sum"), help="복원·critic 점수 결합 방식")
    g.add_argument("--alpha", type=float, default=0.5, help="--comb sum에서 복원 점수 가중치 α")

    g = p.add_argument_group("thresholding (Orion find_anomalies, fixed threshold = mean + 4·std)")
    g.add_argument("--window_size_portion", type=float, default=0.33, help="임계값 창 길이 = ⌈L·portion⌉")
    g.add_argument("--window_step_size_portion", type=float, default=0.1, help="창 이동 간격 = ⌈창 길이·portion⌉")
    g.add_argument("--min_percent", type=float, default=0.1, help="pruning: 다음 구간 대비 최소 감소율")
    g.add_argument("--anomaly_padding", type=nonneg_int, default=50, help="임계 초과 점 앞뒤로 구간을 넓힐 점 수")

    g = p.add_argument_group("runtime")
    g.add_argument("--batch_size", type=positive_int, default=256, help="추론 batch 크기")
    g.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:N")
    g.add_argument("--output_dir", type=str, default=None, help="없으면 <run_dir>/detect/<신호>_<rec_error>_<comb>/")

    args = p.parse_args(argv)
    if args.score_window < 2:
        p.error(f"--score_window는 2 이상이어야 합니다 (Orion 탐색 범위 [2, 200]): {args.score_window}")
    if not 0.0 <= args.alpha <= 1.0:
        p.error(f"--alpha는 [0, 1]: {args.alpha}")
    for name in ("window_size_portion", "window_step_size_portion"):
        if not 0.0 < getattr(args, name) <= 1.0:
            p.error(f"--{name}는 (0, 1]: {getattr(args, name)}")
    if args.smooth_portion < 0 or args.min_percent < 0:
        p.error("--smooth_portion, --min_percent는 0 이상이어야 합니다.")
    return args


# ---------------------------------------------------------------------------
# 1. 복원: 창별 추론 → 겹친 창을 시점별로 모으기
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_windows(
    model: TadGAN, windows: np.ndarray, batch_size: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """창별 복원 x̂ = G(E(x)) (N, W)와 critic score C_x(x) (N,). eval 모드(dropout 끔)로 추론한다.

    x̂는 float32 그대로 둔다 (N·W 배열이라 float64 사본을 만들지 않는다). 시점별 집계 결과는 float64다.
    """
    model.eval()
    recs, critics = [], []
    for s in range(0, len(windows), batch_size):
        x = torch.from_numpy(np.array(windows[s : s + batch_size], dtype=np.float32)).to(device)  # (B, W, 1)
        recs.append(model.reconstruct(x).squeeze(-1).cpu().numpy())  # (B, W) float32
        critics.append(model.critic_x(x).cpu().numpy())  # (B,)
    return np.concatenate(recs), np.concatenate(critics).astype(np.float64)


def aggregate_overlapping(
    per_window: np.ndarray, reducer: Callable[..., np.ndarray], chunk: int = 16384
) -> np.ndarray:
    """(N, W) 창별 값 → (L,) 시점별 값, L = N + W − 1 (step 1 전제).

    시점 t는 창 i ∈ [max(0, t − W + 1), min(t, N − 1)]의 위치 t − i에 들어 있다.
    그 값들 per_window[i, t − i]를 `reducer(values, axis=1)`(np.nanmedian, np.nanmean)로 모은다.
    (T, W) 표를 chunk 단위로 만들어 긴 신호에서도 메모리를 제한한다. 창이 덮지 않는 칸은 NaN이다.
    """
    n, w = per_window.shape
    length = n + w - 1
    offsets = np.arange(w)[None, :]  # (1, W) 창 안 위치 j
    out = np.empty(length, dtype=np.float64)
    for t0 in range(0, length, chunk):
        t = np.arange(t0, min(t0 + chunk, length))[:, None]  # (T, 1)
        i = t - offsets  # (T, W): 시점 t를 위치 j에 담은 창 번호
        valid = (i >= 0) & (i < n)
        vals = np.where(valid, per_window[np.clip(i, 0, n - 1), offsets], np.nan)  # (T, W)
        out[t0 : t0 + len(t)] = reducer(vals, axis=1)
    return out


# ---------------------------------------------------------------------------
# 2. 복원 오차 (Orion timeseries_errors)
# ---------------------------------------------------------------------------


def point_error(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """|x(t) − x̂(t)|."""
    return np.abs(x - x_hat)


def local_area(v: np.ndarray, window: int) -> np.ndarray:
    """t 중심 길이 window 구간의 사다리꼴 적분 (간격 1). 가장자리는 신호 안에 있는 점만 쓴다.

    Orion `_area_error`의 `pd.Series(v).rolling(window, center=True, min_periods=window // 2).apply(trapz)`와
    같은 값을 누적합으로 한꺼번에 계산한다 (점마다 python 호출 없음):
    - 창 경계는 pandas centered fixed window 규칙: [t + 1 + off − window, t + 1 + off), off = (window − 1) // 2,
      [0, L]로 clip (window 10이면 [t − 5, t + 4])
    - 사다리꼴(간격 1) = 구간 합 − (첫 값 + 끝 값) / 2
    - 점 수가 min_periods(= window // 2)보다 적은 창은 pandas처럼 NaN → 가까운 유효값으로 채운다
    """
    n = len(v)
    offset = (window - 1) // 2
    end = np.clip(np.arange(n) + 1 + offset, 0, n)  # 끝 다음 index
    start = np.clip(np.arange(n) + 1 + offset - window, 0, n)
    csum = np.concatenate(([0.0], np.cumsum(v, dtype=np.float64)))
    area = csum[end] - csum[start] - 0.5 * (v[start] + v[end - 1])
    area = np.where(end - start >= max(1, window // 2), area, np.nan)
    return pd.Series(area).bfill().ffill().to_numpy()


def area_error(x: np.ndarray, x_hat: np.ndarray, score_window: int) -> np.ndarray:
    """두 신호의 국소 면적 차 |A_x(t) − A_x̂(t)| (Orion `_area_error`).

    점 하나의 값 차보다 짧은 구간의 누적 편차(완만한 level shift 등)에 민감하다.
    """
    return np.abs(local_area(x, score_window) - local_area(x_hat, score_window))


def dtw_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """행마다 고전 DTW 거리: a, b (T, m) → (T,). 대각·세로·가로 이동, 전역 정렬 (band 없음).

        D[i, j] = (a_i − b_j)² + min(D[i−1, j], D[i, j−1], D[i−1, j−1]),  D[0, 0] = 0, 나머지 경계 = ∞
        DTW(a, b) = √D[m, m]
    Orion이 부르는 pyts `dtw` 기본값(method="classic", 제곱 비용, 결과에 √)과 같은 정의다.
    DP 표 (m+1)×(m+1)을 T개 행에 대해 한꺼번에 채운다 (m = 11이면 121회 vectorized 연산).
    """
    t, m = a.shape
    cost = (a[:, :, None] - b[:, None, :]) ** 2  # (T, m, m)
    acc = np.full((t, m + 1, m + 1), np.inf)
    acc[:, 0, 0] = 0.0
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            best_prev = np.minimum(np.minimum(acc[:, i - 1, j], acc[:, i, j - 1]), acc[:, i - 1, j - 1])
            acc[:, i, j] = cost[:, i - 1, j - 1] + best_prev
    return np.sqrt(acc[:, m, m])


def dtw_error(x: np.ndarray, x_hat: np.ndarray, score_window: int, chunk: int = 16384) -> np.ndarray:
    """시점 t 중심 길이 2h+1 구간끼리의 DTW 거리, h = ⌊score_window / 2⌋.

    DTW는 시간축을 휘어 맞춘 뒤 거리를 재므로, 복원이 조금 앞서거나 늦는 위상차에는 관대하고
    모양 자체가 다른 구간(주파수 변화 등)에서 커진다. 신호 밖은 0으로 padding한다.
    Orion은 같은 zero padding을 쓰되 결과를 h칸 밀어 붙이고 양 끝을 0으로 채운다. 여기서는 t 중심으로 정렬한다.
    """
    length = (score_window // 2) * 2 + 1  # 홀수로 맞춘다: 10 → 11
    half = length // 2
    a = sliding_window_view(np.pad(x, half), length)  # (L, length): a[t] = x[t − h … t + h]
    b = sliding_window_view(np.pad(x_hat, half), length)
    return np.concatenate([dtw_distance(a[s : s + chunk], b[s : s + chunk]) for s in range(0, len(x), chunk)])


def reconstruction_error(x: np.ndarray, x_hat: np.ndarray, kind: str, score_window: int) -> np.ndarray:
    if kind == "point":
        return point_error(x, x_hat)
    if kind == "area":
        return area_error(x, x_hat, score_window)
    if kind == "dtw":
        return dtw_error(x, x_hat, score_window)
    raise ValueError(f"rec_error는 dtw | point | area 중 하나: {kind!r}")


# ---------------------------------------------------------------------------
# 2–4. 평활, 표준화, 결합
# ---------------------------------------------------------------------------


def rolling_smooth(v: np.ndarray, window: int) -> np.ndarray:
    """Orion 평활: `pd.Series.rolling(window, center=True, min_periods=window // 2).mean()`. window < 2면 그대로."""
    if window < 2:
        return v
    return pd.Series(v).rolling(window, center=True, min_periods=max(1, window // 2)).mean().bfill().ffill().to_numpy()


def zscore(v: np.ndarray) -> np.ndarray:
    """(v − mean) / std (모집단 std). 상수 배열은 0."""
    std = v.std()
    return (v - v.mean()) / std if std > 0 else np.zeros_like(v)


def reconstruction_score(rec_error: np.ndarray, smooth_window: int) -> np.ndarray:
    """Orion: 평활 → z-score → 0에서 자르고 +1. 평균보다 오차가 작은 구간은 중립값 1."""
    return np.clip(zscore(rolling_smooth(rec_error, smooth_window)), 0.0, None) + 1.0


def critic_score(critic_t: np.ndarray, smooth_window: int) -> np.ndarray:
    """Orion `_compute_critic_score`: |v − μ_IQR| / σ + 1 → 평활.

    μ_IQR은 25–75 분위 안 값들의 평균(이상 값에 덜 끌리는 중심), σ는 전체 std. 양쪽 방향 모두 이상으로 본다.
    """
    q25, q75 = np.quantile(critic_t, [0.25, 0.75])
    center = critic_t[(critic_t >= q25) & (critic_t <= q75)].mean()
    std = critic_t.std()
    z = np.abs(critic_t - center) / std + 1.0 if std > 0 else np.ones_like(critic_t)
    return rolling_smooth(z, smooth_window)


def combine_scores(rec: np.ndarray, critic: np.ndarray, comb: str, alpha: float) -> np.ndarray:
    """mult: rec × critic,  sum: α·(rec − 1) + (1 − α)·(critic − 1). 두 입력은 모두 1 이상이다."""
    if comb == "mult":
        return rec * critic
    if comb == "sum":
        return alpha * (rec - 1.0) + (1.0 - alpha) * (critic - 1.0)
    raise ValueError(f"comb는 mult | sum 중 하나: {comb!r}")


# ---------------------------------------------------------------------------
# 5. 임계값·구간 (Orion find_anomalies, fixed threshold)
# ---------------------------------------------------------------------------


def _pad_mask(mask: np.ndarray, padding: int) -> np.ndarray:
    """True 점마다 앞뒤 padding 점까지 True로 넓힌다 (Orion `_find_sequences`의 anomaly_padding)."""
    idx = np.flatnonzero(mask)
    if padding == 0 or len(idx) == 0:
        return mask.copy()
    diff = np.zeros(len(mask) + 1, dtype=np.int64)  # 구간 [lo, hi)를 +1/−1 차분으로 표시 후 누적합
    np.add.at(diff, np.maximum(idx - padding, 0), 1)
    np.add.at(diff, np.minimum(idx + padding + 1, len(mask)), -1)
    return np.cumsum(diff[:-1]) > 0


def _prune_anomalies(max_errors: np.ndarray, max_below: float, min_percent: float) -> np.ndarray:
    """Orion `_prune_anomalies`. 남길 구간 번호 (max 점수 내림차순).

    구간 max 점수를 내림차순 정렬하고 끝에 '구간 밖 최댓값'(max_below)을 붙인다. 이웃한 값 사이의 감소율
    (e_k − e_{k+1}) / e_k가 min_percent 이상인 마지막 k까지 남긴다. 즉 뒤쪽의 '고만고만한' 구간을 버린다.
    max_below는 임계값 이하이고 모든 구간 max는 임계값 초과이므로 정렬 뒤에도 항상 맨 끝이다.
    """
    order = np.argsort(-max_errors, kind="stable")
    sorted_errors = np.append(max_errors[order], max_below)
    with np.errstate(divide="ignore", invalid="ignore"):
        decrease = (sorted_errors[:-1] - sorted_errors[1:]) / sorted_errors[:-1]
    significant = np.flatnonzero(decrease >= min_percent)
    if len(significant) == 0:
        return np.array([], dtype=np.int64)
    return order[: significant[-1] + 1]


def _find_window_sequences(window: np.ndarray, offset: int, min_percent: float, padding: int) -> list[Interval]:
    """임계값 창 하나에서 이상 구간을 찾는다 (Orion `_find_window_sequences`, fixed_threshold=True)."""
    threshold = window.mean() + FIXED_THRESHOLD_K * window.std()
    above = _pad_mask(window > threshold, padding)
    if not above.any():
        return []
    edges = np.diff(np.concatenate(([0], above.astype(np.int64), [0])))  # +1: 구간 시작, −1: 구간 끝 다음
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    max_errors = np.array([window[s : e + 1].max() for s, e in zip(starts, ends)])
    max_below = float(window[~above].max()) if not above.all() else 0.0
    keep = _prune_anomalies(max_errors, max_below, min_percent)
    denominator = window.mean() + window.std()  # 구간 점수 정규화 (Orion `_compute_scores`)
    return [
        (int(starts[k]) + offset, int(ends[k]) + offset, float((max_errors[k] - threshold) / denominator)) for k in keep
    ]


def _merge_sequences(sequences: list[Interval]) -> list[Interval]:
    """겹치거나 맞닿은 구간을 합친다. 점수는 합쳐진 구간들의 길이 가중 평균 (Orion `_merge_sequences`)."""
    if not sequences:
        return []
    ordered = sorted(sequences, key=lambda s: s[0])
    merged = [ordered[0]]
    scores, weights = [ordered[0][2]], [ordered[0][1] - ordered[0][0] + 1]
    for start, end, score in ordered[1:]:
        prev_start, prev_end, _ = merged[-1]
        if start <= prev_end + 1:
            scores.append(score)
            weights.append(end - start + 1)
            merged[-1] = (prev_start, max(prev_end, end), float(np.average(scores, weights=weights)))
        else:
            merged.append((start, end, score))
            scores, weights = [score], [end - start + 1]
    return merged


def find_anomalies(
    scores: np.ndarray,
    *,
    window_size_portion: float = 0.33,
    window_step_size_portion: float = 0.1,
    min_percent: float = 0.1,
    anomaly_padding: int = 50,
) -> list[Interval]:
    """점수열 전체를 겹치는 창으로 나눠 창마다 국소 임계값을 적용한다 (점수 수준이 구간마다 달라도 대응).

    반환: [(start index, end index, score)], 양 끝 포함, start 순.
    """
    length = len(scores)
    window_size = int(math.ceil(length * window_size_portion))
    window_step = int(math.ceil(window_size * window_step_size_portion))
    sequences: list[Interval] = []
    window_start = window_end = 0
    while window_end < length:
        window_end = window_start + window_size
        window = scores[window_start:window_end]
        sequences.extend(_find_window_sequences(window, window_start, min_percent, anomaly_padding))
        window_start += window_step
    return _merge_sequences(sequences)


# ---------------------------------------------------------------------------
# 6. 평가
# ---------------------------------------------------------------------------


def _overlaps(a: tuple, b: tuple) -> bool:
    """닫힌 구간 [a0, a1], [b0, b1]이 한 점이라도 겹치는가."""
    return a[0] <= b[1] and b[0] <= a[1]


def overlapping_segment_prf(true: np.ndarray, detected: list[tuple]) -> dict[str, float]:
    """Orion overlapping segment (contextual, weighted=False) 방식.

    TP = 예측 구간과 하나라도 겹친 참 구간 수, FN = 어떤 예측과도 안 겹친 참 구간 수,
    FP = 어떤 참 구간과도 안 겹친 예측 구간 수. 구간 길이는 따지지 않는다.
    분모가 0이면 0.0이다 (Orion은 NaN).
    """
    tp = sum(any(_overlaps(t, d) for d in detected) for t in true)
    fn = len(true) - tp
    fp = sum(not any(_overlaps(d, t) for t in true) for d in detected)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def intervals_to_mask(timestamps: np.ndarray, intervals: np.ndarray, bin_width: float = 0) -> np.ndarray:
    """참 구간 [start, end]와 겹치는 시점을 True로. 집계했으면 시점 t는 [t, t + bin_width)를 대표한다."""
    mask = np.zeros(len(timestamps), dtype=bool)
    for start, end in intervals:
        if bin_width:
            mask |= (timestamps <= end) & (timestamps + bin_width > start)
        else:
            mask |= (timestamps >= start) & (timestamps <= end)
    return mask


# ---------------------------------------------------------------------------
# 7. 저장
# ---------------------------------------------------------------------------


def plot_detection(
    path: Path,
    timestamps: np.ndarray,
    x: np.ndarray,
    x_hat: np.ndarray,
    rec: np.ndarray,
    critic: np.ndarray,
    final: np.ndarray,
    detected_time: list[tuple],
    true: np.ndarray | None,
    labels: dict[str, str],
) -> None:
    """3단 plot: 신호·복원 / 복원·critic 점수 / 최종 점수. 초록 = 참 이상 구간, 빨강 = 검출 구간."""
    fig = Figure(figsize=(14, 9), layout="constrained")
    ax_sig, ax_parts, ax_final = fig.subplots(3, 1, sharex=True)
    ax_sig.plot(timestamps, x, lw=0.7, color="C0", label="signal (scaled)")
    ax_sig.plot(timestamps, x_hat, lw=0.7, color="C1", label="reconstruction (median over windows)")
    ax_parts.plot(timestamps, rec, lw=0.7, color="C2", label=labels["rec"])
    ax_parts.plot(timestamps, critic, lw=0.7, color="C3", label="critic score")
    ax_final.plot(timestamps, final, lw=0.8, color="black", label=labels["final"])

    step = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1.0
    half = 0.5 * step  # 한 점짜리 구간도 보이도록 반 칸씩 넓힌다
    patches = [Patch(color="tab:red", alpha=0.3, label="detected")]
    if true is not None:
        patches.insert(0, Patch(color="tab:green", alpha=0.3, label="true anomaly"))
    for ax in (ax_sig, ax_parts, ax_final):
        if true is not None:
            for start, end in true:
                ax.axvspan(start - half, end + half, color="tab:green", alpha=0.25, lw=0)
        for start, end, _ in detected_time:
            ax.axvspan(start - half, end + half, color="tab:red", alpha=0.2, lw=0)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + patches, loc="upper right", fontsize=8)
    ax_sig.set_title(labels["title"])
    ax_final.set_xlabel("timestamp")
    fig.savefig(path, dpi=110)


def _resolve_source(args: argparse.Namespace, cfg: dict) -> dict:
    """detect 인자가 없으면 학습 config 값을 쓴다. 신호를 새로 지정하면 학습 라벨 경로는 이어받지 않는다."""
    dataset = args.dataset or cfg["dataset"]
    same_source = args.data_path is None and dataset == cfg["dataset"]
    return {
        "dataset": dataset,
        "data_path": args.data_path or (cfg.get("data_path") if same_source else None),
        "labels_path": args.labels_path or (cfg.get("labels_path") if same_source else None),
        "data_root": args.data_root or cfg.get("data_root"),
        "synthetic_n": args.synthetic_n or cfg.get("synthetic_n", 5000),
        "synthetic_seed": args.synthetic_seed if args.synthetic_seed is not None else cfg.get("synthetic_seed", 0),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = get_device(args.device)
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    ckpt = load_checkpoint(ckpt_path)
    cfg, stats = ckpt["config"], ckpt["data_stats"]
    src = _resolve_source(args, cfg)

    # ---- 0. 데이터: 학습 때 fit한 통계(대치 평균·min·max)로 같은 전처리 → step 1 window
    timestamps, raw, true = load_series(
        src["dataset"],
        data_path=src["data_path"],
        labels_path=src["labels_path"],
        data_root=src["data_root"],
        synthetic_n=src["synthetic_n"],
        synthetic_seed=src["synthetic_seed"],
    )
    interval = cfg.get("interval")
    timestamps, values, _ = preprocess(timestamps, raw, interval=interval, stats=stats)
    x = values.astype(np.float64)  # (L,) scaled signal
    window_size = int(cfg["window_size"])
    windows, _ = rolling_windows(values, window_size, step=1)  # (N, W, 1)
    n_windows, length = len(windows), len(x)
    out_of_range = float(np.mean(np.abs(x) > 1.0))

    model = TadGAN(window_size, int(cfg["latent_dim"]), int(cfg.get("critic_z_hidden", 100)))
    model.load_state_dict(ckpt["model"])
    model.to(device)

    source = f"synthetic{src['synthetic_seed']}" if src["dataset"] == "synthetic" else Path(src["data_path"]).stem
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser()
    else:
        run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
        out_dir = run_dir / "detect" / f"{source}_{args.rec_error}_{args.comb}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device} | checkpoint={ckpt_path} (epoch {ckpt['epoch']})\n"
        f"signal {source}: L={length}, windows N={n_windows} (W={window_size}) | "
        f"학습 min/max 범위 밖 비율 {out_of_range:.2%} | labels "
        + ("없음" if true is None else f"{len(true)}개"),
        flush=True,
    )

    # ---- 1. 창별 복원 G(E(x))와 critic C_x(x) → 겹친 창을 시점별로 모은다 (복원은 median)
    rec_windows, critic_windows = predict_windows(model, windows, args.batch_size, device)  # (N, W), (N,)
    x_hat = aggregate_overlapping(rec_windows, np.nanmedian)  # (L,)

    # ---- 2. 복원 오차 → 평활 → z-score → max(z, 0) + 1
    smooth_window = int(n_windows * args.smooth_portion)  # Orion: ⌊N·0.01⌋
    rec_err = reconstruction_error(x, x_hat, args.rec_error, args.score_window)  # (L,)
    rec_sc = reconstruction_score(rec_err, smooth_window)

    # ---- 3. critic: 창 score를 덮는 창 평균으로 시점에 펼침 → |z| + 1 → 평활
    critic_t = aggregate_overlapping(np.broadcast_to(critic_windows[:, None], rec_windows.shape), np.nanmean)  # (L,)
    critic_sc = critic_score(critic_t, smooth_window)

    # ---- 4. 결합
    final = combine_scores(rec_sc, critic_sc, args.comb, args.alpha)  # (L,)

    # ---- 5. 슬라이딩 창 fixed threshold → 구간 → pruning → 병합
    detected = find_anomalies(
        final,
        window_size_portion=args.window_size_portion,
        window_step_size_portion=args.window_step_size_portion,
        min_percent=args.min_percent,
        anomaly_padding=args.anomaly_padding,
    )
    detected_time = [(timestamps[s], timestamps[e], score) for s, e, score in detected]  # 출력용 (구간 시작 시각)
    pred = np.zeros(length, dtype=bool)
    for s, e, _ in detected:
        pred[s : e + 1] = True

    print(f"detected {len(detected)} interval(s) (start, end, score):", flush=True)
    for start, end, score in detected_time:
        print(f"  {start}  {end}  {score:.4f}", flush=True)

    # ---- 6. 평가 (라벨이 있을 때)
    metrics = None
    label_mask = None
    if true is not None:
        bin_width = interval or 0
        # 집계했으면 마지막 구간 [t_e, t_e + interval)까지 덮는 닫힌 구간으로 겹침을 판정한다.
        # 정수 timestamp면 끝 = t_e + interval − 1, 실수 timestamp면 t_e + interval (경계 한 점 차이)
        end_pad = bin_width - 1 if bin_width and np.issubdtype(timestamps.dtype, np.integer) else bin_width
        detected_span = [(timestamps[s], timestamps[e] + end_pad) for s, e, _ in detected]
        segment = overlapping_segment_prf(true, detected_span)
        label_mask = intervals_to_mask(timestamps, true, bin_width)
        p, r, f1 = precision_recall_f1(label_mask, pred)
        metrics = {"segment": segment, "point": {"precision": p, "recall": r, "f1": f1}}
        print(
            f"overlapping segment: P {segment['precision']:.3f} R {segment['recall']:.3f} F1 {segment['f1']:.3f} "
            f"(TP {segment['tp']}, FP {segment['fp']}, FN {segment['fn']})\n"
            f"point-wise         : P {p:.3f} R {r:.3f} F1 {f1:.3f}",
            flush=True,
        )

    # ---- 7. 저장
    pd.DataFrame(detected_time, columns=["start", "end", "score"]).to_csv(out_dir / "anomalies.csv", index=False)
    table = {
        "timestamp": timestamps,
        "value": x,  # scaled
        "reconstruction": x_hat,
        "rec_error": rec_err,  # 평활 전
        "rec_score": rec_sc,
        "critic": critic_t,  # 겹친 창 평균, 표준화 전
        "critic_score": critic_sc,
        "score": final,
        "pred": pred.astype(np.int64),
    }
    if label_mask is not None:
        table["label"] = label_mask.astype(np.int64)
    pd.DataFrame(table).to_csv(out_dir / "scores.csv", index=False)

    plot_detection(
        out_dir / "detection.png",
        timestamps,
        x,
        x_hat,
        rec_sc,
        critic_sc,
        final,
        detected_time,
        true,
        {
            "title": f"TadGAN detection: {source} | rec_error={args.rec_error}, comb={args.comb}",
            "rec": f"reconstruction score ({args.rec_error})",
            "final": f"final score ({args.comb})",
        },
    )
    summary = {
        "checkpoint": str(ckpt_path),
        "source": src,
        "signal_length": length,
        "n_windows": n_windows,
        "smooth_window": smooth_window,
        "n_detected": len(detected),
        "metrics": metrics,
        "args": vars(args),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"saved: {out_dir} (anomalies.csv, scores.csv, detection.png, summary.json)", flush=True)


if __name__ == "__main__":
    main()
