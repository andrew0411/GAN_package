"""TadGAN 데이터: CSV 신호·이상 구간 로드, 합성 신호, Orion식 전처리, rolling window Dataset.

데이터 형식 (Orion과 같다)
- 신호 CSV: `timestamp,value` 두 열. 예: `$DATA_ROOT/tadgan/S-1.csv`
  timestamp는 숫자(unix 초 등)면 그대로, datetime 문자열이면 unix 초(int64)로 바꾼다.
  value의 결측(빈 칸)은 허용한다 (전처리에서 평균으로 채운다). timestamp 결측 행은 경고 후 버린다.
  `--interval` 없이 쓰면 등간격이어야 한다 (중복·불규칙 간격이면 경고. Orion은 집계로 등간격을 강제한다).
- 이상 구간 CSV (선택): `start,end` 두 열, 신호와 같은 timestamp 단위, 양 끝 포함. 예: `$DATA_ROOT/tadgan/S-1_anomalies.csv`
- 상대 경로가 현재 위치에 없으면 `DATA_ROOT`(없으면 `~/data`) 기준으로 찾는다.

전처리 순서 (Orion `tadgan` pipeline)
    (선택) time_segments_aggregate(interval, mean) → 평균 대치(SimpleImputer) → MinMax [-1, 1]
    → rolling window (window_size 100, step 1)
대치 평균·min·max는 학습 신호로 한 번 구해 checkpoint에 저장하고, detect에서 그대로 재사용한다
(sklearn pipeline의 fit/produce와 같은 개념. 테스트 신호가 학습 범위를 벗어나면 [-1, 1] 밖 값이 나온다).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import Tensor
from torch.utils.data import Dataset

from gan_common.data import get_data_root

DATASETS = ("synthetic", "csv")


# ---------------------------------------------------------------------------
# 파일 로드
# ---------------------------------------------------------------------------


def resolve_path(path: str | Path, data_root: str | None = None) -> Path:
    """상대 경로가 현재 위치에 없으면 `<data_root>/path`로 본다. 최종 경로에 파일이 없으면 에러."""
    p = Path(path).expanduser()
    if not p.is_absolute() and not p.exists():
        p = get_data_root(data_root) / p
    if not p.is_file():
        raise FileNotFoundError(f"CSV 파일이 없습니다: '{p}'. 경로 또는 DATA_ROOT 환경변수(--data_root)를 확인하십시오.")
    return p


def _to_numeric_time(col: pd.Series) -> np.ndarray:
    """숫자 timestamp는 그대로, datetime 문자열은 UTC 기준 unix 초로 바꾼다 (tz 없는 문자열은 UTC로 본다).

    결측·해석 불가 값은 NaN이 된다 (이때 배열은 float64). 결측이 없으면 datetime 변환 결과는 int64다.
    """
    if pd.api.types.is_numeric_dtype(col):
        return col.to_numpy()
    dt = pd.to_datetime(col, utc=True, errors="coerce")
    delta = dt - pd.Timestamp("1970-01-01", tz="UTC")
    if dt.isna().any():
        return (delta / pd.Timedelta(seconds=1)).to_numpy(dtype=np.float64)
    return (delta // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)


def _read_columns(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"'{path}'에 {missing} 열이 없습니다. 필요한 열: {list(columns)} (있는 열: {list(df.columns)})")
    return df


def load_signal(csv_path: str | Path, data_root: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`timestamp,value` CSV → (timestamps (L,), values (L,) float64). timestamp 오름차순으로 정렬한다."""
    path = resolve_path(csv_path, data_root)
    df = _read_columns(path, ("timestamp", "value"))
    timestamps = _to_numeric_time(df["timestamp"])
    values = pd.to_numeric(df["value"], errors="coerce").to_numpy(dtype=np.float64)  # 숫자가 아닌 칸은 NaN
    missing = np.isnan(timestamps.astype(np.float64))
    if missing.any():
        warnings.warn(f"'{path}': timestamp 결측·해석 불가 {int(missing.sum())}행을 버립니다.", stacklevel=2)
        timestamps, values = timestamps[~missing], values[~missing]
        if np.all(timestamps == np.round(timestamps)):
            timestamps = timestamps.astype(np.int64)  # 결측 때문에 float가 된 정수 timestamp를 되돌린다
    if len(timestamps) == 0:
        raise ValueError(f"'{path}'에 유효한 timestamp가 없습니다.")
    order = np.argsort(timestamps, kind="stable")
    return timestamps[order], values[order]


def load_anomalies(csv_path: str | Path, data_root: str | None = None) -> np.ndarray:
    """`start,end` CSV → (K, 2) 이상 구간 (양 끝 포함, 신호와 같은 timestamp 단위). start 순으로 정렬한다."""
    path = resolve_path(csv_path, data_root)
    df = _read_columns(path, ("start", "end"))
    intervals = np.stack([_to_numeric_time(df["start"]), _to_numeric_time(df["end"])], axis=1)  # (K, 2)
    if np.isnan(intervals.astype(np.float64)).any():
        raise ValueError(f"'{path}'에 결측이거나 해석할 수 없는 start/end가 있습니다.")
    if (intervals[:, 1] < intervals[:, 0]).any():
        raise ValueError(f"'{path}'에 end < start인 구간이 있습니다.")
    return intervals[np.argsort(intervals[:, 0], kind="stable")]


# ---------------------------------------------------------------------------
# 합성 신호 (다운로드 없이 폴더 전체를 돌려 보기 위한 것)
# ---------------------------------------------------------------------------


def make_synthetic(n: int = 5000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sine 두 개의 합 + Gaussian noise에 세 종류 이상을 주입한 신호.

        x(t) = sin(φ(t)) + 0.4·sin(2πt/173 + θ) + 0.05·ε,   φ(t) = 2π Σ f(s), 기본 주파수 f = 1/40

    주입하는 이상 (위치는 신호 길이 비율로 고정, 부호·위상·noise는 seed로 정한다):
    - point spike 3개 (0.20n, 0.35n, 0.90n): 한 점에 ±1.5
    - level shift (0.55n부터 0.03n 길이): +0.8
    - frequency change (0.78n부터 0.04n 길이): 기본 sine 주파수 3배. 위상을 누적합으로 만들어 경계가 끊기지 않는다
    반환: timestamps (n,) = 0…n-1, values (n,), anomalies (K, 2) 양 끝 포함 index 구간.
    """
    if n < 100:
        raise ValueError(f"합성 신호 길이 n은 100 이상이어야 합니다: {n}")
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    def segment(start_frac: float, len_frac: float) -> tuple[int, int]:
        start = int(start_frac * n)
        return start, start + max(1, int(len_frac * n)) - 1

    freq = np.full(n, 1.0 / 40.0)
    fc_start, fc_end = segment(0.78, 0.04)
    freq[fc_start : fc_end + 1] *= 3.0  # frequency change
    phase = 2.0 * np.pi * np.cumsum(freq) + rng.uniform(0.0, 2.0 * np.pi)
    values = (
        np.sin(phase)
        + 0.4 * np.sin(2.0 * np.pi * t / 173.0 + rng.uniform(0.0, 2.0 * np.pi))
        + 0.05 * rng.standard_normal(n)
    )

    ls_start, ls_end = segment(0.55, 0.03)
    values[ls_start : ls_end + 1] += 0.8  # level shift

    anomalies = [(ls_start, ls_end), (fc_start, fc_end)]
    for frac in (0.20, 0.35, 0.90):  # point spikes
        i = int(frac * n)
        values[i] += rng.choice([-1.5, 1.5])
        anomalies.append((i, i))

    anomalies_arr = np.asarray(sorted(anomalies), dtype=np.int64)  # (5, 2)
    return t.astype(np.int64), values, anomalies_arr


def load_series(
    dataset: str,
    *,
    data_path: str | None = None,
    labels_path: str | None = None,
    data_root: str | None = None,
    synthetic_n: int = 5000,
    synthetic_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """train.py·detect.py 공통 진입점. (timestamps, raw values, anomalies (K, 2) 또는 None)."""
    if dataset == "synthetic":
        return make_synthetic(synthetic_n, synthetic_seed)
    if dataset == "csv":
        if data_path is None:
            raise ValueError("--dataset csv에는 --data_path(`timestamp,value` CSV)가 필요합니다.")
        timestamps, values = load_signal(data_path, data_root)
        anomalies = load_anomalies(labels_path, data_root) if labels_path else None
        return timestamps, values, anomalies
    raise ValueError(f"dataset은 {' | '.join(DATASETS)} 중 하나: {dataset!r}")


# ---------------------------------------------------------------------------
# 전처리 (Orion tadgan pipeline)
# ---------------------------------------------------------------------------


def time_segments_aggregate(
    timestamps: np.ndarray, values: np.ndarray, interval: int | float
) -> tuple[np.ndarray, np.ndarray]:
    """Orion `time_segments_aggregate(method="mean")`: 첫 timestamp부터 [start, start + interval) 구간 평균.

    NaN은 평균에서 뺀다(skipna). 값이 하나도 없는 구간은 NaN으로 남아 다음 단계(평균 대치)에서 채워진다.
    반환 timestamp는 각 구간의 시작 시각이다. timestamps는 오름차순이어야 한다.
    """
    if interval <= 0:
        raise ValueError(f"interval은 양수여야 합니다: {interval}")
    bins = ((timestamps - timestamps[0]) // interval).astype(np.int64)  # (L,) 구간 번호
    n_bins = int(bins[-1]) + 1
    valid = ~np.isnan(values)
    sums = np.bincount(bins[valid], weights=values[valid], minlength=n_bins)
    counts = np.bincount(bins[valid], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return timestamps[0] + np.arange(n_bins) * interval, means


def check_time_grid(timestamps: np.ndarray) -> None:
    """`--interval` 없이 쓸 때 등간격 grid인지 확인하고, 아니면 경고한다.

    Orion은 time_segments_aggregate로 등간격 grid를 강제한다. 집계 없이 쓰면 window 100이 '100개 샘플'이지
    '같은 시간 길이'가 아니게 되므로, 중복 timestamp나 불규칙 간격이 있으면 `--interval`을 권한다.
    """
    if len(timestamps) < 2:
        return
    diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
    n_dup = int((diffs == 0).sum())
    if n_dup:
        warnings.warn(f"중복 timestamp {n_dup}개가 있습니다. --interval로 집계하는 것을 권장합니다.", stacklevel=3)
    positive = diffs[diffs > 0]
    if len(positive) and not np.allclose(positive, positive[0]):
        warnings.warn(
            f"timestamp 간격이 일정하지 않습니다 (최소 {positive.min():g}, 최대 {positive.max():g}). "
            "--interval로 등간격 집계하는 것을 권장합니다.",
            stacklevel=3,
        )


def fit_scaler_stats(values: np.ndarray) -> dict[str, float]:
    """평균 대치 값과 MinMax 범위를 이 신호로 구한다. checkpoint에 넣을 수 있게 python float로 돌려준다."""
    if np.isnan(values).all():
        raise ValueError("신호 값이 모두 결측입니다.")
    impute_mean = float(np.nanmean(values))
    filled = np.where(np.isnan(values), impute_mean, values)
    return {"impute_mean": impute_mean, "min": float(filled.min()), "max": float(filled.max())}


def preprocess(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    interval: int | float | None = None,
    stats: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """(선택) 시간 집계 → 평균 대치 → MinMax [-1, 1]. `stats`가 없으면 이 신호로 fit한다.

    반환: (timestamps (L,), scaled values (L,) float32, stats). 상수 신호(max == min)는 0으로 둔다.
    `interval`이 없으면 등간격 grid인지 검사해 아니면 경고한다 (`check_time_grid`).
    """
    values = np.asarray(values, dtype=np.float64)
    if interval:
        timestamps, values = time_segments_aggregate(timestamps, values, interval)
    else:
        check_time_grid(timestamps)
    if stats is None:
        stats = fit_scaler_stats(values)
    filled = np.where(np.isnan(values), stats["impute_mean"], values)  # SimpleImputer(strategy="mean")
    span = stats["max"] - stats["min"]
    if span > 0:
        scaled = 2.0 * (filled - stats["min"]) / span - 1.0  # MinMaxScaler(feature_range=(-1, 1))
    else:
        scaled = np.zeros_like(filled)
    return timestamps, scaled.astype(np.float32), stats


def rolling_windows(values: np.ndarray, window_size: int = 100, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """1-D 신호 (L,) → windows (N, window_size, 1) float32, 창 시작 index (N,).

    N = (L − window_size) // step + 1. windows는 복사 없는 read-only view다 (메모리 O(L)).
    Orion `rolling_window_sequences`는 target_size=1 예측 target 때문에 마지막 창을 버리지만,
    TadGAN은 target을 쓰지 않으므로 여기서는 마지막 창까지 포함한다 (step 1이면 모든 시점이 창에 덮인다).
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"values는 1-D여야 합니다: shape {values.shape}")
    if len(values) < window_size:
        raise ValueError(f"신호 길이({len(values)})가 window_size({window_size})보다 짧습니다.")
    windows = sliding_window_view(values, window_size)[::step]  # (N, W) view
    starts = np.arange(0, len(values) - window_size + 1, step)
    return windows[..., None], starts  # (N, W, 1), (N,)


class TimeSeriesWindows(Dataset):
    """전처리된 1-D 신호의 rolling window. `__getitem__`은 (window_size, 1) float32 Tensor.

    `starts[i]`는 i번째 창의 시작 index다 (plot·scoring에서 시점 위치를 찾을 때 쓴다).
    """

    def __init__(self, values: np.ndarray, window_size: int = 100, step: int = 1) -> None:
        self.windows, self.starts = rolling_windows(values, window_size, step)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> Tensor:
        return torch.from_numpy(np.array(self.windows[index], dtype=np.float32))  # (W, 1), view를 복사해 writable로
