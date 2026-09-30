"""이상탐지 평가 지표 (numpy만 사용, sklearn 의존 없음). label 1 = anomaly(양성)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_binary(labels: Any) -> np.ndarray:
    """0/1(또는 bool) label을 1-D bool 배열로 바꾼다. 다른 값이 섞이면 에러."""
    y = np.asarray(labels).ravel()
    if y.dtype != bool:
        if not np.isin(y, (0, 1)).all():
            raise ValueError("labels는 0/1(또는 bool)만 허용합니다.")
        y = y.astype(bool)
    return y


def _as_scores(scores: Any, n: int) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.shape[0] != n:
        raise ValueError(f"labels({n})와 scores({s.shape[0]}) 길이가 다릅니다.")
    if np.isnan(s).any():
        raise ValueError("scores에 NaN이 있습니다.")
    return s


def _average_ranks(s: np.ndarray) -> np.ndarray:
    """1부터 시작하는 rank. 동점은 그 구간 rank의 평균을 준다 (예: [5, 7, 7] → [1, 2.5, 2.5])."""
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    is_new = np.r_[True, sorted_s[1:] != sorted_s[:-1]]  # 동점 그룹의 시작 위치
    group = np.cumsum(is_new) - 1
    starts = np.flatnonzero(is_new)  # 그룹 첫 위치 (0-based)
    ends = np.r_[starts[1:], len(s)]  # 그룹 끝 다음 위치
    avg_rank = (starts + 1 + ends) / 2.0  # rank starts+1 … ends 의 평균
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = avg_rank[group]
    return ranks


def roc_auc(labels: Any, scores: Any) -> float:
    """ROC AUC = P(양성 score > 음성 score) (동점은 1/2).

    Mann–Whitney U 통계량으로 계산한다: AUC = (R_pos - n_pos(n_pos+1)/2) / (n_pos·n_neg),
    R_pos는 양성 샘플 rank(동점 평균) 합. score가 클수록 anomaly라는 전제다.
    """
    y = _as_binary(labels)
    s = _as_scores(scores, len(y))
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc: 양성(1)과 음성(0) 두 클래스가 모두 있어야 합니다.")
    ranks = _average_ranks(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: Any, scores: Any) -> float:
    """Average precision = Σ_k (R_k - R_{k-1})·P_k (sklearn과 같은 정의, 보간 없음).

    score 내림차순으로 서로 다른 score 값마다 threshold를 두고 precision P_k, recall R_k를 구한다.
    동점 score는 한 threshold로 함께 처리된다.
    """
    y = _as_binary(labels)
    s = _as_scores(scores, len(y))
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("average_precision: 양성(1) 샘플이 없습니다.")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    last_of_group = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s) - 1]  # 각 threshold의 마지막 index
    tp = np.cumsum(y_sorted)[last_of_group]
    predicted_pos = last_of_group + 1
    precision = tp / predicted_pos
    recall = tp / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def precision_recall_f1(labels: Any, preds: Any) -> tuple[float, float, float]:
    """이진 예측(1 = anomaly)의 precision, recall, F1. 분모가 0이면 0.0."""
    y = _as_binary(labels)
    p = _as_binary(preds)
    if len(p) != len(y):
        raise ValueError(f"labels({len(y)})와 preds({len(p)}) 길이가 다릅니다.")
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(precision), float(recall), float(f1)
