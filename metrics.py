"""DACON 토스 CTR 대회용 로컬 평가 지표."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def weighted_log_loss(y_true, y_pred, eps: float = 1e-15) -> float:
    """클래스 균형 이진 log loss.

    대회 설명의 WLL처럼 양성/음성 클래스가 각각 50%씩 기여하도록 계산한다.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1.0 - eps)

    pos = y_true == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        raise ValueError("weighted_log_loss 계산에는 양성/음성 라벨이 모두 필요합니다.")

    pos_loss = -np.log(y_pred[pos]).mean()
    neg_loss = -np.log1p(-y_pred[neg]).mean()
    return float(0.5 * pos_loss + 0.5 * neg_loss)


def competition_metrics(y_true, y_pred) -> dict[str, float]:
    """AP, WLL, 단순 50/50 로컬 proxy score를 반환한다.

    DACON은 AP와 WLL을 50/50으로 반영한다고 공개했다. leaderboard의 정확한
    정규화 방식은 공개되지 않았으므로 `local_proxy_score`는 빠른 로컬 비교용으로만
    사용하고, 실제 의사결정은 AP와 WLL을 함께 본다.
    """
    ap = float(average_precision_score(y_true, y_pred))
    wll = weighted_log_loss(y_true, y_pred)
    return {
        "ap": ap,
        "wll": wll,
        "local_proxy_score": 0.5 * ap + 0.5 * (1.0 - wll),
    }
