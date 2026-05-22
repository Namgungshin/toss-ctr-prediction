"""Local metrics for the DACON Toss CTR competition."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def weighted_log_loss(y_true, y_pred, eps: float = 1e-15) -> float:
    """Class-balanced binary log loss.

    Each class contributes 50% to the final loss, matching the competition
    description for WLL.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1.0 - eps)

    pos = y_true == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        raise ValueError("weighted_log_loss requires both positive and negative labels.")

    pos_loss = -np.log(y_pred[pos]).mean()
    neg_loss = -np.log1p(-y_pred[neg]).mean()
    return float(0.5 * pos_loss + 0.5 * neg_loss)


def competition_metrics(y_true, y_pred) -> dict[str, float]:
    """Return AP, WLL, and a simple 50/50 local proxy score.

    DACON exposes AP and WLL as 50/50 components. The exact leaderboard
    normalization is not published, so `local_proxy_score` is only for quick
    local comparison. Track AP and WLL directly for serious decisions.
    """
    ap = float(average_precision_score(y_true, y_pred))
    wll = weighted_log_loss(y_true, y_pred)
    return {
        "ap": ap,
        "wll": wll,
        "local_proxy_score": 0.5 * ap + 0.5 * (1.0 - wll),
    }
