from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.data import generate_dataset, load_creditcardfraud


def _safe_div(n: float, d: float) -> float:
    return n / d if d != 0 else 0.0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def effective_class_weights(y: np.ndarray, beta: float) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    weights = (1.0 - beta) / (1.0 - np.power(beta, np.maximum(counts, 1.0)))
    return weights / np.mean(weights)


def forward_scores(w: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    return _sigmoid(X @ w + b)


def train_logreg(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    class_weights: np.ndarray | None,
    loss_type: str,
    focal_alpha: float,
    focal_gamma: float,
) -> tuple[np.ndarray, float]:
    d = X.shape[1]
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = _sigmoid(z)
        y_float = y.astype(float)
        if loss_type == "focal":
            p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
            alpha_t = focal_alpha * y_float + (1 - focal_alpha) * (1 - y_float)
            dL_dp = -alpha_t * (
                y_float * np.power(1 - p_clipped, focal_gamma) * (focal_gamma * (-1) * np.log(p_clipped) + 1)
                - (1 - y_float) * np.power(p_clipped, focal_gamma) * (focal_gamma * (-1) * np.log(1 - p_clipped) + 1)
            )
            dp_dz = p * (1 - p)
            grad = (dL_dp * dp_dz)
        else:
            p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
            if class_weights is None:
                pos_w, neg_w = 1.0, 1.0
            else:
                neg_w, pos_w = class_weights[0], class_weights[1]
            grad = (pos_w * (p_clipped - 1) * y_float + neg_w * (p_clipped) * (1 - y_float))
        grad_w = X.T @ grad / len(X)
        grad_b = float(np.mean(grad))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


@dataclass
class Session:
    seed: int
    rng: np.random.Generator
    max_steps: int
    steps_used: int
    threshold: float
    loss_type: str
    focal_alpha: float
    focal_gamma: float
    class_weights: np.ndarray | None
    tau: float
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    w: np.ndarray
    b: float
    trained: bool = False


_session: Session | None = None


def imbalance_start(
    seed: int = 42,
    max_steps: int = 5,
    tau: float = 0.6,
    minority_frac: float = 0.03,
    source: str = "synthetic",
    sample_n: int | None = None,
) -> dict[str, Any]:
    global _session
    rng = np.random.default_rng(seed)
    if source == "kaggle":
        X_train, y_train, X_val, y_val, X_test, y_test = load_creditcardfraud(rng=rng, sample_n=sample_n)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = generate_dataset(rng, minority_frac=minority_frac)
    w = np.zeros(X_train.shape[1])
    _session = Session(
        seed=seed,
        rng=rng,
        max_steps=max_steps,
        steps_used=0,
        threshold=0.5,
        loss_type="ce",
        focal_alpha=0.25,
        focal_gamma=2.0,
        class_weights=None,
        tau=tau,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        w=w,
        b=0.0,
    )
    scores = forward_scores(w, 0.0, X_val)
    m = metrics_from_scores(y_val, scores, _session.threshold)
    return {"train_counts": {"neg": int(np.sum(y_train == 0)), "pos": int(np.sum(y_train == 1))}, "val_baseline_f1": m["f1"], "tau": tau, "max_steps": max_steps}


def _consume_step() -> bool:
    assert _session is not None
    if _session.steps_used >= _session.max_steps:
        return False
    _session.steps_used += 1
    return True


def smote(ratio: float = 0.2, k: int = 5, seed: int | None = None) -> dict[str, Any]:
    assert _session is not None, "Call imbalance_start first"
    rng = _session.rng if seed is None else np.random.default_rng(int(seed))
    X, y = _session.X_train, _session.y_train
    minority_mask = (y == 1)
    X_min = X[minority_mask]
    n_min = len(X_min)
    n_maj = len(y) - n_min
    # Guard: need at least 2 minority points and some majority
    if n_min < 2 or n_maj <= 0:
        return {"train_counts": {"neg": int(np.sum(y == 0)), "pos": int(np.sum(y == 1))}, "noop": True, "reason": "insufficient_minority", "steps_used": _session.steps_used}
    # Validate and clamp ratio to avoid division by zero and explosion
    try:
        ratio = float(ratio)
    except Exception:
        return {"error": "invalid_ratio", "steps_used": _session.steps_used}
    if not (0.0 < ratio < 1.0):
        return {"error": "invalid_ratio", "ratio": ratio, "steps_used": _session.steps_used}
    ratio = min(ratio, 0.8)
    current_frac = n_min / (n_min + n_maj)
    if ratio <= current_frac:
        return {"train_counts": {"neg": int(np.sum(y == 0)), "pos": int(np.sum(y == 1))}, "noop": True, "reason": "already_at_or_above_ratio", "steps_used": _session.steps_used}
    # Solve for s so (n_min + s) / (n_min + n_maj + s) = ratio
    denom = max(1e-6, 1.0 - ratio)
    synth_needed = int(max(0.0, np.floor((ratio * (n_min + n_maj) - n_min) / denom)))
    # Hard cap to avoid runaway synthesis
    synth_needed = int(min(synth_needed, 10 * (n_min + n_maj)))
    if synth_needed <= 0:
        return {"train_counts": {"neg": int(np.sum(y == 0)), "pos": int(np.sum(y == 1))}, "noop": True, "reason": "no_synthesis_needed", "steps_used": _session.steps_used}
    # Now consume a step since we will actually generate points
    if not _consume_step():
        return {"error": "step_budget_exceeded"}
    from numpy.linalg import norm
    neighbor_k = min(k, max(1, n_min - 1))
    synth = []
    for _ in range(synth_needed):
        i = rng.integers(0, n_min)
        xi = X_min[i]
        dists = norm(X_min - xi, axis=1)
        nn_idx = np.argsort(dists)[1 : neighbor_k + 1]
        if len(nn_idx) == 0:
            nn = xi
        else:
            j = rng.choice(nn_idx)
            nn = X_min[j]
        lam = rng.random()
        x_new = xi + lam * (nn - xi)
        synth.append(x_new)
    if synth:
        X_synth = np.array(synth)
        y_synth = np.ones(len(X_synth), dtype=int)
        _session.X_train = np.vstack([_session.X_train, X_synth])
        _session.y_train = np.concatenate([_session.y_train, y_synth])
    return {"train_counts": {"neg": int(np.sum(_session.y_train == 0)), "pos": int(np.sum(_session.y_train == 1))}, "steps_used": _session.steps_used}


def set_class_weights(beta: float = 0.999) -> dict[str, Any]:
    assert _session is not None
    if not _consume_step():
        return {"error": "step_budget_exceeded"}
    _session.class_weights = effective_class_weights(_session.y_train, beta)
    _session.loss_type = "ce"
    return {"class_weights": _session.class_weights.tolist(), "steps_used": _session.steps_used}


def use_focal(alpha: float = 0.25, gamma: float = 2.0) -> dict[str, Any]:
    assert _session is not None
    if not _consume_step():
        return {"error": "step_budget_exceeded"}
    _session.loss_type = "focal"
    _session.focal_alpha = float(alpha)
    _session.focal_gamma = float(gamma)
    return {"loss_type": _session.loss_type, "alpha": alpha, "gamma": gamma, "steps_used": _session.steps_used}


def train(epochs: int = 50, lr: float = 0.1) -> dict[str, Any]:
    assert _session is not None
    if not _consume_step():
        return {"error": "step_budget_exceeded"}
    w, b = train_logreg(
        _session.X_train,
        _session.y_train,
        int(epochs),
        float(lr),
        _session.class_weights,
        _session.loss_type,
        _session.focal_alpha,
        _session.focal_gamma,
    )
    _session.w = w
    _session.b = b
    _session.trained = True
    scores = forward_scores(w, b, _session.X_val)
    m = metrics_from_scores(_session.y_val, scores, _session.threshold)
    return {"val_f1": m["f1"], "steps_used": _session.steps_used}


def set_threshold(threshold: float) -> dict[str, Any]:
    assert _session is not None
    if not _consume_step():
        return {"error": "step_budget_exceeded"}
    _session.threshold = float(threshold)
    scores = forward_scores(_session.w, _session.b, _session.X_val)
    m = metrics_from_scores(_session.y_val, scores, _session.threshold)
    return {"val_metrics": m, "threshold": _session.threshold, "steps_used": _session.steps_used}


def eval_on_val() -> dict[str, Any]:
    assert _session is not None
    scores = forward_scores(_session.w, _session.b, _session.X_val)
    m = metrics_from_scores(_session.y_val, scores, _session.threshold)
    return m


def submit_and_grade() -> dict[str, Any]:
    assert _session is not None
    if _session.steps_used > _session.max_steps:
        return {"pass": False, "test_f1": 0.0, "reason": "step_budget_exceeded"}
    if not _session.trained:
        return {"pass": False, "test_f1": 0.0, "reason": "not_trained"}
    scores = forward_scores(_session.w, _session.b, _session.X_test)
    m = metrics_from_scores(_session.y_test, scores, _session.threshold)
    return {"pass": m["f1"] >= _session.tau, "test_f1": float(m["f1"]) }


def sweep_thresholds(num_points: int = 101) -> dict[str, Any]:
    """Free helper to choose the best threshold on validation set maximizing F1."""
    assert _session is not None
    scores = forward_scores(_session.w, _session.b, _session.X_val)
    # Scan thresholds in [0,1]
    best_th = _session.threshold
    best = metrics_from_scores(_session.y_val, scores, best_th)
    best_f1 = best["f1"]
    for i in range(max(2, int(num_points))):
        th = i / max(1, num_points - 1)
        m = metrics_from_scores(_session.y_val, scores, th)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = th
            best = m
    _session.threshold = float(best_th)
    return {"best_threshold": float(best_th), "val_metrics": best, "steps_used": _session.steps_used}


