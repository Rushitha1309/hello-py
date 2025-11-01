from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


def generate_dataset(
    rng: np.random.Generator,
    n_train: int = 1200,
    n_val: int = 300,
    n_test: int = 300,
    minority_frac: float = 0.03,
    dim: int = 4,
):
    n1_train = int(n_train * minority_frac)
    n0_train = n_train - n1_train
    n1_val = int(n_val * minority_frac)
    n0_val = n_val - n1_val
    n1_test = int(n_test * minority_frac)
    n0_test = n_test - n1_test

    mu0 = np.zeros(dim)
    mu1 = np.ones(dim) * 1.2
    cov = np.eye(dim) * 1.0

    X0_train = rng.multivariate_normal(mu0, cov, size=n0_train)
    X1_train = rng.multivariate_normal(mu1, cov, size=n1_train)
    X0_val = rng.multivariate_normal(mu0, cov, size=n0_val)
    X1_val = rng.multivariate_normal(mu1, cov, size=n1_val)
    X0_test = rng.multivariate_normal(mu0, cov, size=n0_test)
    X1_test = rng.multivariate_normal(mu1, cov, size=n1_test)

    X_train = np.vstack([X0_train, X1_train])
    y_train = np.concatenate([np.zeros(n0_train, dtype=int), np.ones(n1_train, dtype=int)])
    X_val = np.vstack([X0_val, X1_val])
    y_val = np.concatenate([np.zeros(n0_val, dtype=int), np.ones(n1_val, dtype=int)])
    X_test = np.vstack([X0_test, X1_test])
    y_test = np.concatenate([np.zeros(n0_test, dtype=int), np.ones(n1_test, dtype=int)])

    for X, y in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
        idx = rng.permutation(len(y))
        X[...] = X[idx]
        y[...] = y[idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def download_creditcardfraud(use_cache: bool = True) -> str:
    """
    Download the Kaggle credit card fraud dataset and return the local folder path.

    Prefers kagglehub if available. If not, raises an informative error.
    """
    # Allow override via env var if user already downloaded
    pre = os.environ.get("CREDITCARD_DATA_DIR")
    if pre and os.path.exists(pre):
        return pre

    try:
        import kagglehub  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "kagglehub is not installed or not usable. Install it by adding 'kagglehub' to dependencies, "
            "or set CREDITCARD_DATA_DIR to a local folder containing creditcard.csv"
        ) from e

    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    return path


def _stratified_split_indices(rng: np.random.Generator, y: np.ndarray, train_frac: float, val_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.unique(y)
    train_idx = []
    val_idx = []
    test_idx = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        idx_c = rng.permutation(idx_c)
        n = len(idx_c)
        n_train = int(np.floor(train_frac * n))
        n_val = int(np.floor(val_frac * n))
        train_idx.append(idx_c[:n_train])
        val_idx.append(idx_c[n_train : n_train + n_val])
        test_idx.append(idx_c[n_train + n_val :])
    return np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx)


def load_creditcardfraud(
    rng: np.random.Generator,
    sample_n: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the Kaggle credit card fraud dataset (V1..V28, Amount as features; Class as label).
    Optionally subsample to sample_n rows for faster experiments.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("pandas is required to load the Kaggle dataset. Add 'pandas' to dependencies.") from e

    if cache_dir is None:
        cache_dir = download_creditcardfraud(use_cache=True)
    csv_path = os.path.join(cache_dir, "creditcard.csv")
    if not os.path.exists(csv_path):
        # Some KaggleHub versions nest files differently; search
        for root, _dirs, files in os.walk(cache_dir):
            if "creditcard.csv" in files:
                csv_path = os.path.join(root, "creditcard.csv")
                break
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find creditcard.csv under {cache_dir}")

    df = pd.read_csv(csv_path)
    # Features are V1..V28 plus Amount. Drop Time.
    feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount"]
    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df["Class"].to_numpy(dtype=int)

    # Optional subsample for speed
    n_total = len(y_all)
    if sample_n is not None and sample_n < n_total:
        idx = rng.choice(n_total, size=sample_n, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]

    # Simple standardization per feature
    mu = X_all.mean(axis=0, keepdims=True)
    sigma = X_all.std(axis=0, keepdims=True) + 1e-8
    X_all = (X_all - mu) / sigma

    # Stratified split: 70% train, 15% val, 15% test
    tr_idx, va_idx, te_idx = _stratified_split_indices(rng, y_all, train_frac=0.7, val_frac=0.15)
    X_train, y_train = X_all[tr_idx], y_all[tr_idx]
    X_val, y_val = X_all[va_idx], y_all[va_idx]
    X_test, y_test = X_all[te_idx], y_all[te_idx]
    return X_train, y_train, X_val, y_val, X_test, y_test


