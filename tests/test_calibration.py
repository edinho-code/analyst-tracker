"""
Unit tests for `calibration.py`.

Synthetic data: `raw_prob ~ Uniform(0, 1)`, `y ~ Bernoulli(true_p)` with
``true_p = clip(0.3 + 0.4 * raw_prob, 0, 1)`` (raw is positively
informative but miscalibrated). We require:

1. Platt fit on train beats the identity (raw=pred) baseline on held-out
   Brier score.
2. Isotonic fit on the same data produces monotone output and matches or
   improves on Platt's held-out Brier (within a small tolerance for
   sampling noise, since both are consistent with the true curve).
3. `save_params` / `load_params` is a round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibration import (
    _isotonic_pairs,
    _platt_predict,
    brier_score,
    fit_isotonic,
    fit_platt,
    isotonic_predict,
    load_params,
    log_loss,
    reliability_curve,
    save_params,
)


N_SAMPLES = 500
SEED = 42


def _make_synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    raw = rng.uniform(0.0, 1.0, size=N_SAMPLES)
    true_p = np.clip(0.3 + 0.4 * raw, 0.0, 1.0)
    y = (rng.uniform(0.0, 1.0, size=N_SAMPLES) < true_p).astype(int)
    # fabricate chronological dates so the 70/30 chronological split is
    # well-defined and stable.
    dates = pd.date_range("2020-01-01", periods=N_SAMPLES, freq="D").astype(str)
    return pd.DataFrame(
        {
            "position_id": np.arange(N_SAMPLES),
            "raw_prob": raw,
            "hit_direction": y,
            "open_date": dates,
            "ticker": ["SYN"] * N_SAMPLES,
            "analyst": ["synthetic"] * N_SAMPLES,
        }
    )


def _split_test(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Mirror the 70/30 chronological split used by fit_platt / fit_isotonic
    n_train = max(1, int(round(0.7 * len(df))))
    test = df.iloc[n_train:]
    return (
        test["raw_prob"].to_numpy(dtype=float),
        test["hit_direction"].to_numpy(dtype=float),
    )


def test_brier_and_log_loss_basic():
    # perfect predictions
    y = np.array([0, 1, 0, 1], dtype=float)
    assert brier_score(y, y) == pytest.approx(0.0, abs=1e-12)
    # log_loss for p=0.5 everywhere vs any y is log 2
    p = np.full(4, 0.5)
    assert log_loss(p, y) == pytest.approx(np.log(2.0), abs=1e-9)


def test_reliability_curve_shape():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200)
    y = (rng.uniform(0, 1, 200) < p).astype(int)
    df = reliability_curve(p, y, n_bins=10)
    assert len(df) == 10
    assert set(df.columns) == {"bin_lower", "bin_upper", "mean_pred", "obs_freq", "count"}
    assert int(df["count"].sum()) == 200


def test_platt_beats_identity_on_holdout():
    df = _make_synthetic()
    fit = fit_platt(df)

    raw_te, y_te = _split_test(df)
    # identity baseline: use raw_prob as the predicted probability
    identity_brier = brier_score(raw_te, y_te)
    platt_brier = fit["test_brier"]

    assert platt_brier < identity_brier, (
        f"expected fitted Platt to beat identity: "
        f"platt={platt_brier:.4f} identity={identity_brier:.4f}"
    )
    # Sanity: Platt fit should also beat predicting the global base rate.
    base_rate = float(y_te.mean())
    base_brier = brier_score(np.full_like(y_te, base_rate), y_te)
    assert platt_brier <= base_brier + 1e-6


def test_isotonic_monotone_and_competitive_with_platt():
    df = _make_synthetic()
    iso = fit_isotonic(df)
    platt = fit_platt(df)

    # monotone non-decreasing
    calibs = [c for (_u, c) in iso["pairs"]]
    assert all(calibs[i] <= calibs[i + 1] + 1e-12 for i in range(len(calibs) - 1))

    # held-out isotonic brier should be roughly as good as Platt.
    # Small positive tolerance for sampling noise — both methods are
    # consistent with the true mapping.
    assert iso["test_brier"] <= platt["test_brier"] + 0.01


def test_isotonic_predict_piecewise_constant():
    pairs = [(0.2, 0.1), (0.5, 0.3), (0.8, 0.6), (1.0, 0.9)]
    xs = np.array([0.0, 0.1, 0.2, 0.25, 0.5, 0.51, 0.8, 0.9, 1.0, 2.0])
    got = isotonic_predict(xs, pairs)
    expected = np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 0.9])
    np.testing.assert_allclose(got, expected)


def test_platt_predict_range():
    raw = np.linspace(0.0, 1.0, 11)
    p = _platt_predict(raw, A=-2.0, B=0.5)
    assert np.all((p >= 0.0) & (p <= 1.0))
    # A<0 ⇒ higher raw ⇒ higher calibrated probability
    assert np.all(np.diff(p) >= -1e-12)


def test_save_load_roundtrip_platt(tmp_path: Path):
    params = {
        "method": "platt",
        "A": -3.21,
        "B": 0.77,
        "n_train": 350,
        "n_test": 150,
        "train_brier": 0.21,
        "test_brier": 0.22,
        "fit_date": "2026-04-19",
    }
    out = save_params(params, tmp_path / "calibration_params.json")
    assert out.exists()

    loaded = load_params(out)
    assert loaded == params

    # raw JSON is parseable independently
    raw = json.loads(out.read_text())
    assert raw["method"] == "platt"


def test_save_load_roundtrip_isotonic(tmp_path: Path):
    params = {
        "method": "isotonic",
        "pairs": [[0.1, 0.05], [0.4, 0.3], [0.7, 0.55], [1.0, 0.9]],
        "n_train": 300,
        "n_test": 100,
    }
    out = save_params(params, tmp_path / "calibration_params.json")
    loaded = load_params(out)
    assert loaded == params


def test_load_params_missing_file(tmp_path: Path):
    assert load_params(tmp_path / "does_not_exist.json") is None


def test_isotonic_pairs_monotone_on_noisy_input():
    rng = np.random.default_rng(0)
    raw = np.sort(rng.uniform(0, 1, 200))
    # non-monotone noise; isotonic must still produce monotone fit
    y = (rng.uniform(0, 1, 200) < raw).astype(int)
    pairs = _isotonic_pairs(raw, y)
    calibs = [c for (_u, c) in pairs]
    assert all(calibs[i] <= calibs[i + 1] + 1e-12 for i in range(len(calibs) - 1))
