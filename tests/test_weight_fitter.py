"""
Tests for weight_fitter.py + composite_score backward-compatibility.

Covers:
  1. IC + ridge fits on a synthetic panel recover the true dominant weight.
  2. Weights from both fits sum to 1.
  3. Held-out R² from ridge is positive on synthetic data.
  4. save_weights / load_weights round-trip preserves dicts.
  5. composite_score() with no composite_weights.json returns the same
     numeric value as a version that uses only the hard-coded
     0.40/0.25/0.25/0.10 weights.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the repo root importable when tests are run from any cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scoring_engine  # noqa: E402
import weight_fitter   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic panel fixture
# ─────────────────────────────────────────────────────────────────────────────

TRUE_COEFS = {
    "direction":   0.6,
    "target":      0.1,
    "alpha":       0.2,
    "consistency": 0.1,
}


def _make_synthetic_panel(
    n_analysts: int = 30,
    n_months: int = 12,
    seed: int = 7,
) -> pd.DataFrame:
    """
    30 analysts × 12 monthly snapshots. Components drawn on comparable
    scales; `forward_direction_score` = linear combo of TRUE_COEFS + noise.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-01-31", periods=n_months, freq="ME")
    rows = []
    # Per-analyst "skill" shifts so direction is the strongest driver
    # cross-sectionally as well as time-wise.
    analyst_skill = rng.normal(0.5, 0.15, size=n_analysts)
    for a_idx in range(n_analysts):
        for t in months:
            direction = float(np.clip(analyst_skill[a_idx] + rng.normal(0, 0.05), 0, 1))
            target    = float(np.clip(rng.normal(0.5, 0.20), 0, 1.5))
            alpha     = float(rng.normal(0.0, 0.20))
            consist   = float(np.clip(rng.normal(0.5, 0.15), 0, 1))
            fwd = (
                TRUE_COEFS["direction"]   * direction
                + TRUE_COEFS["target"]    * target
                + TRUE_COEFS["alpha"]     * alpha
                + TRUE_COEFS["consistency"] * consist
                + rng.normal(0, 0.05)
            )
            rows.append({
                "analyst_id":              a_idx,
                "as_of_date":              t.date().isoformat(),
                "avg_direction_score":     direction,
                "avg_target_score":        target,
                "avg_alpha":               alpha,
                "consistency":             consist,
                "n_t":                     int(rng.integers(5, 25)),
                "forward_direction_score": float(fwd),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def panel() -> pd.DataFrame:
    return _make_synthetic_panel()


# ─────────────────────────────────────────────────────────────────────────────
# IC / Ridge fits
# ─────────────────────────────────────────────────────────────────────────────

def _public_weights(w: dict) -> dict:
    return {k: w[k] for k in ("direction", "target", "alpha", "consistency")}


def test_ic_weights_direction_dominates(panel):
    w = weight_fitter.fit_ic_weights(panel)
    pub = _public_weights(w)
    assert pub["direction"] == max(pub.values()), pub
    assert abs(sum(pub.values()) - 1.0) < 1e-6


def test_ridge_weights_direction_dominates(panel):
    w = weight_fitter.fit_ridge_weights(panel, alpha=1.0)
    pub = _public_weights(w)
    assert pub["direction"] == max(pub.values()), pub
    assert abs(sum(pub.values()) - 1.0) < 1e-6


def test_ridge_holdout_r2_positive(panel):
    w = weight_fitter.fit_ridge_weights(panel, alpha=1.0)
    meta = w.get("_metadata", {})
    assert meta.get("fallback") is not True, meta
    assert meta["holdout_r2"] > 0.0, meta


# ─────────────────────────────────────────────────────────────────────────────
# save / load round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    w = {
        "direction":   0.55,
        "target":      0.15,
        "alpha":       0.20,
        "consistency": 0.10,
        "_metadata":   {"method": "ic", "fallback": False},
    }
    p = tmp_path / "composite_weights.json"
    weight_fitter.save_weights(w, path=str(p))
    loaded = weight_fitter.load_weights(path=str(p))
    assert loaded["direction"]   == pytest.approx(w["direction"])
    assert loaded["target"]      == pytest.approx(w["target"])
    assert loaded["alpha"]       == pytest.approx(w["alpha"])
    assert loaded["consistency"] == pytest.approx(w["consistency"])
    assert loaded["_metadata"]   == w["_metadata"]


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat: composite_score output with no composite_weights.json
# ─────────────────────────────────────────────────────────────────────────────

def _hardcoded_composite_score(row: dict, priors: dict | None = None) -> float:
    """Exact copy of the previous (pre-weight-fitter) composite_score body.

    Kept here as the ground truth for the backward-compat assertion. If
    this function drifts, update scoring_engine.DEFAULT_WEIGHTS in lockstep.
    """
    ds_raw  = row.get("avg_direction_score")
    ts_raw  = row.get("avg_target_score")
    alp_raw = row.get("avg_alpha")
    con_raw = row.get("consistency")
    n       = row.get("total_positions") or 0
    ds  = ds_raw if ds_raw is not None else 0.0
    ts  = (ts_raw / scoring_engine.MAX_TARGET_SCORE) if ts_raw is not None else 0.0
    alp = alp_raw if alp_raw is not None else 0.0
    con = con_raw if con_raw is not None else 0.5
    if priors is not None:
        ds  = scoring_engine._shrink(ds,  priors.get("avg_direction_score"),   n)
        ts  = scoring_engine._shrink(ts,  priors.get("avg_target_score_norm"), n)
        alp = scoring_engine._shrink(alp, priors.get("avg_alpha"),             n)
        con = scoring_engine._shrink(con, priors.get("consistency"),           n)
    ds100 = (ds  or 0.0) * 100
    ts100 = (ts  or 0.0) * 100
    con100 = (con or 0.0) * 100
    alp_norm = max(0.0, min(100.0, ((alp or 0.0) + 30) / 60 * 100))
    return round(ds100 * 0.40 + ts100 * 0.25 + alp_norm * 0.25 + con100 * 0.10, 2)


@pytest.fixture
def _no_weights_file(tmp_path, monkeypatch):
    """
    Run each test in a cwd that has no composite_weights.json, and reset
    the module-level cache so _load_weights_once() exercises the fallback.
    """
    monkeypatch.chdir(tmp_path)
    scoring_engine._WEIGHTS_CACHE = None
    scoring_engine._WEIGHTS_CACHE_LOGGED = False
    assert not (tmp_path / "composite_weights.json").exists()
    yield
    scoring_engine._WEIGHTS_CACHE = None
    scoring_engine._WEIGHTS_CACHE_LOGGED = False


@pytest.mark.parametrize("row", [
    {"avg_direction_score": 0.72, "avg_target_score": 1.05, "avg_alpha": 4.2,
     "consistency": 0.83, "total_positions": 27},
    {"avg_direction_score": 0.10, "avg_target_score": 0.20, "avg_alpha": -8.0,
     "consistency": 0.40, "total_positions": 3},
    {"avg_direction_score": None, "avg_target_score": None, "avg_alpha": None,
     "consistency": None, "total_positions": 0},
    {"avg_direction_score": 0.55, "avg_target_score": 0.80, "avg_alpha": 2.0,
     "consistency": 0.50, "total_positions": 15},
])
def test_composite_score_backward_compat_no_weights_file(_no_weights_file, row):
    priors = {
        "avg_direction_score":     0.50,
        "avg_target_score_norm":   0.55,
        "avg_alpha":                0.0,
        "consistency":              0.55,
    }
    got_plain    = scoring_engine.composite_score(row)
    got_priors   = scoring_engine.composite_score(row, priors=priors)
    want_plain   = _hardcoded_composite_score(row)
    want_priors  = _hardcoded_composite_score(row, priors=priors)
    assert got_plain  == pytest.approx(want_plain,  abs=1e-6)
    assert got_priors == pytest.approx(want_priors, abs=1e-6)


def test_composite_score_explicit_weights_overrides_file(tmp_path, monkeypatch):
    """When callers pass `weights=...` explicitly, they win over the file."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "composite_weights.json").write_text(json.dumps({
        "direction": 0.90, "target": 0.05, "alpha": 0.03, "consistency": 0.02,
    }))
    scoring_engine._WEIGHTS_CACHE = None
    scoring_engine._WEIGHTS_CACHE_LOGGED = False

    row = {"avg_direction_score": 0.6, "avg_target_score": 0.9,
           "avg_alpha": 1.0, "consistency": 0.5, "total_positions": 10}

    explicit = {"direction": 0.4, "target": 0.25, "alpha": 0.25, "consistency": 0.10}
    got = scoring_engine.composite_score(row, weights=explicit)
    want = _hardcoded_composite_score(row)
    assert got == pytest.approx(want, abs=1e-6)

    scoring_engine._WEIGHTS_CACHE = None
    scoring_engine._WEIGHTS_CACHE_LOGGED = False
