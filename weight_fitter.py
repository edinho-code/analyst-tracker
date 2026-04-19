"""
Analyst Tracker — Weight Fitter (composite_score weight learner)
================================================================

Fits the four composite_score weights (direction, target, alpha, consistency)
from data, instead of using the fixed 0.40 / 0.25 / 0.25 / 0.10 defaults.

Two estimators are provided:

  1. Forward-IC (Grinold–Kahn style):
        For each month-end t, compute the cross-sectional Spearman rank
        correlation of each component (evaluated as of t) with the
        forward realised direction_score over the next 6 months. Average
        the ICs across t, floor negatives at 0, and normalise positive
        ICs to sum to 1.

  2. Ridge regression:
        Pool all (analyst, t) observations into a single regression of
        forward_direction_score on the four components, weighted by
        sqrt(n_t) (an analyst with more positions at t carries more
        signal). Use ridge (L2) with intercept un-penalised. Normalise
        the four coefficients to sum to 1 after clipping negatives to 0.
        Report the out-of-sample R² on a chronological 30% holdout.

Both estimators output a dict of the form::

    {"direction": 0.52, "target": 0.20, "alpha": 0.20, "consistency": 0.08,
     "_metadata": {...}}

The `_metadata` key is carried through save/load but ignored by callers
that only consume the four numeric weights.

CLI::

    python weight_fitter.py --fit                  # fits IC + ridge,
                                                   # prints both, writes
                                                   # composite_weights.json
                                                   # from the IC fit
    python weight_fitter.py --fit --method=ridge   # same, writes ridge
    python weight_fitter.py --report               # compare deployed vs fit

Dependencies: pandas + numpy only. No scipy, no sklearn.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_DB_PATH      = "analyst_tracker.db"
DEFAULT_WEIGHTS_PATH = "composite_weights.json"
DEFAULT_WEIGHTS      = {"direction": 0.40, "target": 0.25, "alpha": 0.25, "consistency": 0.10}
COMPONENT_COLS       = ["avg_direction_score", "avg_target_score", "avg_alpha", "consistency"]
COMPONENT_TO_KEY     = {
    "avg_direction_score": "direction",
    "avg_target_score":    "target",
    "avg_alpha":            "alpha",
    "consistency":          "consistency",
}
FORWARD_COL          = "forward_direction_score"
FORWARD_HORIZON_MONTHS = 6


# ─────────────────────────────────────────────────────────────────────────────
# Panel construction
# ─────────────────────────────────────────────────────────────────────────────

def _compute_consistency(scores: list[float]) -> float:
    """Mirror of scoring_engine.compute_consistency (kept local to avoid an
    import cycle and so the panel builder has no dependency on the CLI)."""
    if len(scores) < 5:
        return 0.5
    window = 10
    windows = [
        scores[i:i + window]
        for i in range(0, len(scores), window)
        if len(scores[i:i + window]) >= 3
    ]
    if len(windows) < 2:
        return 0.5
    rates    = [sum(w) / len(w) for w in windows]
    mean     = sum(rates) / len(rates)
    variance = sum((r - mean) ** 2 for r in rates) / len(rates)
    std      = math.sqrt(variance)
    return round(max(0.0, min(1.0, 1 - (std / 0.5))), 4)


def build_panel(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Build a long-format panel (analyst_id, as_of_date) with trailing
    composite components at t and forward direction_score over (t, t+6m].

    Returns a DataFrame with columns::

        analyst_id, as_of_date,
        avg_direction_score, avg_target_score, avg_alpha, consistency,
        n_t, forward_direction_score

    `as_of_date` is a month-end ISO date. `forward_direction_score` is
    None when the analyst opened no positions in the forward window.
    """
    query = """
        SELECT pos.analyst_id,
               pos.open_date,
               perf.eval_date,
               perf.direction_score,
               perf.target_score,
               perf.alpha_vs_spy,
               perf.alpha_vs_ibov
          FROM performance perf
          JOIN positions pos ON pos.id = perf.position_id
         WHERE perf.direction_score IS NOT NULL
           AND perf.eval_date IS NOT NULL
           AND pos.open_date IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    cols = ["analyst_id", "as_of_date"] + COMPONENT_COLS + ["n_t", FORWARD_COL]
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["open_date"] = pd.to_datetime(df["open_date"], errors="coerce")
    df["eval_date"] = pd.to_datetime(df["eval_date"], errors="coerce")
    df = df.dropna(subset=["open_date", "eval_date"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    # Composite alpha (matches scoring_engine.compute_analyst_score: flat
    # average of non-null alpha_vs_spy and alpha_vs_ibov).
    df["alpha"] = df[["alpha_vs_spy", "alpha_vs_ibov"]].mean(axis=1, skipna=True)

    start = df["eval_date"].min()
    end   = df["open_date"].max()
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame(columns=cols)
    as_of_dates = pd.date_range(start=start, end=end, freq="ME")
    analysts = sorted(df["analyst_id"].unique())

    rows: list[dict[str, Any]] = []
    for t in as_of_dates:
        t_fwd = t + pd.DateOffset(months=FORWARD_HORIZON_MONTHS)
        past = df[df["eval_date"] <= t]
        fwd  = df[(df["open_date"] > t) & (df["open_date"] <= t_fwd)]
        for a in analysts:
            past_a = past[past["analyst_id"] == a]
            if past_a.empty:
                continue
            fwd_a = fwd[fwd["analyst_id"] == a]
            fwd_ds = fwd_a["direction_score"].mean() if not fwd_a.empty else None
            rows.append({
                "analyst_id":           int(a),
                "as_of_date":           t.date().isoformat(),
                "avg_direction_score":  _nan_to_none(past_a["direction_score"].mean()),
                "avg_target_score":     _nan_to_none(past_a["target_score"].mean()),
                "avg_alpha":            _nan_to_none(past_a["alpha"].mean()),
                "consistency":          _compute_consistency(
                                            past_a["direction_score"].dropna().tolist()
                                        ),
                "n_t":                  int(len(past_a)),
                FORWARD_COL:            _nan_to_none(fwd_ds),
            })
    return pd.DataFrame(rows, columns=cols)


def _nan_to_none(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return float(v)


# ─────────────────────────────────────────────────────────────────────────────
# Spearman IC (rank correlation) — pandas only, no scipy
# ─────────────────────────────────────────────────────────────────────────────

def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    """Cross-sectional Spearman rank correlation. Returns None when the
    pair has <2 joint observations or either side is constant."""
    joint = pd.concat([x, y], axis=1).dropna()
    if len(joint) < 2:
        return None
    a = joint.iloc[:, 0].rank(method="average")
    b = joint.iloc[:, 1].rank(method="average")
    if a.nunique() < 2 or b.nunique() < 2:
        return None
    corr = a.corr(b)
    if corr is None or (isinstance(corr, float) and math.isnan(corr)):
        return None
    return float(corr)


def fit_ic_weights(panel: pd.DataFrame) -> dict:
    """
    Forward-IC weighting.

    For each month-end `as_of_date`, compute the cross-sectional Spearman
    rank correlation of each component vs `forward_direction_score`.
    Average ICs across t, floor negatives at 0, normalise to sum to 1.

    When every component has non-positive mean IC (or the panel is empty),
    falls back to DEFAULT_WEIGHTS and marks `_metadata.fallback=True`.
    """
    if panel is None or panel.empty:
        return _weights_fallback(reason="empty panel", method="ic")

    sub = panel.dropna(subset=[FORWARD_COL]).copy()
    if sub.empty:
        return _weights_fallback(reason="no forward observations", method="ic")

    per_t_ic: dict[str, list[float]] = {c: [] for c in COMPONENT_COLS}
    n_periods_used = 0
    for _, group in sub.groupby("as_of_date"):
        if len(group) < 2:
            continue
        had_any = False
        for c in COMPONENT_COLS:
            ic = _spearman(group[c], group[FORWARD_COL])
            if ic is not None:
                per_t_ic[c].append(ic)
                had_any = True
        if had_any:
            n_periods_used += 1

    mean_ic = {
        c: (sum(vs) / len(vs) if vs else 0.0) for c, vs in per_t_ic.items()
    }
    floored = {c: max(0.0, v) for c, v in mean_ic.items()}
    total = sum(floored.values())
    if total <= 0:
        fallback = _weights_fallback(
            reason="all mean ICs non-positive", method="ic"
        )
        fallback["_metadata"]["mean_ic"] = mean_ic
        fallback["_metadata"]["n_periods_used"] = n_periods_used
        return fallback

    weights = {
        COMPONENT_TO_KEY[c]: floored[c] / total for c in COMPONENT_COLS
    }
    weights["_metadata"] = {
        "method":           "ic",
        "mean_ic":          mean_ic,
        "n_periods_used":   n_periods_used,
        "n_obs":            int(len(sub)),
        "fallback":         False,
    }
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Ridge regression — numpy only
# ─────────────────────────────────────────────────────────────────────────────

def fit_ridge_weights(panel: pd.DataFrame, alpha: float = 1.0) -> dict:
    """
    Ridge regression of forward_direction_score on the four components,
    weighted by sqrt(n_t) (per-sample, not per-feature).

    Solves::

        (Xᵀ W X + alpha · P) β = Xᵀ W y

    where W = diag(sqrt(n_t)) and P is the identity but with 0 on the
    intercept position so the intercept is un-penalised. Reports R² on a
    chronological 30% held-out slice and returns normalised (clipped-at-0,
    sum-to-1) weights over the four components.
    """
    if panel is None or panel.empty:
        return _weights_fallback(reason="empty panel", method="ridge")

    sub = panel.dropna(subset=COMPONENT_COLS + [FORWARD_COL, "n_t", "as_of_date"]).copy()
    if sub.empty:
        return _weights_fallback(reason="no complete rows", method="ridge")
    sub = sub.sort_values("as_of_date", kind="mergesort").reset_index(drop=True)
    if len(sub) < 5:
        return _weights_fallback(reason="fewer than 5 complete rows", method="ridge")

    split = max(1, int(round(len(sub) * 0.7)))
    split = min(split, len(sub) - 1)
    train = sub.iloc[:split]
    test  = sub.iloc[split:]

    X_tr = np.column_stack([
        np.ones(len(train)),
        train[COMPONENT_COLS].to_numpy(dtype=float),
    ])
    y_tr = train[FORWARD_COL].to_numpy(dtype=float)
    w_tr = np.sqrt(np.clip(train["n_t"].to_numpy(dtype=float), 1e-9, None))
    W_tr = np.diag(w_tr)

    p = X_tr.shape[1]
    P = np.eye(p)
    P[0, 0] = 0.0  # do not penalise intercept
    A = X_tr.T @ W_tr @ X_tr + alpha * P
    b = X_tr.T @ W_tr @ y_tr
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]

    # Holdout R²
    X_te = np.column_stack([
        np.ones(len(test)),
        test[COMPONENT_COLS].to_numpy(dtype=float),
    ])
    y_te = test[FORWARD_COL].to_numpy(dtype=float)
    y_hat = X_te @ beta
    ss_res = float(((y_te - y_hat) ** 2).sum())
    ss_tot = float(((y_te - y_te.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    coefs = beta[1:].copy()
    coefs_clipped = np.clip(coefs, 0.0, None)
    total = float(coefs_clipped.sum())
    if total <= 0:
        fallback = _weights_fallback(reason="all ridge coefs non-positive", method="ridge")
        fallback["_metadata"].update({
            "raw_coefs":   _coef_dict(beta),
            "holdout_r2":  round(r2, 4),
            "alpha":       alpha,
            "n_train":     int(len(train)),
            "n_test":      int(len(test)),
        })
        return fallback

    normalised = coefs_clipped / total
    weights = {
        COMPONENT_TO_KEY[c]: float(normalised[i]) for i, c in enumerate(COMPONENT_COLS)
    }
    weights["_metadata"] = {
        "method":     "ridge",
        "alpha":      alpha,
        "holdout_r2": round(r2, 4),
        "n_train":    int(len(train)),
        "n_test":     int(len(test)),
        "raw_coefs":  _coef_dict(beta),
        "fallback":   False,
    }
    return weights


def _coef_dict(beta: np.ndarray) -> dict:
    return {
        "intercept": float(beta[0]),
        "direction":    float(beta[1]),
        "target":       float(beta[2]),
        "alpha":        float(beta[3]),
        "consistency":  float(beta[4]),
    }


def _weights_fallback(*, reason: str, method: str) -> dict:
    w = dict(DEFAULT_WEIGHTS)
    w["_metadata"] = {
        "method":   method,
        "fallback": True,
        "reason":   reason,
    }
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_weights(weights: dict, path: str = DEFAULT_WEIGHTS_PATH) -> None:
    """Write `weights` to `path` as pretty-printed JSON."""
    Path(path).write_text(json.dumps(weights, indent=2, sort_keys=True) + "\n")


def load_weights(path: str = DEFAULT_WEIGHTS_PATH) -> dict:
    """Read and return the weights dict from `path`."""
    return json.loads(Path(path).read_text())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _format_weights(w: dict) -> str:
    keys = ["direction", "target", "alpha", "consistency"]
    return "  ".join(f"{k}={w.get(k, 0.0):.4f}" for k in keys)


def _public_weights(w: dict) -> dict:
    return {k: w[k] for k in ("direction", "target", "alpha", "consistency")}


def _cli_fit(db_path: str, out_path: str, method: str, ridge_alpha: float) -> int:
    conn = sqlite3.connect(db_path)
    try:
        panel = build_panel(conn)
    finally:
        conn.close()

    print(f"\n{'═' * 72}")
    print(f"  Weight fitter — panel: {len(panel)} rows "
          f"({panel['analyst_id'].nunique() if not panel.empty else 0} analysts, "
          f"{panel['as_of_date'].nunique() if not panel.empty else 0} month-ends)")
    print(f"{'═' * 72}")

    ic_w = fit_ic_weights(panel)
    rg_w = fit_ridge_weights(panel, alpha=ridge_alpha)

    print(f"\n  [IC]    {_format_weights(ic_w)}")
    meta = ic_w.get("_metadata", {})
    if meta.get("fallback"):
        print(f"          (fallback: {meta.get('reason')})")
    else:
        print(f"          mean IC: {meta.get('mean_ic')}")

    print(f"\n  [Ridge] {_format_weights(rg_w)}")
    meta = rg_w.get("_metadata", {})
    if meta.get("fallback"):
        print(f"          (fallback: {meta.get('reason')})")
    else:
        print(f"          holdout R² = {meta.get('holdout_r2')}  "
              f"(train n={meta.get('n_train')}, test n={meta.get('n_test')})")

    chosen = ic_w if method == "ic" else rg_w
    save_weights(chosen, out_path)
    print(f"\n  ✅ Wrote {method} weights to {out_path}\n")
    return 0


def _cli_report(db_path: str, weights_path: str, ridge_alpha: float) -> int:
    # Deployed (or default)
    if Path(weights_path).exists():
        deployed = load_weights(weights_path)
        deployed_src = weights_path
    else:
        deployed = dict(DEFAULT_WEIGHTS)
        deployed_src = "DEFAULT_WEIGHTS (no composite_weights.json)"

    conn = sqlite3.connect(db_path)
    try:
        panel = build_panel(conn)
    finally:
        conn.close()

    ic_w = fit_ic_weights(panel)
    rg_w = fit_ridge_weights(panel, alpha=ridge_alpha)

    print(f"\n{'═' * 72}")
    print(f"  Deployed weights ({deployed_src})")
    print(f"{'═' * 72}")
    print(f"  {_format_weights(deployed)}")
    print(f"\n  Candidate fits from current data ({len(panel)} panel rows)")
    print(f"  {'─' * 68}")
    print(f"  [IC]    {_format_weights(ic_w)}")
    print(f"  [Ridge] {_format_weights(rg_w)}")
    meta = rg_w.get("_metadata", {})
    if not meta.get("fallback"):
        print(f"          holdout R² = {meta.get('holdout_r2')}")
    print(f"{'═' * 72}\n")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit composite_score weights from data (forward-IC or ridge)."
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH,
                        help=f"SQLite DB path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--out", default=DEFAULT_WEIGHTS_PATH,
                        help=f"Output JSON path (default: {DEFAULT_WEIGHTS_PATH})")
    parser.add_argument("--fit", action="store_true",
                        help="Fit IC + ridge, write one of them to --out.")
    parser.add_argument("--method", choices=["ic", "ridge"], default="ic",
                        help="Which fit to persist when --fit (default: ic).")
    parser.add_argument("--ridge-alpha", type=float, default=1.0,
                        help="L2 penalty for ridge (default: 1.0).")
    parser.add_argument("--report", action="store_true",
                        help="Print deployed weights vs candidate fits.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.fit:
        return _cli_fit(args.db, args.out, args.method, args.ridge_alpha)
    if args.report:
        return _cli_report(args.db, args.out, args.ridge_alpha)
    # Default: print help when invoked with no args, like scoring_engine does.
    print("Nothing to do. Pass --fit or --report.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
