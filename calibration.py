"""
Analyst Tracker — Risk Calibration
===================================
Real calibration of `risk_engine` raw probabilities against realized
outcomes. Replaces the previous hard-coded affine "squeeze" in
`risk_engine.calibrate_probability` with either Platt (1999) or
isotonic regression, fit on held-out `(raw_prob, hit_direction)`
pairs.

Two calibrators supported:

- **Platt (1999)**: fits ``P = 1 / (1 + exp(A * raw + B))`` by
  minimizing log-loss. Parametric, robust on small data.
- **Isotonic**: monotone non-parametric, more flexible when enough
  labels are available.

Both are fit on a chronological 70/30 split so we report held-out
Brier and log-loss on the *later* 30% of realized positions (no look-
ahead). Also produces a reliability diagram (10 bins by default).

CLI
---

    python calibration.py --fit                # fits both, writes calibration_params.json
    python calibration.py --fit --method platt
    python calibration.py --fit --method isotonic
    python calibration.py --report             # Brier/log-loss for current deployed calibration
    python calibration.py --ascii-reliability  # text reliability diagram

The `--fit` run automatically picks the method with lower held-out
Brier score (unless `--method` is explicitly passed) and writes
``calibration_params.json`` next to `risk_engine.py`. On next import,
`risk_engine.calibrate_probability` will pick the file up.

Dependencies: pandas, numpy, scipy.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize, isotonic_regression
except ImportError:  # pragma: no cover - handled at CLI entry
    print("ERROR: scipy is required. pip install scipy", file=sys.stderr)
    raise

import risk_engine
from risk_engine import DB_PATH, WEIGHTS, evaluate_call, get_connection

PARAMS_PATH_DEFAULT = Path(__file__).with_name("calibration_params.json")


# ─────────────────────────────────────────────
# Raw probability helper
# ─────────────────────────────────────────────

def _raw_probability(dimensions: dict) -> float:
    """
    Recompute the raw (pre-calibration) weighted probability from the
    dimension breakdown returned by `evaluate_call`. Mirrors the
    weighted average at the top of `risk_engine.calibrate_probability`,
    without applying any post-hoc calibration.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for dim_name, dim_weight in WEIGHTS.items():
        dim = dimensions.get(dim_name, {})
        score = dim.get("score", 0.5)
        conf_w = dim.get("weight", 0.5)
        effective = dim_weight * conf_w
        weighted_sum += score * effective
        total_weight += effective
    if total_weight == 0:
        return 0.5
    return float(weighted_sum / total_weight)


# ─────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────

def _db_path_from_conn(conn: sqlite3.Connection) -> str:
    row = list(conn.execute("PRAGMA database_list"))
    # ('main', name, file)
    for r in row:
        if r[1] == "main" and r[2]:
            return r[2]
    return DB_PATH


def collect_labelled_pairs(
    conn: sqlite3.Connection,
    min_days_matured: int = 90,
) -> pd.DataFrame:
    """
    Build a training set of `(raw_prob, hit_direction)` pairs.

    For every closed/evaluated position with a non-null
    `hit_direction` whose `open_date` is at least `min_days_matured`
    days old, recompute the risk engine's raw probability as of the
    position's open date. `exclude_position_id` is passed to
    `evaluate_call` so the position is not in its own training data.

    Returns a DataFrame with columns:
        position_id, raw_prob, hit_direction, open_date, ticker, analyst
    """
    cursor = conn.cursor()
    cutoff = (date.today() - timedelta(days=min_days_matured)).isoformat()
    rows = cursor.execute(
        """SELECT pos.id              AS position_id,
                  pos.open_date       AS open_date,
                  pos.direction       AS direction,
                  pos.price_at_open   AS price_at_open,
                  pos.initial_target  AS price_target,
                  a.ticker            AS ticker,
                  an.name             AS analyst_name,
                  perf.hit_direction  AS hit_direction
             FROM positions pos
             JOIN performance perf ON perf.position_id = pos.id
             JOIN assets      a   ON a.id  = pos.asset_id
             JOIN analysts    an  ON an.id = pos.analyst_id
            WHERE perf.hit_direction IS NOT NULL
              AND pos.open_date <= ?
            ORDER BY pos.open_date""",
        (cutoff,),
    ).fetchall()

    db_path = _db_path_from_conn(conn)
    records: list[dict] = []
    for r in rows:
        price = r["price_at_open"]
        if price is None or price <= 0:
            continue
        try:
            result = evaluate_call(
                ticker=r["ticker"],
                analyst_name=r["analyst_name"],
                direction=r["direction"],
                price_current=float(price),
                price_target=r["price_target"],
                rec_date=r["open_date"],
                verbose=False,
                db_path=db_path,
                exclude_position_id=r["position_id"],
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"warn: evaluate_call failed for position {r['position_id']}: {exc}",
                file=sys.stderr,
            )
            continue
        raw = _raw_probability(result["dimensions"])
        records.append(
            {
                "position_id": int(r["position_id"]),
                "raw_prob": float(raw),
                "hit_direction": int(r["hit_direction"]),
                "open_date": r["open_date"],
                "ticker": r["ticker"],
                "analyst": r["analyst_name"],
            }
        )

    df = pd.DataFrame.from_records(
        records,
        columns=["position_id", "raw_prob", "hit_direction", "open_date", "ticker", "analyst"],
    )
    if not df.empty:
        df = df.sort_values("open_date").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of predicted probability vs binary outcome."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Binary cross-entropy. Predictions clipped to [eps, 1-eps]."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def reliability_curve(
    p: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """
    Bin predictions into equal-width [0, 1] bins and compute:
        - mean predicted probability per bin
        - observed hit frequency per bin
        - count per bin
    Empty bins are included with NaN mean/frequency and count=0.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize into 1..n_bins; clip so p==1.0 lands in last bin
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin_lower": float(edges[b]),
                    "bin_upper": float(edges[b + 1]),
                    "mean_pred": float("nan"),
                    "obs_freq": float("nan"),
                    "count": 0,
                }
            )
        else:
            rows.append(
                {
                    "bin_lower": float(edges[b]),
                    "bin_upper": float(edges[b + 1]),
                    "mean_pred": float(np.mean(p[mask])),
                    "obs_freq": float(np.mean(y[mask])),
                    "count": n,
                }
            )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Platt calibration
# ─────────────────────────────────────────────

def _platt_predict(raw: np.ndarray, A: float, B: float) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    # Platt 1999 convention: P = 1 / (1 + exp(A*raw + B))
    z = A * raw + B
    # numerically stable sigmoid(-z)
    return np.where(z >= 0, np.exp(-z) / (1.0 + np.exp(-z)), 1.0 / (1.0 + np.exp(z)))


def _platt_nll(params: np.ndarray, raw: np.ndarray, y: np.ndarray) -> float:
    A, B = float(params[0]), float(params[1])
    z = A * raw + B
    # -log p(y): y=1 -> log(1+e^z); y=0 -> log(1+e^z) - z
    return float(np.sum(np.logaddexp(0.0, z) - (1.0 - y) * z))


def fit_platt(pairs_df: pd.DataFrame) -> dict:
    """
    Fit Platt parameters (A, B) on a chronological 70/30 split.
    Returns a dict with method/params/metrics/split sizes.
    """
    df = pairs_df.sort_values("open_date").reset_index(drop=True)
    n = len(df)
    if n < 10:
        raise ValueError(f"need at least 10 labelled pairs, got {n}")
    n_train = max(1, int(round(0.7 * n)))
    train = df.iloc[:n_train]
    test = df.iloc[n_train:]
    raw_tr = train["raw_prob"].to_numpy(dtype=float)
    y_tr = train["hit_direction"].to_numpy(dtype=float)
    raw_te = test["raw_prob"].to_numpy(dtype=float)
    y_te = test["hit_direction"].to_numpy(dtype=float)

    res = minimize(
        _platt_nll,
        x0=np.array([-1.0, 0.0]),
        args=(raw_tr, y_tr),
        method="BFGS",
    )
    A, B = float(res.x[0]), float(res.x[1])

    p_tr = _platt_predict(raw_tr, A, B)
    p_te = _platt_predict(raw_te, A, B) if len(raw_te) else np.array([])

    return {
        "method": "platt",
        "A": A,
        "B": B,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_brier": brier_score(p_tr, y_tr),
        "train_log_loss": log_loss(p_tr, y_tr),
        "test_brier": brier_score(p_te, y_te) if len(raw_te) else float("nan"),
        "test_log_loss": log_loss(p_te, y_te) if len(raw_te) else float("nan"),
        "fit_date": date.today().isoformat(),
    }


# ─────────────────────────────────────────────
# Isotonic calibration
# ─────────────────────────────────────────────

def _isotonic_pairs(raw: np.ndarray, y: np.ndarray) -> list[tuple[float, float]]:
    """
    Fit monotone non-decreasing regression of y on raw, return
    deduplicated breakpoints as (raw_upper, calibrated_prob) pairs.
    """
    raw = np.asarray(raw, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(raw, kind="mergesort")
    raw_s = raw[order]
    y_s = y[order]
    res = isotonic_regression(y_s, increasing=True)
    fit = np.asarray(res.x, dtype=float)

    # Collapse runs that share the same fitted value to minimal
    # breakpoints (raw_upper, calibrated). For each block of constant
    # fit, keep the right-most raw as the upper bound.
    pairs: list[tuple[float, float]] = []
    i = 0
    n = len(fit)
    while i < n:
        j = i
        while j + 1 < n and fit[j + 1] == fit[i]:
            j += 1
        pairs.append((float(raw_s[j]), float(fit[i])))
        i = j + 1
    return pairs


def fit_isotonic(pairs_df: pd.DataFrame) -> dict:
    """
    Fit isotonic calibration on a chronological 70/30 split.
    Returns a dict with the monotone `pairs` list and metrics.
    """
    df = pairs_df.sort_values("open_date").reset_index(drop=True)
    n = len(df)
    if n < 10:
        raise ValueError(f"need at least 10 labelled pairs, got {n}")
    n_train = max(1, int(round(0.7 * n)))
    train = df.iloc[:n_train]
    test = df.iloc[n_train:]
    raw_tr = train["raw_prob"].to_numpy(dtype=float)
    y_tr = train["hit_direction"].to_numpy(dtype=float)
    raw_te = test["raw_prob"].to_numpy(dtype=float)
    y_te = test["hit_direction"].to_numpy(dtype=float)

    pairs = _isotonic_pairs(raw_tr, y_tr)

    p_tr = isotonic_predict(raw_tr, pairs)
    p_te = isotonic_predict(raw_te, pairs) if len(raw_te) else np.array([])

    return {
        "method": "isotonic",
        "pairs": [[u, c] for (u, c) in pairs],
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_brier": brier_score(p_tr, y_tr),
        "train_log_loss": log_loss(p_tr, y_tr),
        "test_brier": brier_score(p_te, y_te) if len(raw_te) else float("nan"),
        "test_log_loss": log_loss(p_te, y_te) if len(raw_te) else float("nan"),
        "fit_date": date.today().isoformat(),
    }


def isotonic_predict(
    raw: np.ndarray | float,
    pairs: list[tuple[float, float]] | list[list[float]],
) -> np.ndarray:
    """Piecewise-constant lookup: first pair whose upper >= raw."""
    if not pairs:
        return np.asarray(raw, dtype=float)
    uppers = np.asarray([p[0] for p in pairs], dtype=float)
    calibs = np.asarray([p[1] for p in pairs], dtype=float)
    raw_arr = np.atleast_1d(np.asarray(raw, dtype=float))
    # searchsorted on sorted uppers; any raw above max upper clips to last
    idx = np.searchsorted(uppers, raw_arr, side="left")
    idx = np.clip(idx, 0, len(uppers) - 1)
    out = calibs[idx]
    return out


# ─────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────

def save_params(params: dict, path: str | Path = PARAMS_PATH_DEFAULT) -> Path:
    p = Path(path)
    p.write_text(json.dumps(params, indent=2))
    return p


def load_params(path: str | Path = PARAMS_PATH_DEFAULT) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


# ─────────────────────────────────────────────
# Reporting / ASCII
# ─────────────────────────────────────────────

def _format_reliability_table(df: pd.DataFrame) -> str:
    lines = [
        f"{'bin':>12}  {'mean_pred':>9}  {'obs_freq':>9}  {'count':>6}",
        "-" * 44,
    ]
    for _, row in df.iterrows():
        bin_lbl = f"[{row['bin_lower']:.2f},{row['bin_upper']:.2f})"
        mp = "  n/a  " if pd.isna(row["mean_pred"]) else f"{row['mean_pred']:.4f}"
        of = "  n/a  " if pd.isna(row["obs_freq"]) else f"{row['obs_freq']:.4f}"
        lines.append(
            f"{bin_lbl:>12}  {mp:>9}  {of:>9}  {int(row['count']):>6}"
        )
    return "\n".join(lines)


def _format_ascii_reliability(df: pd.DataFrame, width: int = 40) -> str:
    lines = ["Reliability diagram (observed hit frequency per predicted-prob bin):", ""]
    for _, row in df.iterrows():
        bin_lbl = f"[{row['bin_lower']:.2f},{row['bin_upper']:.2f})"
        if row["count"] == 0 or pd.isna(row["obs_freq"]):
            lines.append(f"  {bin_lbl}  {'':{width}}  (empty)")
            continue
        bar_len = int(round(row["obs_freq"] * width))
        bar = "*" * bar_len + "." * (width - bar_len)
        lines.append(
            f"  {bin_lbl}  {bar}  p={row['mean_pred']:.3f}  obs={row['obs_freq']:.3f}  n={int(row['count'])}"
        )
    lines.append("")
    lines.append("  (bar length ∝ observed frequency; perfectly calibrated ⇔ bar ≈ mean_pred)")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Deployed calibration evaluation
# ─────────────────────────────────────────────

def _deployed_calibrated_prob(raw: float) -> float:
    """Apply whatever `risk_engine.calibrate_probability` currently does."""
    # Build a minimal dimensions dict that yields the desired raw: a
    # single dimension with weighted=1 and score=raw reproduces the
    # internal weighted-sum exactly.
    dims = {name: {"score": raw, "weight": 1.0} for name in WEIGHTS}
    return float(risk_engine.calibrate_probability(dims))


def report_deployed(pairs_df: pd.DataFrame, n_bins: int = 10) -> dict:
    raw = pairs_df["raw_prob"].to_numpy(dtype=float)
    y = pairs_df["hit_direction"].to_numpy(dtype=float)
    p_deployed = np.array([_deployed_calibrated_prob(r) for r in raw])
    return {
        "n": int(len(raw)),
        "brier": brier_score(p_deployed, y),
        "log_loss": log_loss(p_deployed, y),
        "reliability": reliability_curve(p_deployed, y, n_bins=n_bins),
        "p": p_deployed,
        "y": y,
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _cli_fit(args: argparse.Namespace) -> int:
    conn = get_connection(args.db)
    try:
        pairs = collect_labelled_pairs(conn, min_days_matured=args.min_days_matured)
    finally:
        conn.close()

    if len(pairs) < 10:
        print(
            f"ERROR: only {len(pairs)} labelled pairs available "
            f"(need ≥10). Let positions mature and try again.",
            file=sys.stderr,
        )
        return 2

    print(f"Collected {len(pairs)} labelled (raw_prob, hit_direction) pairs.")
    print(f"  date range: {pairs['open_date'].min()} → {pairs['open_date'].max()}")
    print(f"  positive rate: {pairs['hit_direction'].mean():.3f}")
    print()

    platt = fit_platt(pairs)
    iso = fit_isotonic(pairs)

    # Also report uncalibrated baseline on same split for context
    df_sorted = pairs.sort_values("open_date").reset_index(drop=True)
    n_train = max(1, int(round(0.7 * len(df_sorted))))
    test = df_sorted.iloc[n_train:]
    raw_te = test["raw_prob"].to_numpy(dtype=float)
    y_te = test["hit_direction"].to_numpy(dtype=float)
    deployed_te = np.array([_deployed_calibrated_prob(r) for r in raw_te]) if len(raw_te) else np.array([])

    def fmt(v: float) -> str:
        return f"{v:.4f}" if not (v is None or (isinstance(v, float) and np.isnan(v))) else "  n/a "

    print(f"{'method':<14}{'test_brier':>12}{'test_log_loss':>16}{'n_test':>8}")
    print("-" * 50)
    if len(deployed_te):
        print(
            f"{'affine(deployed)':<14}"
            f"{fmt(brier_score(deployed_te, y_te)):>12}"
            f"{fmt(log_loss(deployed_te, y_te)):>16}"
            f"{len(y_te):>8}"
        )
    print(f"{'platt':<14}{fmt(platt['test_brier']):>12}{fmt(platt['test_log_loss']):>16}{platt['n_test']:>8}")
    print(f"{'isotonic':<14}{fmt(iso['test_brier']):>12}{fmt(iso['test_log_loss']):>16}{iso['n_test']:>8}")
    print()

    if args.method == "platt":
        chosen = platt
    elif args.method == "isotonic":
        chosen = iso
    else:
        # auto: lower held-out Brier wins; fallback to train Brier if test is NaN
        def _key(p: dict) -> float:
            v = p["test_brier"]
            return p["train_brier"] if (v is None or np.isnan(v)) else v
        chosen = min((platt, iso), key=_key)

    out_path = save_params(chosen, args.out)
    print(f"wrote {chosen['method']} params → {out_path}")
    return 0


def _cli_report(args: argparse.Namespace) -> int:
    conn = get_connection(args.db)
    try:
        pairs = collect_labelled_pairs(conn, min_days_matured=args.min_days_matured)
    finally:
        conn.close()
    if len(pairs) == 0:
        print("No labelled pairs available.", file=sys.stderr)
        return 2

    rep = report_deployed(pairs, n_bins=args.bins)
    params = load_params(args.out)
    method_name = params.get("method") if params else "affine (fallback)"
    print(f"Deployed calibration: {method_name}")
    print(f"  n = {rep['n']}")
    print(f"  Brier     = {rep['brier']:.4f}")
    print(f"  log-loss  = {rep['log_loss']:.4f}")
    print()
    print(_format_reliability_table(rep["reliability"]))
    return 0


def _cli_ascii(args: argparse.Namespace) -> int:
    conn = get_connection(args.db)
    try:
        pairs = collect_labelled_pairs(conn, min_days_matured=args.min_days_matured)
    finally:
        conn.close()
    if len(pairs) == 0:
        print("No labelled pairs available.", file=sys.stderr)
        return 2

    rep = report_deployed(pairs, n_bins=args.bins)
    params = load_params(args.out)
    method_name = params.get("method") if params else "affine (fallback)"
    print(f"Deployed calibration: {method_name}  "
          f"(Brier={rep['brier']:.4f}  log-loss={rep['log_loss']:.4f}  n={rep['n']})")
    print()
    print(_format_ascii_reliability(rep["reliability"]))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit / report risk-engine calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fit", action="store_true", help="fit Platt+isotonic, write params")
    parser.add_argument("--report", action="store_true", help="Brier/log-loss/reliability for deployed calibration")
    parser.add_argument("--ascii-reliability", action="store_true", help="ASCII reliability diagram")
    parser.add_argument("--method", choices=("platt", "isotonic", "auto"), default="auto",
                        help="calibrator to persist (only with --fit)")
    parser.add_argument("--db", default=DB_PATH, help="sqlite path")
    parser.add_argument("--out", default=str(PARAMS_PATH_DEFAULT),
                        help="calibration_params.json path")
    parser.add_argument("--min-days-matured", type=int, default=90,
                        help="minimum age of position to include")
    parser.add_argument("--bins", type=int, default=10, help="reliability bins")

    args = parser.parse_args(argv)
    if not (args.fit or args.report or args.ascii_reliability):
        parser.error("pass --fit, --report, or --ascii-reliability")

    if args.fit:
        rc = _cli_fit(args)
        if rc != 0:
            return rc
    if args.report:
        rc = _cli_report(args)
        if rc != 0:
            return rc
    if args.ascii_reliability:
        rc = _cli_ascii(args)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
