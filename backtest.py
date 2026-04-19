"""
Analyst Tracker — Backtest Module
==================================
Point-in-time (PIT) evaluation infrastructure for the position-based scoring
system. Lets us answer:

  • As of date t, what would the leaderboard have looked like using ONLY
    information available at t?                         → score_at(conn, t)
  • Do composite_score rankings predict realized direction_score over the
    next N months?                                      → ic_series(...)
  • Would a long-top-decile / short-bottom-decile portfolio of analysts'
    open BUY calls have generated risk-adjusted alpha?  → decile_backtest(...)

All scoring logic is delegated to `scoring_engine` — this module only
reindexes the inputs (positions eligible at t, truncated at the horizon
or at t) and reuses:

    scoring_engine.evaluate_position
    scoring_engine.compute_analyst_score
    scoring_engine.composite_score / cohort_priors

No changes to scoring_engine.py / risk_engine.py / dashboard.py.

CLI:
    python backtest.py --ic                # IC series + summary
    python backtest.py --decile            # decile portfolio backtest
    python backtest.py --all               # both
    python backtest.py --as-of 2024-06-30  # ad-hoc PIT ranking snapshot

Detailed per-period output is written to:
    backtest_results/ic_series_{yyyymmdd}.csv
    backtest_results/decile_returns_{yyyymmdd}.csv

Dependencies: stdlib + pandas + numpy. No scipy / sklearn.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is importable when tests run backtest from a subdirectory.
_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import scoring_engine as se  # noqa: E402

DB_PATH = "analyst_tracker.db"
RESULTS_DIR = Path("backtest_results")
DEFAULT_HORIZON_DAYS = se.DEFAULT_HORIZON  # 180


# ─────────────────────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────────────────────

def _month_ends(start: date, end: date) -> list[date]:
    """Inclusive list of calendar month-ends between `start` and `end`."""
    if end < start:
        return []
    y, m = start.year, start.month
    out: list[date] = []
    while True:
        first_next = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        month_end = first_next - timedelta(days=1)
        if month_end > end:
            break
        if month_end >= start:
            out.append(month_end)
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def _add_months(d: date, months: int) -> date:
    idx = d.month - 1 + months
    y = d.year + idx // 12
    m = idx % 12 + 1
    first_next = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    last_day = (first_next - timedelta(days=1)).day
    return date(y, m, min(d.day, last_day))


# ─────────────────────────────────────────────────────────────────────────────
# Point-in-time position row
# ─────────────────────────────────────────────────────────────────────────────

def _mk_pit_pos(pos, as_of: date,
                default_horizon: int = DEFAULT_HORIZON_DAYS) -> dict | None:
    """
    Build a dict that mimics the `positions`-joined-with-assets row expected
    by scoring_engine.evaluate_position, but with close_date (and thus the
    evaluation terminal) clipped to the earliest of:
        1. the actual close_date (if ≤ as_of)
        2. open_date + horizon_days (if ≤ as_of)

    A position is eligible at `as_of` iff one of those two dates is ≤ as_of
    AND open_date ≤ as_of. Otherwise returns None.

    For case (2) we pass price_at_close=None so evaluate_position re-fetches
    the terminal price from price_history at horizon expiry (not today).

    Note: target_upgrades / target_downgrades are left as stored. They feed
    into conviction_score but NOT into composite_score, so a minor PIT leak
    there does not affect IC / decile metrics.
    """
    open_date = pos["open_date"]
    close_date = pos["close_date"]
    horizon = pos["horizon_days"] or default_horizon
    try:
        open_dt = datetime.strptime(open_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    if open_dt > as_of:
        return None

    horizon_expiry = open_dt + timedelta(days=horizon)

    real_close_dt: date | None = None
    if close_date:
        try:
            real_close_dt = datetime.strptime(close_date, "%Y-%m-%d").date()
        except ValueError:
            real_close_dt = None

    if real_close_dt and real_close_dt <= as_of:
        pit_close_dt = real_close_dt
        pit_price_at_close = pos["price_at_close"]
    elif horizon_expiry <= as_of:
        pit_close_dt = horizon_expiry
        pit_price_at_close = None  # force lookup at horizon expiry
    else:
        return None

    return {
        "pos_id":            pos["pos_id"],
        "asset_id":          pos["asset_id"],
        "direction":         pos["direction"],
        "price_at_open":     pos["price_at_open"],
        "price_at_close":    pit_price_at_close,
        "open_date":         open_date,
        "close_date":        pit_close_dt.isoformat(),
        "final_target":      pos["final_target"],
        "target_upgrades":   pos["target_upgrades"] or 0,
        "target_downgrades": pos["target_downgrades"] or 0,
        "horizon_days":      horizon,
        "country":           pos["country"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# In-memory PIT mirror DB — lets compute_analyst_score work on a filtered set
# of positions / performance without mutating the real DB.
# ─────────────────────────────────────────────────────────────────────────────

_PIT_SCHEMA = """
CREATE TABLE positions (
    id INTEGER PRIMARY KEY,
    analyst_id INTEGER, asset_id INTEGER, direction TEXT,
    open_date TEXT, close_date TEXT,
    price_at_open REAL, price_at_close REAL,
    target_upgrades INTEGER DEFAULT 0, target_downgrades INTEGER DEFAULT 0,
    final_target REAL, horizon_days INTEGER
);
CREATE TABLE performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER, eval_date TEXT,
    price_open REAL, price_eval REAL, return_pct REAL,
    direction_score REAL, target_score REAL,
    hit_direction INTEGER, hit_target INTEGER, touched_target INTEGER,
    alpha_vs_spy REAL, alpha_vs_ibov REAL,
    days_to_target INTEGER, conviction_score REAL
);
CREATE TABLE analysts (id INTEGER PRIMARY KEY, name TEXT, source_id INTEGER);
CREATE TABLE sources  (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE assets   (id INTEGER PRIMARY KEY, ticker TEXT, country TEXT);
"""


def _build_pit_mirror(conn: sqlite3.Connection) -> sqlite3.Connection:
    mem = sqlite3.connect(":memory:")
    mem.row_factory = sqlite3.Row
    mem.executescript(_PIT_SCHEMA)
    for row in conn.execute("SELECT id, name, source_id FROM analysts"):
        mem.execute("INSERT INTO analysts (id, name, source_id) VALUES (?,?,?)",
                    (row["id"], row["name"], row["source_id"]))
    for row in conn.execute("SELECT id, name, country FROM sources"):
        mem.execute("INSERT INTO sources (id, name, country) VALUES (?,?,?)",
                    (row["id"], row["name"], row["country"]))
    for row in conn.execute("SELECT id, ticker, country FROM assets"):
        mem.execute("INSERT INTO assets (id, ticker, country) VALUES (?,?,?)",
                    (row["id"], row["ticker"], row["country"]))
    return mem


def _fetch_positions_for_scoring(conn: sqlite3.Connection, as_of: date):
    return conn.execute(
        """SELECT
               pos.id           AS pos_id,
               pos.analyst_id   AS analyst_id,
               pos.asset_id     AS asset_id,
               pos.direction    AS direction,
               pos.open_date    AS open_date,
               pos.close_date   AS close_date,
               pos.price_at_open  AS price_at_open,
               pos.price_at_close AS price_at_close,
               pos.final_target   AS final_target,
               pos.target_upgrades   AS target_upgrades,
               pos.target_downgrades AS target_downgrades,
               pos.horizon_days   AS horizon_days,
               ast.country        AS country,
               a.name             AS analyst_name,
               s.country          AS source_country
           FROM positions pos
           JOIN assets    ast ON ast.id = pos.asset_id
           JOIN analysts  a   ON a.id   = pos.analyst_id
           JOIN sources   s   ON s.id   = a.source_id
           WHERE pos.price_at_open IS NOT NULL
             AND pos.open_date <= ?""",
        (as_of.isoformat(),),
    ).fetchall()


# ─────────────────────────────────────────────────────────────────────────────
# score_at — PIT ranking
# ─────────────────────────────────────────────────────────────────────────────

def score_at(conn: sqlite3.Connection, as_of: date) -> pd.DataFrame:
    """
    Rebuild `analyst_scores`-equivalent rows as of `as_of`, using only
    positions whose close_date ≤ as_of OR whose open_date + horizon_days
    ≤ as_of. All numeric fields come from scoring_engine.compute_analyst_score;
    composite is scoring_engine.composite_score with empirical-Bayes shrinkage
    against cohort_priors over the PIT cohort.

    Returns an empty DataFrame if no analyst has any eligible position.
    """
    all_positions = _fetch_positions_for_scoring(conn, as_of)
    if not all_positions:
        return pd.DataFrame()

    mem = _build_pit_mirror(conn)
    analyst_ids: set[int] = set()

    try:
        for pos in all_positions:
            pit = _mk_pit_pos(pos, as_of)
            if pit is None:
                continue
            perf = se.evaluate_position(conn, pit)
            if perf is None or perf.get("direction_score") is None:
                continue
            mem.execute(
                """INSERT INTO positions
                   (id, analyst_id, asset_id, direction, open_date, close_date,
                    price_at_open, price_at_close, target_upgrades, target_downgrades,
                    final_target, horizon_days)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (pit["pos_id"], pos["analyst_id"], pit["asset_id"], pit["direction"],
                 pit["open_date"], pit["close_date"],
                 pit["price_at_open"], pit["price_at_close"],
                 pit["target_upgrades"], pit["target_downgrades"],
                 pit["final_target"], pit["horizon_days"]),
            )
            mem.execute(
                """INSERT INTO performance
                   (position_id, eval_date, price_open, price_eval, return_pct,
                    direction_score, target_score, hit_direction, hit_target, touched_target,
                    alpha_vs_spy, alpha_vs_ibov, days_to_target, conviction_score)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (perf["position_id"], perf["eval_date"], perf["price_open"],
                 perf["price_eval"], perf["return_pct"],
                 perf["direction_score"], perf["target_score"],
                 perf["hit_direction"], perf["hit_target"], perf.get("touched_target"),
                 perf["alpha_vs_spy"], perf["alpha_vs_ibov"],
                 perf["days_to_target"], perf["conviction_score"]),
            )
            analyst_ids.add(pos["analyst_id"])
        mem.commit()

        records: list[dict] = []
        for aid in analyst_ids:
            sc = se.compute_analyst_score(mem, aid)
            if not sc:
                continue
            ainfo = mem.execute(
                """SELECT a.name AS analyst, s.country AS country
                   FROM analysts a JOIN sources s ON s.id = a.source_id
                   WHERE a.id = ?""",
                (aid,),
            ).fetchone()
            sc["analyst_id"] = aid
            sc["analyst"] = ainfo["analyst"] if ainfo else None
            sc["country"] = ainfo["country"] if ainfo else None
            records.append(sc)
    finally:
        mem.close()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    priors = se.cohort_priors(records)
    df["composite"] = df.apply(
        lambda r: se.composite_score(r.to_dict(), priors=priors), axis=1
    )
    df["as_of"] = as_of.isoformat()
    return df.sort_values("composite", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Information Coefficient (IC) series
# ─────────────────────────────────────────────────────────────────────────────

def _spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation = Pearson correlation of ranks."""
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return float("nan")
    rx = x[mask].rank(method="average")
    ry = y[mask].rank(method="average")
    c = rx.corr(ry)
    return float(c) if c is not None and not pd.isna(c) else float("nan")


def _truncated_pos_at(pos, terminal: date,
                      default_horizon: int = DEFAULT_HORIZON_DAYS) -> dict | None:
    """
    Like _mk_pit_pos, but ALWAYS treats the position as closed at
    min(actual close, open + horizon, terminal) — even if horizon has not
    fully expired by `terminal`. Used for the forward leg of IC, where the
    whole point is to measure partial-horizon realized direction score.
    """
    open_date = pos["open_date"]
    close_date = pos["close_date"]
    horizon = pos["horizon_days"] or default_horizon
    try:
        open_dt = datetime.strptime(open_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    if open_dt >= terminal:
        return None
    horizon_expiry = open_dt + timedelta(days=horizon)

    candidates: list[date] = [terminal, horizon_expiry]
    price_at_close_override = None
    if close_date:
        try:
            real_close = datetime.strptime(close_date, "%Y-%m-%d").date()
            candidates.append(real_close)
            if real_close == min(candidates):
                price_at_close_override = pos["price_at_close"]
        except ValueError:
            pass
    pit_close_dt = min(candidates)
    if pit_close_dt <= open_dt:
        return None
    return {
        "pos_id":            pos["pos_id"],
        "asset_id":          pos["asset_id"],
        "direction":         pos["direction"],
        "price_at_open":     pos["price_at_open"],
        "price_at_close":    price_at_close_override,
        "open_date":         open_date,
        "close_date":        pit_close_dt.isoformat(),
        "final_target":      pos["final_target"],
        "target_upgrades":   pos["target_upgrades"] or 0,
        "target_downgrades": pos["target_downgrades"] or 0,
        "horizon_days":      horizon,
        "country":           pos["country"],
    }


def _forward_direction_scores(
    conn: sqlite3.Connection, t: date, horizon_months: int
) -> pd.DataFrame:
    """
    For each analyst, mean direction_score over positions opened in (t, t+Nm],
    each evaluated at min(actual close, open + horizon_days, t + Nm). Note
    we use `_truncated_pos_at` so partial-horizon evaluations are valid —
    otherwise almost no positions would be eligible (horizon usually > Nm).
    """
    end = _add_months(t, horizon_months)
    rows = conn.execute(
        """SELECT
               pos.id           AS pos_id,
               pos.analyst_id   AS analyst_id,
               pos.asset_id     AS asset_id,
               pos.direction    AS direction,
               pos.open_date    AS open_date,
               pos.close_date   AS close_date,
               pos.price_at_open  AS price_at_open,
               pos.price_at_close AS price_at_close,
               pos.final_target   AS final_target,
               pos.target_upgrades   AS target_upgrades,
               pos.target_downgrades AS target_downgrades,
               pos.horizon_days   AS horizon_days,
               ast.country        AS country
           FROM positions pos
           JOIN assets ast ON ast.id = pos.asset_id
           WHERE pos.open_date > ? AND pos.open_date <= ?
             AND pos.price_at_open IS NOT NULL
             AND pos.direction IN ('buy', 'sell')""",
        (t.isoformat(), end.isoformat()),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["analyst_id", "fwd_direction_score", "n_fwd"])

    by_analyst: dict[int, list[float]] = {}
    for pos in rows:
        pit = _truncated_pos_at(pos, end)
        if pit is None:
            continue
        perf = se.evaluate_position(conn, pit)
        if perf is None or perf.get("direction_score") is None:
            continue
        by_analyst.setdefault(pos["analyst_id"], []).append(perf["direction_score"])

    rec = [
        {"analyst_id": aid,
         "fwd_direction_score": sum(v) / len(v),
         "n_fwd": len(v)}
        for aid, v in by_analyst.items()
    ]
    return pd.DataFrame(rec)


def ic_series(
    conn: sqlite3.Connection,
    freq: str = "M",
    min_positions_per_analyst: int = 5,
    horizon_months: int = 6,
) -> pd.DataFrame:
    """
    Month-end IC of composite_score_t vs realized avg direction_score over
    (t, t+horizon_months]. Spearman rank correlation across analysts at
    each t. Returns DataFrame with columns [date, ic, n_analysts].

    Summary stats attached to df.attrs:
        mean_ic, std_ic, ir (= mean/std * sqrt(n)), t_stat (= ir), n_periods.
    """
    row = conn.execute("SELECT MIN(open_date) AS mn FROM positions").fetchone()
    if not row or not row["mn"]:
        out = pd.DataFrame(columns=["date", "ic", "n_analysts"])
        out.attrs.update(mean_ic=float("nan"), std_ic=float("nan"),
                         ir=float("nan"), t_stat=float("nan"), n_periods=0)
        return out

    start = datetime.strptime(row["mn"], "%Y-%m-%d").date()
    end = _add_months(date.today(), -horizon_months)
    ends = _month_ends(start, end)

    records: list[dict] = []
    for t in ends:
        scored = score_at(conn, t)
        if scored.empty:
            continue
        scored = scored[scored["total_positions"] >= min_positions_per_analyst]
        if scored.empty:
            continue

        fwd = _forward_direction_scores(conn, t, horizon_months)
        if fwd.empty:
            continue

        merged = scored.merge(fwd, on="analyst_id", how="inner")
        if len(merged) < 5:
            continue

        ic = _spearman(merged["composite"], merged["fwd_direction_score"])
        if pd.isna(ic):
            continue
        records.append({"date": t.isoformat(), "ic": ic, "n_analysts": int(len(merged))})

    df = pd.DataFrame(records, columns=["date", "ic", "n_analysts"])
    if df.empty:
        df.attrs.update(mean_ic=float("nan"), std_ic=float("nan"),
                        ir=float("nan"), t_stat=float("nan"), n_periods=0)
        return df

    mean_ic = float(df["ic"].mean())
    if len(df) > 1:
        std_ic = float(df["ic"].std(ddof=1))
    else:
        std_ic = float("nan")
    n = int(len(df))
    if std_ic and not pd.isna(std_ic) and std_ic > 0:
        ir = mean_ic / std_ic * (n ** 0.5)
    else:
        ir = float("nan")
    df.attrs.update(mean_ic=mean_ic, std_ic=std_ic, ir=ir, t_stat=ir, n_periods=n)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Decile portfolio backtest
# ─────────────────────────────────────────────────────────────────────────────

def _decile_returns(
    conn: sqlite3.Connection, analyst_ids: set[int], t: date, horizon_months: int
) -> tuple[float | None, float | None]:
    """
    Equal-weight total return (%) of each analyst's open BUY calls at t,
    held for horizon_months or until close (whichever first). Entry price
    is the price ON t (we enter when the ranking is formed), NOT the
    position's original open price. Benchmark: matching country (SPY/^BVSP)
    over the same window, equal-weighted across positions.
    """
    if not analyst_ids:
        return None, None
    end = _add_months(t, horizon_months)
    t_str = t.isoformat()
    placeholders = ",".join("?" for _ in analyst_ids)
    rows = conn.execute(
        f"""SELECT pos.id AS pos_id, pos.analyst_id, pos.asset_id, pos.direction,
                   pos.open_date, pos.close_date,
                   pos.price_at_open, pos.price_at_close,
                   pos.target_upgrades, pos.target_downgrades,
                   pos.final_target, pos.horizon_days,
                   ast.country
            FROM positions pos
            JOIN assets ast ON ast.id = pos.asset_id
            WHERE pos.analyst_id IN ({placeholders})
              AND pos.direction = 'buy'
              AND pos.price_at_open IS NOT NULL
              AND pos.open_date <= ?
              AND (pos.close_date IS NULL OR pos.close_date > ?)""",
        (*list(analyst_ids), t_str, t_str),
    ).fetchall()
    if not rows:
        return None, None

    returns: list[float] = []
    bench_returns: list[float] = []
    for pos in rows:
        p_entry = se.get_price_on_date(conn, pos["asset_id"], t_str)
        if p_entry is None or p_entry <= 0:
            continue
        candidates = [end, date.today()]
        if pos["close_date"]:
            try:
                candidates.append(
                    datetime.strptime(pos["close_date"], "%Y-%m-%d").date()
                )
            except ValueError:
                pass
        exit_dt = min(candidates)
        if exit_dt <= t:
            continue
        p_exit = se.get_price_on_date(conn, pos["asset_id"], exit_dt.isoformat())
        if p_exit is None or p_exit <= 0:
            continue
        returns.append((p_exit - p_entry) / p_entry * 100)

        bench = se.get_benchmark_return(
            conn, pos["country"], t_str, exit_dt.isoformat()
        )
        if bench is not None:
            bench_returns.append(bench)

    if not returns:
        return None, None
    avg_ret = sum(returns) / len(returns)
    avg_bench = sum(bench_returns) / len(bench_returns) if bench_returns else None
    return avg_ret, avg_bench


def _stats_block(rets: pd.Series, label: str, periods_per_year: float,
                 stride: int = 1) -> dict:
    """
    Cumulative, annualized, vol, Sharpe, MDD on a series of period returns
    (fractional). Because rebalancing is monthly but holding period is
    horizon_months, consecutive returns overlap. We therefore compute:
        - mean / vol / Sharpe over ALL periods (unbiased population stats)
        - cumulative / MDD on the NON-OVERLAPPING subset (stride = horizon_months)
          so compounding reflects actual capital deployed, not leveraged overlap.
    """
    if rets.empty:
        return {f"{label}_cum": None, f"{label}_ann": None,
                f"{label}_vol": None, f"{label}_sharpe": None, f"{label}_mdd": None}
    non_overlap = rets.iloc[::max(1, stride)].reset_index(drop=True)
    if non_overlap.empty:
        non_overlap = rets
    cum = float((1 + non_overlap).prod() - 1)
    mean = float(rets.mean())
    ann = float((1 + mean) ** periods_per_year - 1)
    vol = float(rets.std(ddof=1) * (periods_per_year ** 0.5)) if len(rets) > 1 else float("nan")
    sharpe = float(ann / vol) if vol and not pd.isna(vol) and vol > 0 else float("nan")
    equity = (1 + non_overlap).cumprod()
    peak = equity.cummax()
    mdd = float((equity / peak - 1).min()) if len(non_overlap) > 1 else float("nan")
    return {f"{label}_cum": cum, f"{label}_ann": ann, f"{label}_vol": vol,
            f"{label}_sharpe": sharpe, f"{label}_mdd": mdd}


def _summarize_backtest(df: pd.DataFrame, horizon_months: int) -> dict:
    if df.empty:
        return {}
    periods_per_year = 12.0 / max(1, horizon_months)
    lo = (df["long_ret"].dropna() / 100.0).reset_index(drop=True)
    ls = (df["long_short"].dropna() / 100.0).reset_index(drop=True)
    bench = (df["bench_ret"].dropna() / 100.0).reset_index(drop=True)

    stride = max(1, horizon_months)  # rebalance monthly, cohorts last horizon_months
    summary: dict = {}
    summary.update(_stats_block(lo, "long", periods_per_year, stride=stride))
    summary.update(_stats_block(ls, "longshort", periods_per_year, stride=stride))
    summary.update(_stats_block(bench, "bench", periods_per_year, stride=stride))
    summary["turnover"] = float(df["turnover"].mean()) if len(df) else None
    summary["n_periods"] = int(len(df))

    df2 = df.copy()
    df2["year"] = df2["date"].str.slice(0, 4)
    per_year = df2.groupby("year").agg(
        months=("date", "count"),
        long_avg=("long_ret", "mean"),
        short_avg=("short_ret", "mean"),
        ls_avg=("long_short", "mean"),
        bench_avg=("bench_ret", "mean"),
    ).reset_index()
    summary["per_year"] = per_year
    return summary


def decile_backtest(conn: sqlite3.Connection, horizon_months: int = 6) -> dict:
    """
    Monthly rebalance, long top-decile / short bottom-decile analysts' open
    BUY calls. Equal-weight within decile. Hold for horizon_months or until
    position close, whichever first. Returns:

        {"per_month": DataFrame with [date, long_ret, short_ret, long_short,
                                       bench_ret, long_alpha, n_top, n_bot, turnover],
         "summary":   dict of cumulative / annualized / vol / Sharpe / MDD /
                       turnover / n_periods / per_year breakdown}
    """
    row = conn.execute("SELECT MIN(open_date) AS mn FROM positions").fetchone()
    if not row or not row["mn"]:
        return {"per_month": pd.DataFrame(), "summary": {}}

    start = datetime.strptime(row["mn"], "%Y-%m-%d").date()
    end = date.today()
    ends = _month_ends(start, end)

    records: list[dict] = []
    prev_long: set[int] = set()
    prev_short: set[int] = set()

    for t in ends:
        scored = score_at(conn, t)
        if scored.empty or len(scored) < 10:
            continue
        n = len(scored)
        k = max(1, n // 10)
        top_ids = set(scored.head(k)["analyst_id"].tolist())
        bot_ids = set(scored.tail(k)["analyst_id"].tolist())

        long_ret, long_bench = _decile_returns(conn, top_ids, t, horizon_months)
        short_ret, short_bench = _decile_returns(conn, bot_ids, t, horizon_months)

        denom = max(1, 2 * k)
        churn = (len(top_ids.symmetric_difference(prev_long))
                 + len(bot_ids.symmetric_difference(prev_short)))
        turnover = churn / denom
        prev_long, prev_short = top_ids, bot_ids

        long_short = None
        if long_ret is not None and short_ret is not None:
            long_short = long_ret - short_ret
        long_alpha = None
        if long_ret is not None and long_bench is not None:
            long_alpha = long_ret - long_bench

        records.append({
            "date": t.isoformat(),
            "long_ret": long_ret,
            "short_ret": short_ret,
            "long_short": long_short,
            "bench_ret": long_bench,
            "long_alpha": long_alpha,
            "n_top": k,
            "n_bot": k,
            "turnover": turnover,
        })

    per_month = pd.DataFrame(records)
    summary = _summarize_backtest(per_month, horizon_months)
    return {"per_month": per_month, "summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _save_csv(df: pd.DataFrame, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    path = RESULTS_DIR / f"{name}_{stamp}.csv"
    df.to_csv(path, index=False)
    return path


def _fmt(x, pct: bool = True, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    if not pct:
        return f"{x:.3f}"
    return f"{x:+.2%}" if signed else f"{x:.2%}"


def _print_ic(df: pd.DataFrame) -> None:
    print("\n═══ Information Coefficient (IC) series ═══")
    if df.empty:
        print("  (no IC data — not enough history in DB)")
        return
    print(df.to_string(index=False))
    a = df.attrs
    print("\nSummary:")
    print(f"  Mean IC:        {a.get('mean_ic', float('nan')):+.4f}")
    print(f"  Std  IC:        {a.get('std_ic',  float('nan')):.4f}")
    print(f"  IR (t-stat):    {a.get('ir',      float('nan')):+.3f}")
    print(f"  N periods:      {a.get('n_periods', 0)}")
    path = _save_csv(df, "ic_series")
    print(f"  → saved: {path}")


def _print_decile(res: dict) -> None:
    print("\n═══ Decile portfolio backtest ═══")
    per_month = res.get("per_month")
    if per_month is None:
        per_month = pd.DataFrame()
    summary: dict = res.get("summary") or {}
    if per_month.empty:
        print("  (no data — not enough history in DB)")
        return
    print(per_month.to_string(index=False))
    print("\nSummary:")
    print(f"  Long-only     cum: {_fmt(summary.get('long_cum'))}  "
          f"ann: {_fmt(summary.get('long_ann'))}  "
          f"vol: {_fmt(summary.get('long_vol'), signed=False)}  "
          f"Sharpe: {_fmt(summary.get('long_sharpe'), pct=False)}  "
          f"MDD: {_fmt(summary.get('long_mdd'))}")
    print(f"  Long-short    cum: {_fmt(summary.get('longshort_cum'))}  "
          f"ann: {_fmt(summary.get('longshort_ann'))}  "
          f"vol: {_fmt(summary.get('longshort_vol'), signed=False)}  "
          f"Sharpe: {_fmt(summary.get('longshort_sharpe'), pct=False)}  "
          f"MDD: {_fmt(summary.get('longshort_mdd'))}")
    print(f"  Benchmark     cum: {_fmt(summary.get('bench_cum'))}  "
          f"ann: {_fmt(summary.get('bench_ann'))}")
    print(f"  Turnover avg:   {_fmt(summary.get('turnover'), signed=False)}")
    print(f"  N periods:      {summary.get('n_periods', 0)}")
    py = summary.get("per_year")
    if py is not None and not py.empty:
        print("\nPer-year:")
        print(py.to_string(index=False))
    path = _save_csv(per_month, "decile_returns")
    print(f"  → saved: {path}")


def _print_as_of(df: pd.DataFrame, as_of: date) -> None:
    print(f"\n═══ Ranking as of {as_of.isoformat()} ═══")
    if df.empty:
        print("  (no data — no eligible positions as of that date)")
        return
    cols = [c for c in ["analyst", "country", "total_positions",
                        "avg_direction_score", "avg_target_score",
                        "avg_alpha", "consistency", "composite"]
            if c in df.columns]
    print(df[cols].to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Analyst Tracker — PIT backtest (IC + decile portfolio sim)"
    )
    p.add_argument("--ic", action="store_true", help="Compute IC series")
    p.add_argument("--decile", action="store_true", help="Run decile backtest")
    p.add_argument("--all", action="store_true", help="Both --ic and --decile")
    p.add_argument("--as-of", type=str, default=None, metavar="YYYY-MM-DD",
                   help="Print PIT ranking snapshot at this date")
    p.add_argument("--db", type=str, default=DB_PATH, help="SQLite DB path")
    p.add_argument("--min-positions", type=int, default=5,
                   help="Minimum PIT positions per analyst for IC eligibility")
    p.add_argument("--horizon-months", type=int, default=6,
                   help="Forward evaluation / holding horizon in months")
    args = p.parse_args(argv)

    if not (args.ic or args.decile or args.all or args.as_of):
        p.error("pick one of --ic / --decile / --all / --as-of")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        if args.as_of:
            d = datetime.strptime(args.as_of, "%Y-%m-%d").date()
            _print_as_of(score_at(conn, d), d)
        if args.ic or args.all:
            _print_ic(ic_series(
                conn,
                min_positions_per_analyst=args.min_positions,
                horizon_months=args.horizon_months,
            ))
        if args.decile or args.all:
            _print_decile(decile_backtest(conn, horizon_months=args.horizon_months))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
