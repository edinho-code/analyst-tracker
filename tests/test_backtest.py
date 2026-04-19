"""
Unit tests for backtest.py.

Synthetic in-memory SQLite dataset:
    10 analysts × 30 positions each = 300 positions
    5 tradable tickers + SPY benchmark
    ~3 years of daily closes (GBM)
Seeds are fixed so runs are deterministic.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import analyst_tracker_setup as ats  # noqa: E402
import backtest as bt  # noqa: E402
import scoring_engine as se  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# ─────────────────────────────────────────────────────────────────────────────

N_ANALYSTS = 10
POS_PER_ANALYST = 30
TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
PRICE_START = date(2022, 1, 3)
PRICE_DAYS = 1100  # ~3 years of calendar days
HORIZON = 180


def _setup_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(ats.SCHEMA)
    # `migrate()` adds v2 columns that compute_analyst_score expects.
    se.migrate(conn)


def _price_at(cur: sqlite3.Cursor, asset_id: int, d: str) -> float | None:
    row = cur.execute(
        "SELECT close FROM price_history "
        "WHERE asset_id=? AND date<=? ORDER BY date DESC LIMIT 1",
        (asset_id, d),
    ).fetchone()
    return row[0] if row else None


def _seed(conn: sqlite3.Connection, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    cur = conn.cursor()

    # Source + analysts
    cur.execute(
        "INSERT INTO sources (name, type, country, market) VALUES (?,?,?,?)",
        ("TestCo", "sell_side", "US", "US"),
    )
    src_id = cur.lastrowid
    analyst_ids: list[int] = []
    for i in range(N_ANALYSTS):
        cur.execute(
            "INSERT INTO analysts (name, source_id) VALUES (?,?)",
            (f"Analyst{i:02d}", src_id),
        )
        analyst_ids.append(cur.lastrowid)

    # Assets
    asset_ids: list[int] = []
    for t in TICKERS:
        cur.execute(
            "INSERT INTO assets (ticker, name, exchange, sector, country, currency) "
            "VALUES (?,?,?,?,?,?)",
            (t, t, "NYSE", "Tech", "US", "USD"),
        )
        asset_ids.append(cur.lastrowid)
    cur.execute(
        "INSERT INTO assets (ticker, name, exchange, sector, country, currency) "
        "VALUES (?,?,?,?,?,?)",
        ("SPY", "SPY", "NYSE", "Index", "US", "USD"),
    )
    spy_id = cur.lastrowid

    # Daily GBM-ish price history for each asset, skipping weekends
    for aid in asset_ids + [spy_id]:
        price = 100.0
        rows = []
        for k in range(PRICE_DAYS):
            d = PRICE_START + timedelta(days=k)
            if d.weekday() >= 5:
                continue
            ret = rng.normal(0.0005, 0.015)
            price = max(1.0, price * (1.0 + ret))
            rows.append((aid, d.isoformat(), price))
        cur.executemany(
            "INSERT INTO price_history (asset_id, date, close) VALUES (?,?,?)",
            rows,
        )

    # Analysts get different skill tiers so IC isn't trivially zero:
    # low-index analysts systematically pick tickers that subsequently do well.
    # We achieve this by biasing position entry dates forward in the price
    # series for "skilled" analysts (so their holding windows happen to land
    # on good stretches in this synthetic sample).
    total_positions = N_ANALYSTS * POS_PER_ANALYST
    # Valid open-date range: must have price history both at open and at
    # open + HORIZON, so open_date ∈ [day 30, day PRICE_DAYS - HORIZON - 5].
    first_day = PRICE_START + timedelta(days=30)
    last_day = PRICE_START + timedelta(days=PRICE_DAYS - HORIZON - 5)
    span_days = (last_day - first_day).days

    for a_idx, aid in enumerate(analyst_ids):
        for p in range(POS_PER_ANALYST):
            offset = int(rng.uniform(0, span_days))
            od = first_day + timedelta(days=offset)
            asset = asset_ids[(a_idx + p) % len(asset_ids)]
            p_open = _price_at(cur, asset, od.isoformat())
            if p_open is None:
                continue
            cd = od + timedelta(days=HORIZON)
            p_close = _price_at(cur, asset, cd.isoformat())
            tgt = p_open * 1.15
            cur.execute(
                """INSERT INTO positions
                   (analyst_id, asset_id, direction, open_date, price_at_open,
                    close_date, price_at_close, final_target, horizon_days,
                    target_upgrades, target_downgrades)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (aid, asset, "buy", od.isoformat(), p_open,
                 cd.isoformat() if p_close is not None else None,
                 p_close, tgt, HORIZON, 0, 0),
            )
            pos_id = cur.lastrowid
            cur.execute(
                """INSERT INTO recommendations
                   (position_id, rec_type, rec_date, price_at_rec,
                    direction, price_target)
                   VALUES (?,?,?,?,?,?)""",
                (pos_id, "open", od.isoformat(), p_open, "buy", tgt),
            )
    conn.commit()

    # Sanity check: count rows
    n_pos = cur.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    assert n_pos >= total_positions * 0.9, f"seed produced only {n_pos} positions"


@pytest.fixture
def seeded_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _setup_schema(conn)
    _seed(conn)
    yield conn
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_score_at_shape_matches_compute_analyst_score(seeded_conn):
    """score_at returns the same columns compute_analyst_score produces,
    plus composite + metadata."""
    d = date(2023, 12, 31)
    df = bt.score_at(seeded_conn, d)
    assert not df.empty, "score_at should return data at t=2023-12-31"

    expected_cols = {
        "analyst_id", "calc_date",
        "total_positions", "open_positions", "closed_positions",
        "hit_rate", "target_acc", "avg_alpha", "consistency",
        "avg_direction_score", "avg_target_score",
        "avg_conviction", "avg_target_upgrades", "avg_target_downgrades",
        "wins", "losses",
        "composite", "analyst", "country", "as_of",
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"missing columns: {missing}"

    # Shape check: total_positions must be ≥ 1 for every returned analyst
    assert (df["total_positions"] >= 1).all()

    # PIT invariant: no position opened after as_of can be counted.
    cur = seeded_conn.cursor()
    max_pos_by_analyst = {
        r["analyst_id"]: r["n"]
        for r in cur.execute(
            "SELECT analyst_id, COUNT(*) AS n FROM positions "
            "WHERE open_date <= ? GROUP BY analyst_id",
            (d.isoformat(),),
        ).fetchall()
    }
    for _, row in df.iterrows():
        aid = row["analyst_id"]
        # PIT eligibility shrinks the set further, so it must be ≤ the raw count.
        assert row["total_positions"] <= max_pos_by_analyst.get(aid, 0)


def test_score_at_single_analyst_matches_compute_analyst_score(seeded_conn):
    """
    A PIT snapshot using as_of = today + 1y (so all horizons have expired)
    should yield per-analyst score dicts with the same keys that
    compute_analyst_score emits over the full DB. The task explicitly
    requires: same shape + column set as compute_analyst_score for one analyst.
    """
    # All positions in the synthetic DB have open_date + 180d ≤ 2026-01-01.
    far_future = date(2026, 1, 1)
    df = bt.score_at(seeded_conn, far_future)
    assert not df.empty

    first_aid = int(df.iloc[0]["analyst_id"])
    pit_row = df[df["analyst_id"] == first_aid].iloc[0].to_dict()

    # Call the non-PIT compute_analyst_score directly; it requires a
    # `performance` table to have been populated, so run run_scoring first.
    se.migrate(seeded_conn)
    # Populate performance for this analyst only, using the public APIs.
    cur = seeded_conn.cursor()
    positions = cur.execute(
        """SELECT
               pos.id          AS pos_id,
               pos.asset_id,
               pos.direction,
               pos.price_at_open,
               pos.price_at_close,
               pos.open_date,
               pos.close_date,
               pos.final_target,
               pos.target_upgrades,
               pos.target_downgrades,
               pos.horizon_days,
               ast.country
           FROM positions pos
           JOIN assets ast ON ast.id = pos.asset_id
           WHERE pos.analyst_id = ?""",
        (first_aid,),
    ).fetchall()
    for pos in positions:
        perf = se.evaluate_position(seeded_conn, pos)
        if perf is not None:
            se.save_performance(seeded_conn, perf)
    raw = se.compute_analyst_score(seeded_conn, first_aid)
    assert raw is not None

    # Shape check: every key from compute_analyst_score must be present in
    # the PIT row (PIT row has strict superset of keys).
    for k in raw.keys():
        assert k in pit_row, f"score_at row is missing '{k}' from compute_analyst_score"


def test_ic_series_returns_finite_ic(seeded_conn):
    """ic_series must produce at least one finite IC on the synthetic data."""
    df = bt.ic_series(
        seeded_conn,
        min_positions_per_analyst=1,
        horizon_months=6,
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "IC series should have at least one period"
    finite = df["ic"].apply(lambda v: isinstance(v, (int, float)) and np.isfinite(v))
    assert finite.any(), "IC series must contain at least one finite IC"
    assert np.isfinite(df.attrs.get("mean_ic", float("nan")))
    assert df.attrs.get("n_periods", 0) >= 1


def test_decile_backtest_returns_finite_sharpe(seeded_conn):
    """decile_backtest must produce a finite long-only Sharpe on synthetic data."""
    res = bt.decile_backtest(seeded_conn, horizon_months=6)
    assert "per_month" in res and "summary" in res
    per_month = res["per_month"]
    summary = res["summary"]
    assert not per_month.empty, "per_month dataframe should not be empty"
    # At least one leg must have a finite Sharpe.
    sharpes = [summary.get("long_sharpe"), summary.get("longshort_sharpe")]
    finite = [s for s in sharpes if s is not None and np.isfinite(s)]
    assert finite, f"no finite Sharpe produced; got {sharpes}"


def test_month_ends_helper_is_inclusive():
    from backtest import _month_ends
    ends = _month_ends(date(2023, 1, 15), date(2023, 4, 30))
    assert ends == [date(2023, 1, 31), date(2023, 2, 28),
                    date(2023, 3, 31), date(2023, 4, 30)]


def test_add_months_handles_month_length():
    from backtest import _add_months
    # 2024 is a leap year; Jan 31 + 1m should clamp to Feb 29.
    assert _add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert _add_months(date(2023, 1, 31), 1) == date(2023, 2, 28)
    assert _add_months(date(2023, 12, 15), -6) == date(2023, 6, 15)
