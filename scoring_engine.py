"""
Analyst Tracker — Scoring Engine v2
=====================================
Evaluates performance per POSITION, not per individual revision.

  • direction_score  — continuous score 0→1: how far price moved in the right direction
  • target_score     — continuous score 0→1.5: how close it got to the target
  • conviction_score — (target_upgrades - target_downgrades) / max(revisions, 1)
  • avg_alpha        — average return vs benchmark
  • consistency      — stability of direction_score over time
  • composite_score  — weighted final ranking

Score logic:
  direction_score:
    BUY:  clamp(return_pct / expected_return, 0, 1)
    SELL: clamp(-return_pct / expected_return, 0, 1)

  target_score (terminal-price basis — how close the TERMINAL price got to
  the target; avoids rewarding transient intraday-to-eval highs/lows):
    BUY:  (price_eval - price_at_open) / (final_target - price_at_open)
    SELL: (price_at_open - price_eval) / (price_at_open - final_target)

  touched_target (binary, auxiliary signal):
    1 if the extreme price over [open, eval] reached the target, else 0.
    Kept as a SEPARATE signal — not part of target_score — because using
    max/min over the window roughly doubles apparent hit rates (reflection
    principle) and is confounded with volatility and horizon.

  conviction_score:
    (target_upgrades - target_downgrades) / max(total_revisions, 1)
    Positive = analyst conviction increasing | Negative = decreasing

Usage:
    python scoring_engine.py                   # compute all + ranking
    python scoring_engine.py --analyst "Dan Ives"
    python scoring_engine.py --ranking
    python scoring_engine.py --ticker NVDA
    python scoring_engine.py --migrate

Dependencies:
    pip install pandas
"""

from __future__ import annotations
import sqlite3
import argparse
import sys
import math
from datetime import date, datetime, timedelta

try:
    import pandas as pd
except ImportError:
    print("❌ Run: pip install pandas")
    sys.exit(1)

DB_PATH          = "analyst_tracker.db"
DEFAULT_HORIZON  = 180
DEFAULT_EXPECTED = 15.0
MAX_TARGET_SCORE = 1.5


# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ─────────────────────────────────────────────
# MIGRATION — adds columns if they don't exist
# ─────────────────────────────────────────────

def migrate(conn: sqlite3.Connection):
    """
    Ensures v2 schema columns exist.
    Idempotent — safe to run multiple times.
    """
    cursor = conn.cursor()

    # performance — ensure v2 columns
    cursor.execute("PRAGMA table_info(performance)")
    perf_cols = {row["name"] for row in cursor.fetchall()}
    new_perf_cols = [
        ("direction_score",  "REAL"),
        ("target_score",     "REAL"),
        ("conviction_score", "REAL"),
        ("price_open",       "REAL"),
        ("touched_target",   "INTEGER"),
    ]
    for col, typ in new_perf_cols:
        if col not in perf_cols:
            cursor.execute(f"ALTER TABLE performance ADD COLUMN {col} {typ}")
            print(f"  ✅ performance.{col} added")

    # analyst_scores — ensure v2 columns
    cursor.execute("PRAGMA table_info(analyst_scores)")
    score_cols = {row["name"] for row in cursor.fetchall()}
    new_score_cols = [
        ("avg_direction_score",   "REAL"),
        ("avg_target_score",      "REAL"),
        ("avg_conviction",        "REAL"),
        ("avg_target_upgrades",   "REAL"),
        ("avg_target_downgrades", "REAL"),
        ("total_positions",       "INTEGER"),
        ("open_positions",        "INTEGER"),
        ("closed_positions",      "INTEGER"),
    ]
    for col, typ in new_score_cols:
        if col not in score_cols:
            cursor.execute(f"ALTER TABLE analyst_scores ADD COLUMN {col} {typ}")
            print(f"  ✅ analyst_scores.{col} added")

    conn.commit()


# ─────────────────────────────────────────────
# PRICE HELPERS
# ─────────────────────────────────────────────

def get_price_on_date(
    conn: sqlite3.Connection,
    asset_id: int,
    target_date: str,
    tolerance_days: int = 7
) -> float | None:
    cursor = conn.cursor()
    cursor.execute(
        """SELECT close FROM price_history
           WHERE asset_id = ?
             AND date <= ?
             AND date >= date(?, ?)
           ORDER BY date DESC LIMIT 1""",
        (asset_id, target_date, target_date, f"-{tolerance_days} days")
    )
    row = cursor.fetchone()
    return row["close"] if row else None


def get_extreme_price_in_period(
    conn: sqlite3.Connection,
    asset_id: int,
    start_date: str,
    end_date: str,
    mode: str = "max"
) -> tuple[float | None, str | None]:
    func = "MAX" if mode == "max" else "MIN"
    cursor = conn.cursor()
    cursor.execute(
        f"""SELECT {func}(close) as extreme_price, date
            FROM price_history
            WHERE asset_id = ? AND date BETWEEN ? AND ?""",
        (asset_id, start_date, end_date)
    )
    row = cursor.fetchone()
    if row and row["extreme_price"]:
        return row["extreme_price"], row["date"]
    return None, None


def get_benchmark_return(
    conn: sqlite3.Connection,
    country: str,
    start_date: str,
    end_date: str
) -> float | None:
    ticker = "SPY" if country == "US" else "^BVSP"
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM assets WHERE ticker = ?", (ticker,))
    row = cursor.fetchone()
    if not row:
        return None
    p_start = get_price_on_date(conn, row["id"], start_date)
    p_end   = get_price_on_date(conn, row["id"], end_date)
    if p_start and p_end and p_start > 0:
        return round(((p_end - p_start) / p_start) * 100, 4)
    return None


# ─────────────────────────────────────────────
# CONTINUOUS SCORES
# ─────────────────────────────────────────────

def calc_direction_score(
    direction: str,
    return_pct: float,
    expected_return: float = DEFAULT_EXPECTED
) -> float | None:
    if direction not in ("buy", "sell"):
        return None
    if expected_return <= 0:
        expected_return = DEFAULT_EXPECTED
    raw = (return_pct if direction == "buy" else -return_pct) / expected_return
    return round(max(0.0, min(1.0, raw)), 4)


def calc_target_score(
    direction: str,
    price_at_open: float,
    price_target: float,
    price_eval: float
) -> float | None:
    """
    Target score on a TERMINAL-price basis: how close the price at evaluation
    (close / horizon / today) is to the analyst's target.

    Rationale: using max/min over [open, eval] rewards transient moves that
    reverse before the thesis matures. Empirically, P(max ≥ target) ≈ 2·P(X_T ≥ target)
    under random-walk-like dynamics, so max-based scores roughly double the
    apparent hit rate and are confounded with volatility and horizon length.
    Use `calc_touched_target` for the max-based binary signal instead.
    """
    if not price_target or not price_at_open:
        return None
    if direction == "buy":
        distance = price_target - price_at_open
        if distance <= 0:
            return None
        progress = (price_eval - price_at_open) / distance
    elif direction == "sell":
        distance = price_at_open - price_target
        if distance <= 0:
            return None
        progress = (price_at_open - price_eval) / distance
    else:
        return None
    return round(max(0.0, min(MAX_TARGET_SCORE, progress)), 4)


def calc_touched_target(
    direction: str,
    price_at_open: float,
    price_target: float,
    extreme_price: float
) -> int | None:
    """
    Binary: did the extreme price over the evaluation window reach the target?
    Kept separate from target_score so it does not inflate the primary metric.
    """
    if not price_target or not price_at_open or extreme_price is None:
        return None
    if direction == "buy":
        return 1 if extreme_price >= price_target else 0
    if direction == "sell":
        return 1 if extreme_price <= price_target else 0
    return None


# ─────────────────────────────────────────────
# POSITION EVALUATION
# ─────────────────────────────────────────────

def evaluate_position(
    conn: sqlite3.Connection,
    pos: sqlite3.Row
) -> dict | None:
    """
    Evaluates a complete position (from open to close or today).
    Computes direction_score, target_score and conviction_score.
    Returns dict with metrics or None if insufficient data.
    """
    pos_id          = pos["pos_id"]
    asset_id        = pos["asset_id"]
    direction       = pos["direction"]
    price_at_open   = pos["price_at_open"]
    price_at_close  = pos["price_at_close"]   # None if position is open
    open_date       = pos["open_date"]
    close_date      = pos["close_date"]       # None if position is open
    final_target    = pos["final_target"]
    target_upgrades   = pos["target_upgrades"]   or 0
    target_downgrades = pos["target_downgrades"] or 0
    horizon         = pos["horizon_days"] or DEFAULT_HORIZON
    country         = pos["country"]

    # Count revisions for conviction_score
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) as n FROM recommendations WHERE position_id = ?",
        (pos_id,)
    )
    total_revisions = max(cursor.fetchone()["n"] or 1, 1)

    # Evaluation date
    if close_date:
        eval_date = close_date
    else:
        eval_dt   = datetime.strptime(open_date, "%Y-%m-%d") + timedelta(days=horizon)
        eval_date = min(eval_dt.date(), date.today()).isoformat()

    # Evaluation price
    if close_date and price_at_close:
        price_eval = price_at_close
    else:
        price_eval = get_price_on_date(conn, asset_id, eval_date)

    if not price_eval or not price_at_open:
        return None

    # Position return
    return_pct = round(((price_eval - price_at_open) / price_at_open) * 100, 4)

    # Expected return (derived from target or default)
    if final_target and price_at_open:
        expected = abs(((final_target - price_at_open) / price_at_open) * 100)
        if expected < 1.0:
            expected = DEFAULT_EXPECTED
    else:
        expected = DEFAULT_EXPECTED

    # ── Direction Score ──────────────────────────────
    direction_score = calc_direction_score(direction, return_pct, expected)

    # ── Target Score (terminal price) ────────────────
    # Primary target metric uses price_eval — NOT the max/min over the window —
    # so volatility and horizon length don't mechanically inflate scores.
    target_score   = None
    touched_target = None
    days_to_target = None

    if final_target and price_at_open and direction in ("buy", "sell"):
        target_score = calc_target_score(direction, price_at_open, final_target, price_eval)

        # Auxiliary signal: did the extreme over the window reach the target?
        mode = "max" if direction == "buy" else "min"
        extreme_price, extreme_date = get_extreme_price_in_period(
            conn, asset_id, open_date, eval_date, mode=mode
        )
        if extreme_price is not None:
            touched_target = calc_touched_target(direction, price_at_open, final_target, extreme_price)
            if touched_target == 1 and extreme_date:
                d = datetime.strptime(extreme_date, "%Y-%m-%d")
                days_to_target = (d - datetime.strptime(open_date, "%Y-%m-%d")).days

    # ── Alpha vs benchmark ───────────────────────────
    bench_return  = get_benchmark_return(conn, country, open_date, eval_date)
    alpha_vs_spy  = None
    alpha_vs_ibov = None

    if bench_return is not None:
        alpha = round(return_pct - bench_return, 4)
        if country == "US":
            alpha_vs_spy  = alpha
        else:
            alpha_vs_ibov = alpha

    # ── Conviction Score ─────────────────────────────
    # Positive = analyst raised conviction | Negative = lowered
    conviction_score = round(
        (target_upgrades - target_downgrades) / total_revisions, 4
    )

    # Binary fields
    # hit_direction: did the position move in the predicted direction by eval?
    # hit_target:    did the TERMINAL price close within 10% of the target?
    # touched_target: did the price EVER reach target during the window?
    hit_direction = None
    hit_target    = None
    if direction is not None and return_pct is not None:
        if direction == "buy":
            hit_direction = 1 if return_pct > 0 else 0
        elif direction == "sell":
            hit_direction = 1 if return_pct < 0 else 0
    if target_score is not None:
        hit_target = 1 if target_score >= 0.9 else 0

    return {
        "position_id":     pos_id,
        "eval_date":       eval_date,
        "price_open":      price_at_open,
        "price_eval":      price_eval,
        "return_pct":      return_pct,
        "direction_score": direction_score,
        "target_score":    target_score,
        "hit_direction":   hit_direction,
        "hit_target":      hit_target,
        "touched_target":  touched_target,
        "alpha_vs_spy":    alpha_vs_spy,
        "alpha_vs_ibov":   alpha_vs_ibov,
        "days_to_target":  days_to_target,
        "conviction_score": conviction_score,
    }


# ─────────────────────────────────────────────
# SAVE PERFORMANCE
# ─────────────────────────────────────────────

def save_performance(conn: sqlite3.Connection, perf: dict):
    """Upsert performance for position_id (UNIQUE)."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM performance WHERE position_id = ?",
        (perf["position_id"],)
    )
    existing = cursor.fetchone()

    fields = (
        perf["eval_date"],
        perf["price_open"], perf["price_eval"],
        perf["return_pct"],
        perf["direction_score"], perf["target_score"],
        perf["hit_direction"], perf["hit_target"],
        perf.get("touched_target"),
        perf["alpha_vs_spy"], perf["alpha_vs_ibov"],
        perf["days_to_target"], perf["conviction_score"],
    )

    if existing:
        cursor.execute(
            """UPDATE performance SET
               eval_date=?, price_open=?, price_eval=?, return_pct=?,
               direction_score=?, target_score=?,
               hit_direction=?, hit_target=?, touched_target=?,
               alpha_vs_spy=?, alpha_vs_ibov=?,
               days_to_target=?, conviction_score=?,
               updated_at=date('now')
               WHERE id=?""",
            fields + (existing["id"],)
        )
    else:
        cursor.execute(
            """INSERT INTO performance
               (position_id, eval_date, price_open, price_eval, return_pct,
                direction_score, target_score,
                hit_direction, hit_target, touched_target,
                alpha_vs_spy, alpha_vs_ibov,
                days_to_target, conviction_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (perf["position_id"],) + fields
        )
    conn.commit()


# ─────────────────────────────────────────────
# AGGREGATED ANALYST SCORE
# ─────────────────────────────────────────────

def compute_consistency(scores: list[float]) -> float:
    if len(scores) < 5:
        return 0.5
    window  = 10
    windows = [
        scores[i:i+window]
        for i in range(0, len(scores), window)
        if len(scores[i:i+window]) >= 3
    ]
    if len(windows) < 2:
        return 0.5
    rates    = [sum(w) / len(w) for w in windows]
    mean     = sum(rates) / len(rates)
    variance = sum((r - mean) ** 2 for r in rates) / len(rates)
    std      = math.sqrt(variance)
    return round(max(0.0, min(1.0, 1 - (std / 0.5))), 4)


def compute_analyst_score(conn: sqlite3.Connection, analyst_id: int) -> dict | None:
    """Aggregates all performances of an analyst into unified position-based scores."""
    cursor = conn.cursor()

    cursor.execute(
        """SELECT
               perf.return_pct,
               perf.direction_score,
               perf.target_score,
               perf.hit_direction,
               perf.hit_target,
               perf.alpha_vs_spy,
               perf.alpha_vs_ibov,
               perf.conviction_score,
               pos.open_date,
               pos.close_date,
               pos.target_upgrades,
               pos.target_downgrades
           FROM performance perf
           JOIN positions pos ON pos.id = perf.position_id
           WHERE pos.analyst_id = ?
             AND perf.direction_score IS NOT NULL
           ORDER BY pos.open_date ASC""",
        (analyst_id,)
    )
    rows = cursor.fetchall()

    if not rows:
        return None

    # Count positions (all, open, closed)
    cursor.execute(
        "SELECT COUNT(*) as total FROM positions WHERE analyst_id=?",
        (analyst_id,)
    )
    total_positions = cursor.fetchone()["total"] or 0
    cursor.execute(
        "SELECT COUNT(*) as n FROM positions WHERE analyst_id=? AND close_date IS NULL",
        (analyst_id,)
    )
    open_positions   = cursor.fetchone()["n"] or 0
    closed_positions = total_positions - open_positions

    dir_scores     = [r["direction_score"] for r in rows]
    tgt_scores     = [r["target_score"]    for r in rows if r["target_score"] is not None]
    hit_list       = [r["hit_direction"]   for r in rows if r["hit_direction"] is not None]
    conv_scores    = [r["conviction_score"] for r in rows if r["conviction_score"] is not None]
    all_alpha      = (
        [r["alpha_vs_spy"]  for r in rows if r["alpha_vs_spy"]  is not None] +
        [r["alpha_vs_ibov"] for r in rows if r["alpha_vs_ibov"] is not None]
    )
    upgrades_list   = [r["target_upgrades"]   for r in rows]
    downgrades_list = [r["target_downgrades"] for r in rows]

    total  = len(dir_scores)
    wins   = sum(hit_list)
    losses = total - wins

    hit_rate            = round(wins / total, 4)                             if total       else 0.0
    avg_direction_score = round(sum(dir_scores) / len(dir_scores), 4)       if dir_scores  else None
    avg_target_score    = round(sum(tgt_scores)  / len(tgt_scores),  4)     if tgt_scores  else None
    avg_alpha           = round(sum(all_alpha)    / len(all_alpha),   4)     if all_alpha   else None
    consistency         = compute_consistency(dir_scores)
    avg_conviction      = round(sum(conv_scores) / len(conv_scores), 4)     if conv_scores else None
    avg_tgt_up          = round(sum(upgrades_list)   / len(upgrades_list),   4) if upgrades_list   else None
    avg_tgt_down        = round(sum(downgrades_list) / len(downgrades_list), 4) if downgrades_list else None
    target_acc          = round(sum(1 for s in tgt_scores if s >= 0.9) / len(tgt_scores), 4) if tgt_scores else None

    return {
        "analyst_id":           analyst_id,
        "calc_date":            date.today().isoformat(),
        "total_positions":      total_positions,
        "open_positions":       open_positions,
        "closed_positions":     closed_positions,
        "hit_rate":             hit_rate,
        "target_acc":           target_acc,
        "avg_alpha":            avg_alpha,
        "consistency":          consistency,
        "avg_direction_score":  avg_direction_score,
        "avg_target_score":     avg_target_score,
        "avg_conviction":       avg_conviction,
        "avg_target_upgrades":  avg_tgt_up,
        "avg_target_downgrades": avg_tgt_down,
        "wins":                 wins,
        "losses":               losses,
    }


def save_analyst_score(conn: sqlite3.Connection, score: dict):
    """Upsert analyst score for today."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM analyst_scores WHERE analyst_id = ? AND calc_date = ?",
        (score["analyst_id"], score["calc_date"])
    )
    existing = cursor.fetchone()

    fields = (
        score["total_positions"], score["open_positions"], score["closed_positions"],
        score["hit_rate"], score["target_acc"], score["avg_alpha"],
        score["consistency"], score["avg_direction_score"], score["avg_target_score"],
        score["avg_conviction"], score["avg_target_upgrades"], score["avg_target_downgrades"],
        score["wins"], score["losses"],
    )

    if existing:
        cursor.execute(
            """UPDATE analyst_scores SET
               total_positions=?, open_positions=?, closed_positions=?,
               hit_rate=?, target_acc=?, avg_alpha=?,
               consistency=?, avg_direction_score=?, avg_target_score=?,
               avg_conviction=?, avg_target_upgrades=?, avg_target_downgrades=?,
               wins=?, losses=?, updated_at=date('now')
               WHERE id=?""",
            fields + (existing["id"],)
        )
    else:
        cursor.execute(
            """INSERT INTO analyst_scores
               (analyst_id, calc_date,
                total_positions, open_positions, closed_positions,
                hit_rate, target_acc, avg_alpha,
                consistency, avg_direction_score, avg_target_score,
                avg_conviction, avg_target_upgrades, avg_target_downgrades,
                wins, losses)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (score["analyst_id"], score["calc_date"]) + fields
        )
    conn.commit()


# ─────────────────────────────────────────────
# YEARLY SCORE BREAKDOWN
# ─────────────────────────────────────────────

def compute_yearly_scores(conn: sqlite3.Connection, analyst_id: int) -> list[dict]:
    """
    Computes separate scores per year for an analyst.
    Allows detecting ascending vs declining analysts.

    Returns list of dicts, one per year with sufficient data:
      [{year, positions, hit_rate, avg_direction_score, avg_target_score,
        avg_alpha, wins, losses, trend}]
    """
    cursor = conn.cursor()
    cursor.execute(
        """SELECT
               perf.direction_score,
               perf.target_score,
               perf.hit_direction,
               perf.return_pct,
               perf.alpha_vs_spy,
               perf.alpha_vs_ibov,
               pos.open_date
           FROM performance perf
           JOIN positions pos ON pos.id = perf.position_id
           WHERE pos.analyst_id = ?
             AND perf.direction_score IS NOT NULL
           ORDER BY pos.open_date ASC""",
        (analyst_id,)
    )
    rows = cursor.fetchall()

    if not rows:
        return []

    # Group by year
    by_year: dict[int, list] = {}
    for r in rows:
        try:
            year = int(r["open_date"][:4])
        except (ValueError, TypeError):
            continue
        by_year.setdefault(year, []).append(r)

    results = []
    for year in sorted(by_year.keys()):
        year_rows = by_year[year]
        n = len(year_rows)
        if n == 0:
            continue

        dir_scores = [r["direction_score"] for r in year_rows]
        tgt_scores = [r["target_score"] for r in year_rows if r["target_score"] is not None]
        hit_list   = [r["hit_direction"] for r in year_rows if r["hit_direction"] is not None]
        all_alpha  = (
            [r["alpha_vs_spy"]  for r in year_rows if r["alpha_vs_spy"]  is not None] +
            [r["alpha_vs_ibov"] for r in year_rows if r["alpha_vs_ibov"] is not None]
        )

        wins   = sum(hit_list) if hit_list else 0
        losses = len(hit_list) - wins if hit_list else 0

        results.append({
            "year":                year,
            "positions":           n,
            "hit_rate":            round(wins / len(hit_list), 4) if hit_list else None,
            "avg_direction_score": round(sum(dir_scores) / len(dir_scores), 4) if dir_scores else None,
            "avg_target_score":    round(sum(tgt_scores) / len(tgt_scores), 4) if tgt_scores else None,
            "avg_alpha":           round(sum(all_alpha) / len(all_alpha), 4) if all_alpha else None,
            "wins":                wins,
            "losses":              losses,
        })

    # Compute trend based on direction score progression
    dir_by_year = [(r["year"], r["avg_direction_score"]) for r in results if r["avg_direction_score"] is not None]
    for r in results:
        r["trend"] = "stable"

    if len(dir_by_year) >= 2:
        first_score = dir_by_year[0][1]
        last_score  = dir_by_year[-1][1]
        delta = last_score - first_score

        if delta > 0.05:
            trend = "ascending"
        elif delta < -0.05:
            trend = "declining"
        else:
            trend = "stable"

        for r in results:
            r["trend"] = trend

    return results


def print_yearly_scores(analyst_name: str):
    """Prints yearly scores table for an analyst."""
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, name FROM analysts WHERE name LIKE ?",
        (f"%{analyst_name}%",)
    )
    analyst = cursor.fetchone()
    if not analyst:
        print(f"❌ Analyst not found: '{analyst_name}'")
        conn.close()
        return

    yearly = compute_yearly_scores(conn, analyst["id"])
    conn.close()

    if not yearly:
        print(f"❌ No performance data for {analyst['name']}")
        return

    trend_icon = {"ascending": "📈", "declining": "📉", "stable": "➡️"}
    trend = yearly[0]["trend"]

    print(f"\n{'═'*72}")
    print(f"  📊  Yearly Scores — {analyst['name']}  {trend_icon.get(trend, '')} {trend}")
    print(f"{'═'*72}")
    print(f"  {'Year':<6} {'Pos.':>5} {'Hit%':>6} {'Dir':>6} {'Tgt':>6} {'Alpha':>8} {'W/L':>7}")
    print(f"  {'─'*6} {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*7}")

    for r in yearly:
        hr  = f"{r['hit_rate']:.0%}" if r["hit_rate"] is not None else "—"
        ds  = f"{r['avg_direction_score']:.2f}" if r["avg_direction_score"] is not None else "—"
        ts  = f"{r['avg_target_score']:.2f}" if r["avg_target_score"] is not None else "—"
        alp = f"{r['avg_alpha']:+.1f}%" if r["avg_alpha"] is not None else "—"
        wl  = f"{r['wins']}/{r['losses']}"
        print(f"  {r['year']:<6} {r['positions']:>5} {hr:>6} {ds:>6} {ts:>6} {alp:>8} {wl:>7}")

    print(f"{'═'*72}\n")


# ─────────────────────────────────────────────
# SIMULATED PORTFOLIO
# ─────────────────────────────────────────────

def simulate_portfolio(
    conn: sqlite3.Connection,
    analyst_id: int,
    year: int | None = None
) -> dict | None:
    """
    Simulates the return of an equal-weight portfolio following an analyst's calls.

    Rules:
      - Each position receives 1/N weight (equal-weight)
      - No new call on the same asset if position already exists
      - Return measured from open to close (or today/horizon if still open)
      - Compare vs benchmark per year (SPY for US, ^BVSP for BR)

    If year=None, computes all years. If year specified, only that year.

    Returns dict with:
      {years: [{year, n_positions, return_pct, benchmark_return, alpha,
                best_call, worst_call, monthly_equity}],
       cumulative_return, cumulative_alpha, total_positions}
    """
    cursor = conn.cursor()

    year_filter = ""
    params = [analyst_id]
    if year:
        year_filter = "AND substr(pos.open_date, 1, 4) = ?"
        params.append(str(year))

    cursor.execute(
        f"""SELECT
               pos.id          AS pos_id,
               pos.asset_id,
               pos.direction,
               pos.price_at_open,
               pos.price_at_close,
               pos.open_date,
               pos.close_date,
               pos.final_target,
               pos.horizon_days,
               ast.ticker,
               ast.name         AS asset_name,
               ast.country,
               perf.return_pct,
               perf.direction_score,
               perf.alpha_vs_spy,
               perf.alpha_vs_ibov
           FROM positions pos
           JOIN assets ast ON ast.id = pos.asset_id
           LEFT JOIN performance perf ON perf.position_id = pos.id
           WHERE pos.analyst_id = ?
             {year_filter}
             AND pos.price_at_open IS NOT NULL
           ORDER BY pos.open_date ASC""",
        params
    )
    positions = cursor.fetchall()

    if not positions:
        return None

    # Group by year
    by_year: dict[int, list] = {}
    for p in positions:
        try:
            y = int(p["open_date"][:4])
        except (ValueError, TypeError):
            continue
        by_year.setdefault(y, []).append(p)

    yearly_results = []
    cumulative_value = 1.0  # starts at 1.0 (100%)
    total_positions = 0

    for yr in sorted(by_year.keys()):
        year_positions = by_year[yr]

        # Filter: skip duplicate assets (only first position per asset per year)
        seen_assets: set[int] = set()
        filtered = []
        for p in year_positions:
            if p["asset_id"] not in seen_assets:
                seen_assets.add(p["asset_id"])
                filtered.append(p)
        year_positions = filtered

        n = len(year_positions)
        if n == 0:
            continue

        # Calculate equal-weight portfolio return
        returns = []
        best_call  = None
        worst_call = None
        best_ret   = -float("inf")
        worst_ret  = float("inf")

        for p in year_positions:
            ret = p["return_pct"]
            if ret is None:
                # Calculate from prices if performance not computed yet
                if p["price_at_close"] and p["price_at_open"] and p["price_at_open"] > 0:
                    ret = ((p["price_at_close"] - p["price_at_open"]) / p["price_at_open"]) * 100
                elif p["price_at_open"] and p["price_at_open"] > 0:
                    # Open position — use current/horizon price
                    horizon = p["horizon_days"] or DEFAULT_HORIZON
                    eval_dt = datetime.strptime(p["open_date"], "%Y-%m-%d") + timedelta(days=horizon)
                    eval_date = min(eval_dt.date(), date.today()).isoformat()
                    price_eval = get_price_on_date(conn, p["asset_id"], eval_date)
                    if price_eval:
                        ret = ((price_eval - p["price_at_open"]) / p["price_at_open"]) * 100

            if ret is not None:
                # Adjust for direction: sell positions gain when price drops
                if p["direction"] == "sell":
                    ret = -ret
                returns.append(ret)

                if ret > best_ret:
                    best_ret  = ret
                    best_call = {"ticker": p["ticker"], "return_pct": round(ret, 2), "direction": p["direction"]}
                if ret < worst_ret:
                    worst_ret  = ret
                    worst_call = {"ticker": p["ticker"], "return_pct": round(ret, 2), "direction": p["direction"]}

        if not returns:
            continue

        total_positions += len(returns)

        # Equal-weight average return for the year
        avg_return = sum(returns) / len(returns)

        # Benchmark return for the year
        year_start = f"{yr}-01-01"
        year_end   = f"{yr}-12-31" if yr < date.today().year else date.today().isoformat()

        # Determine dominant market from positions
        us_count = sum(1 for p in year_positions if p["country"] == "US")
        br_count = sum(1 for p in year_positions if p["country"] == "BR")
        primary_country = "US" if us_count >= br_count else "BR"

        bench_return = get_benchmark_return(conn, primary_country, year_start, year_end)
        alpha = round(avg_return - bench_return, 2) if bench_return is not None else None

        # Monthly equity curve for this year
        monthly_equity = _compute_monthly_equity(conn, year_positions, yr)

        cumulative_value *= (1 + avg_return / 100)

        yearly_results.append({
            "year":             yr,
            "n_positions":      len(returns),
            "return_pct":       round(avg_return, 2),
            "benchmark_return": round(bench_return, 2) if bench_return is not None else None,
            "alpha":            alpha,
            "best_call":        best_call,
            "worst_call":       worst_call,
            "monthly_equity":   monthly_equity,
        })

    if not yearly_results:
        return None

    cumulative_return = round((cumulative_value - 1) * 100, 2)

    # Cumulative benchmark — only compare over years with benchmark data
    has_bench = [r for r in yearly_results if r["benchmark_return"] is not None]
    if has_bench and len(has_bench) == len(yearly_results):
        # All years have benchmark data — straightforward comparison
        cumulative_bench = 1.0
        for r in has_bench:
            cumulative_bench *= (1 + r["benchmark_return"] / 100)
        cumulative_bench_return = round((cumulative_bench - 1) * 100, 2)
        cumulative_alpha = round(cumulative_return - cumulative_bench_return, 2)
    elif has_bench:
        # Some years missing benchmark — compute alpha only over matched years
        matched_port = 1.0
        matched_bench = 1.0
        for r in has_bench:
            matched_port  *= (1 + r["return_pct"] / 100)
            matched_bench *= (1 + r["benchmark_return"] / 100)
        cumulative_bench_return = round((matched_bench - 1) * 100, 2)
        cumulative_alpha = round((matched_port - 1) * 100 - cumulative_bench_return, 2)
    else:
        cumulative_bench_return = None
        cumulative_alpha = None

    return {
        "years":              yearly_results,
        "cumulative_return":  cumulative_return,
        "cumulative_bench":   cumulative_bench_return,
        "cumulative_alpha":   cumulative_alpha,
        "total_positions":    total_positions,
    }


def _compute_monthly_equity(
    conn: sqlite3.Connection,
    positions: list,
    year: int
) -> list[dict]:
    """
    Computes monthly equity curve for a set of positions in a year.
    Returns list of {month: 'YYYY-MM', equity: float} where equity starts at 100.
    """
    monthly = []
    equity = 100.0

    for month in range(1, 13):
        month_str = f"{year}-{month:02d}"
        month_end = f"{year}-{month:02d}-28"  # approximate last day

        # For each position open during this month, compute partial return
        month_returns = []
        for p in positions:
            open_date = p["open_date"]
            close_date = p["close_date"] or date.today().isoformat()

            # Check if position was active during this month
            if open_date > f"{month_str}-28" or close_date < f"{month_str}-01":
                continue

            # Get price at start and end of month (or open/close)
            period_start = max(open_date, f"{month_str}-01")
            period_end   = min(close_date, month_end)

            p_start = get_price_on_date(conn, p["asset_id"], period_start, tolerance_days=10)
            p_end   = get_price_on_date(conn, p["asset_id"], period_end, tolerance_days=10)

            if p_start and p_end and p_start > 0:
                ret = ((p_end - p_start) / p_start) * 100
                if p["direction"] == "sell":
                    ret = -ret
                month_returns.append(ret)

        if month_returns:
            avg_ret = sum(month_returns) / len(month_returns)
            equity *= (1 + avg_ret / 100)

        monthly.append({"month": month_str, "equity": round(equity, 2)})

        # Stop if we're past the current date
        if year == date.today().year and month >= date.today().month:
            break

    return monthly


def print_portfolio(analyst_name: str):
    """Prints portfolio simulation for an analyst."""
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, name FROM analysts WHERE name LIKE ?",
        (f"%{analyst_name}%",)
    )
    analyst = cursor.fetchone()
    if not analyst:
        print(f"❌ Analyst not found: '{analyst_name}'")
        conn.close()
        return

    result = simulate_portfolio(conn, analyst["id"])
    conn.close()

    if not result:
        print(f"❌ No data to simulate portfolio for {analyst['name']}")
        return

    print(f"\n{'═'*80}")
    print(f"  💼  Simulated Portfolio — {analyst['name']}")
    print(f"  \"If you had followed this analyst, how much would you have made?\"")
    print(f"{'═'*80}")
    print(f"  {'Year':<6} {'Pos.':>5} {'Return':>9} {'Bench':>9} {'Alpha':>9} {'Best':>18} {'Worst':>18}")
    print(f"  {'─'*6} {'─'*5} {'─'*9} {'─'*9} {'─'*9} {'─'*18} {'─'*18}")

    for yr in result["years"]:
        ret   = f"{yr['return_pct']:+.1f}%"
        bench = f"{yr['benchmark_return']:+.1f}%" if yr["benchmark_return"] is not None else "—"
        alpha = f"{yr['alpha']:+.1f}%" if yr["alpha"] is not None else "—"
        best  = f"{yr['best_call']['ticker']} {yr['best_call']['return_pct']:+.1f}%" if yr["best_call"] else "—"
        worst = f"{yr['worst_call']['ticker']} {yr['worst_call']['return_pct']:+.1f}%" if yr["worst_call"] else "—"
        print(f"  {yr['year']:<6} {yr['n_positions']:>5} {ret:>9} {bench:>9} {alpha:>9} {best:>18} {worst:>18}")

    print(f"  {'─'*80}")
    print(f"  Cumulative return:    {result['cumulative_return']:+.1f}%")
    bench_str = f"{result['cumulative_bench']:+.1f}%" if result['cumulative_bench'] is not None else "—"
    alpha_str = f"{result['cumulative_alpha']:+.1f}%" if result['cumulative_alpha'] is not None else "—"
    print(f"  Cumulative benchmark: {bench_str}")
    print(f"  Cumulative alpha:     {alpha_str}")
    print(f"  Total positions:      {result['total_positions']}")
    print(f"{'═'*80}\n")


# ─────────────────────────────────────────────
# AUTO-CLOSE EXPIRED POSITIONS
# ─────────────────────────────────────────────

def auto_close_expired_positions(conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """
    Closes positions whose evaluation window has expired:
      open_date + horizon_days <= today

    Uses the price on the expiry date as price_at_close and registers
    a 'close' revision in recommendations.
    Returns the number of closed positions.
    """
    cursor = conn.cursor()

    cursor.execute(
        """SELECT
               pos.id          as pos_id,
               pos.asset_id,
               pos.direction,
               pos.open_date,
               COALESCE(pos.horizon_days, ?) as horizon,
               ast.ticker
           FROM positions pos
           JOIN assets ast ON ast.id = pos.asset_id
           WHERE pos.close_date IS NULL
             AND date(pos.open_date, '+' || COALESCE(pos.horizon_days, ?) || ' days') <= date('now')""",
        (DEFAULT_HORIZON, DEFAULT_HORIZON)
    )
    expired = cursor.fetchall()

    if not expired:
        print("  ✅ No expired positions.")
        return 0

    closed = 0
    for pos in expired:
        expiry = (
            datetime.strptime(pos["open_date"], "%Y-%m-%d") + timedelta(days=pos["horizon"])
        ).date().isoformat()

        price_close = get_price_on_date(conn, pos["asset_id"], expiry)
        if not price_close:
            continue

        if dry_run:
            print(f"  [dry] {pos['ticker']:<6} opened={pos['open_date']}  expires={expiry}  price={price_close:.2f}")
            continue

        cursor.execute(
            "UPDATE positions SET close_date=?, price_at_close=? WHERE id=?",
            (expiry, price_close, pos["pos_id"])
        )
        cursor.execute(
            """INSERT INTO recommendations
               (position_id, rec_type, rec_date, price_at_rec, direction, notes)
               VALUES (?, 'close', ?, ?, ?, 'Auto-closed: horizon expired')""",
            (pos["pos_id"], expiry, price_close, pos["direction"])
        )
        closed += 1

    if not dry_run:
        conn.commit()

    return closed


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_scoring(analyst_filter: str = None, auto_close: bool = False):
    conn   = get_connection()
    cursor = conn.cursor()

    migrate(conn)

    if auto_close:
            print("\n🔒 Auto-closing expired positions...")
            n = auto_close_expired_positions(conn)
            print(f"  → {n} positions closed.\n")

    if analyst_filter:
        cursor.execute(
            """SELECT DISTINCT a.id, a.name FROM analysts a
               JOIN positions p ON p.analyst_id = a.id
               WHERE a.name LIKE ?""",
            (f"%{analyst_filter}%",)
        )
    else:
        cursor.execute(
            """SELECT DISTINCT a.id, a.name FROM analysts a
               JOIN positions p ON p.analyst_id = a.id"""
        )

    analysts = cursor.fetchall()

    if not analysts:
        print("❌ No analysts with positions found.")
        print("   Run collector_us.py or collector_br.py to add data.")
        conn.close()
        return

    print(f"\n🔢 Computing scores for {len(analysts)} analyst(s) by position...\n")
    print(f"  {'Analyst':<28} {'Eval.':>5} {'Skip':>5} {'Dir':>6} {'Tgt':>6} {'Conv':>6} {'Alpha':>7}")
    print(f"  {'─'*28} {'─'*5} {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*7}")

    for analyst in analysts:
        analyst_id   = analyst["id"]
        analyst_name = analyst["name"]

        cursor.execute(
            """SELECT
                   pos.id          as pos_id,
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
            (analyst_id,)
        )
        positions = cursor.fetchall()

        perf_count = 0
        skip_count = 0

        for pos in positions:
            perf = evaluate_position(conn, pos)
            if perf:
                save_performance(conn, perf)
                perf_count += 1
            else:
                skip_count += 1

        score = compute_analyst_score(conn, analyst_id)
        if score:
            save_analyst_score(conn, score)
            ds   = f"{score['avg_direction_score']:.2f}" if score["avg_direction_score"] is not None else "—"
            ts   = f"{score['avg_target_score']:.2f}"    if score["avg_target_score"]    is not None else "—"
            conv = f"{score['avg_conviction']:+.2f}"     if score["avg_conviction"]       is not None else "—"
            alp  = f"{score['avg_alpha']:+.1f}%"         if score["avg_alpha"]            is not None else "—"
        else:
            ds = ts = conv = alp = "—"

        print(f"  {analyst_name[:27]:<28} {perf_count:>5} {skip_count:>5} {ds:>6} {ts:>6} {conv:>6} {alp:>7}")

    print(f"\n✅ Position-based scoring complete.\n")
    conn.close()


# ─────────────────────────────────────────────
# COMPOSITE SCORE AND RANKING
# ─────────────────────────────────────────────

def composite_score(row: dict) -> float:
    """
    Composite score for final ranking (0→100).

    Weights:
      avg_direction_score  40%
      avg_target_score     25%
      avg_alpha (norm)     25%
      consistency          10%
    """
    ds  = (row.get("avg_direction_score") or 0) * 100
    ts  = (row.get("avg_target_score")    or 0) / MAX_TARGET_SCORE * 100
    alp = (row.get("avg_alpha")           or 0)
    con = (row.get("consistency")         or 0.5) * 100

    alp_norm = max(0.0, min(100.0, (alp + 30) / 60 * 100))
    return round(ds * 0.40 + ts * 0.25 + alp_norm * 0.25 + con * 0.10, 2)


def print_ranking(top_n: int = 20):
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """SELECT
               a.name                as analyst,
               s.name                as source,
               s.country,
               sc.hit_rate,
               sc.avg_direction_score,
               sc.avg_target_score,
               sc.avg_alpha,
               sc.consistency,
               sc.avg_conviction,
               sc.total_positions,
               sc.open_positions,
               sc.closed_positions,
               sc.wins,
               sc.losses,
               sc.calc_date
           FROM analyst_scores sc
           JOIN analysts a ON a.id = sc.analyst_id
           JOIN sources  s ON s.id = a.source_id
           WHERE sc.calc_date = (
               SELECT MAX(calc_date) FROM analyst_scores WHERE analyst_id = sc.analyst_id
           )
           LIMIT ?""",
        (top_n * 2,)
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("❌ No scores computed. Run: python scoring_engine.py")
        return

    data = sorted(
        [dict(r) for r in rows],
        key=lambda x: composite_score(x),
        reverse=True
    )[:top_n]

    print(f"\n{'═'*96}")
    print(f"  🏆  ANALYST RANKING — Analyst Tracker  (position-based evaluation)")
    print(f"  Computed on: {data[0]['calc_date']}")
    print(f"{'═'*96}")
    print(f"  {'#':<3} {'Analyst':<26} {'Firm':<20} {'Dir':>5} {'Tgt':>5} {'Conv':>6} {'Alpha':>7} {'Pos.':>5} {'Score':>7}")
    print(f"  {'─'*3} {'─'*26} {'─'*20} {'─'*5} {'─'*5} {'─'*6} {'─'*7} {'─'*5} {'─'*7}")

    for i, d in enumerate(data, 1):
        ds   = f"{d['avg_direction_score']:.2f}" if d.get("avg_direction_score") is not None else "—"
        ts   = f"{d['avg_target_score']:.2f}"    if d.get("avg_target_score")    is not None else "—"
        conv = f"{d['avg_conviction']:+.2f}"     if d.get("avg_conviction")       is not None else "—"
        alp  = f"{d['avg_alpha']:+.1f}%"         if d.get("avg_alpha")            is not None else "—"
        pos  = d.get("total_positions") or 0
        cmp  = f"{composite_score(d):.1f}"
        flag = "🇧🇷" if d["country"] == "BR" else "🇺🇸" if d["country"] == "US" else "🌐"

        print(
            f"  {i:<3} {d['analyst'][:25]:<26} {d['source'][:19]:<20} "
            f"{ds:>5} {ts:>5} {conv:>6} {alp:>7} {pos:>5} {cmp:>7}  {flag}"
        )

    print(f"{'═'*96}")
    print(f"  Dir = direction_score (0→1) | Tgt = target_score (0→1.5) | Conv = conviction")
    print(f"  Composite score: Dir×40% + Tgt×25% + Alpha×25% + Consistency×10%")
    print(f"{'═'*96}\n")


# ─────────────────────────────────────────────
# ANALYSIS BY ASSET
# ─────────────────────────────────────────────

def best_analysts_for_ticker(ticker: str, top_n: int = 5):
    """Who performed best on positions for a specific asset."""
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """SELECT
               an.name as analyst,
               s.name  as source,
               COUNT(DISTINCT pos.id) as total,
               AVG(perf.direction_score) as avg_dir,
               AVG(perf.target_score)    as avg_tgt,
               AVG(perf.return_pct)      as avg_return,
               AVG(COALESCE(perf.alpha_vs_spy, 0) + COALESCE(perf.alpha_vs_ibov, 0)) as avg_alpha,
               AVG(perf.conviction_score) as avg_conv
           FROM performance perf
           JOIN positions pos ON pos.id = perf.position_id
           JOIN assets    a   ON a.id   = pos.asset_id
           JOIN analysts  an  ON an.id  = pos.analyst_id
           JOIN sources   s   ON s.id   = an.source_id
           WHERE a.ticker = ?
             AND perf.direction_score IS NOT NULL
           GROUP BY an.id
           HAVING total >= 1
           ORDER BY avg_dir DESC
           LIMIT ?""",
        (ticker.upper(), top_n)
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"❌ No performance data for {ticker}")
        return

    print(f"\n{'─'*76}")
    print(f"  📊  Best analysts for {ticker.upper()}")
    print(f"{'─'*76}")
    print(f"  {'Analyst':<24} {'Firm':<18} {'Dir':>5} {'Tgt':>5} {'Conv':>6} {'Ret%':>7} {'Pos.':>5}")
    print(f"  {'─'*24} {'─'*18} {'─'*5} {'─'*5} {'─'*6} {'─'*7} {'─'*5}")
    for row in rows:
        ds   = f"{row['avg_dir']:.2f}"      if row["avg_dir"]    is not None else "—"
        ts   = f"{row['avg_tgt']:.2f}"      if row["avg_tgt"]    is not None else "—"
        conv = f"{row['avg_conv']:+.2f}"    if row["avg_conv"]   is not None else "—"
        ret  = f"{row['avg_return']:+.1f}%" if row["avg_return"] is not None else "—"
        print(f"  {row['analyst'][:23]:<24} {row['source'][:17]:<18} {ds:>5} {ts:>5} {conv:>6} {ret:>7} {row['total']:>5}")
    print(f"{'─'*76}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyst Tracker — Scoring Engine v2 (position-based evaluation)"
    )
    parser.add_argument("--analyst",    "-a", type=str,  default=None)
    parser.add_argument("--ranking",    "-r", action="store_true")
    parser.add_argument("--ticker",     "-t", type=str,  default=None)
    parser.add_argument("--top",        "-n", type=int,  default=20)
    parser.add_argument("--migrate",    "-m", action="store_true")
    parser.add_argument("--auto-close", "-c", action="store_true",
                        help="Close expired positions before computing scores")
    parser.add_argument("--dry-run",    "-d", action="store_true",
                        help="Show which positions would be closed without modifying the database")
    parser.add_argument("--yearly",     "-y", type=str,  default=None, metavar="ANALYST",
                        help="Show yearly scores for an analyst (e.g.: --yearly 'Dan Ives')")
    parser.add_argument("--portfolio",  "-p", type=str,  default=None, metavar="ANALYST",
                        help="Simulate portfolio for an analyst (e.g.: --portfolio 'Dan Ives')")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.migrate:
        print("\n🔧 Migrating database...")
        conn = get_connection()
        migrate(conn)
        conn.close()
        print("✅ Migration complete.\n")

    elif args.dry_run:
        print("\n🔍 Dry-run: positions that would be closed today...")
        conn = get_connection()
        auto_close_expired_positions(conn, dry_run=True)
        conn.close()

    elif args.ranking:
        print_ranking(top_n=args.top)

    elif args.ticker:
        best_analysts_for_ticker(args.ticker, top_n=args.top)

    elif args.yearly:
        print_yearly_scores(args.yearly)

    elif args.portfolio:
        print_portfolio(args.portfolio)

    else:
        print("\n🚀 Analyst Tracker — Scoring Engine v2 (position-based)")
        run_scoring(analyst_filter=args.analyst, auto_close=args.auto_close)
        print_ranking(top_n=args.top)
        print("✅ Next step: streamlit run dashboard.py")
