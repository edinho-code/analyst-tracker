"""
Analyst Tracker — Risk Engine
================================
Calculates the calibrated probability of a recommendation being correct
BEFORE you follow it — the real product differentiator.

Unlike the scoring_engine which evaluates the past, the risk_engine
answers the question: "given what I know about this analyst and this context,
what is the probability that this specific call is correct?"

Model dimensions:
  1. Analyst × Asset       — specific history of this analyst on this ticker
  2. Analyst × Sector      — if no history on asset, how does analyst perform in sector?
  3. Upside magnitude      — calls of +50% have a much lower hit rate
  4. Consensus vs contrarian — call aligned or against the market?
  5. Timing/recency         — how long since last revision? old calls lose strength
  6. Volatility context     — does analyst perform better in bull or bear market?
  7. Coverage volume        — analyst with 3 calls has much more uncertainty

Output: calibrated probability (0–100%) + breakdown by dimension + confidence rating

Usage:
    python risk_engine.py --ticker NVDA --analyst "Dan Ives" --direction buy --target 300 --price 182
    python risk_engine.py --ticker VALE3 --analyst "BTG Pactual" --direction buy --target 85 --price 65
    python risk_engine.py --calc-all          # calculate risk profiles for all recent calls
    python risk_engine.py --profile "Dan Ives" # complete risk profile of an analyst

Dependencies:
    pip install pandas scipy
"""

from __future__ import annotations
import sqlite3
import argparse
import sys
import math
import json
from datetime import date, datetime, timedelta
from typing import Optional

try:
    import pandas as pd
except ImportError:
    print("❌ Run: pip install pandas")
    sys.exit(1)

DB_PATH = "analyst_tracker.db"

# ─────────────────────────────────────────────
# MODEL WEIGHTS
# Sum = 1.0
# ─────────────────────────────────────────────

WEIGHTS = {
    "analyst_asset":    0.30,   # specific history analyst × asset
    "analyst_sector":   0.20,   # analyst history × sector (fallback)
    "magnitude":        0.20,   # implied upside magnitude
    "consensus":        0.10,   # alignment with market consensus
    "recency":          0.10,   # how recent is the call
    "volatility_fit":   0.10,   # does analyst succeed in current vol regime?
}

# Minimum historical calls to use data with confidence
MIN_CALLS_CONFIDENT  = 10
MIN_CALLS_MODERATE   = 3

# "Normal" market upside — calls beyond this have penalty
NORMAL_UPSIDE_PCT    = 20.0
MAX_RELIABLE_UPSIDE  = 60.0   # above 60% upside, maximum penalty

# Window for "recent call" (days)
RECENT_WINDOW_DAYS   = 90
STALE_WINDOW_DAYS    = 365


# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ─────────────────────────────────────────────
# DIMENSION 1 — Analyst × Asset
# ─────────────────────────────────────────────

def score_analyst_asset(
    conn: sqlite3.Connection,
    analyst_id: int,
    asset_id: int,
    direction: str,
    exclude_position_id: Optional[int] = None,
) -> dict:
    """
    Empirical hit rate of this analyst on this specific asset.

    Uses `perf.hit_direction` (binary realized outcome) so AVG is a true
    probability in [0, 1]. Previously used AVG(perf.direction_score), which
    is a clipped continuous score, not a probability; E[clipped ratio] is
    not P(direction correct).

    `exclude_position_id` omits the currently-evaluated position from its
    own training set to avoid evidence leakage (mostly relevant for bulk
    scoring of already-stored open positions).
    """
    cursor = conn.cursor()
    cursor.execute(
        """SELECT
               COUNT(*)                                                              AS total,
               AVG(CAST(perf.hit_direction AS REAL))                                  AS hit_rate,
               SUM(CASE WHEN pos.direction = ? THEN 1 ELSE 0 END)                     AS same_dir,
               AVG(CASE WHEN pos.direction = ? THEN CAST(perf.hit_direction AS REAL)
                        ELSE NULL END)                                                AS hit_rate_same
           FROM performance perf
           JOIN positions pos ON pos.id = perf.position_id
           WHERE pos.analyst_id = ?
             AND pos.asset_id   = ?
             AND perf.hit_direction IS NOT NULL
             AND (? IS NULL OR pos.id <> ?)""",
        (direction, direction, analyst_id, asset_id, exclude_position_id, exclude_position_id)
    )
    row = cursor.fetchone()

    total = row["total"] or 0

    if total == 0:
        return {"score": 0.5, "n": 0, "weight": 0.3, "label": "no history on this asset"}

    hit_rate      = row["hit_rate"]      if row["hit_rate"]      is not None else 0.5
    hit_rate_same = row["hit_rate_same"] if row["hit_rate_same"] is not None else hit_rate

    # Use direction-specific hit rate when there is sufficient same-direction data
    score = hit_rate_same if row["same_dir"] and row["same_dir"] >= 2 else hit_rate

    # Confidence weight based on data volume
    if total >= MIN_CALLS_CONFIDENT:
        weight = 1.0
    elif total >= MIN_CALLS_MODERATE:
        weight = 0.6
    else:
        weight = 0.3

    label = f"{total} hist. calls on this asset | hit rate: {score:.0%}"
    return {"score": score, "n": total, "weight": weight, "label": label}


# ─────────────────────────────────────────────
# DIMENSION 2 — Analyst × Sector
# ─────────────────────────────────────────────

def score_analyst_sector(
    conn: sqlite3.Connection,
    analyst_id: int,
    sector: str,
    direction: str,
    exclude_position_id: Optional[int] = None,
) -> dict:
    """
    Empirical hit rate of this analyst in the asset's sector.
    Fallback when there is no (or little) history on the specific asset.

    Same rationale as score_analyst_asset: uses the binary hit_direction
    so AVG is a proper probability, and optionally excludes the currently
    evaluated position to prevent leakage.
    """
    if not sector:
        return {"score": 0.5, "n": 0, "weight": 0.2, "label": "unknown sector"}

    cursor = conn.cursor()
    cursor.execute(
        """SELECT
               COUNT(*)                                                              AS total,
               AVG(CAST(perf.hit_direction AS REAL))                                  AS hit_rate,
               AVG(CASE WHEN pos.direction = ? THEN CAST(perf.hit_direction AS REAL)
                        ELSE NULL END)                                                AS hit_rate_same
           FROM performance perf
           JOIN positions pos ON pos.id  = perf.position_id
           JOIN assets    a   ON a.id    = pos.asset_id
           WHERE pos.analyst_id = ?
             AND a.sector       = ?
             AND perf.hit_direction IS NOT NULL
             AND (? IS NULL OR pos.id <> ?)""",
        (direction, analyst_id, sector, exclude_position_id, exclude_position_id)
    )
    row = cursor.fetchone()

    total = row["total"] or 0

    if total == 0:
        return {"score": 0.5, "n": 0, "weight": 0.1, "label": f"no history in {sector}"}

    score = (
        row["hit_rate_same"] if row["hit_rate_same"] is not None
        else row["hit_rate"] if row["hit_rate"] is not None
        else 0.5
    )

    if total >= MIN_CALLS_CONFIDENT:
        weight = 0.8
    elif total >= MIN_CALLS_MODERATE:
        weight = 0.5
    else:
        weight = 0.2

    label = f"{total} calls in {sector} | hit rate: {score:.0%}"
    return {"score": score, "n": total, "weight": weight, "label": label}


# ─────────────────────────────────────────────
# DIMENSION 3 — Upside Magnitude
# ─────────────────────────────────────────────

def score_magnitude(
    direction: str,
    price_current: float,
    price_target: Optional[float]
) -> dict:
    """
    Penalizes calls with very high implied upside/downside.
    Historically, +50% calls have a much lower hit rate.

    Curve: score = 1.0 for upside <= 20%
                   decays linearly to 0.4 at 60%+
    """
    if not price_target or not price_current or price_current <= 0:
        return {"score": 0.65, "upside_pct": None, "weight": 0.5,
                "label": "no price target — unknown upside"}

    if direction == "buy":
        upside_pct = ((price_target - price_current) / price_current) * 100
    elif direction == "sell":
        upside_pct = ((price_current - price_target) / price_current) * 100
    else:
        return {"score": 0.65, "upside_pct": 0, "weight": 0.5, "label": "hold — magnitude N/A"}

    # Penalty for very aggressive upsides
    if upside_pct <= 0:
        # Target below current price on buy call — bad signal
        score = 0.2
        label = f"invalid target ({upside_pct:.1f}% upside)"
    elif upside_pct <= NORMAL_UPSIDE_PCT:
        score = 1.0
        label = f"+{upside_pct:.1f}% upside — moderate ✅"
    elif upside_pct <= MAX_RELIABLE_UPSIDE:
        # Linear decay between 20% and 60%
        decay = (upside_pct - NORMAL_UPSIDE_PCT) / (MAX_RELIABLE_UPSIDE - NORMAL_UPSIDE_PCT)
        score = 1.0 - (0.6 * decay)
        label = f"+{upside_pct:.1f}% upside — aggressive ⚠️"
    else:
        score = 0.4
        label = f"+{upside_pct:.1f}% upside — very aggressive 🔴"

    return {"score": round(score, 3), "upside_pct": round(upside_pct, 1),
            "weight": 1.0, "label": label}


# ─────────────────────────────────────────────
# DIMENSION 4 — Consensus vs Contrarian
# ─────────────────────────────────────────────

def score_consensus(
    conn: sqlite3.Connection,
    asset_id: int,
    direction: str,
    days: int = 90
) -> dict:
    """
    Checks if the call is aligned with recent consensus or is contrarian.

    Consensus-aligned calls have a slightly higher win rate,
    but correct contrarian calls have much higher returns.
    We slightly penalize very contrarian calls for having a lower base rate.
    """
    since = (date.today() - timedelta(days=days)).isoformat()

    cursor = conn.cursor()
    cursor.execute(
        """SELECT
               COUNT(*) AS total,
               SUM(CASE WHEN pos.direction = 'buy'  THEN 1 ELSE 0 END) AS buys,
               SUM(CASE WHEN pos.direction = 'sell' THEN 1 ELSE 0 END) AS sells,
               SUM(CASE WHEN pos.direction = 'hold' THEN 1 ELSE 0 END) AS holds
           FROM positions pos
           WHERE pos.asset_id = ? AND pos.open_date >= ?""",
        (asset_id, since)
    )
    row = cursor.fetchone()

    total = row["total"] or 0
    if total < 3:
        return {"score": 0.65, "consensus_pct": None, "weight": 0.3,
                "label": "insufficient consensus (< 3 recent calls)"}

    buys  = row["buys"]  or 0
    sells = row["sells"] or 0
    holds = row["holds"] or 0

    if direction == "buy":
        aligned_pct = buys / total
    elif direction == "sell":
        aligned_pct = sells / total
    else:
        aligned_pct = holds / total

    # Score: calls with 60-80% consensus have the best balance
    # Very consensual (>90%): slightly penalized (already priced in)
    # Very contrarian (<20%): penalized (lower base rate)
    if aligned_pct >= 0.9:
        score = 0.70
        label = f"herd call — {aligned_pct:.0%} of market agrees ⚠️ (may already be priced in)"
    elif aligned_pct >= 0.60:
        score = 0.80
        label = f"solid consensus — {aligned_pct:.0%} agrees ✅"
    elif aligned_pct >= 0.35:
        score = 0.70
        label = f"split — {aligned_pct:.0%} agrees, market divided"
    elif aligned_pct >= 0.15:
        score = 0.55
        label = f"contrarian — only {aligned_pct:.0%} agrees ⚠️"
    else:
        score = 0.40
        label = f"very contrarian — {aligned_pct:.0%} agrees 🔴 (lone call)"

    return {"score": score, "consensus_pct": round(aligned_pct, 3),
            "weight": 0.7, "label": label}


# ─────────────────────────────────────────────
# DIMENSION 5 — Call Recency
# ─────────────────────────────────────────────

def score_recency(rec_date: Optional[str]) -> dict:
    """
    More recent calls are more reliable.
    A 2-year-old call without revision = context has changed significantly.
    """
    if not rec_date:
        return {"score": 0.60, "days_old": None, "weight": 0.5,
                "label": "unknown call date"}

    try:
        rec_dt  = datetime.strptime(rec_date, "%Y-%m-%d").date()
        days_old = (date.today() - rec_dt).days
    except ValueError:
        return {"score": 0.60, "days_old": None, "weight": 0.5, "label": "invalid date"}

    if days_old <= 7:
        score = 1.0
        label = f"call {days_old}d ago — very fresh ✅"
    elif days_old <= RECENT_WINDOW_DAYS:
        score = 0.90
        label = f"call {days_old}d ago — recent ✅"
    elif days_old <= 180:
        score = 0.75
        label = f"call {days_old}d ago — moderately recent"
    elif days_old <= STALE_WINDOW_DAYS:
        score = 0.55
        label = f"call {days_old}d ago — aging ⚠️"
    else:
        score = 0.35
        label = f"call {days_old}d ago — outdated 🔴"

    return {"score": score, "days_old": days_old, "weight": 1.0, "label": label}


# ─────────────────────────────────────────────
# DIMENSION 6 — Volatility Fit
# ─────────────────────────────────────────────

def score_volatility_fit(
    conn: sqlite3.Connection,
    analyst_id: int,
    asset_id: int,
    exclude_position_id: Optional[int] = None,
) -> dict:
    """
    Checks if the analyst tends to be more accurate in high or low volatility periods,
    and compares with the asset's current volatility (proxy: std dev of last 30 days).

    Volatility proxy: std of daily returns over last 30 days.
    High vol: std > 2.5% | Low vol: std < 1.0%
    """
    cursor = conn.cursor()

    # Current asset volatility (last 30 days)
    cursor.execute(
        """SELECT close, date FROM price_history
           WHERE asset_id = ?
             AND date >= date('now', '-35 days')
           ORDER BY date""",
        (asset_id,)
    )
    prices = cursor.fetchall()

    current_vol = None
    vol_regime  = "unknown"

    if len(prices) >= 10:
        closes  = [p["close"] for p in prices]
        returns = [(closes[i] - closes[i-1]) / closes[i-1] * 100
                   for i in range(1, len(closes))]
        mean_r  = sum(returns) / len(returns)
        # Unbiased sample variance (Bessel's correction: n-1)
        variance = (sum((r - mean_r) ** 2 for r in returns) /
                    (len(returns) - 1)) if len(returns) > 1 else 0.0
        current_vol = math.sqrt(variance)

        if current_vol > 2.5:
            vol_regime = "high"
        elif current_vol > 1.0:
            vol_regime = "medium"
        else:
            vol_regime = "low"

    if vol_regime == "unknown":
        return {"score": 0.60, "current_vol": None, "weight": 0.3,
                "label": "current volatility unknown (no prices)"}

    # Analyst's historical hit rate (binary outcome), not clipped score.
    # NOTE: this dimension still applies a *uniform* vol-regime multiplier
    # rather than a regime-conditional hit rate (which the docstring
    # promises). Proper regime-conditional fitting is out of scope for
    # this PR and is tracked as a follow-up.
    cursor.execute(
        """SELECT
               AVG(CAST(perf.hit_direction AS REAL)) AS hit_rate,
               COUNT(*)                              AS total
           FROM performance perf
           JOIN positions pos ON pos.id = perf.position_id
           WHERE pos.analyst_id = ?
             AND perf.hit_direction IS NOT NULL
             AND (? IS NULL OR pos.id <> ?)""",
        (analyst_id, exclude_position_id, exclude_position_id)
    )
    overall = cursor.fetchone()

    if not overall or overall["hit_rate"] is None:
        return {"score": 0.60, "current_vol": current_vol, "weight": 0.3,
                "label": f"current vol: {current_vol:.2f}% — insufficient history"}

    # Without detailed regime-conditional history, apply a simple adjustment:
    # high volatility historically compresses analyst accuracy by ~15%.
    base_score = overall["hit_rate"]

    if vol_regime == "high":
        adj_score = base_score * 0.85
        label = f"high current vol ({current_vol:.2f}%/day) — accuracy tends to drop ⚠️"
    elif vol_regime == "medium":
        adj_score = base_score * 1.00
        label = f"moderate current vol ({current_vol:.2f}%/day) — normal conditions ✅"
    else:
        adj_score = base_score * 1.05
        label = f"low current vol ({current_vol:.2f}%/day) — calm market ✅"

    adj_score = max(0.0, min(1.0, adj_score))
    weight    = 0.7 if overall["total"] >= MIN_CALLS_MODERATE else 0.3

    return {"score": round(adj_score, 3), "current_vol": round(current_vol, 3),
            "weight": weight, "label": label}


# ─────────────────────────────────────────────
# FINAL CALIBRATION
# ─────────────────────────────────────────────

def calibrate_probability(dimensions: dict) -> float:
    """
    Combines scores from the 6 dimensions into a final calibrated probability.

    Uses weighted average of scores × dimension_weight × confidence_weight.
    The confidence_weight reflects how much real data we have for that dimension.
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for dim_name, dim_weight in WEIGHTS.items():
        dim = dimensions.get(dim_name, {})
        score  = dim.get("score", 0.5)
        conf_w = dim.get("weight", 0.5)   # internal confidence of dimension

        effective_weight = dim_weight * conf_w
        weighted_sum    += score * effective_weight
        total_weight    += effective_weight

    if total_weight == 0:
        return 0.50

    raw_prob = weighted_sum / total_weight

    # Platt calibration — smooths extremes (avoids 95%+ or <10%)
    # Mapping: [0, 1] → [0.15, 0.88]
    # NOTE: this is an affine squeeze, NOT Platt/isotonic calibration.
    # Platt calibration would fit P = sigmoid(A * raw_prob + B) with
    # parameters learned from realized (raw_prob, outcome) pairs on a
    # holdout. Treat the output as an UNCALIBRATED composite signal
    # until a proper calibration step is added (tracked as follow-up).
    calibrated = 0.15 + (raw_prob * 0.73)

    return round(calibrated, 4)


def confidence_rating(prob: float, n_calls: int) -> tuple[str, str]:
    """
    Converts probability + data volume into qualitative rating.
    Returns (rating, emoji).
    """
    # High uncertainty when few data points
    if n_calls < MIN_CALLS_MODERATE:
        return "UNCERTAIN", "⚪"

    if prob >= 0.75:
        return "HIGH", "🟢"
    elif prob >= 0.62:
        return "MODERATE-HIGH", "🟡"
    elif prob >= 0.50:
        return "MODERATE", "🟠"
    elif prob >= 0.38:
        return "MODERATE-LOW", "🔴"
    else:
        return "LOW", "🔴"


# ─────────────────────────────────────────────
# MAIN INTERFACE
# ─────────────────────────────────────────────

def evaluate_call(
    ticker: str,
    analyst_name: str,
    direction: str,
    price_current: float,
    price_target: Optional[float] = None,
    rec_date: Optional[str] = None,
    verbose: bool = True,
    db_path: str = DB_PATH,
    exclude_position_id: Optional[int] = None,
) -> dict:
    """
    Evaluates the risk of a specific call.
    Returns dict with probability, breakdown by dimension and rating.

    Example:
        result = evaluate_call(
            ticker="NVDA",
            analyst_name="Dan Ives",
            direction="buy",
            price_current=182.0,
            price_target=300.0,
            rec_date="2024-03-15"
        )
        print(result["probability_pct"])  # e.g.: 67.3
        print(result["rating"])           # e.g.: "MODERATE-HIGH"
    """
    conn   = get_connection(db_path)
    cursor = conn.cursor()

    # Find analyst_id
    cursor.execute(
        "SELECT id FROM analysts WHERE name LIKE ?",
        (f"%{analyst_name}%",)
    )
    analyst_row = cursor.fetchone()
    analyst_id  = analyst_row["id"] if analyst_row else None

    # Find asset_id + sector
    cursor.execute(
        "SELECT id, sector, country FROM assets WHERE ticker = ?",
        (ticker.upper(),)
    )
    asset_row = cursor.fetchone()
    asset_id  = asset_row["id"]     if asset_row else None
    sector    = asset_row["sector"] if asset_row else None

    # Total historical positions of analyst
    n_calls_total = 0
    if analyst_id:
        cursor.execute(
            "SELECT COUNT(*) AS n FROM positions WHERE analyst_id = ?",
            (analyst_id,)
        )
        n_calls_total = cursor.fetchone()["n"] or 0

    # ── Calculate each dimension ──────────────────────

    dimensions = {}

    # 1. Analyst × Asset
    if analyst_id and asset_id:
        dimensions["analyst_asset"] = score_analyst_asset(
            conn, analyst_id, asset_id, direction,
            exclude_position_id=exclude_position_id,
        )
    else:
        dimensions["analyst_asset"] = {
            "score": 0.5, "n": 0, "weight": 0.1,
            "label": f"analyst '{analyst_name}' or ticker '{ticker}' not found in database"
        }

    # 2. Analyst × Sector
    if analyst_id and sector:
        dimensions["analyst_sector"] = score_analyst_sector(
            conn, analyst_id, sector, direction,
            exclude_position_id=exclude_position_id,
        )
    else:
        dimensions["analyst_sector"] = {
            "score": 0.5, "n": 0, "weight": 0.1,
            "label": "no sector data"
        }

    # 3. Upside magnitude
    dimensions["magnitude"] = score_magnitude(direction, price_current, price_target)

    # 4. Consensus
    if asset_id:
        dimensions["consensus"] = score_consensus(conn, asset_id, direction)
    else:
        dimensions["consensus"] = {"score": 0.65, "weight": 0.2, "label": "asset not found"}

    # 5. Recency
    dimensions["recency"] = score_recency(rec_date or date.today().isoformat())

    # 6. Volatility
    if analyst_id and asset_id:
        dimensions["volatility_fit"] = score_volatility_fit(
            conn, analyst_id, asset_id,
            exclude_position_id=exclude_position_id,
        )
    else:
        dimensions["volatility_fit"] = {"score": 0.60, "weight": 0.2, "label": "insufficient data"}

    # ── Final probability ─────────────────────────────

    probability = calibrate_probability(dimensions)
    prob_pct    = round(probability * 100, 1)
    rating, emoji = confidence_rating(probability, n_calls_total)

    # Implied upside
    upside_pct = dimensions["magnitude"].get("upside_pct")

    result = {
        "ticker":          ticker.upper(),
        "analyst":         analyst_name,
        "direction":       direction,
        "price_current":   price_current,
        "price_target":    price_target,
        "upside_pct":      upside_pct,
        "rec_date":        rec_date,
        "probability":     probability,
        "probability_pct": prob_pct,
        "rating":          rating,
        "rating_emoji":    emoji,
        "n_calls_history": n_calls_total,
        "dimensions":      dimensions,
    }

    if verbose:
        _print_result(result)

    conn.close()
    return result


def _print_result(r: dict):
    """Prints the formatted result to the terminal."""
    dir_icon = {"buy": "📈 BUY", "sell": "📉 SELL", "hold": "➡️  HOLD"}.get(r["direction"], r["direction"])
    upside   = f"+{r['upside_pct']:.1f}%" if r["upside_pct"] else "no target"

    print(f"\n{'═'*62}")
    print(f"  🎯  RISK ASSESSMENT — Analyst Tracker")
    print(f"{'═'*62}")
    print(f"  Analyst:    {r['analyst']}")
    print(f"  Call:       {r['ticker']} {dir_icon}  |  current price: ${r['price_current']:.2f}")
    if r["price_target"]:
        print(f"  Target:     ${r['price_target']:.2f}  ({upside})")
    if r["rec_date"]:
        print(f"  Date:       {r['rec_date']}")
    print(f"  History:    {r['n_calls_history']} total analyst calls in database")
    print(f"{'─'*62}")
    print(f"\n  HIT PROBABILITY:  {r['probability_pct']:.1f}%  {r['rating_emoji']} {r['rating']}\n")
    print(f"{'─'*62}")
    print(f"  Breakdown by dimension:")
    print()

    dim_labels = {
        "analyst_asset":  "Hist. Analyst × Asset",
        "analyst_sector": "Hist. Analyst × Sector",
        "magnitude":      "Upside Magnitude",
        "consensus":      "Consensus Alignment",
        "recency":        "Call Recency",
        "volatility_fit": "Volatility Fit",
    }

    for key, label in dim_labels.items():
        dim   = r["dimensions"].get(key, {})
        score = dim.get("score", 0.5)
        text  = dim.get("label", "—")
        bar   = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"  {label:<26}  [{bar}] {score:.2f}")
        print(f"  {'':26}  → {text}")
        print()

    print(f"{'─'*62}")

    # Interpretation
    prob = r["probability_pct"]
    if prob >= 75:
        msg = "Strong signal — history and context favor this call."
    elif prob >= 62:
        msg = "Moderate-positive signal — reasonable call, but monitor closely."
    elif prob >= 50:
        msg = "Neutral signal — considerable uncertainty, size position carefully."
    elif prob >= 38:
        msg = "Weak signal — unfavorable history or context. High caution."
    else:
        msg = "Negative signal — multiple factors against. Avoid or hedge."

    print(f"  💬 {msg}")
    print(f"{'═'*62}\n")


# ─────────────────────────────────────────────
# CALCULATE RISK FOR ALL RECENT CALLS
# ─────────────────────────────────────────────

def calc_all_recent(days: int = 90, db_path: str = DB_PATH):
    """
    Calculates and saves risk scores for recent open positions.
    Persists to risk_assessments table (position_id) for dashboard use.
    """
    conn = get_connection(db_path)

    # Create/ensure table with position_id
    conn.execute("""
        CREATE TABLE IF NOT EXISTS risk_assessments (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            position_id         INTEGER NOT NULL REFERENCES positions(id),
            probability         REAL NOT NULL,
            rating              TEXT,
            dim_analyst_asset   REAL,
            dim_analyst_sector  REAL,
            dim_magnitude       REAL,
            dim_consensus       REAL,
            dim_recency         REAL,
            dim_volatility      REAL,
            upside_pct          REAL,
            calc_date           TEXT DEFAULT (date('now')),
            UNIQUE(position_id)
        )
    """)
    conn.commit()

    since  = (date.today() - timedelta(days=days)).isoformat()
    cursor = conn.cursor()

    cursor.execute(
        """SELECT
               pos.id, pos.open_date, pos.direction, pos.price_at_open,
               pos.final_target, pos.horizon_days,
               a.name   AS analyst_name,
               ast.ticker, ast.sector, ast.country
           FROM positions pos
           JOIN analysts a   ON a.id   = pos.analyst_id
           JOIN assets   ast ON ast.id = pos.asset_id
           WHERE pos.open_date >= ?
             AND pos.price_at_open IS NOT NULL
             AND pos.close_date IS NULL
           ORDER BY pos.open_date DESC""",
        (since,)
    )
    positions = cursor.fetchall()

    print(f"\n🔢 Calculating risk for {len(positions)} open positions (since {since})...\n")

    calculated = 0
    for pos in positions:
        try:
            result = evaluate_call(
                ticker=pos["ticker"],
                analyst_name=pos["analyst_name"],
                direction=pos["direction"],
                price_current=pos["price_at_open"],
                price_target=pos["final_target"],
                rec_date=pos["open_date"],
                verbose=False,
                db_path=db_path,
                # Exclude this very position from its own training data to
                # avoid evidence leakage in the "forward" probability estimate.
                exclude_position_id=pos["id"],
            )

            dims = result["dimensions"]
            conn.execute(
                """INSERT OR REPLACE INTO risk_assessments
                   (position_id, probability, rating,
                    dim_analyst_asset, dim_analyst_sector, dim_magnitude,
                    dim_consensus, dim_recency, dim_volatility, upside_pct)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pos["id"],
                    result["probability"],
                    result["rating"],
                    dims.get("analyst_asset",  {}).get("score"),
                    dims.get("analyst_sector", {}).get("score"),
                    dims.get("magnitude",      {}).get("score"),
                    dims.get("consensus",      {}).get("score"),
                    dims.get("recency",        {}).get("score"),
                    dims.get("volatility_fit", {}).get("score"),
                    result.get("upside_pct"),
                )
            )
            conn.commit()
            calculated += 1

            prob  = result["probability_pct"]
            emoji = result["rating_emoji"]
            print(f"  {pos['ticker']:<8} {pos['analyst_name'][:25]:<26} "
                  f"{pos['direction']:<5} {prob:>5.1f}%  {emoji} {result['rating']}")

        except Exception as e:
            print(f"  ⚠️  Error on position #{pos['id']}: {e}")

    print(f"\n✅ {calculated} risk assessments calculated and saved.\n")
    conn.close()


# ─────────────────────────────────────────────
# ANALYST RISK PROFILE
# ─────────────────────────────────────────────

def analyst_risk_profile(analyst_name: str, db_path: str = DB_PATH):
    """
    Shows the complete risk profile of an analyst:
    which types of calls have higher/lower calibrated probability.
    """
    conn   = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """SELECT
               ra.probability, ra.upside_pct,
               pos.direction, pos.open_date AS rec_date,
               ast.ticker, ast.sector
           FROM risk_assessments ra
           JOIN positions pos ON pos.id  = ra.position_id
           JOIN analysts  a   ON a.id    = pos.analyst_id
           JOIN assets    ast ON ast.id  = pos.asset_id
           WHERE a.name LIKE ?
           ORDER BY ra.probability DESC""",
        (f"%{analyst_name}%",)
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"\n❌ No risk assessments for '{analyst_name}'.")
        print("   Run: python risk_engine.py --calc-all\n")
        return

    df = pd.DataFrame([dict(r) for r in rows])

    print(f"\n{'═'*62}")
    print(f"  🧠  Risk Profile — {analyst_name}")
    print(f"{'═'*62}")
    print(f"  Calls evaluated: {len(df)}")
    print(f"  Avg probability: {df['probability'].mean()*100:.1f}%")
    print(f"  Med probability: {df['probability'].median()*100:.1f}%")
    print(f"\n  By call type:")

    for direction in ["buy", "sell", "hold"]:
        sub = df[df["direction"] == direction]
        if sub.empty:
            continue
        print(f"    {direction.upper():<5}: {sub['probability'].mean()*100:.1f}% avg prob. "
              f"({len(sub)} calls)")

    if df["sector"].notna().any():
        print(f"\n  By sector:")
        sector_stats = (df.groupby("sector")["probability"]
                        .agg(["mean", "count"])
                        .sort_values("mean", ascending=False))
        for sector, row in sector_stats.iterrows():
            print(f"    {sector:<20} {row['mean']*100:.1f}%  ({int(row['count'])} calls)")

    # Top e bottom calls
    print(f"\n  Calls with highest probability:")
    top5 = df.nlargest(5, "probability")
    for _, r in top5.iterrows():
        print(f"    {r['ticker']:<8} {r['direction']:<5} {r['rec_date']}  "
              f"→ {r['probability']*100:.1f}%")

    print(f"\n  Calls with lowest probability:")
    bot5 = df.nsmallest(5, "probability")
    for _, r in bot5.iterrows():
        print(f"    {r['ticker']:<8} {r['direction']:<5} {r['rec_date']}  "
              f"→ {r['probability']*100:.1f}%")

    print(f"{'═'*62}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyst Tracker — Risk Engine"
    )
    # Evaluate a specific call
    parser.add_argument("--ticker",    "-t", type=str,  default=None)
    parser.add_argument("--analyst",   "-a", type=str,  default=None)
    parser.add_argument("--direction", "-d", type=str,  default="buy",
                        choices=["buy", "sell", "hold"])
    parser.add_argument("--price",     "-p", type=float, default=None,
                        help="Current asset price")
    parser.add_argument("--target",          type=float, default=None,
                        help="Recommendation price target")
    parser.add_argument("--date",            type=str,   default=None,
                        help="Recommendation date (YYYY-MM-DD)")

    # Batch operations
    parser.add_argument("--calc-all",  action="store_true",
                        help="Calculate risk for all recent calls")
    parser.add_argument("--days",      type=int, default=90,
                        help="Day window for --calc-all (default: 90)")
    parser.add_argument("--profile",   type=str, default=None,
                        help="Risk profile of an analyst")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.calc_all:
        calc_all_recent(days=args.days)

    elif args.profile:
        analyst_risk_profile(args.profile)

    elif args.ticker and args.analyst and args.price:
        evaluate_call(
            ticker=args.ticker,
            analyst_name=args.analyst,
            direction=args.direction,
            price_current=args.price,
            price_target=args.target,
            rec_date=args.date,
            verbose=True,
        )

    else:
        print("\n🚀 Analyst Tracker — Risk Engine")
        print()
        print("Examples:")
        print("  # Evaluate a specific call:")
        print('  python risk_engine.py --ticker NVDA --analyst "Dan Ives" --direction buy --price 182 --target 300')
        print()
        print("  # Calculate risk for all recent calls:")
        print("  python risk_engine.py --calc-all --days 90")
        print()
        print("  # Risk profile of an analyst:")
        print('  python risk_engine.py --profile "Dan Ives"')
        print()
