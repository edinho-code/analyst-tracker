"""
Analyst Tracker — Collector US
================================
Collects historical ratings and price targets from US analysts.

Data sources (in priority order):
  1. Yahoo Finance (yfinance) — full historical upgrades/downgrades with
     firm, grade, action, and price targets going back to ~2012.
  2. StockAnalysis.com (fallback) — scrapes /stocks/{ticker}/ratings/
     but is limited to ~16 most recent ratings per ticker.

Flow v2 (position model):
  1. Fetches ratings for each ticker (Yahoo Finance primary, SA fallback)
  2. Normalizes direction (Strong Buy/Buy → buy, Hold → hold, Sell → sell)
  3. Processes ratings from oldest to newest:
     - No open position → opens new position
     - Same direction → revision (target_up, target_down or reiterate)
     - Different direction → closes current position + opens new
  4. Creates analysts/sources automatically if they don't exist

Usage:
    python collector_us.py                        # all tickers
    python collector_us.py --ticker NVDA          # specific ticker
    python collector_us.py --since 2019-01-01     # filter by date
    python collector_us.py --stats                # summary of collected data

Dependencies:
    pip install requests beautifulsoup4 yfinance
"""

from __future__ import annotations
import sqlite3
import argparse
import time
import re
import sys
from datetime import date, datetime
from urllib.parse import urljoin

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("❌ Run: pip install requests beautifulsoup4")
    sys.exit(1)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

DB_PATH       = "analyst_tracker.db"
BASE_URL      = "https://stockanalysis.com/stocks/{ticker}/ratings/"
SLEEP_BETWEEN = 1.0   # seconds between requests
DEFAULT_SINCE = "2019-01-01"

US_TICKERS = [
    # Big Tech / Semis
    "NVDA", "AAPL", "MSFT", "META", "TSLA",
    "AMZN", "GOOGL", "AMD", "NFLX", "ORCL",
    "AVGO", "QCOM", "INTC", "MU", "TXN",
    # Software / Cloud
    "CRM", "NOW", "ADBE", "SNOW", "UBER", "PLTR",
    # Financials
    "JPM", "GS", "BAC", "V", "MA", "AXP",
    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "PFE",
    # Energy
    "XOM", "CVX",
    # Consumer / Retail
    "WMT", "COST", "HD", "NKE",
    # Industrial
    "CAT", "BA", "HON",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://stockanalysis.com/",
}

RATING_MAP = {
    "strong buy":    "buy",
    "buy":           "buy",
    "outperform":    "buy",
    "overweight":    "buy",
    "accumulate":    "buy",
    "positive":      "buy",
    "market outperform": "buy",
    "sector outperform": "buy",
    "top pick":      "buy",
    "hold":          "hold",
    "neutral":       "hold",
    "market perform": "hold",
    "sector perform": "hold",
    "equal weight":  "hold",
    "equal-weight":  "hold",
    "peer perform":  "hold",
    "in-line":       "hold",
    "inline":        "hold",
    "mixed":         "hold",
    "sector weight": "hold",
    "perform":       "hold",
    "market weight": "hold",
    "fair value":    "hold",
    "sell":          "sell",
    "strong sell":   "sell",
    "underperform":  "sell",
    "underweight":   "sell",
    "reduce":        "sell",
    "negative":      "sell",
    "market underperform": "sell",
    "sector underperform": "sell",
}


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
# NORMALIZATION
# ─────────────────────────────────────────────

def normalize_direction(rating_text: str) -> str | None:
    clean = rating_text.lower().strip()
    if clean in RATING_MAP:
        return RATING_MAP[clean]
    # Try with hyphens replaced by spaces (handles yfinance grades like "Equal-Weight")
    dehyphenated = clean.replace("-", " ")
    if dehyphenated in RATING_MAP:
        return RATING_MAP[dehyphenated]
    for key, val in RATING_MAP.items():
        if key in clean:
            return val
    return None


def parse_price_target(text: str) -> float | None:
    if not text or text.strip() in ("-", "—", ""):
        return None
    parts = re.split(r"→|->|to", text)
    last  = parts[-1].strip()
    match = re.search(r"\$?([\d,]+\.?\d*)", last.replace(",", ""))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def parse_date(text: str) -> str | None:
    if not text:
        return None
    text = text.strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────
# YAHOO FINANCE DATA SOURCE (primary)
# ─────────────────────────────────────────────

def fetch_ratings_yfinance(ticker: str, since: str = DEFAULT_SINCE) -> list[dict]:
    """Fetches full historical ratings from Yahoo Finance upgrades_downgrades.
    Returns list of rating dicts ready for save_ratings()."""
    if not HAS_YFINANCE:
        return []

    try:
        tk = yf.Ticker(ticker)
        ud = tk.upgrades_downgrades
    except Exception as e:
        print(f"yfinance error: {e}")
        return []

    if ud is None or ud.empty:
        return []

    results = []
    for grade_date, row in ud.iterrows():
        rec_date = grade_date.strftime("%Y-%m-%d")
        if rec_date < since:
            continue

        raw_rating = row.get("ToGrade", "")
        direction  = normalize_direction(raw_rating)
        if not direction:
            continue

        firm_name    = row.get("Firm", "Unknown")
        analyst_name = "Research Team"  # yfinance doesn't provide analyst names

        pt_now = row.get("currentPriceTarget", None)
        price_target = float(pt_now) if pt_now and pt_now > 0 else None

        results.append({
            "ticker":       ticker.upper(),
            "analyst_name": analyst_name,
            "firm_name":    firm_name or "Unknown",
            "direction":    direction,
            "price_target": price_target,
            "rec_date":     rec_date,
            "raw_rating":   raw_rating,
            "source_url":   f"https://finance.yahoo.com/quote/{ticker.upper()}/",
        })

    return results


# ─────────────────────────────────────────────
# STOCKANALYSIS.COM SCRAPER (fallback)
# ─────────────────────────────────────────────

def fetch_ratings_page(ticker: str) -> str | None:
    url = BASE_URL.format(ticker=ticker.lower())
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.text
    except requests.HTTPError as e:
        print(f"  ❌ HTTP {e.response.status_code} for {ticker}")
        return None
    except Exception as e:
        print(f"  ❌ Error fetching {ticker}: {e}")
        return None


def parse_ratings_table(html: str, ticker: str, since: str = DEFAULT_SINCE) -> list[dict]:
    soup    = BeautifulSoup(html, "html.parser")
    results = []

    tables = soup.find_all("table")
    target_table = None
    for tbl in tables:
        headers = [th.get_text(strip=True).lower() for th in tbl.find_all("th")]
        if any(h in headers for h in ["analyst", "firm", "rating", "price target", "date"]):
            target_table = tbl
            break

    if not target_table:
        print(f"  ⚠️  Table not found for {ticker} — may be JS rendering")
        return []

    headers = [th.get_text(strip=True).lower() for th in target_table.find_all("th")]
    col_map = {}
    for i, h in enumerate(headers):
        if "analyst" in h:
            col_map["analyst"] = i
        elif "firm" in h or "company" in h or "broker" in h:
            col_map["firm"] = i
        elif "rating" in h:
            col_map["rating"] = i
        elif "price target" in h or "target" in h:
            col_map["price_target"] = i
        elif "date" in h:
            col_map["date"] = i
        elif "action" in h:
            col_map["action"] = i

    rows = target_table.find_all("tr")[1:]

    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) < 3:
            continue

        def cell(key: str) -> str:
            idx = col_map.get(key)
            if idx is None or idx >= len(cells):
                return ""
            return cells[idx].get_text(separator=" ", strip=True)

        raw_date     = cell("date")
        rec_date     = parse_date(raw_date)

        if rec_date and rec_date < since:
            continue

        raw_rating   = cell("rating")
        raw_target   = cell("price_target")
        analyst_name = cell("analyst")
        firm_name    = cell("firm")

        if not firm_name and analyst_name:
            lines = [l.strip() for l in analyst_name.split("\n") if l.strip()]
            if len(lines) >= 2:
                analyst_name = lines[0]
                firm_name    = lines[1]

        direction = normalize_direction(raw_rating)
        if not direction:
            continue

        price_target = parse_price_target(raw_target)

        results.append({
            "ticker":       ticker.upper(),
            "analyst_name": analyst_name or "Research Team",
            "firm_name":    firm_name    or "Unknown",
            "direction":    direction,
            "price_target": price_target,
            "rec_date":     rec_date or date.today().isoformat(),
            "raw_rating":   raw_rating,
            "source_url":   BASE_URL.format(ticker=ticker.lower()),
        })

    return results


# ─────────────────────────────────────────────
# DATABASE HELPERS — ANALYSTS / SOURCES
# ─────────────────────────────────────────────

def get_or_create_source(conn: sqlite3.Connection, firm_name: str) -> int:
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM sources WHERE name LIKE ?", (f"%{firm_name}%",))
    row = cursor.fetchone()
    if row:
        return row["id"]
    cursor.execute(
        """INSERT INTO sources (name, type, country, market, language, url)
           VALUES (?, 'sell_side', 'US', 'US', 'en', ?)""",
        (firm_name, "https://stockanalysis.com/stocks/")
    )
    conn.commit()
    return cursor.lastrowid


def get_or_create_analyst(conn: sqlite3.Connection, analyst_name: str, source_id: int) -> int:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM analysts WHERE name = ? AND source_id = ?",
        (analyst_name, source_id)
    )
    row = cursor.fetchone()
    if row:
        return row["id"]
    cursor.execute(
        "INSERT INTO analysts (name, source_id, role) VALUES (?, ?, 'analyst')",
        (analyst_name, source_id)
    )
    conn.commit()
    return cursor.lastrowid


def get_asset_id(conn: sqlite3.Connection, ticker: str) -> int | None:
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM assets WHERE ticker = ?", (ticker.upper(),))
    row = cursor.fetchone()
    return row["id"] if row else None


def get_price_at_date(conn: sqlite3.Connection, asset_id: int, rec_date: str) -> float | None:
    cursor = conn.cursor()
    cursor.execute(
        """SELECT close FROM price_history
           WHERE asset_id = ? AND date <= ? AND date >= date(?, '-7 days')
           ORDER BY date DESC LIMIT 1""",
        (asset_id, rec_date, rec_date)
    )
    row = cursor.fetchone()
    return row["close"] if row else None


# ─────────────────────────────────────────────
# POSITION HELPERS (by direct ID)
# ─────────────────────────────────────────────

def revision_exists(
    conn: sqlite3.Connection,
    analyst_id: int,
    asset_id: int,
    rec_date: str
) -> bool:
    """Checks if a revision already exists for this analyst+asset on the exact date."""
    cursor = conn.cursor()
    cursor.execute(
        """SELECT r.id FROM recommendations r
           JOIN positions p ON p.id = r.position_id
           WHERE p.analyst_id = ? AND p.asset_id = ? AND r.rec_date = ?
           LIMIT 1""",
        (analyst_id, asset_id, rec_date)
    )
    return cursor.fetchone() is not None


def get_open_position(
    conn: sqlite3.Connection,
    analyst_id: int,
    asset_id: int
) -> sqlite3.Row | None:
    """Returns the analyst's open position on the asset, or None."""
    cursor = conn.cursor()
    cursor.execute(
        """SELECT id, direction, final_target, target_upgrades, target_downgrades
           FROM positions
           WHERE analyst_id = ? AND asset_id = ? AND close_date IS NULL""",
        (analyst_id, asset_id)
    )
    return cursor.fetchone()


def open_position_by_id(
    conn: sqlite3.Connection,
    analyst_id: int,
    asset_id: int,
    direction: str,
    open_date: str,
    price_at_open: float | None,
    price_target: float | None = None,
    source_url: str | None = None,
    notes: str | None = None
) -> int:
    """Creates new position and 'open' record in recommendations. Returns position_id."""
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO positions
           (analyst_id, asset_id, direction, open_date, price_at_open,
            initial_target, final_target, source_url, notes)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (analyst_id, asset_id, direction.lower(), open_date, price_at_open,
         price_target, price_target, source_url, notes)
    )
    position_id = cursor.lastrowid

    cursor.execute(
        """INSERT INTO recommendations
           (position_id, rec_type, rec_date, price_at_rec, direction,
            price_target, source_url, notes)
           VALUES (?,?,?,?,?,?,?,?)""",
        (position_id, "open", open_date, price_at_open, direction.lower(),
         price_target, source_url, notes)
    )
    conn.commit()
    return position_id


def update_position_by_id(
    conn: sqlite3.Connection,
    position_id: int,
    rec_date: str,
    price_at_rec: float | None,
    old_target: float | None,
    new_target: float | None,
    direction: str,
    source_url: str | None = None,
    notes: str | None = None
) -> str:
    """Adds revision to existing position. Returns detected rec_type."""
    cursor = conn.cursor()

    if new_target is None or old_target is None:
        rec_type        = "reiterate"
        target_delta    = None
        effective_target = old_target
    elif new_target > old_target:
        rec_type        = "target_up"
        target_delta    = round(((new_target - old_target) / old_target) * 100, 2)
        effective_target = new_target
    elif new_target < old_target:
        rec_type        = "target_down"
        target_delta    = round(((new_target - old_target) / old_target) * 100, 2)
        effective_target = new_target
    else:
        rec_type        = "reiterate"
        target_delta    = 0.0
        effective_target = old_target

    cursor.execute(
        """INSERT INTO recommendations
           (position_id, rec_type, rec_date, price_at_rec, direction,
            price_target, target_delta_pct, source_url, notes)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (position_id, rec_type, rec_date, price_at_rec, direction,
         effective_target, target_delta, source_url, notes)
    )

    if rec_type == "target_up":
        cursor.execute(
            "UPDATE positions SET final_target=?, target_upgrades=target_upgrades+1 WHERE id=?",
            (new_target, position_id)
        )
    elif rec_type == "target_down":
        cursor.execute(
            "UPDATE positions SET final_target=?, target_downgrades=target_downgrades+1 WHERE id=?",
            (new_target, position_id)
        )

    conn.commit()
    return rec_type


def close_position_by_id(
    conn: sqlite3.Connection,
    position_id: int,
    close_date: str,
    price_at_close: float | None,
    old_direction: str,
    new_direction: str | None = None,
    new_target: float | None = None,
    analyst_id: int | None = None,
    asset_id: int | None = None,
    source_url: str | None = None,
    notes: str | None = None
) -> int | None:
    """
    Closes position and records 'close' revision.
    If new_direction differs from current, opens new position. Returns new position_id or None.
    """
    cursor = conn.cursor()

    cursor.execute(
        """INSERT INTO recommendations
           (position_id, rec_type, rec_date, price_at_rec, direction,
            price_target, source_url, notes)
           VALUES (?,?,?,?,?,?,?,?)""",
        (position_id, "close", close_date, price_at_close,
         new_direction or old_direction, new_target, source_url, notes)
    )
    cursor.execute(
        """UPDATE positions SET
           close_date=?, price_at_close=?, close_reason='rating_change'
           WHERE id=?""",
        (close_date, price_at_close, position_id)
    )
    conn.commit()

    # If direction changed, open new position
    new_pos_id = None
    if new_direction and new_direction.lower() != old_direction.lower() and analyst_id and asset_id:
        new_pos_id = open_position_by_id(
            conn, analyst_id, asset_id, new_direction, close_date,
            price_at_close, new_target, source_url, notes
        )
    return new_pos_id


# ─────────────────────────────────────────────
# PERSISTENCE — POSITIONS AND REVISIONS
# ─────────────────────────────────────────────

def save_ratings(conn: sqlite3.Connection, ratings: list[dict]) -> tuple[int, int]:
    """
    Processes list of ratings and creates/updates positions in the database.
    Ratings are processed from oldest to newest.
    Returns (inserted, duplicates).
    """
    inserted   = 0
    duplicates = 0

    # Process from oldest to newest
    ratings_sorted = sorted(ratings, key=lambda r: r["rec_date"])

    for r in ratings_sorted:
        asset_id = get_asset_id(conn, r["ticker"])
        if not asset_id:
            continue

        source_id  = get_or_create_source(conn, r["firm_name"])
        analyst_id = get_or_create_analyst(conn, r["analyst_name"], source_id)

        # Deduplication: does a revision already exist on this date for this analyst+asset?
        if revision_exists(conn, analyst_id, asset_id, r["rec_date"]):
            duplicates += 1
            continue

        price_at_rec = get_price_at_date(conn, asset_id, r["rec_date"])
        direction    = r["direction"]
        price_target = r["price_target"]
        rec_date     = r["rec_date"]
        source_url   = r["source_url"]
        notes        = f"Raw rating: {r['raw_rating']}"

        open_pos = get_open_position(conn, analyst_id, asset_id)

        if not open_pos:
            # No open position → open new
            open_position_by_id(
                conn, analyst_id, asset_id, direction, rec_date,
                price_at_rec, price_target, source_url, notes
            )
            inserted += 1

        elif open_pos["direction"] == direction:
            # Same direction → revision (target up/down/reiterate)
            update_position_by_id(
                conn, open_pos["id"], rec_date, price_at_rec,
                old_target=open_pos["final_target"],
                new_target=price_target,
                direction=direction,
                source_url=source_url,
                notes=notes
            )
            inserted += 1

        else:
            # Direction changed → close + open new
            close_position_by_id(
                conn, open_pos["id"], rec_date, price_at_rec,
                old_direction=open_pos["direction"],
                new_direction=direction,
                new_target=price_target,
                analyst_id=analyst_id,
                asset_id=asset_id,
                source_url=source_url,
                notes=notes
            )
            inserted += 1

    return inserted, duplicates


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def run_collector(ticker: str, since: str = DEFAULT_SINCE):
    conn   = get_connection()
    ticker = ticker.upper()

    asset_id = get_asset_id(conn, ticker)
    if not asset_id:
        print(f"  ⚠️  Ticker {ticker} not found in database.")
        print(f"     Run analyst_tracker_setup.py first.")
        conn.close()
        return 0, 0

    print(f"  📥 {ticker:<6} ", end="", flush=True)

    # Primary: Yahoo Finance (full history)
    ratings = fetch_ratings_yfinance(ticker, since=since)
    source = "yf"

    # Fallback: StockAnalysis.com (limited to ~16 recent ratings)
    if not ratings:
        html = fetch_ratings_page(ticker)
        if html:
            ratings = parse_ratings_table(html, ticker, since=since)
            source = "sa"

    if not ratings:
        print(f"0 ratings found (no data since {since})")
        conn.close()
        return 0, 0

    inserted, duplicates = save_ratings(conn, ratings)
    print(f"{len(ratings):>4} ratings [{source}] → {inserted:>4} inserted/revised, {duplicates:>4} duplicates")

    conn.close()
    return inserted, duplicates


def run_all(since: str = DEFAULT_SINCE):
    src_label = "Yahoo Finance" if HAS_YFINANCE else "StockAnalysis.com"
    print(f"\n🚀 Collector US v2 — {len(US_TICKERS)} tickers | since {since} | source: {src_label}\n")

    total_inserted   = 0
    total_duplicates = 0
    failed           = []

    for i, ticker in enumerate(US_TICKERS, 1):
        prefix = f"[{i:>2}/{len(US_TICKERS)}]"
        print(f"  {prefix} ", end="", flush=True)

        try:
            ins, dup = run_collector(ticker, since=since)
            total_inserted   += ins
            total_duplicates += dup
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            failed.append(ticker)

        if i < len(US_TICKERS):
            time.sleep(SLEEP_BETWEEN)

    print(f"\n{'─'*55}")
    print(f"  ✅ Total inserted:    {total_inserted:>5} revisions/positions")
    print(f"  ♻️  Duplicates:        {total_duplicates:>5}")
    if failed:
        print(f"  ❌ Failures:         {', '.join(failed)}")
    print(f"{'─'*55}")

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT COUNT(*) as n FROM analysts
           WHERE source_id IN (SELECT id FROM sources WHERE country='US')"""
    )
    n_analysts = cursor.fetchone()["n"]
    cursor.execute("SELECT COUNT(*) as n FROM sources WHERE country='US'")
    n_sources = cursor.fetchone()["n"]
    cursor.execute("SELECT COUNT(*) as n FROM positions p JOIN assets a ON a.id=p.asset_id WHERE a.country='US'")
    n_positions = cursor.fetchone()["n"]
    conn.close()

    print(f"  📊 US positions in database: {n_positions}")
    print(f"  👤 US analysts:             {n_analysts}")
    print(f"  🏢 US firms:                {n_sources}")
    print(f"{'─'*55}\n")


# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────

def show_stats():
    conn   = get_connection()
    cursor = conn.cursor()

    print(f"\n{'─'*65}")
    print(f"  📊  Collector US — Summary (position model)")
    print(f"{'─'*65}")

    cursor.execute(
        """SELECT a.ticker,
                  COUNT(DISTINCT p.id) as posicoes,
                  SUM(CASE WHEN p.direction='buy'  THEN 1 ELSE 0 END) as buys,
                  SUM(CASE WHEN p.direction='hold' THEN 1 ELSE 0 END) as holds,
                  SUM(CASE WHEN p.direction='sell' THEN 1 ELSE 0 END) as sells,
                  SUM(CASE WHEN p.close_date IS NULL THEN 1 ELSE 0 END) as abertas,
                  MIN(p.open_date) as earliest,
                  MAX(p.open_date) as latest
           FROM positions p
           JOIN assets a ON a.id = p.asset_id
           WHERE a.country = 'US' AND a.ticker != 'SPY'
           GROUP BY a.ticker
           ORDER BY posicoes DESC"""
    )
    rows = cursor.fetchall()

    if not rows:
        print("  No US positions found.")
        print("  Run: python collector_us.py\n")
        conn.close()
        return

    print(f"  {'Ticker':<8} {'Pos.':>5} {'Buy':>5} {'Hold':>5} {'Sell':>5} {'Open':>5}  {'Period'}")
    print(f"  {'─'*8} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5}  {'─'*22}")

    for row in rows:
        period = f"{row['earliest']} → {row['latest']}"
        print(f"  {row['ticker']:<8} {row['posicoes']:>5} {row['buys']:>5} "
              f"{row['holds']:>5} {row['sells']:>5} {row['abertas']:>5}  {period}")


    cursor.execute(
        """SELECT COUNT(DISTINCT an.id) as analysts, COUNT(DISTINCT s.id) as firms
           FROM positions p
           JOIN analysts an ON an.id = p.analyst_id
           JOIN sources   s  ON s.id  = an.source_id
           JOIN assets    a  ON a.id  = p.asset_id
           WHERE a.country = 'US'"""
    )
    agg = cursor.fetchone()
    print(f"\n  Unique analysts: {agg['analysts']}  |  Firms: {agg['firms']}")
    print(f"{'─'*65}\n")

    conn.close()


def show_top_analysts(ticker: str = None, top_n: int = 15):
    conn   = get_connection()
    cursor = conn.cursor()

    if ticker:
        cursor.execute(
            """SELECT an.name as analyst, s.name as firm,
                      COUNT(DISTINCT p.id) as posicoes,
                      SUM(CASE WHEN p.direction='buy' THEN 1 ELSE 0 END) as buys
               FROM positions p
               JOIN analysts an ON an.id = p.analyst_id
               JOIN sources   s  ON s.id  = an.source_id
               JOIN assets    a  ON a.id  = p.asset_id
               WHERE a.ticker = ? AND a.country = 'US'
               GROUP BY an.id
               ORDER BY posicoes DESC
               LIMIT ?""",
            (ticker.upper(), top_n)
        )
    else:
        cursor.execute(
            """SELECT an.name as analyst, s.name as firm,
                      COUNT(DISTINCT p.id) as posicoes,
                      SUM(CASE WHEN p.direction='buy' THEN 1 ELSE 0 END) as buys
               FROM positions p
               JOIN analysts an ON an.id = p.analyst_id
               JOIN sources   s  ON s.id  = an.source_id
               JOIN assets    a  ON a.id  = p.asset_id
               WHERE a.country = 'US'
               GROUP BY an.id
               ORDER BY posicoes DESC
               LIMIT ?""",
            (top_n,)
        )

    rows = cursor.fetchall()
    conn.close()

    scope = f" — {ticker.upper()}" if ticker else ""
    print(f"\n{'─'*60}")
    print(f"  🏆  Top {top_n} US Analysts{scope}")
    print(f"{'─'*60}")
    print(f"  {'Analyst':<28} {'Firm':<18} {'Pos.':>5} {'Buys':>5}")
    print(f"  {'─'*28} {'─'*18} {'─'*5} {'─'*5}")
    for row in rows:
        print(f"  {row['analyst'][:27]:<28} {row['firm'][:17]:<18} {row['posicoes']:>5} {row['buys']:>5}")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyst Tracker — Collector US v2 (position model)"
    )
    parser.add_argument("--ticker",  "-t", type=str, default=None,
                        help=f"US Ticker (e.g.: NVDA). Default: all ({', '.join(US_TICKERS)})")
    parser.add_argument("--since",   "-s", type=str, default=DEFAULT_SINCE,
                        help=f"Start date YYYY-MM-DD. Default: {DEFAULT_SINCE}")
    parser.add_argument("--stats",   action="store_true",
                        help="Summary of US positions in database")
    parser.add_argument("--top",     "-n", type=int, default=15,
                        help="Top N analysts (default: 15)")
    parser.add_argument("--analysts", action="store_true",
                        help="List top analysts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.stats:
        show_stats()

    elif args.analysts:
        show_top_analysts(ticker=args.ticker, top_n=args.top)

    elif args.ticker:
        print(f"\n🚀 Collector US v2 — {args.ticker.upper()}")
        run_collector(ticker=args.ticker, since=args.since)
        show_stats()

    else:
        run_all(since=args.since)
        print("✅ Next step: python scoring_engine.py")
