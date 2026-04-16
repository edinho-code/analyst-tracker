"""
Analyst Tracker — Setup v2
===========================
Redesigned schema with positions as the central evaluation unit.

Previous model (wrong):
  Each row in recommendations = independent call
  → Dan Ives BUY NVDA Jan, Mar, Jun = 3 calls = hit rate inflation

New model (correct):
  positions    = the thesis (analyst is LONG/SHORT/NEUTRAL on an asset)
  recommendations = revisions within a position (target updates, reiterations)
  performance  = computed on the entire POSITION, from open to close

What opens a new position:
  - Rating change (Neutral→Buy, Buy→Sell)
  - First rating on an asset
  - Resumption after dropping coverage

What is just a revision (within the position):
  - Same rating with new target (e.g.: BUY NVDA $300 → BUY NVDA $450)
  - Reiteration without target change

Conviction score (new):
  - Analyst raising target = conviction increasing (+)
  - Analyst cutting target but maintaining Buy = conviction decreasing (-)
  - Series of raises followed by downgrade = local top signal

Usage:
    python analyst_tracker_setup.py

Dependencies:
    pip install yfinance pandas
"""

from __future__ import annotations
import sqlite3
import os
from datetime import datetime, date

DB_PATH = "analyst_tracker.db"


# ─────────────────────────────────────────────
# SCHEMA v2
# ─────────────────────────────────────────────

SCHEMA = """
-- ── BASE ENTITIES ────────────────────────────────────────────────

-- Sources: brokerages, channels, newsletters, etc.
CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL CHECK(type IN (
                    'corretora','consultoria','canal_youtube',
                    'newsletter','banco','fundo','influencer','sell_side')),
    country     TEXT NOT NULL,
    market      TEXT NOT NULL CHECK(market IN ('BR','US','EU','LATAM','GLOBAL')),
    language    TEXT DEFAULT 'pt',
    url         TEXT,
    description TEXT,
    active      INTEGER DEFAULT 1,
    created_at  TEXT DEFAULT (date('now'))
);

-- Individual analysts within each source
CREATE TABLE IF NOT EXISTS analysts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    source_id  INTEGER NOT NULL REFERENCES sources(id),
    role       TEXT,
    active     INTEGER DEFAULT 1,
    twitter    TEXT,
    linkedin   TEXT,
    created_at TEXT DEFAULT (date('now'))
);

-- Financial assets (stocks, ETFs, indices)
CREATE TABLE IF NOT EXISTS assets (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker   TEXT NOT NULL UNIQUE,
    name     TEXT NOT NULL,
    exchange TEXT NOT NULL,
    sector   TEXT,
    country  TEXT NOT NULL,
    currency TEXT DEFAULT 'USD'
);

-- OHLCV price history
CREATE TABLE IF NOT EXISTS price_history (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL REFERENCES assets(id),
    date     TEXT NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL NOT NULL,
    volume   INTEGER,
    UNIQUE(asset_id, date)
);

-- ── POSITION MODEL ──────────────────────────────────────────────

-- Positions: the analyst's thesis on an asset (evaluation unit)
-- A position is opened when the analyst initiates or changes rating.
-- It is closed when rating changes again or coverage is dropped.
CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Who and what
    analyst_id      INTEGER NOT NULL REFERENCES analysts(id),
    asset_id        INTEGER NOT NULL REFERENCES assets(id),

    -- Position direction (rating at the time of opening)
    direction       TEXT NOT NULL CHECK(direction IN ('buy','sell','hold','neutral')),

    -- Position open date and price
    open_date       TEXT NOT NULL,
    price_at_open   REAL,

    -- Close date and price (NULL = position still open)
    -- Close occurs when: rating changes, coverage dropped, or we evaluate
    close_date      TEXT,
    price_at_close  REAL,

    -- Close reason
    close_reason    TEXT CHECK(close_reason IN (
                        'rating_change',   -- analyst changed from Buy to Hold, etc.
                        'coverage_dropped',-- analyst dropped coverage
                        'horizon_reached', -- horizon deadline reached
                        'target_hit',      -- price target hit
                        'evaluated'        -- periodic evaluation (position still active)
                    )),

    -- Declared horizon (days) — if analyst specified a timeframe
    horizon_days    INTEGER,

    -- Conviction tracking
    -- How many times target was raised / cut during the position
    target_upgrades   INTEGER DEFAULT 0,
    target_downgrades INTEGER DEFAULT 0,

    -- Initial and final price target (to compute conviction journey)
    initial_target  REAL,
    final_target    REAL,

    -- URL of the first publication that opened the position
    source_url      TEXT,
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Revisions: each update within a position (target changes, reiterations)
-- These are NOT independent calls — they are chapters of the same thesis
CREATE TABLE IF NOT EXISTS recommendations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id     INTEGER NOT NULL REFERENCES positions(id),

    -- Revision type
    rec_type        TEXT NOT NULL CHECK(rec_type IN (
                        'open',        -- opens the position (first record)
                        'target_up',   -- raises price target, same rating → conviction +
                        'target_down', -- cuts price target, same rating → conviction -
                        'reiterate',   -- reiterates without target change
                        'close'        -- closes position (rating change or dropped)
                    )),

    -- Date and price at the time of the revision
    rec_date        TEXT NOT NULL,
    price_at_rec    REAL,

    -- Rating and target at this point
    direction       TEXT NOT NULL CHECK(direction IN ('buy','sell','hold','neutral')),
    price_target    REAL,

    -- Target delta vs previous revision (NULL on open)
    target_delta_pct REAL,

    -- Link to the original publication
    source_url      TEXT,
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ── PERFORMANCE ────────────────────────────────────────────────────

-- Performance computed per POSITION (not per individual revision)
-- Return is measured from open_date to close_date of the position
CREATE TABLE IF NOT EXISTS performance (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id     INTEGER NOT NULL REFERENCES positions(id) UNIQUE,

    -- Evaluation dates
    eval_date       TEXT NOT NULL,

    -- Prices
    price_open      REAL,
    price_eval      REAL,

    -- Gross position return
    return_pct      REAL,

    -- Continuous scores (0→1, >1 = exceeded the target)
    direction_score REAL,   -- how far it went in the right direction
    target_score    REAL,   -- how close it got to the final price target

    -- Alpha vs benchmark
    alpha_vs_spy    REAL,
    alpha_vs_ibov   REAL,

    -- Days to reach target (NULL if not reached)
    days_to_target  INTEGER,

    -- Conviction score of the complete position
    -- Positive = analyst raised conviction over time
    -- Negative = cut conviction
    conviction_score REAL,

    -- Legacy binary (derived from continuous scores)
    hit_direction   INTEGER,
    hit_target      INTEGER,

    updated_at      TEXT DEFAULT (datetime('now'))
);

-- ── AGGREGATED SCORES ───────────────────────────────────────────────

-- Aggregated score per analyst (computed periodically)
-- Based on closed or evaluated positions, not individual revisions
CREATE TABLE IF NOT EXISTS analyst_scores (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    analyst_id          INTEGER NOT NULL REFERENCES analysts(id),
    calc_date           TEXT NOT NULL,

    -- Metrics per position (correct evaluation unit)
    total_positions     INTEGER DEFAULT 0,
    open_positions      INTEGER DEFAULT 0,
    closed_positions    INTEGER DEFAULT 0,

    -- Performance
    hit_rate            REAL,   -- % positions that got direction right
    target_acc          REAL,   -- % positions that hit target
    avg_alpha           REAL,   -- average alpha vs benchmark
    consistency         REAL,   -- stability over time

    -- Continuous scores
    avg_direction_score REAL,
    avg_target_score    REAL,

    -- Conviction
    avg_conviction      REAL,   -- does analyst tend to raise or cut conviction?
    avg_target_upgrades REAL,   -- average upgrades per position
    avg_target_downgrades REAL, -- average downgrades per position

    -- Legacy
    wins                INTEGER DEFAULT 0,
    losses              INTEGER DEFAULT 0,

    updated_at          TEXT DEFAULT (datetime('now')),
    UNIQUE(analyst_id, calc_date)
);

-- ── CLIPPING (BR collection) ───────────────────────────────────────────

CREATE TABLE IF NOT EXISTS clipping_raw (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    source_name  TEXT NOT NULL,
    article_url  TEXT NOT NULL,
    article_date TEXT,
    title        TEXT,
    body_text    TEXT,
    url_hash     TEXT UNIQUE,
    scraped_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS clipping_extractions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    clipping_id     INTEGER NOT NULL REFERENCES clipping_raw(id),
    raw_json        TEXT NOT NULL,

    -- Fields extracted by the LLM
    analyst_name    TEXT,
    source_house    TEXT,
    ticker          TEXT,
    direction       TEXT,
    price_target    REAL,
    currency        TEXT DEFAULT 'BRL',
    rec_date        TEXT,
    horizon_days    INTEGER,

    -- Revision type detected by the LLM
    rec_type        TEXT DEFAULT 'open'
                    CHECK(rec_type IN ('open','target_up','target_down','reiterate','close')),

    notes           TEXT,
    confidence      REAL,
    status          TEXT DEFAULT 'pending'
                    CHECK(status IN ('pending','approved','rejected','imported')),

    -- Reference to position/revision created after approval
    position_id     INTEGER REFERENCES positions(id),
    rec_id          INTEGER REFERENCES recommendations(id),

    extracted_at    TEXT DEFAULT (datetime('now')),
    reviewed_at     TEXT
);

-- ── RISK ASSESSMENTS ───────────────────────────────────────────────

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
);

-- ── INDEXES ────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_pos_analyst  ON positions(analyst_id);
CREATE INDEX IF NOT EXISTS idx_pos_asset    ON positions(asset_id);
CREATE INDEX IF NOT EXISTS idx_pos_open     ON positions(open_date);
CREATE INDEX IF NOT EXISTS idx_pos_close    ON positions(close_date);
CREATE INDEX IF NOT EXISTS idx_pos_dir      ON positions(direction);
CREATE INDEX IF NOT EXISTS idx_rec_position ON recommendations(position_id);
CREATE INDEX IF NOT EXISTS idx_rec_date     ON recommendations(rec_date);
CREATE INDEX IF NOT EXISTS idx_rec_type     ON recommendations(rec_type);
CREATE INDEX IF NOT EXISTS idx_perf_pos     ON performance(position_id);
CREATE INDEX IF NOT EXISTS idx_price_asset  ON price_history(asset_id);
CREATE INDEX IF NOT EXISTS idx_price_date   ON price_history(date);
CREATE INDEX IF NOT EXISTS idx_score_analyst ON analyst_scores(analyst_id);
CREATE INDEX IF NOT EXISTS idx_clip_ticker  ON clipping_raw(ticker);
CREATE INDEX IF NOT EXISTS idx_clip_status  ON clipping_extractions(status);
"""


# ─────────────────────────────────────────────
# SEED DATA
# ─────────────────────────────────────────────

SEED_SOURCES = [
    # Brasil
    ("XP Investimentos",     "corretora",   "BR", "BR",     "pt", "https://xpi.com.br"),
    ("BTG Pactual",          "banco",       "BR", "BR",     "pt", "https://btgpactual.com"),
    ("Itaú BBA",             "banco",       "BR", "BR",     "pt", "https://itaubba.com.br"),
    ("Safra",                "banco",       "BR", "BR",     "pt", "https://safra.com.br"),
    ("Genial Investimentos",  "corretora",  "BR", "BR",     "pt", "https://genialinvestimentos.com.br"),
    ("Nord Research",        "consultoria", "BR", "BR",     "pt", "https://nordresearch.com.br"),
    ("Empiricus",            "newsletter",  "BR", "BR",     "pt", "https://empiricus.com.br"),
    ("Hike Capital",         "canal_youtube","BR","BR",     "pt", "https://youtube.com/@hikecapital"),
    # EUA / Global
    ("Goldman Sachs",        "banco",       "US", "GLOBAL", "en", "https://goldmansachs.com"),
    ("Morgan Stanley",       "banco",       "US", "GLOBAL", "en", "https://morganstanley.com"),
    ("JP Morgan",            "banco",       "US", "GLOBAL", "en", "https://jpmorgan.com"),
    ("Bank of America",      "banco",       "US", "US",     "en", "https://bankofamerica.com"),
    ("Wedbush Securities",   "sell_side",   "US", "US",     "en", "https://wedbush.com"),
    ("Bernstein",            "sell_side",   "US", "US",     "en", "https://bernstein.com"),
    ("ARK Invest",           "fundo",       "US", "US",     "en", "https://ark-invest.com"),
    ("Motley Fool",          "newsletter",  "US", "US",     "en", "https://fool.com"),
    ("The Street",           "newsletter",  "US", "US",     "en", "https://thestreet.com"),
]

SEED_ANALYSTS = [
    ("Equipe Research XP",    "XP Investimentos",  "research team"),
    ("Equipe Research BTG",   "BTG Pactual",       "research team"),
    ("Equipe Research Itaú",  "Itaú BBA",          "research team"),
    ("Felipe Miranda",        "Empiricus",          "chief strategist"),
    ("Equipe Nord",           "Nord Research",      "research team"),
    ("Dan Ives",              "Wedbush Securities", "senior analyst - tech"),
    ("Stacy Rasgon",          "Bernstein",          "senior analyst - semiconductor"),
    ("Cathie Wood",           "ARK Invest",         "portfolio manager"),
    ("David Kostin",          "Goldman Sachs",      "chief equity strategist"),
    ("Michael Wilson",        "Morgan Stanley",     "chief equity strategist"),
]

SEED_ASSETS = [
    # US — Big Tech / Semis
    ("NVDA",  "NVIDIA Corporation",        "NASDAQ", "Technology",  "US", "USD"),
    ("AAPL",  "Apple Inc.",                "NASDAQ", "Technology",  "US", "USD"),
    ("MSFT",  "Microsoft Corporation",     "NASDAQ", "Technology",  "US", "USD"),
    ("AMZN",  "Amazon.com Inc.",           "NASDAQ", "Technology",  "US", "USD"),
    ("TSLA",  "Tesla Inc.",                "NASDAQ", "Automotive",  "US", "USD"),
    ("META",  "Meta Platforms Inc.",       "NASDAQ", "Technology",  "US", "USD"),
    ("GOOGL", "Alphabet Inc.",             "NASDAQ", "Technology",  "US", "USD"),
    ("AMD",   "Advanced Micro Devices",    "NASDAQ", "Technology",  "US", "USD"),
    ("NFLX",  "Netflix Inc.",              "NASDAQ", "Technology",  "US", "USD"),
    ("ORCL",  "Oracle Corporation",        "NYSE",   "Technology",  "US", "USD"),
    ("AVGO",  "Broadcom Inc.",             "NASDAQ", "Technology",  "US", "USD"),
    ("QCOM",  "Qualcomm Inc.",             "NASDAQ", "Technology",  "US", "USD"),
    ("INTC",  "Intel Corporation",         "NASDAQ", "Technology",  "US", "USD"),
    ("MU",    "Micron Technology",         "NASDAQ", "Technology",  "US", "USD"),
    ("TXN",   "Texas Instruments",         "NASDAQ", "Technology",  "US", "USD"),
    # US — Software / Cloud
    ("CRM",   "Salesforce Inc.",           "NYSE",   "Technology",  "US", "USD"),
    ("NOW",   "ServiceNow Inc.",           "NYSE",   "Technology",  "US", "USD"),
    ("ADBE",  "Adobe Inc.",                "NASDAQ", "Technology",  "US", "USD"),
    ("SNOW",  "Snowflake Inc.",            "NYSE",   "Technology",  "US", "USD"),
    ("UBER",  "Uber Technologies",         "NYSE",   "Technology",  "US", "USD"),
    ("PLTR",  "Palantir Technologies",     "NYSE",   "Technology",  "US", "USD"),
    # US — Financials
    ("JPM",   "JPMorgan Chase",            "NYSE",   "Financials",  "US", "USD"),
    ("GS",    "Goldman Sachs",             "NYSE",   "Financials",  "US", "USD"),
    ("BAC",   "Bank of America",           "NYSE",   "Financials",  "US", "USD"),
    ("V",     "Visa Inc.",                 "NYSE",   "Financials",  "US", "USD"),
    ("MA",    "Mastercard Inc.",           "NYSE",   "Financials",  "US", "USD"),
    ("AXP",   "American Express",          "NYSE",   "Financials",  "US", "USD"),
    # US — Healthcare
    ("LLY",   "Eli Lilly",                 "NYSE",   "Healthcare",  "US", "USD"),
    ("UNH",   "UnitedHealth Group",        "NYSE",   "Healthcare",  "US", "USD"),
    ("JNJ",   "Johnson & Johnson",         "NYSE",   "Healthcare",  "US", "USD"),
    ("ABBV",  "AbbVie Inc.",               "NYSE",   "Healthcare",  "US", "USD"),
    ("PFE",   "Pfizer Inc.",               "NYSE",   "Healthcare",  "US", "USD"),
    # US — Energy
    ("XOM",   "Exxon Mobil",               "NYSE",   "Energy",      "US", "USD"),
    ("CVX",   "Chevron Corp.",             "NYSE",   "Energy",      "US", "USD"),
    # US — Consumer / Retail
    ("WMT",   "Walmart Inc.",              "NYSE",   "Consumer",    "US", "USD"),
    ("COST",  "Costco Wholesale",          "NASDAQ", "Consumer",    "US", "USD"),
    ("HD",    "Home Depot",                "NYSE",   "Consumer",    "US", "USD"),
    ("NKE",   "Nike Inc.",                 "NYSE",   "Consumer",    "US", "USD"),
    # US — Industrials
    ("CAT",   "Caterpillar Inc.",          "NYSE",   "Industrials", "US", "USD"),
    ("BA",    "Boeing Co.",                "NYSE",   "Industrials", "US", "USD"),
    ("HON",   "Honeywell International",   "NASDAQ", "Industrials", "US", "USD"),
    # US — Benchmarks
    ("SPY",   "SPDR S&P 500 ETF",          "NYSE",   "ETF",         "US", "USD"),
    # Brasil — Petróleo / Energia
    ("PETR4.SA", "Petrobras PN",           "BVMF",  "Energy",      "BR", "BRL"),
    ("PRIO3.SA", "PetroRio ON",            "BVMF",  "Energy",      "BR", "BRL"),
    ("CPLE3.SA", "Copel ON",               "BVMF",  "Energy",      "BR", "BRL"),
    ("VBBR3.SA", "Vibra Energia ON",       "BVMF",  "Energy",      "BR", "BRL"),
    # Brasil — Mineração / Siderurgia
    ("VALE3.SA", "Vale S.A. ON",           "BVMF",  "Materials",   "BR", "BRL"),
    ("GGBR4.SA", "Gerdau PN",              "BVMF",  "Materials",   "BR", "BRL"),
    ("CSNA3.SA", "CSN ON",                 "BVMF",  "Materials",   "BR", "BRL"),
    ("SUZB3.SA", "Suzano S.A. ON",         "BVMF",  "Materials",   "BR", "BRL"),
    # Brasil — Bancos / Financeiro
    ("ITUB4.SA", "Itaú Unibanco PN",       "BVMF",  "Financials",  "BR", "BRL"),
    ("BBDC4.SA", "Bradesco PN",            "BVMF",  "Financials",  "BR", "BRL"),
    ("BBAS3.SA", "Banco do Brasil ON",     "BVMF",  "Financials",  "BR", "BRL"),
    ("B3SA3.SA", "B3 S.A. ON",             "BVMF",  "Financials",  "BR", "BRL"),
    # Brasil — Consumo / Varejo
    ("WEGE3.SA", "WEG S.A. ON",            "BVMF",  "Industrials", "BR", "BRL"),
    ("ABEV3.SA", "Ambev S.A. ON",          "BVMF",  "Consumer",    "BR", "BRL"),
    ("LREN3.SA", "Lojas Renner ON",        "BVMF",  "Consumer",    "BR", "BRL"),
    ("RENT3.SA", "Localiza ON",            "BVMF",  "Consumer",    "BR", "BRL"),
    ("MGLU3.SA", "Magazine Luiza ON",      "BVMF",  "Consumer",    "BR", "BRL"),
    # Brasil — Saúde
    ("RDOR3.SA", "Rede D'Or ON",           "BVMF",  "Healthcare",  "BR", "BRL"),
    ("HAPV3.SA", "Hapvida ON",             "BVMF",  "Healthcare",  "BR", "BRL"),
    # Brasil — Proteína / Agro
    ("BEEF3.SA", "Minerva Foods ON",        "BVMF",  "Consumer",    "BR", "BRL"),
    ("SLCE3.SA", "SLC Agrícola ON",        "BVMF",  "Materials",   "BR", "BRL"),
    # Brasil — Industrial
    ("RAIL3.SA", "Rumo Logística ON",       "BVMF",  "Industrials", "BR", "BRL"),
    # Brasil — Benchmark
    ("^BVSP",    "Ibovespa",               "BVMF",  "Index",       "BR", "BRL"),
]


# ─────────────────────────────────────────────
# POSITION HELPERS
# ─────────────────────────────────────────────

def open_position(
    conn: sqlite3.Connection,
    analyst_name: str,
    ticker: str,
    direction: str,
    open_date: str,
    price_at_open: float,
    price_target: float = None,
    horizon_days: int = None,
    source_url: str = None,
    notes: str = None
) -> int:
    """
    Opens a new position for an analyst on an asset.
    Also creates the first record in recommendations (rec_type='open').
    Returns the position_id.

    Example:
        pos_id = open_position(conn, "Dan Ives", "NVDA", "buy",
                               "2023-01-15", 145.0, price_target=220.0,
                               horizon_days=365,
                               notes="AI supercycle — NVDA beneficiary #1")
    """
    cursor = conn.cursor()

    # Buscar IDs
    cursor.execute("SELECT id FROM analysts WHERE name LIKE ?", (f"%{analyst_name}%",))
    analyst = cursor.fetchone()
    if not analyst:
        raise ValueError(f"Analyst not found: '{analyst_name}'")

    cursor.execute("SELECT id FROM assets WHERE ticker = ?", (ticker.upper(),))
    asset = cursor.fetchone()
    if not asset:
        raise ValueError(f"Asset not found: '{ticker}'")

    # Check if there's already an open position on the same asset
    cursor.execute(
        """SELECT id, direction FROM positions
           WHERE analyst_id=? AND asset_id=? AND close_date IS NULL""",
        (analyst["id"], asset["id"])
    )
    existing = cursor.fetchone()
    if existing:
        print(f"  ⚠️  {analyst_name} already has open position on {ticker} "
              f"({existing['direction'].upper()}) — use update_position() to revise")
        return existing["id"]

    # Create position
    cursor.execute(
        """INSERT INTO positions
           (analyst_id, asset_id, direction, open_date, price_at_open,
            horizon_days, initial_target, final_target, source_url, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (analyst["id"], asset["id"], direction.lower(), open_date, price_at_open,
         horizon_days, price_target, price_target, source_url, notes)
    )
    position_id = cursor.lastrowid

    # Create opening revision
    cursor.execute(
        """INSERT INTO recommendations
           (position_id, rec_type, rec_date, price_at_rec, direction,
            price_target, source_url, notes)
           VALUES (?,?,?,?,?,?,?,?)""",
        (position_id, "open", open_date, price_at_open, direction.lower(),
         price_target, source_url, notes)
    )
    conn.commit()
    print(f"  ✅ Position opened: {analyst_name} → {ticker} {direction.upper()} @ "
          f"${price_at_open} | target: {'$'+str(price_target) if price_target else 'no target'}")
    return position_id


def update_position(
    conn: sqlite3.Connection,
    position_id: int,
    rec_date: str,
    price_at_rec: float,
    new_target: float = None,
    notes: str = None,
    source_url: str = None
) -> str:
    """
    Updates an existing position with a new target revision.
    Automatically detects whether it's target_up, target_down or reiterate.
    Returns the detected rec_type.

    Example:
        # Dan Ives raises NVDA target from $220 to $300
        update_position(conn, pos_id, "2023-05-25", 320.0,
                        new_target=300.0, notes="Blackwell demand beat estimates")
    """
    cursor = conn.cursor()

    cursor.execute(
        """SELECT p.direction, p.final_target, p.target_upgrades, p.target_downgrades,
                  a.name as analyst_name, ast.ticker
           FROM positions p
           JOIN analysts a   ON a.id   = p.analyst_id
           JOIN assets   ast ON ast.id = p.asset_id
           WHERE p.id=?""",
        (position_id,)
    )
    pos = cursor.fetchone()
    if not pos:
        raise ValueError(f"Position #{position_id} not found")

    old_target = pos["final_target"]

    # Detect revision type
    if new_target is None or old_target is None:
        rec_type = "reiterate"
        target_delta_pct = None
    elif new_target > old_target:
        rec_type = "target_up"
        target_delta_pct = round(((new_target - old_target) / old_target) * 100, 2)
    elif new_target < old_target:
        rec_type = "target_down"
        target_delta_pct = round(((new_target - old_target) / old_target) * 100, 2)
    else:
        rec_type = "reiterate"
        target_delta_pct = 0.0

    # Insert revision
    cursor.execute(
        """INSERT INTO recommendations
           (position_id, rec_type, rec_date, price_at_rec, direction,
            price_target, target_delta_pct, source_url, notes)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (position_id, rec_type, rec_date, price_at_rec, pos["direction"],
         new_target or old_target, target_delta_pct, source_url, notes)
    )

    # Update counters and final target on the position
    if rec_type == "target_up":
        cursor.execute(
            "UPDATE positions SET final_target=?, target_upgrades=target_upgrades+1 WHERE id=?",
            (new_target, position_id)
        )
        icon = "📈"
    elif rec_type == "target_down":
        cursor.execute(
            "UPDATE positions SET final_target=?, target_downgrades=target_downgrades+1 WHERE id=?",
            (new_target, position_id)
        )
        icon = "📉"
    else:
        icon = "↗️"

    conn.commit()

    delta_str = f" ({'+' if target_delta_pct and target_delta_pct>0 else ''}{target_delta_pct:.1f}%)" if target_delta_pct else ""
    print(f"  {icon} Position #{position_id} revised ({rec_type}): "
          f"{pos['analyst_name']} → {pos['ticker']} "
          f"target: ${old_target} → ${new_target or old_target}{delta_str}")
    return rec_type


def close_position(
    conn: sqlite3.Connection,
    position_id: int,
    close_date: str,
    price_at_close: float,
    new_direction: str = None,
    new_target: float = None,
    close_reason: str = "rating_change",
    source_url: str = None,
    notes: str = None
) -> int | None:
    """
    Closes an existing position.
    If new_direction is provided (e.g.: Buy → Hold), opens a new position automatically.
    Returns the new position_id if a new position was opened, None otherwise.

    Example:
        # Dan Ives downgrades NVDA from Buy to Hold
        new_pos = close_position(conn, pos_id, "2024-08-15", 109.0,
                                 new_direction="hold",
                                 notes="Valuation stretched post-rally",
                                 close_reason="rating_change")
    """
    cursor = conn.cursor()

    cursor.execute(
        """SELECT p.direction, p.analyst_id, p.asset_id,
                  a.name as analyst_name, ast.ticker
           FROM positions p
           JOIN analysts a   ON a.id   = p.analyst_id
           JOIN assets   ast ON ast.id = p.asset_id
           WHERE p.id=?""",
        (position_id,)
    )
    pos = cursor.fetchone()
    if not pos:
        raise ValueError(f"Position #{position_id} not found")

    # Record closing revision
    cursor.execute(
        """INSERT INTO recommendations
           (position_id, rec_type, rec_date, price_at_rec, direction,
            price_target, source_url, notes)
           VALUES (?,?,?,?,?,?,?,?)""",
        (position_id, "close", close_date, price_at_close,
         new_direction or pos["direction"], new_target, source_url, notes)
    )

    # Close position
    cursor.execute(
        """UPDATE positions SET
           close_date=?, price_at_close=?, close_reason=?
           WHERE id=?""",
        (close_date, price_at_close, close_reason, position_id)
    )
    conn.commit()
    print(f"  🔴 Position #{position_id} closed: {pos['analyst_name']} → "
          f"{pos['ticker']} ({pos['direction'].upper()} → "
          f"{(new_direction or 'dropped').upper()}) @ ${price_at_close}")

    # If rating changed, open new position
    new_pos_id = None
    if new_direction and new_direction.lower() != pos["direction"]:
        new_pos_id = open_position(
            conn,
            analyst_name=pos["analyst_name"],
            ticker=pos["ticker"],
            direction=new_direction.lower(),
            open_date=close_date,
            price_at_open=price_at_close,
            price_target=new_target,
            source_url=source_url,
            notes=notes,
        )

    return new_pos_id


def get_open_positions(conn: sqlite3.Connection, analyst_name: str = None) -> list:
    """Lists open positions (no close_date)."""
    cursor = conn.cursor()
    if analyst_name:
        cursor.execute(
            """SELECT p.id, a.name as analyst, ast.ticker, p.direction,
                      p.open_date, p.price_at_open, p.final_target,
                      p.target_upgrades, p.target_downgrades
               FROM positions p
               JOIN analysts a   ON a.id   = p.analyst_id
               JOIN assets   ast ON ast.id = p.asset_id
               WHERE p.close_date IS NULL AND a.name LIKE ?
               ORDER BY p.open_date DESC""",
            (f"%{analyst_name}%",)
        )
    else:
        cursor.execute(
            """SELECT p.id, a.name as analyst, ast.ticker, p.direction,
                      p.open_date, p.price_at_open, p.final_target,
                      p.target_upgrades, p.target_downgrades
               FROM positions p
               JOIN analysts a   ON a.id   = p.analyst_id
               JOIN assets   ast ON ast.id = p.asset_id
               WHERE p.close_date IS NULL
               ORDER BY a.name, p.open_date DESC"""
        )
    return cursor.fetchall()


def get_position_history(conn: sqlite3.Connection, position_id: int) -> dict:
    """Returns complete position with all revisions."""
    cursor = conn.cursor()

    cursor.execute(
        """SELECT p.*, a.name as analyst_name, ast.ticker, ast.name as asset_name
           FROM positions p
           JOIN analysts a   ON a.id   = p.analyst_id
           JOIN assets   ast ON ast.id = p.asset_id
           WHERE p.id=?""",
        (position_id,)
    )
    pos = cursor.fetchone()

    cursor.execute(
        "SELECT * FROM recommendations WHERE position_id=? ORDER BY rec_date",
        (position_id,)
    )
    revisions = cursor.fetchall()

    return {"position": dict(pos), "revisions": [dict(r) for r in revisions]}


# ─────────────────────────────────────────────
# MAIN SETUP
# ─────────────────────────────────────────────

def create_database(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.commit()
    print(f"✅ Schema v2 created in: {db_path}")
    return conn


def insert_seed_sources(conn: sqlite3.Connection):
    cursor = conn.cursor()
    inserted = 0
    for (name, type_, country, market, lang, url) in SEED_SOURCES:
        cursor.execute("SELECT id FROM sources WHERE name=?", (name,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO sources (name,type,country,market,language,url) VALUES(?,?,?,?,?,?)",
                (name, type_, country, market, lang, url)
            )
            inserted += 1
    conn.commit()
    print(f"✅ {inserted} sources inserted")


def insert_seed_analysts(conn: sqlite3.Connection):
    cursor = conn.cursor()
    inserted = 0
    for (name, source_name, role) in SEED_ANALYSTS:
        cursor.execute("SELECT id FROM sources WHERE name=?", (source_name,))
        source = cursor.fetchone()
        if not source:
            print(f"  ⚠️  Source not found: {source_name}")
            continue
        cursor.execute(
            "SELECT id FROM analysts WHERE name=? AND source_id=?",
            (name, source["id"])
        )
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO analysts (name,source_id,role) VALUES(?,?,?)",
                (name, source["id"], role)
            )
            inserted += 1
    conn.commit()
    print(f"✅ {inserted} analysts inserted")


def insert_seed_assets(conn: sqlite3.Connection):
    cursor = conn.cursor()
    inserted = 0
    for (ticker, name, exchange, sector, country, currency) in SEED_ASSETS:
        cursor.execute("SELECT id FROM assets WHERE ticker=?", (ticker,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO assets (ticker,name,exchange,sector,country,currency) VALUES(?,?,?,?,?,?)",
                (ticker, name, exchange, sector, country, currency)
            )
            inserted += 1
    conn.commit()
    print(f"✅ {inserted} assets inserted")


def show_summary(conn: sqlite3.Connection):
    cursor = conn.cursor()
    print("\n─── Database Summary ───────────────────────────")
    tables = [
        "sources", "analysts", "assets",
        "positions", "recommendations",
        "performance", "analyst_scores", "price_history",
        "clipping_raw", "clipping_extractions"
    ]
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) as n FROM {table}")
            print(f"  {table:<26} {cursor.fetchone()['n']:>6} records")
        except Exception:
            pass
    print("────────────────────────────────────────────────")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 Analyst Tracker v2 — Initializing database\n")

    conn = create_database(DB_PATH)
    insert_seed_sources(conn)
    insert_seed_analysts(conn)
    insert_seed_assets(conn)

    # ── Example usage with positions ──────────────────────────
    print("\n── Example: complete position cycle ──────────────")

    # Dan Ives opens BUY on NVDA
    pos1 = open_position(
        conn, "Dan Ives", "NVDA", "buy", "2023-01-15", 145.0,
        price_target=220.0, horizon_days=365,
        notes="AI supercycle — NVDA beneficiary #1"
    )

    # Raises target after strong results
    update_position(
        conn, pos1, "2023-05-25", 320.0, new_target=350.0,
        notes="Data center demand accelerating post-ChatGPT"
    )

    # Raises target again — conviction increasing
    update_position(
        conn, pos1, "2023-08-23", 435.0, new_target=550.0,
        notes="Blackwell pipeline stronger than expected"
    )

    # Reiterates in Jan/2024
    update_position(
        conn, pos1, "2024-01-10", 480.0,
        notes="Reiterate — AI capex cycle intact"
    )

    # Downgrades in Aug/2024 (valuation concern)
    new_pos = close_position(
        conn, pos1, "2024-08-15", 109.0,
        new_direction="hold",
        notes="Valuation stretched — moving to sidelines",
        close_reason="rating_change"
    )

    # Show position history
    print("\n── Position History ───────────────────────────────")
    hist = get_position_history(conn, pos1)
    p    = hist["position"]
    print(f"  Position #{p['id']}: {p['analyst_name']} → {p['ticker']} {p['direction'].upper()}")
    print(f"  Opened: {p['open_date']} @ ${p['price_at_open']}")
    print(f"  Closed: {p['close_date']} @ ${p['price_at_close']}")
    print(f"  Target upgrades: {p['target_upgrades']} | downgrades: {p['target_downgrades']}")
    print(f"  Initial target: ${p['initial_target']} → final: ${p['final_target']}")
    print(f"\n  Revisions ({len(hist['revisions'])}):")
    for r in hist["revisions"]:
        icon = {"open":"🟢","target_up":"📈","target_down":"📉","reiterate":"↗️","close":"🔴"}.get(r["rec_type"],"·")
        tgt  = f" → ${r['price_target']}" if r["price_target"] else ""
        delt = f" ({'+' if r['target_delta_pct'] and r['target_delta_pct']>0 else ''}{r['target_delta_pct']:.1f}%)" if r["target_delta_pct"] else ""
        print(f"    {icon} {r['rec_date']}  {r['rec_type']:<12}  @ ${r['price_at_rec']}{tgt}{delt}")

    show_summary(conn)
    conn.close()
    print(f"\n✅ Done! Database v2: {os.path.abspath(DB_PATH)}")
    print("   Next step: python price_fetcher.py")
