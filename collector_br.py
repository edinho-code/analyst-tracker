"""
Analyst Tracker — Collector BR (Clipping + LLM Extractor)
===========================================================
Collects Brazilian analyst recommendations via financial press clipping.

Flow:
  1. Scraper searches articles by ticker from BR sources (InfoMoney, Seu Dinheiro, etc.)
  2. Filters articles with recommendation language
  3. LLM (Claude) extracts structure: ticker, direction, price target, analyst, date
  4. Saves to database with confidence score — below 0.75 goes to manual review

Usage:
    python collector_br.py --ticker VALE3
    python collector_br.py --ticker PETR4 --since 2022-01-01
    python collector_br.py --all                        # all BR tickers in database
    python collector_br.py --review                     # review pending extractions
    python collector_br.py --approve <clipping_id>      # manually approve
    python collector_br.py --reject  <clipping_id>      # manually reject

Dependencies:
    pip install requests beautifulsoup4 anthropic
    Environment variable: ANTHROPIC_API_KEY
"""

from __future__ import annotations
import sqlite3
import argparse
import json
import time
import os
import re
import sys
import hashlib
from datetime import date, datetime
from urllib.parse import quote_plus, urljoin

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("❌ Run: pip install requests beautifulsoup4")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("❌ Run: pip install anthropic")
    sys.exit(1)

DB_PATH            = "analyst_tracker.db"
CONFIDENCE_AUTO    = 0.75   # above this → auto-approved
CONFIDENCE_REVIEW  = 0.50   # between 0.50–0.75 → manual review
SLEEP_BETWEEN      = 1.5    # seconds between requests (respect rate limit)
MAX_ARTICLES       = 10     # articles per ticker per source

# BR tickers we collect
BR_TICKERS = [
    # Oil / Energy
    "PETR4", "PRIO3", "CPLE3", "VBBR3",
    # Mining / Steel
    "VALE3", "GGBR4", "CSNA3",
    # Banks / Financial
    "ITUB4", "BBDC4", "BBAS3", "B3SA3",
    # Consumer / Retail
    "WEGE3", "ABEV3", "LREN3", "RENT3",
    # Healthcare
    "RDOR3", "HAPV3",
    # Protein / Agro
    "BEEF3", "SLCE3",
    # Other
    "SUZB3", "MGLU3", "RAIL3",
]

# Keywords indicating recommendation presence in article
REC_KEYWORDS = [
    "preço-alvo", "preco alvo", "preço alvo",
    "recomenda", "recomendação", "recomendacao",
    "compra", "venda", "neutro", "manter",
    "eleva recomendação", "rebaixa", "upgrade", "downgrade",
    "price target", "target price",
    "potencial de alta", "potencial de valorização",
    "BBA", "BTG", "XP Investimentos", "Safra", "Genial",
    "Goldman Sachs", "Morgan Stanley", "JPMorgan",
]


# ─────────────────────────────────────────────
# CONNECTION AND MIGRATION
# ─────────────────────────────────────────────

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def migrate_clipping_table(conn: sqlite3.Connection):
    """
    Creates clipping table if it doesn't exist (schema v2 with position_id and rec_type).
    Idempotent — safe to run multiple times.
    """
    conn.execute("""
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
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clipping_extractions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            clipping_id     INTEGER NOT NULL REFERENCES clipping_raw(id),
            raw_json        TEXT NOT NULL,
            analyst_name    TEXT,
            source_house    TEXT,
            ticker          TEXT,
            direction       TEXT,
            price_target    REAL,
            currency        TEXT DEFAULT 'BRL',
            rec_date        TEXT,
            horizon_days    INTEGER,
            rec_type        TEXT DEFAULT 'open'
                            CHECK(rec_type IN ('open','target_up','target_down','reiterate','close')),
            notes           TEXT,
            confidence      REAL,
            status          TEXT DEFAULT 'pending'
                            CHECK(status IN ('pending', 'approved', 'rejected', 'imported')),
            position_id     INTEGER REFERENCES positions(id),
            rec_id          INTEGER REFERENCES recommendations(id),
            extracted_at    TEXT DEFAULT (datetime('now')),
            reviewed_at     TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_clip_ticker ON clipping_raw(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_clip_status ON clipping_extractions(status)")
    conn.commit()

    # Add new columns if table existed without them (migration)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(clipping_extractions)")
    cols = {row[1] for row in cursor.fetchall()}
    if "rec_type" not in cols:
        conn.execute("ALTER TABLE clipping_extractions ADD COLUMN rec_type TEXT DEFAULT 'open'")
    if "position_id" not in cols:
        conn.execute("ALTER TABLE clipping_extractions ADD COLUMN position_id INTEGER REFERENCES positions(id)")
    conn.commit()


# ─────────────────────────────────────────────
# SCRAPERS BY SOURCE
# ─────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9",
}


def url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def scrape_infomoney(ticker: str, since: str = "2022-01-01") -> list[dict]:
    """
    InfoMoney — searches articles by ticker via search.
    URL: https://www.infomoney.com.br/busca/?q=VALE3+recomendação
    """
    results = []
    query   = f"{ticker} recomendação preço-alvo analista"
    url     = f"https://www.infomoney.com.br/busca/?q={quote_plus(query)}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # InfoMoney search results — articles in <article> or <div class="search-item">
        articles = soup.find_all("article", limit=MAX_ARTICLES)
        if not articles:
            articles = soup.find_all("div", class_=re.compile(r"search|result|item"), limit=MAX_ARTICLES)

        for art in articles:
            a_tag = art.find("a", href=True)
            if not a_tag:
                continue

            href  = a_tag["href"]
            title = a_tag.get_text(strip=True) or art.get_text(strip=True)[:120]

            if not href.startswith("http"):
                href = "https://www.infomoney.com.br" + href

            results.append({
                "source": "InfoMoney",
                "url":    href,
                "title":  title,
                "date":   None,
            })

    except Exception as e:
        print(f"    ⚠️  InfoMoney scrape error: {e}")

    return results


def scrape_seudinheiro(ticker: str, since: str = "2022-01-01") -> list[dict]:
    """
    Seu Dinheiro — searches by ticker.
    """
    results = []
    query   = f"{ticker} recomendação preço alvo"
    url     = f"https://www.seudinheiro.com/?s={quote_plus(query)}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for art in soup.find_all("article", limit=MAX_ARTICLES):
            a_tag = art.find("a", href=True)
            if not a_tag:
                continue
            href  = a_tag["href"]
            title = art.find("h2") or art.find("h3")
            title = title.get_text(strip=True) if title else a_tag.get_text(strip=True)

            # Try to get date
            time_tag = art.find("time")
            art_date = None
            if time_tag and time_tag.get("datetime"):
                art_date = time_tag["datetime"][:10]

            if art_date and art_date < since:
                continue

            results.append({
                "source": "Seu Dinheiro",
                "url":    href,
                "title":  title[:120],
                "date":   art_date,
            })

    except Exception as e:
        print(f"    ⚠️  Seu Dinheiro scrape error: {e}")

    return results


def scrape_moneytimes(ticker: str, since: str = "2022-01-01") -> list[dict]:
    """
    Money Times — covers upgrades/downgrades quickly.
    """
    results = []
    query   = f"{ticker} recomendação"
    url     = f"https://www.moneytimes.com.br/?s={quote_plus(query)}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for art in soup.find_all("article", limit=MAX_ARTICLES):
            a_tag = art.find("a", href=True)
            if not a_tag:
                continue
            href  = a_tag["href"]
            title = art.find("h2") or art.find("h3")
            title = title.get_text(strip=True) if title else a_tag.get_text(strip=True)

            time_tag = art.find("time")
            art_date = None
            if time_tag and time_tag.get("datetime"):
                art_date = time_tag["datetime"][:10]

            if art_date and art_date < since:
                continue

            results.append({
                "source": "Money Times",
                "url":    href,
                "title":  title[:120],
                "date":   art_date,
            })

    except Exception as e:
        print(f"    ⚠️  Money Times scrape error: {e}")

    return results


def scrape_einvestidor(ticker: str, since: str = "2022-01-01") -> list[dict]:
    """
    E-Investidor (Estadão) — good coverage of blue chips.
    """
    results = []
    query   = f"{ticker} preço-alvo analista"
    url     = f"https://einvestidor.estadao.com.br/?s={quote_plus(query)}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for art in soup.find_all("article", limit=MAX_ARTICLES):
            a_tag = art.find("a", href=True)
            if not a_tag:
                continue
            href  = a_tag["href"]
            title = art.find("h2") or art.find("h3")
            title = title.get_text(strip=True) if title else a_tag.get_text(strip=True)

            time_tag = art.find("time")
            art_date = None
            if time_tag and time_tag.get("datetime"):
                art_date = time_tag["datetime"][:10]

            if art_date and art_date < since:
                continue

            results.append({
                "source": "E-Investidor",
                "url":    href,
                "title":  title[:120],
                "date":   art_date,
            })

    except Exception as e:
        print(f"    ⚠️  E-Investidor scrape error: {e}")

    return results


def scrape_guiadoinvestidor(ticker: str, since: str = "2022-01-01") -> list[dict]:
    """
    Guia do Investidor — aggregates price targets, great source.
    """
    results = []
    query   = f"{ticker} preço alvo recomendação"
    url     = f"https://guiadoinvestidor.com.br/?s={quote_plus(query)}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for art in soup.find_all("article", limit=MAX_ARTICLES):
            a_tag = art.find("a", href=True)
            if not a_tag:
                continue
            href  = a_tag["href"]
            title = art.find("h2") or art.find("h3")
            title = title.get_text(strip=True) if title else a_tag.get_text(strip=True)

            time_tag = art.find("time")
            art_date = None
            if time_tag and time_tag.get("datetime"):
                art_date = time_tag["datetime"][:10]

            if art_date and art_date < since:
                continue

            results.append({
                "source": "Guia do Investidor",
                "url":    href,
                "title":  title[:120],
                "date":   art_date,
            })

    except Exception as e:
        print(f"    ⚠️  Guia do Investidor scrape error: {e}")

    return results


ALL_SCRAPERS = [
    scrape_infomoney,
    scrape_seudinheiro,
    scrape_moneytimes,
    scrape_einvestidor,
    scrape_guiadoinvestidor,
]


# ─────────────────────────────────────────────
# FETCH ARTICLE BODY
# ─────────────────────────────────────────────

def fetch_article_body(url: str) -> str | None:
    """
    Downloads and extracts the main text of an article.
    Returns None if it fails or if the article has no relevant content.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts, styles, navs
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "iframe"]):
            tag.decompose()

        # Try to extract the main article body
        body = (
            soup.find("article") or
            soup.find("div", class_=re.compile(r"content|article|post|body|texto")) or
            soup.find("main")
        )

        text = body.get_text(separator=" ", strip=True) if body else soup.get_text(separator=" ", strip=True)

        # Clean extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Only return if it has minimum content
        return text[:6000] if len(text) > 200 else None

    except Exception:
        return None


def has_recommendation_keywords(text: str) -> bool:
    """Checks if text contains recommendation language."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in REC_KEYWORDS)


# ─────────────────────────────────────────────
# LLM EXTRACTION (CLAUDE)
# ─────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a specialized extractor of financial analyst recommendations.

Analyze the text below and extract ALL analyst recommendations mentioned.

For each recommendation found, return a JSON object with:
- analyst_name: name of the brokerage/bank/analyst (e.g.: "Itaú BBA", "BTG Pactual", "XP Investimentos")
- ticker: stock code (e.g.: "VALE3", "PETR4")
- direction: "buy", "sell" or "hold" (compra=buy, venda=sell, neutro/manter=hold)
- price_target: numeric price target (null if not mentioned)
- currency: "BRL" or "USD"
- rec_date: recommendation date in YYYY-MM-DD format (use article date if no specific date)
- horizon_days: timeframe in days (null if not mentioned; 365 if it says "12 meses")
- notes: relevant excerpt from text supporting the extraction (max 200 chars)
- confidence: number between 0 and 1 indicating your certainty in the extraction
  - 1.0: exact date, named analyst, explicit price target
  - 0.8: named analyst, clear direction, no price target
  - 0.6: implicit analyst or ambiguous direction
  - below 0.5: very uncertain — preferable not to include

Return ONLY valid JSON in the format:
{
  "recommendations": [
    { ...fields above... },
    ...
  ],
  "article_date": "YYYY-MM-DD or null"
}

If there are no clear recommendations, return:
{ "recommendations": [], "article_date": null }

Do NOT include explanations, markdown or text outside the JSON.

TICKER SEARCHED: {ticker}
ARTICLE DATE: {article_date}

ARTICLE TEXT:
{article_text}
"""


def extract_with_llm(
    ticker: str,
    article_text: str,
    article_date: str | None,
    client: anthropic.Anthropic
) -> dict | None:
    """
    Uses Claude to extract structured recommendations from article text.
    Returns parsed JSON or None on failure.
    """
    prompt = EXTRACTION_PROMPT.format(
        ticker=ticker,
        article_date=article_date or "unknown",
        article_text=article_text[:5000]
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()

        # Clean possible markdown
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"    ⚠️  JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"    ⚠️  LLM error: {e}")
        return None


# ─────────────────────────────────────────────
# SAVE TO DATABASE
# ─────────────────────────────────────────────

def save_clipping(conn: sqlite3.Connection, ticker: str, article: dict, body: str) -> int | None:
    """Saves raw article. Returns id or None if duplicate."""
    h = url_hash(article["url"])
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM clipping_raw WHERE url_hash = ?", (h,))
    if cursor.fetchone():
        return None  # already exists

    cursor.execute(
        """INSERT INTO clipping_raw (ticker, source_name, article_url, article_date, title, body_text, url_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (ticker, article["source"], article["url"], article.get("date"),
         article.get("title"), body, h)
    )
    conn.commit()
    return cursor.lastrowid


def save_extraction(conn: sqlite3.Connection, clipping_id: int, rec: dict, raw_json: str):
    """Saves an LLM extraction. Initial status depends on confidence."""
    confidence = rec.get("confidence", 0)
    status     = "approved" if confidence >= CONFIDENCE_AUTO else "pending"

    conn.execute(
        """INSERT INTO clipping_extractions
           (clipping_id, raw_json, analyst_name, source_house, ticker, direction,
            price_target, currency, rec_date, horizon_days, notes, confidence, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            clipping_id, raw_json,
            rec.get("analyst_name"), rec.get("analyst_name"),
            rec.get("ticker"), rec.get("direction"),
            rec.get("price_target"), rec.get("currency", "BRL"),
            rec.get("rec_date"), rec.get("horizon_days"),
            rec.get("notes"), confidence, status
        )
    )
    conn.commit()


def _get_or_create_analyst_id(conn: sqlite3.Connection, analyst_name: str, source_house: str) -> int:
    """Finds or creates analyst + source. Returns analyst_id."""
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM analysts WHERE name LIKE ?", (f"%{analyst_name}%",))
    analyst = cursor.fetchone()
    if analyst:
        return analyst["id"]

    # Create or find source
    cursor.execute("SELECT id FROM sources WHERE name LIKE ?", (f"%{source_house}%",))
    source = cursor.fetchone()
    if not source:
        cursor.execute(
            "INSERT INTO sources (name, type, country, market, language) VALUES (?, 'banco', 'BR', 'BR', 'pt')",
            (source_house,)
        )
        conn.commit()
        source_id = cursor.lastrowid
    else:
        source_id = source["id"]

    cursor.execute(
        "INSERT INTO analysts (name, source_id, role) VALUES (?, ?, 'research team')",
        (analyst_name, source_id)
    )
    conn.commit()
    return cursor.lastrowid


def _get_price_at_date(conn: sqlite3.Connection, asset_id: int, rec_date: str) -> float | None:
    cursor = conn.cursor()
    cursor.execute(
        """SELECT close FROM price_history
           WHERE asset_id = ? AND date <= ? AND date >= date(?, '-7 days')
           ORDER BY date DESC LIMIT 1""",
        (asset_id, rec_date, rec_date)
    )
    row = cursor.fetchone()
    return row["close"] if row else None


def _open_position_br(
    conn: sqlite3.Connection,
    analyst_id: int, asset_id: int,
    direction: str, rec_date: str,
    price_at_rec: float | None, price_target: float | None,
    source_url: str | None, notes: str | None
) -> tuple[int, int]:
    """Opens new position and 'open' revision. Returns (position_id, rec_id)."""
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO positions
           (analyst_id, asset_id, direction, open_date, price_at_open,
            initial_target, final_target, source_url, notes)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (analyst_id, asset_id, direction.lower(), rec_date, price_at_rec,
         price_target, price_target, source_url, notes)
    )
    position_id = cursor.lastrowid
    cursor.execute(
        """INSERT INTO recommendations
           (position_id, rec_type, rec_date, price_at_rec, direction,
            price_target, source_url, notes)
           VALUES (?,?,?,?,?,?,?,?)""",
        (position_id, "open", rec_date, price_at_rec, direction.lower(),
         price_target, source_url, notes)
    )
    rec_id = cursor.lastrowid
    conn.commit()
    return position_id, rec_id


def _update_position_br(
    conn: sqlite3.Connection,
    position_id: int, old_target: float | None,
    rec_type_hint: str, direction: str,
    rec_date: str, price_at_rec: float | None,
    price_target: float | None, source_url: str | None, notes: str | None
) -> tuple[str, int]:
    """Adds revision to existing position. Returns (rec_type, rec_id)."""
    cursor = conn.cursor()

    # Detect rec_type from LLM hint and target change
    if rec_type_hint in ("target_up", "target_down", "reiterate"):
        rec_type = rec_type_hint
    elif price_target and old_target:
        if price_target > old_target:
            rec_type = "target_up"
        elif price_target < old_target:
            rec_type = "target_down"
        else:
            rec_type = "reiterate"
    else:
        rec_type = "reiterate"

    target_delta = None
    if price_target and old_target and old_target != 0:
        target_delta = round(((price_target - old_target) / old_target) * 100, 2)

    effective_target = price_target or old_target
    cursor.execute(
        """INSERT INTO recommendations
           (position_id, rec_type, rec_date, price_at_rec, direction,
            price_target, target_delta_pct, source_url, notes)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (position_id, rec_type, rec_date, price_at_rec, direction,
         effective_target, target_delta, source_url, notes)
    )
    rec_id = cursor.lastrowid

    if rec_type == "target_up" and price_target:
        cursor.execute(
            "UPDATE positions SET final_target=?, target_upgrades=target_upgrades+1 WHERE id=?",
            (price_target, position_id)
        )
    elif rec_type == "target_down" and price_target:
        cursor.execute(
            "UPDATE positions SET final_target=?, target_downgrades=target_downgrades+1 WHERE id=?",
            (price_target, position_id)
        )
    conn.commit()
    return rec_type, rec_id


def import_approved_to_recommendations(conn: sqlite3.Connection) -> int:
    """
    Imports approved extractions to positions + recommendations (v2 model).
    Checks if an open position already exists for analyst+asset and creates revision if so.
    Returns number of imported records.
    """
    cursor   = conn.cursor()
    imported = 0

    cursor.execute(
        """SELECT e.*, c.article_url
           FROM clipping_extractions e
           JOIN clipping_raw c ON c.id = e.clipping_id
           WHERE e.status = 'approved' AND e.rec_id IS NULL"""
    )
    rows = cursor.fetchall()

    for row in rows:
        if not row["analyst_name"] or not row["ticker"] or not row["direction"]:
            continue

        analyst_id = _get_or_create_analyst_id(conn, row["analyst_name"], row["source_house"] or row["analyst_name"])

        cursor.execute("SELECT id FROM assets WHERE ticker = ?", (row["ticker"],))
        asset = cursor.fetchone()
        if not asset:
            # Try with .SA suffix
            cursor.execute("SELECT id FROM assets WHERE ticker = ?", (row["ticker"] + ".SA",))
            asset = cursor.fetchone()
        if not asset:
            print(f"  ⚠️  Asset not found: {row['ticker']} — skipping")
            continue

        asset_id     = asset["id"]
        price_at_rec = _get_price_at_date(conn, asset_id, row["rec_date"]) if row["rec_date"] else None
        source_url   = row["article_url"]
        notes        = row["notes"]
        direction    = row["direction"].lower()
        price_target = row["price_target"]
        rec_date     = row["rec_date"]
        rec_type_hint = row.get("rec_type") or "open"

        # Check if open position already exists
        cursor.execute(
            """SELECT id, direction, final_target FROM positions
               WHERE analyst_id=? AND asset_id=? AND close_date IS NULL""",
            (analyst_id, asset_id)
        )
        open_pos = cursor.fetchone()

        position_id = None
        rec_id      = None

        if not open_pos:
            # Open new position
            position_id, rec_id = _open_position_br(
                conn, analyst_id, asset_id, direction, rec_date,
                price_at_rec, price_target, source_url, notes
            )
        elif open_pos["direction"] == direction:
            # Same direction → revision
            _, rec_id = _update_position_br(
                conn, open_pos["id"], open_pos["final_target"],
                rec_type_hint, direction, rec_date, price_at_rec,
                price_target, source_url, notes
            )
            position_id = open_pos["id"]
        else:
            # Different direction → close current position
            cursor.execute(
                """INSERT INTO recommendations
                   (position_id, rec_type, rec_date, price_at_rec, direction,
                    price_target, source_url, notes)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (open_pos["id"], "close", rec_date, price_at_rec,
                 direction, price_target, source_url, notes)
            )
            cursor.execute(
                """UPDATE positions SET close_date=?, price_at_close=?, close_reason='rating_change'
                   WHERE id=?""",
                (rec_date, price_at_rec, open_pos["id"])
            )
            conn.commit()
            # Open new position
            position_id, rec_id = _open_position_br(
                conn, analyst_id, asset_id, direction, rec_date,
                price_at_rec, price_target, source_url, notes
            )

        # Mark extraction as imported
        cursor.execute(
            """UPDATE clipping_extractions
               SET status='imported', position_id=?, rec_id=?, reviewed_at=datetime('now')
               WHERE id=?""",
            (position_id, rec_id, row["id"])
        )
        conn.commit()
        imported += 1

    return imported


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_collector(ticker: str, since: str = "2022-01-01"):
    """Complete pipeline for a single ticker."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set.")
        print("   export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    conn   = get_connection()
    migrate_clipping_table(conn)

    ticker = ticker.upper()
    print(f"\n📰 Collecting clippings for {ticker} (since {since})\n")

    articles_found    = 0
    articles_relevant = 0
    extractions_total = 0
    extractions_auto  = 0

    for scraper in ALL_SCRAPERS:
        source_name = scraper.__name__.replace("scrape_", "").replace("_", " ").title()
        print(f"  🔍 {source_name}...", end=" ", flush=True)

        articles = scraper(ticker, since=since)
        print(f"{len(articles)} articles found")
        articles_found += len(articles)

        for article in articles:
            time.sleep(SLEEP_BETWEEN)

            # Download article body
            body = fetch_article_body(article["url"])
            if not body:
                continue

            # Filter by recommendation keywords
            if not has_recommendation_keywords(body):
                continue

            articles_relevant += 1

            # Save raw clipping
            clipping_id = save_clipping(conn, ticker, article, body)
            if not clipping_id:
                continue  # duplicate article

            # Extract with LLM
            result = extract_with_llm(
                ticker,
                body,
                article.get("date"),
                client
            )

            if not result or not result.get("recommendations"):
                continue

            # Update article date if LLM detected it
            if result.get("article_date") and not article.get("date"):
                conn.execute(
                    "UPDATE clipping_raw SET article_date=? WHERE id=?",
                    (result["article_date"], clipping_id)
                )
                conn.commit()

            for rec in result["recommendations"]:
                # Only save recommendations above minimum threshold
                if rec.get("confidence", 0) < CONFIDENCE_REVIEW:
                    continue

                # Only save if it's about the searched ticker (or close ticker)
                rec_ticker = rec.get("ticker", "").upper()
                if rec_ticker and rec_ticker != ticker and ticker not in rec_ticker:
                    continue

                # Ensure correct ticker
                rec["ticker"] = ticker

                save_extraction(conn, clipping_id, rec, json.dumps(rec, ensure_ascii=False))
                extractions_total += 1

                if rec.get("confidence", 0) >= CONFIDENCE_AUTO:
                    extractions_auto += 1

    # Auto-import approved extractions
    imported = import_approved_to_recommendations(conn)

    print(f"\n{'─'*55}")
    print(f"  Articles found:         {articles_found:>4}")
    print(f"  With recommendations:   {articles_relevant:>4}")
    print(f"  Extractions saved:      {extractions_total:>4}")
    print(f"  Auto-approved:          {extractions_auto:>4}")
    print(f"  Imported to database:   {imported:>4}")
    pending = extractions_total - extractions_auto
    if pending > 0:
        print(f"  ⚠️  Pending review:       {pending:>4}  → run: python collector_br.py --review")
    print(f"{'─'*55}\n")

    conn.close()


def run_all(since: str = "2022-01-01"):
    """Collects for all BR tickers."""
    for ticker in BR_TICKERS:
        run_collector(ticker, since=since)
        time.sleep(2)


# ─────────────────────────────────────────────
# MANUAL REVIEW
# ─────────────────────────────────────────────

def show_pending_reviews():
    """Lists extractions pending review."""
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """SELECT e.id, e.ticker, e.analyst_name, e.direction,
                  e.price_target, e.currency, e.rec_date,
                  e.confidence, e.notes, c.article_url, c.source_name
           FROM clipping_extractions e
           JOIN clipping_raw c ON c.id = e.clipping_id
           WHERE e.status = 'pending'
           ORDER BY e.confidence DESC, e.extracted_at DESC"""
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("\n✅ No extractions pending review.\n")
        return

    print(f"\n{'═'*80}")
    print(f"  📋  EXTRACTIONS PENDING REVIEW ({len(rows)} items)")
    print(f"{'═'*80}")

    for row in rows:
        pt  = f"R$ {row['price_target']:.2f}" if row["price_target"] else "no target"
        dir_icon = "📈" if row["direction"] == "buy" else "📉" if row["direction"] == "sell" else "➡️"
        print(f"\n  ID #{row['id']}  [{row['confidence']:.0%} confidence]")
        print(f"  {dir_icon} {row['analyst_name']} → {row['ticker']} {row['direction'].upper()} @ {pt}")
        print(f"  📅 {row['rec_date'] or 'unknown date'}  |  {row['source_name']}")
        print(f"  💬 {row['notes'][:100] if row['notes'] else '—'}")
        print(f"  🔗 {row['article_url'][:80]}")
        print(f"  {'─'*76}")
        print(f"  Approve:  python collector_br.py --approve {row['id']}")
        print(f"  Reject:   python collector_br.py --reject  {row['id']}")

    print(f"\n{'═'*80}\n")


def approve_extraction(extraction_id: int):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE clipping_extractions SET status='approved', reviewed_at=datetime('now') WHERE id=?",
        (extraction_id,)
    )
    if cursor.rowcount:
        conn.commit()
        imported = import_approved_to_recommendations(conn)
        print(f"✅ Extraction #{extraction_id} approved and imported ({imported} rec inserted).")
    else:
        print(f"❌ Extraction #{extraction_id} not found.")
    conn.close()


def reject_extraction(extraction_id: int):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE clipping_extractions SET status='rejected', reviewed_at=datetime('now') WHERE id=?",
        (extraction_id,)
    )
    if cursor.rowcount:
        conn.commit()
        print(f"🗑️  Extraction #{extraction_id} rejected.")
    else:
        print(f"❌ Extraction #{extraction_id} not found.")
    conn.close()


def show_clipping_stats():
    """General clipping summary."""
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as n FROM clipping_raw")
    total_articles = cursor.fetchone()["n"]

    cursor.execute(
        """SELECT status, COUNT(*) as n FROM clipping_extractions GROUP BY status"""
    )
    status_counts = {row["status"]: row["n"] for row in cursor.fetchall()}

    cursor.execute(
        """SELECT ticker, COUNT(*) as n FROM clipping_extractions
           WHERE status IN ('approved', 'imported')
           GROUP BY ticker ORDER BY n DESC"""
    )
    by_ticker = cursor.fetchall()

    conn.close()

    print(f"\n{'─'*50}")
    print(f"  📊  Clipping BR — Summary")
    print(f"{'─'*50}")
    print(f"  Articles collected:  {total_articles:>5}")
    print(f"  Approved:            {status_counts.get('approved', 0):>5}")
    print(f"  Imported:            {status_counts.get('imported', 0):>5}")
    print(f"  Pending review:      {status_counts.get('pending', 0):>5}")
    print(f"  Rejected:            {status_counts.get('rejected', 0):>5}")
    if by_ticker:
        print(f"\n  By ticker:")
        for row in by_ticker:
            print(f"    {row['ticker']:<10} {row['n']:>4} recommendations")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyst Tracker — Collector BR (clipping + LLM)"
    )
    parser.add_argument("--ticker",  "-t", type=str,  default=None,
                        help="BR Ticker (e.g.: VALE3, PETR4)")
    parser.add_argument("--all",     "-a", action="store_true",
                        help=f"Collect all BR tickers: {', '.join(BR_TICKERS)}")
    parser.add_argument("--since",   "-s", type=str,  default="2022-01-01",
                        help="Start date (YYYY-MM-DD). Default: 2022-01-01")
    parser.add_argument("--review",  "-r", action="store_true",
                        help="List extractions pending review")
    parser.add_argument("--approve", type=int, default=None, metavar="ID",
                        help="Approve extraction by ID")
    parser.add_argument("--reject",  type=int, default=None, metavar="ID",
                        help="Reject extraction by ID")
    parser.add_argument("--stats",   action="store_true",
                        help="Show clipping summary")
    parser.add_argument("--import-approved", action="store_true",
                        help="Import all approved extractions to recommendations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.review:
        show_pending_reviews()

    elif args.approve:
        approve_extraction(args.approve)

    elif args.reject:
        reject_extraction(args.reject)

    elif args.stats:
        show_clipping_stats()

    elif args.import_approved:
        conn = get_connection()
        migrate_clipping_table(conn)
        n = import_approved_to_recommendations(conn)
        conn.close()
        print(f"✅ {n} recommendations imported.")

    elif args.all:
        print(f"\n🚀 Collector BR — all tickers")
        print(f"   Tickers: {', '.join(BR_TICKERS)}")
        print(f"   Since:   {args.since}\n")
        run_all(since=args.since)

    elif args.ticker:
        run_collector(ticker=args.ticker, since=args.since)

    else:
        print("\n🚀 Analyst Tracker — Collector BR")
        print("   Use --ticker VALE3, --all, --review or --stats")
        print("   Example: python collector_br.py --ticker VALE3 --since 2022-01-01\n")
