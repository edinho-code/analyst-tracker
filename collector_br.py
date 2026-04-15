"""
Analyst Tracker — Collector BR (Clipping + LLM Extractor)
===========================================================
Coleta recomendações de analistas brasileiros via clipping de imprensa financeira.

Fluxo:
  1. Scraper busca artigos por ticker em fontes BR (InfoMoney, Seu Dinheiro, etc.)
  2. Filtra artigos com linguagem de recomendação
  3. LLM (Claude) extrai estrutura: ticker, direção, preço-alvo, analista, data
  4. Salva no banco com confidence score — abaixo de 0.75 vai para revisão manual

Uso:
    python collector_br.py --ticker VALE3
    python collector_br.py --ticker PETR4 --since 2022-01-01
    python collector_br.py --all                        # todos os tickers BR do banco
    python collector_br.py --review                     # revisar extrações pendentes
    python collector_br.py --approve <clipping_id>      # aprovar manualmente
    python collector_br.py --reject  <clipping_id>      # rejeitar manualmente

Dependências:
    pip install requests beautifulsoup4 anthropic
    Variável de ambiente: ANTHROPIC_API_KEY
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
    print("❌ Rode: pip install requests beautifulsoup4")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("❌ Rode: pip install anthropic")
    sys.exit(1)

DB_PATH            = "analyst_tracker.db"
CONFIDENCE_AUTO    = 0.75   # acima disso → entra automaticamente
CONFIDENCE_REVIEW  = 0.50   # entre 0.50–0.75 → revisão manual
SLEEP_BETWEEN      = 1.5    # segundos entre requests (respeitar rate limit)
MAX_ARTICLES       = 10     # artigos por ticker por fonte

# Tickers BR que coletamos
BR_TICKERS = [
    # Petróleo / Energia
    "PETR4", "PRIO3", "CPLE3", "VBBR3",
    # Mineração / Siderurgia
    "VALE3", "GGBR4", "CSNA3",
    # Bancos / Financeiro
    "ITUB4", "BBDC4", "BBAS3", "B3SA3",
    # Consumo / Varejo
    "WEGE3", "ABEV3", "LREN3", "RENT3",
    # Saúde
    "RDOR3", "HAPV3",
    # Proteína / Agro
    "BEEF3", "SLCE3",
    # Outros
    "SUZB3", "MGLU3", "RAIL3",
]

# Palavras-chave que indicam presença de recomendação no artigo
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
# CONEXÃO E MIGRAÇÃO
# ─────────────────────────────────────────────

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def migrate_clipping_table(conn: sqlite3.Connection):
    """
    Cria tabela de clipping se não existir (schema v2 com position_id e rec_type).
    Idempotente — seguro rodar várias vezes.
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

    # Adicionar colunas novas se tabela existia sem elas (migração)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(clipping_extractions)")
    cols = {row[1] for row in cursor.fetchall()}
    if "rec_type" not in cols:
        conn.execute("ALTER TABLE clipping_extractions ADD COLUMN rec_type TEXT DEFAULT 'open'")
    if "position_id" not in cols:
        conn.execute("ALTER TABLE clipping_extractions ADD COLUMN position_id INTEGER REFERENCES positions(id)")
    conn.commit()


# ─────────────────────────────────────────────
# SCRAPERS POR FONTE
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
    InfoMoney — busca artigos por ticker via search.
    URL: https://www.infomoney.com.br/busca/?q=VALE3+recomendação
    """
    results = []
    query   = f"{ticker} recomendação preço-alvo analista"
    url     = f"https://www.infomoney.com.br/busca/?q={quote_plus(query)}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # InfoMoney search results — artigos em <article> ou <div class="search-item">
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
    Seu Dinheiro — busca por ticker.
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

            # Tentar pegar data
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
    Money Times — cobre upgrades/downgrades rapidamente.
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
    E-Investidor (Estadão) — boa cobertura de blue chips.
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
    Guia do Investidor — agrega preços-alvo, ótima fonte.
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
# BUSCAR CORPO DO ARTIGO
# ─────────────────────────────────────────────

def fetch_article_body(url: str) -> str | None:
    """
    Baixa e extrai o texto principal de um artigo.
    Retorna None se falhar ou se o artigo não tiver conteúdo relevante.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remover scripts, styles, navs
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "iframe"]):
            tag.decompose()

        # Tentar extrair o corpo principal do artigo
        body = (
            soup.find("article") or
            soup.find("div", class_=re.compile(r"content|article|post|body|texto")) or
            soup.find("main")
        )

        text = body.get_text(separator=" ", strip=True) if body else soup.get_text(separator=" ", strip=True)

        # Limpar espaços extras
        text = re.sub(r"\s+", " ", text).strip()

        # Só retornar se tiver conteúdo mínimo
        return text[:6000] if len(text) > 200 else None

    except Exception:
        return None


def has_recommendation_keywords(text: str) -> bool:
    """Verifica se o texto contém linguagem de recomendação."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in REC_KEYWORDS)


# ─────────────────────────────────────────────
# EXTRAÇÃO LLM (CLAUDE)
# ─────────────────────────────────────────────

EXTRACTION_PROMPT = """Você é um extrator especializado em recomendações de analistas financeiros.

Analise o texto abaixo e extraia TODAS as recomendações de analistas mencionadas.

Para cada recomendação encontrada, retorne um objeto JSON com:
- analyst_name: nome da corretora/banco/analista (ex: "Itaú BBA", "BTG Pactual", "XP Investimentos")
- ticker: código da ação (ex: "VALE3", "PETR4")
- direction: "buy", "sell" ou "hold" (compra=buy, venda=sell, neutro/manter=hold)
- price_target: preço-alvo numérico (null se não mencionado)
- currency: "BRL" ou "USD"
- rec_date: data da recomendação no formato YYYY-MM-DD (use a data do artigo se não houver data específica)
- horizon_days: prazo em dias (null se não mencionado; 365 se disser "12 meses")
- notes: trecho relevante do texto que suporta a extração (máx 200 chars)
- confidence: número entre 0 e 1 indicando sua certeza na extração
  - 1.0: data exata, analista nomeado, preço-alvo explícito
  - 0.8: analista nomeado, direção clara, sem preço-alvo
  - 0.6: analista implícito ou direção ambígua
  - abaixo de 0.5: muito incerto — preferível não incluir

Retorne APENAS um JSON válido no formato:
{
  "recommendations": [
    { ...campos acima... },
    ...
  ],
  "article_date": "YYYY-MM-DD ou null"
}

Se não houver nenhuma recomendação clara, retorne:
{ "recommendations": [], "article_date": null }

NÃO inclua explicações, markdown ou texto fora do JSON.

TICKER BUSCADO: {ticker}
DATA DO ARTIGO: {article_date}

TEXTO DO ARTIGO:
{article_text}
"""


def extract_with_llm(
    ticker: str,
    article_text: str,
    article_date: str | None,
    client: anthropic.Anthropic
) -> dict | None:
    """
    Usa Claude para extrair recomendações estruturadas de um texto de artigo.
    Retorna o JSON parseado ou None se falhar.
    """
    prompt = EXTRACTION_PROMPT.format(
        ticker=ticker,
        article_date=article_date or "desconhecida",
        article_text=article_text[:5000]
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()

        # Limpar possível markdown
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
# SALVAR NO BANCO
# ─────────────────────────────────────────────

def save_clipping(conn: sqlite3.Connection, ticker: str, article: dict, body: str) -> int | None:
    """Salva artigo bruto. Retorna id ou None se duplicado."""
    h = url_hash(article["url"])
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM clipping_raw WHERE url_hash = ?", (h,))
    if cursor.fetchone():
        return None  # já existe

    cursor.execute(
        """INSERT INTO clipping_raw (ticker, source_name, article_url, article_date, title, body_text, url_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (ticker, article["source"], article["url"], article.get("date"),
         article.get("title"), body, h)
    )
    conn.commit()
    return cursor.lastrowid


def save_extraction(conn: sqlite3.Connection, clipping_id: int, rec: dict, raw_json: str):
    """Salva uma extração LLM. Status inicial depende do confidence."""
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
    """Busca ou cria analista + fonte. Retorna analyst_id."""
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM analysts WHERE name LIKE ?", (f"%{analyst_name}%",))
    analyst = cursor.fetchone()
    if analyst:
        return analyst["id"]

    # Criar ou buscar fonte
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
    """Abre nova posição e revisão 'open'. Retorna (position_id, rec_id)."""
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
    """Adiciona revisão a posição existente. Retorna (rec_type, rec_id)."""
    cursor = conn.cursor()

    # Detectar rec_type a partir do hint do LLM e da mudança de target
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
    Importa extrações aprovadas para positions + recommendations (modelo v2).
    Verifica se já existe posição aberta para analista+ativo e cria revisão se sim.
    Retorna número de registros importados.
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
            # Tentar com sufixo .SA
            cursor.execute("SELECT id FROM assets WHERE ticker = ?", (row["ticker"] + ".SA",))
            asset = cursor.fetchone()
        if not asset:
            print(f"  ⚠️  Ativo não encontrado: {row['ticker']} — pulando")
            continue

        asset_id     = asset["id"]
        price_at_rec = _get_price_at_date(conn, asset_id, row["rec_date"]) if row["rec_date"] else None
        source_url   = row["article_url"]
        notes        = row["notes"]
        direction    = row["direction"].lower()
        price_target = row["price_target"]
        rec_date     = row["rec_date"]
        rec_type_hint = row.get("rec_type") or "open"

        # Verificar se já existe posição aberta
        cursor.execute(
            """SELECT id, direction, final_target FROM positions
               WHERE analyst_id=? AND asset_id=? AND close_date IS NULL""",
            (analyst_id, asset_id)
        )
        open_pos = cursor.fetchone()

        position_id = None
        rec_id      = None

        if not open_pos:
            # Abrir nova posição
            position_id, rec_id = _open_position_br(
                conn, analyst_id, asset_id, direction, rec_date,
                price_at_rec, price_target, source_url, notes
            )
        elif open_pos["direction"] == direction:
            # Mesma direção → revisão
            _, rec_id = _update_position_br(
                conn, open_pos["id"], open_pos["final_target"],
                rec_type_hint, direction, rec_date, price_at_rec,
                price_target, source_url, notes
            )
            position_id = open_pos["id"]
        else:
            # Direção diferente → fechar posição atual
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
            # Abrir nova posição
            position_id, rec_id = _open_position_br(
                conn, analyst_id, asset_id, direction, rec_date,
                price_at_rec, price_target, source_url, notes
            )

        # Marcar extração como importada
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
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_collector(ticker: str, since: str = "2022-01-01"):
    """Pipeline completo para um ticker."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY não definida.")
        print("   export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    conn   = get_connection()
    migrate_clipping_table(conn)

    ticker = ticker.upper()
    print(f"\n📰 Coletando clipping para {ticker} (desde {since})\n")

    articles_found    = 0
    articles_relevant = 0
    extractions_total = 0
    extractions_auto  = 0

    for scraper in ALL_SCRAPERS:
        source_name = scraper.__name__.replace("scrape_", "").replace("_", " ").title()
        print(f"  🔍 {source_name}...", end=" ", flush=True)

        articles = scraper(ticker, since=since)
        print(f"{len(articles)} artigos encontrados")
        articles_found += len(articles)

        for article in articles:
            time.sleep(SLEEP_BETWEEN)

            # Baixar corpo do artigo
            body = fetch_article_body(article["url"])
            if not body:
                continue

            # Filtrar por keywords de recomendação
            if not has_recommendation_keywords(body):
                continue

            articles_relevant += 1

            # Salvar clipping bruto
            clipping_id = save_clipping(conn, ticker, article, body)
            if not clipping_id:
                continue  # artigo duplicado

            # Extrair com LLM
            result = extract_with_llm(
                ticker,
                body,
                article.get("date"),
                client
            )

            if not result or not result.get("recommendations"):
                continue

            # Atualizar data do artigo se LLM detectou
            if result.get("article_date") and not article.get("date"):
                conn.execute(
                    "UPDATE clipping_raw SET article_date=? WHERE id=?",
                    (result["article_date"], clipping_id)
                )
                conn.commit()

            for rec in result["recommendations"]:
                # Só salvar recomendações acima do threshold mínimo
                if rec.get("confidence", 0) < CONFIDENCE_REVIEW:
                    continue

                # Só salvar se for sobre o ticker buscado (ou ticker próximo)
                rec_ticker = rec.get("ticker", "").upper()
                if rec_ticker and rec_ticker != ticker and ticker not in rec_ticker:
                    continue

                # Garantir ticker correto
                rec["ticker"] = ticker

                save_extraction(conn, clipping_id, rec, json.dumps(rec, ensure_ascii=False))
                extractions_total += 1

                if rec.get("confidence", 0) >= CONFIDENCE_AUTO:
                    extractions_auto += 1

    # Importar aprovadas automaticamente
    imported = import_approved_to_recommendations(conn)

    print(f"\n{'─'*55}")
    print(f"  Artigos encontrados:    {articles_found:>4}")
    print(f"  Com recomendações:      {articles_relevant:>4}")
    print(f"  Extrações salvas:       {extractions_total:>4}")
    print(f"  Auto-aprovadas:         {extractions_auto:>4}")
    print(f"  Importadas p/ banco:    {imported:>4}")
    pending = extractions_total - extractions_auto
    if pending > 0:
        print(f"  ⚠️  Pendentes revisão:   {pending:>4}  → rode: python collector_br.py --review")
    print(f"{'─'*55}\n")

    conn.close()


def run_all(since: str = "2022-01-01"):
    """Coleta para todos os tickers BR."""
    for ticker in BR_TICKERS:
        run_collector(ticker, since=since)
        time.sleep(2)


# ─────────────────────────────────────────────
# REVISÃO MANUAL
# ─────────────────────────────────────────────

def show_pending_reviews():
    """Lista extrações pendentes de revisão."""
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
        print("\n✅ Nenhuma extração pendente de revisão.\n")
        return

    print(f"\n{'═'*80}")
    print(f"  📋  EXTRAÇÕES PENDENTES DE REVISÃO ({len(rows)} itens)")
    print(f"{'═'*80}")

    for row in rows:
        pt  = f"R$ {row['price_target']:.2f}" if row["price_target"] else "sem target"
        dir_icon = "📈" if row["direction"] == "buy" else "📉" if row["direction"] == "sell" else "➡️"
        print(f"\n  ID #{row['id']}  [{row['confidence']:.0%} confiança]")
        print(f"  {dir_icon} {row['analyst_name']} → {row['ticker']} {row['direction'].upper()} @ {pt}")
        print(f"  📅 {row['rec_date'] or 'data desconhecida'}  |  {row['source_name']}")
        print(f"  💬 {row['notes'][:100] if row['notes'] else '—'}")
        print(f"  🔗 {row['article_url'][:80]}")
        print(f"  {'─'*76}")
        print(f"  Aprovar:  python collector_br.py --approve {row['id']}")
        print(f"  Rejeitar: python collector_br.py --reject  {row['id']}")

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
        print(f"✅ Extração #{extraction_id} aprovada e importada ({imported} rec inserida).")
    else:
        print(f"❌ Extração #{extraction_id} não encontrada.")
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
        print(f"🗑️  Extração #{extraction_id} rejeitada.")
    else:
        print(f"❌ Extração #{extraction_id} não encontrada.")
    conn.close()


def show_clipping_stats():
    """Resumo geral do clipping."""
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
    print(f"  📊  Clipping BR — Resumo")
    print(f"{'─'*50}")
    print(f"  Artigos coletados:  {total_articles:>5}")
    print(f"  Aprovadas:          {status_counts.get('approved', 0):>5}")
    print(f"  Importadas:         {status_counts.get('imported', 0):>5}")
    print(f"  Pendentes revisão:  {status_counts.get('pending', 0):>5}")
    print(f"  Rejeitadas:         {status_counts.get('rejected', 0):>5}")
    if by_ticker:
        print(f"\n  Por ticker:")
        for row in by_ticker:
            print(f"    {row['ticker']:<10} {row['n']:>4} recomendações")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyst Tracker — Collector BR (clipping + LLM)"
    )
    parser.add_argument("--ticker",  "-t", type=str,  default=None,
                        help="Ticker BR (ex: VALE3, PETR4)")
    parser.add_argument("--all",     "-a", action="store_true",
                        help=f"Coletar todos os tickers BR: {', '.join(BR_TICKERS)}")
    parser.add_argument("--since",   "-s", type=str,  default="2022-01-01",
                        help="Data de início (YYYY-MM-DD). Padrão: 2022-01-01")
    parser.add_argument("--review",  "-r", action="store_true",
                        help="Listar extrações pendentes de revisão")
    parser.add_argument("--approve", type=int, default=None, metavar="ID",
                        help="Aprovar extração pelo ID")
    parser.add_argument("--reject",  type=int, default=None, metavar="ID",
                        help="Rejeitar extração pelo ID")
    parser.add_argument("--stats",   action="store_true",
                        help="Mostrar resumo do clipping")
    parser.add_argument("--import-approved", action="store_true",
                        help="Importar todas as extrações aprovadas para recommendations")
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
        print(f"✅ {n} recomendações importadas.")

    elif args.all:
        print(f"\n🚀 Collector BR — todos os tickers")
        print(f"   Tickers: {', '.join(BR_TICKERS)}")
        print(f"   Desde:   {args.since}\n")
        run_all(since=args.since)

    elif args.ticker:
        run_collector(ticker=args.ticker, since=args.since)

    else:
        print("\n🚀 Analyst Tracker — Collector BR")
        print("   Use --ticker VALE3, --all, --review ou --stats")
        print("   Exemplo: python collector_br.py --ticker VALE3 --since 2022-01-01\n")
