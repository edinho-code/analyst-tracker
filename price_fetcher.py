"""
Analyst Tracker — Price Fetcher
================================
Baixa histórico de preços (2016–hoje) via Yahoo Finance
para todos os ativos cadastrados no banco e salva em price_history.

Também baixa os benchmarks SPY e ^BVSP automaticamente.

Uso:
    python price_fetcher.py                   # todos os ativos
    python price_fetcher.py --ticker NVDA     # ativo específico
    python price_fetcher.py --since 2020-01-01 # a partir de uma data

Dependências:
    pip install yfinance pandas
"""

from __future__ import annotations
import sqlite3
import argparse
import time
import sys
from datetime import datetime, date

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("❌ Dependências faltando. Rode: pip install yfinance pandas")
    sys.exit(1)

DB_PATH        = "analyst_tracker.db"
DEFAULT_START  = "2016-01-01"
DEFAULT_END    = date.today().isoformat()
SLEEP_BETWEEN  = 0.8   # segundos entre chamadas (evita rate limit)


# ─────────────────────────────────────────────
# CONEXÃO
# ─────────────────────────────────────────────

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ─────────────────────────────────────────────
# FETCH DE PREÇOS
# ─────────────────────────────────────────────

def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Baixa OHLCV histórico via yfinance.
    Retorna DataFrame com colunas: date, open, high, low, close, volume.
    Retorna DataFrame vazio em caso de erro.
    """
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(start=start, end=end, auto_adjust=True)

        if df.empty:
            print(f"  ⚠️  Sem dados para {ticker}")
            return pd.DataFrame()

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        # yfinance retorna datetime com timezone — normalizar para date string
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

        # Renomear colunas para bater com o schema
        rename = {
            "open":   "open",
            "high":   "high",
            "low":    "low",
            "close":  "close",
            "volume": "volume",
        }
        cols = ["date"] + [c for c in rename if c in df.columns]
        df = df[cols].rename(columns=rename)

        # Arredondar preços para 4 casas
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].round(4)

        return df

    except Exception as e:
        print(f"  ❌ Erro ao buscar {ticker}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# SALVAR NO BANCO
# ─────────────────────────────────────────────

def save_prices(conn: sqlite3.Connection, asset_id: int, df: pd.DataFrame) -> int:
    """
    Insere ou ignora preços no banco (ON CONFLICT IGNORE para não duplicar).
    Retorna número de linhas novas inseridas.
    """
    if df.empty:
        return 0

    cursor   = conn.cursor()
    inserted = 0

    for _, row in df.iterrows():
        try:
            cursor.execute(
                """INSERT OR IGNORE INTO price_history
                   (asset_id, date, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    asset_id,
                    row["date"],
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row["close"],
                    int(row["volume"]) if pd.notna(row.get("volume")) else None,
                )
            )
            if cursor.rowcount:
                inserted += 1
        except Exception as e:
            print(f"  ⚠️  Erro ao salvar linha {row['date']}: {e}")

    conn.commit()
    return inserted


# ─────────────────────────────────────────────
# GARANTIR BENCHMARKS
# ─────────────────────────────────────────────

BENCHMARKS = [
    ("SPY",   "SPDR S&P 500 ETF",  "NYSE",  "ETF",   "US", "USD"),
    ("^BVSP", "Ibovespa",           "BVMF",  "Index", "BR", "BRL"),
]

def ensure_benchmarks(conn: sqlite3.Connection):
    """Garante que SPY e ^BVSP estão cadastrados como ativos."""
    cursor = conn.cursor()
    for (ticker, name, exchange, sector, country, currency) in BENCHMARKS:
        cursor.execute("SELECT id FROM assets WHERE ticker = ?", (ticker,))
        if not cursor.fetchone():
            cursor.execute(
                """INSERT INTO assets (ticker, name, exchange, sector, country, currency)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ticker, name, exchange, sector, country, currency)
            )
            print(f"  ➕ Benchmark cadastrado: {ticker}")
    conn.commit()


# ─────────────────────────────────────────────
# BUSCAR ÚLTIMO PREÇO DISPONÍVEL
# ─────────────────────────────────────────────

def get_price_on_date(
    conn: sqlite3.Connection,
    ticker: str,
    target_date: str,
    tolerance_days: int = 5
) -> float | None:
    """
    Retorna o preço de fechamento mais próximo de target_date (até tolerance_days antes).
    Útil para calcular performance de recomendações em datas específicas.

    Exemplo:
        price = get_price_on_date(conn, "NVDA", "2024-03-15")
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM assets WHERE ticker = ?", (ticker,))
    asset = cursor.fetchone()
    if not asset:
        return None

    cursor.execute(
        """SELECT close FROM price_history
           WHERE asset_id = ? AND date <= ? AND date >= date(?, ?)
           ORDER BY date DESC LIMIT 1""",
        (asset["id"], target_date, target_date, f"-{tolerance_days} days")
    )
    row = cursor.fetchone()
    return row["close"] if row else None


def get_return_between(
    conn: sqlite3.Connection,
    ticker: str,
    start_date: str,
    end_date: str
) -> float | None:
    """
    Retorna o retorno percentual de um ativo entre duas datas.
    Útil para calcular alpha vs benchmark.

    Exemplo:
        ret = get_return_between(conn, "SPY", "2024-01-15", "2024-07-15")
        # → 12.3  (significa +12.3%)
    """
    p_start = get_price_on_date(conn, ticker, start_date)
    p_end   = get_price_on_date(conn, ticker, end_date)

    if p_start and p_end and p_start > 0:
        return round(((p_end - p_start) / p_start) * 100, 2)
    return None


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_fetch(ticker_filter: str = None, start: str = DEFAULT_START, end: str = DEFAULT_END):
    """
    Executa o fetch para todos os ativos (ou um específico).
    """
    conn = get_connection()
    ensure_benchmarks(conn)

    cursor = conn.cursor()

    if ticker_filter:
        cursor.execute(
            "SELECT id, ticker, name FROM assets WHERE ticker = ?",
            (ticker_filter.upper(),)
        )
    else:
        cursor.execute("SELECT id, ticker, name FROM assets ORDER BY country, ticker")

    assets = cursor.fetchall()

    if not assets:
        print(f"❌ Nenhum ativo encontrado{' para ' + ticker_filter if ticker_filter else ''}.")
        conn.close()
        return

    print(f"\n📥 Baixando preços de {len(assets)} ativo(s) | {start} → {end}\n")
    total_new = 0
    errors    = []

    for i, asset in enumerate(assets, 1):
        ticker = asset["ticker"]
        name   = asset["name"]
        prefix = f"[{i:>2}/{len(assets)}]"

        print(f"{prefix} {ticker:<12} {name[:35]:<35} ", end="", flush=True)

        df = fetch_prices(ticker, start, end)

        if df.empty:
            errors.append(ticker)
            print("sem dados")
        else:
            n = save_prices(conn, asset["id"], df)
            total_new += n
            print(f"{len(df):>5} candles → {n:>5} novos")

        if i < len(assets):
            time.sleep(SLEEP_BETWEEN)

    # Resumo final
    print(f"\n{'─'*55}")
    print(f"✅ Concluído: {total_new} novos registros inseridos")

    if errors:
        print(f"⚠️  Falhas ({len(errors)}): {', '.join(errors)}")

    # Contar total no banco
    cursor.execute("SELECT COUNT(*) as n FROM price_history")
    total = cursor.fetchone()["n"]
    print(f"📊 Total de candles no banco: {total:,}")
    print(f"{'─'*55}\n")

    conn.close()


# ─────────────────────────────────────────────
# UTILITÁRIO — MOSTRAR DADOS DE UM ATIVO
# ─────────────────────────────────────────────

def show_asset_prices(ticker: str, limit: int = 10):
    """Mostra os últimos N preços de um ativo. Útil para debug."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ph.date, ph.open, ph.close, ph.volume
        FROM price_history ph
        JOIN assets a ON a.id = ph.asset_id
        WHERE a.ticker = ?
        ORDER BY ph.date DESC
        LIMIT ?
    """, (ticker.upper(), limit))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"Sem dados para {ticker}")
        return

    print(f"\n{'─'*50}")
    print(f"  {ticker} — últimos {limit} fechamentos")
    print(f"{'─'*50}")
    print(f"  {'Data':<12} {'Abertura':>10} {'Fechamento':>12} {'Volume':>14}")
    print(f"  {'─'*10:<12} {'─'*9:>10} {'─'*11:>12} {'─'*13:>14}")
    for row in rows:
        vol = f"{int(row['volume']):,}" if row["volume"] else "—"
        print(f"  {row['date']:<12} {row['open']:>10.2f} {row['close']:>12.2f} {vol:>14}")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyst Tracker — Price Fetcher"
    )
    parser.add_argument(
        "--ticker", "-t",
        type=str, default=None,
        help="Ticker específico (ex: NVDA, PETR4.SA). Padrão: todos os ativos."
    )
    parser.add_argument(
        "--since", "-s",
        type=str, default=DEFAULT_START,
        help=f"Data de início (YYYY-MM-DD). Padrão: {DEFAULT_START}"
    )
    parser.add_argument(
        "--until", "-u",
        type=str, default=DEFAULT_END,
        help=f"Data de fim (YYYY-MM-DD). Padrão: hoje ({DEFAULT_END})"
    )
    parser.add_argument(
        "--show", "-S",
        type=str, default=None,
        metavar="TICKER",
        help="Mostra os últimos preços de um ticker (ex: --show NVDA)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.show:
        show_asset_prices(args.show)
    else:
        print("\n🚀 Analyst Tracker — Price Fetcher")
        run_fetch(
            ticker_filter=args.ticker,
            start=args.since,
            end=args.until,
        )
        print("✅ Próximo passo: python scoring_engine.py")
