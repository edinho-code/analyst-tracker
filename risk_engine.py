"""
Analyst Tracker — Risk Engine
================================
Calcula a probabilidade calibrada de uma recomendação estar certa
ANTES de você seguir ela — o diferencial real do produto.

Diferente do scoring_engine que avalia o passado, o risk_engine
responde a pergunta: "dado o que sei sobre esse analista e esse contexto,
qual a probabilidade de essa call específica estar certa?"

Dimensões do modelo:
  1. Analista × Ativo      — histórico específico desse analista nesse ticker
  2. Analista × Setor      — se sem histórico no ativo, como vai no setor?
  3. Magnitude do upside   — calls de +50% têm taxa de acerto muito menor
  4. Consenso vs contrarian — call alinhada ou na contramão do mercado?
  5. Timing/recência        — há quanto tempo não revisa? call velha perde força
  6. Contexto de volatilidade — analista acerta mais em bull ou bear market?
  7. Volume de cobertura    — analista com 3 calls tem muito mais incerteza

Output: probabilidade calibrada (0–100%) + breakdown por dimensão + rating de confiança

Uso:
    python risk_engine.py --ticker NVDA --analyst "Dan Ives" --direction buy --target 300 --price 182
    python risk_engine.py --ticker VALE3 --analyst "BTG Pactual" --direction buy --target 85 --price 65
    python risk_engine.py --calc-all          # calcula risk profiles de todas as calls recentes
    python risk_engine.py --profile "Dan Ives" # perfil de risco completo de um analista

Dependências:
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
    print("❌ Rode: pip install pandas")
    sys.exit(1)

DB_PATH = "analyst_tracker.db"

# ─────────────────────────────────────────────
# PESOS DO MODELO
# Soma = 1.0
# ─────────────────────────────────────────────

WEIGHTS = {
    "analyst_asset":    0.30,   # histórico específico analista × ativo
    "analyst_sector":   0.20,   # histórico analista × setor (fallback)
    "magnitude":        0.20,   # tamanho do upside implícito
    "consensus":        0.10,   # alinhamento com consenso de mercado
    "recency":          0.10,   # quão recente é a call
    "volatility_fit":   0.10,   # analista acerta em regime de vol atual?
}

# Mínimo de calls históricas para usar o dado com confiança
MIN_CALLS_CONFIDENT  = 10
MIN_CALLS_MODERATE   = 3

# Upside "normal" de mercado — calls além disso têm penalidade
NORMAL_UPSIDE_PCT    = 20.0
MAX_RELIABLE_UPSIDE  = 60.0   # acima de 60% de upside, penalidade máxima

# Janela para "call recente" (dias)
RECENT_WINDOW_DAYS   = 90
STALE_WINDOW_DAYS    = 365


# ─────────────────────────────────────────────
# CONEXÃO
# ─────────────────────────────────────────────

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ─────────────────────────────────────────────
# DIMENSÃO 1 — Analista × Ativo
# ─────────────────────────────────────────────

def score_analyst_asset(
    conn: sqlite3.Connection,
    analyst_id: int,
    asset_id: int,
    direction: str
) -> dict:
    """
    Probabilidade baseada no histórico específico desse analista nesse ativo.
    Retorna score (0–1), n_calls, confidence_weight.
    """
    cursor = conn.cursor()
    cursor.execute(
        """SELECT
               COUNT(*) AS total,
               AVG(perf.direction_score) AS avg_dir,
               AVG(perf.target_score)    AS avg_tgt,
               SUM(CASE WHEN pos.direction = ? THEN 1 ELSE 0 END) AS same_dir,
               AVG(CASE WHEN pos.direction = ? THEN perf.direction_score ELSE NULL END) AS avg_dir_same
           FROM performance perf
           JOIN positions pos ON pos.id = perf.position_id
           WHERE pos.analyst_id = ?
             AND pos.asset_id   = ?
             AND perf.direction_score IS NOT NULL""",
        (direction, direction, analyst_id, asset_id)
    )
    row = cursor.fetchone()

    total = row["total"] or 0

    if total == 0:
        return {"score": 0.5, "n": 0, "weight": 0.3, "label": "sem histórico neste ativo"}

    avg_dir      = row["avg_dir"]      or 0.5
    avg_dir_same = row["avg_dir_same"] or avg_dir

    # Usar avg_dir_same (performance em calls do mesmo tipo — buy/sell)
    # quando há dados suficientes
    score = avg_dir_same if row["same_dir"] and row["same_dir"] >= 2 else avg_dir

    # Peso de confiança baseado no volume de dados
    if total >= MIN_CALLS_CONFIDENT:
        weight = 1.0
    elif total >= MIN_CALLS_MODERATE:
        weight = 0.6
    else:
        weight = 0.3

    label = f"{total} calls hist. neste ativo | dir score médio: {score:.2f}"
    return {"score": score, "n": total, "weight": weight, "label": label}


# ─────────────────────────────────────────────
# DIMENSÃO 2 — Analista × Setor
# ─────────────────────────────────────────────

def score_analyst_sector(
    conn: sqlite3.Connection,
    analyst_id: int,
    sector: str,
    direction: str
) -> dict:
    """
    Probabilidade baseada no histórico do analista no setor do ativo.
    Fallback quando não há histórico no ativo específico.
    """
    if not sector:
        return {"score": 0.5, "n": 0, "weight": 0.2, "label": "setor desconhecido"}

    cursor = conn.cursor()
    cursor.execute(
        """SELECT
               COUNT(*) AS total,
               AVG(perf.direction_score) AS avg_dir,
               AVG(CASE WHEN pos.direction = ? THEN perf.direction_score ELSE NULL END) AS avg_dir_same
           FROM performance perf
           JOIN positions pos ON pos.id  = perf.position_id
           JOIN assets    a   ON a.id    = pos.asset_id
           WHERE pos.analyst_id = ?
             AND a.sector       = ?
             AND perf.direction_score IS NOT NULL""",
        (direction, analyst_id, sector)
    )
    row = cursor.fetchone()

    total = row["total"] or 0

    if total == 0:
        return {"score": 0.5, "n": 0, "weight": 0.1, "label": f"sem histórico em {sector}"}

    avg_dir_same = row["avg_dir_same"] or row["avg_dir"] or 0.5
    score = avg_dir_same

    if total >= MIN_CALLS_CONFIDENT:
        weight = 0.8
    elif total >= MIN_CALLS_MODERATE:
        weight = 0.5
    else:
        weight = 0.2

    label = f"{total} calls em {sector} | dir score médio: {score:.2f}"
    return {"score": score, "n": total, "weight": weight, "label": label}


# ─────────────────────────────────────────────
# DIMENSÃO 3 — Magnitude do Upside
# ─────────────────────────────────────────────

def score_magnitude(
    direction: str,
    price_current: float,
    price_target: Optional[float]
) -> dict:
    """
    Penaliza calls com upside/downside implícito muito alto.
    Historicamente, calls de +50% têm taxa de acerto muito menor.

    Curva: score = 1.0 para upside <= 20%
                   decai linearmente até 0.4 em 60%+
    """
    if not price_target or not price_current or price_current <= 0:
        return {"score": 0.65, "upside_pct": None, "weight": 0.5,
                "label": "sem preço-alvo — upside desconhecido"}

    if direction == "buy":
        upside_pct = ((price_target - price_current) / price_current) * 100
    elif direction == "sell":
        upside_pct = ((price_current - price_target) / price_current) * 100
    else:
        return {"score": 0.65, "upside_pct": 0, "weight": 0.5, "label": "hold — magnitude N/A"}

    # Penalidade para upsides muito agressivos
    if upside_pct <= 0:
        # Target abaixo do preço atual em call de buy — sinal ruim
        score = 0.2
        label = f"target inválido ({upside_pct:.1f}% upside)"
    elif upside_pct <= NORMAL_UPSIDE_PCT:
        score = 1.0
        label = f"+{upside_pct:.1f}% upside — moderado ✅"
    elif upside_pct <= MAX_RELIABLE_UPSIDE:
        # Decaimento linear entre 20% e 60%
        decay = (upside_pct - NORMAL_UPSIDE_PCT) / (MAX_RELIABLE_UPSIDE - NORMAL_UPSIDE_PCT)
        score = 1.0 - (0.6 * decay)
        label = f"+{upside_pct:.1f}% upside — agressivo ⚠️"
    else:
        score = 0.4
        label = f"+{upside_pct:.1f}% upside — muito agressivo 🔴"

    return {"score": round(score, 3), "upside_pct": round(upside_pct, 1),
            "weight": 1.0, "label": label}


# ─────────────────────────────────────────────
# DIMENSÃO 4 — Consenso vs Contrarian
# ─────────────────────────────────────────────

def score_consensus(
    conn: sqlite3.Connection,
    asset_id: int,
    direction: str,
    days: int = 90
) -> dict:
    """
    Verifica se a call está alinhada com o consenso recente ou é contrarian.

    Calls alinhadas com consenso têm win rate ligeiramente maior,
    mas calls contrarian corretas têm retorno muito maior.
    Penalizamos levemente calls muito contrarian por ter menor base rate.
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
                "label": "consenso insuficiente (< 3 calls recentes)"}

    buys  = row["buys"]  or 0
    sells = row["sells"] or 0
    holds = row["holds"] or 0

    if direction == "buy":
        aligned_pct = buys / total
    elif direction == "sell":
        aligned_pct = sells / total
    else:
        aligned_pct = holds / total

    # Score: calls com 60-80% de consenso têm melhor equilíbrio
    # Muito consensual (>90%): levemente penalizado (já precificado)
    # Muito contrarian (<20%): penalizado (menor base rate)
    if aligned_pct >= 0.9:
        score = 0.70
        label = f"herd call — {aligned_pct:.0%} do mercado concorda ⚠️ (pode já estar precificado)"
    elif aligned_pct >= 0.60:
        score = 0.80
        label = f"consenso sólido — {aligned_pct:.0%} concorda ✅"
    elif aligned_pct >= 0.35:
        score = 0.70
        label = f"split — {aligned_pct:.0%} concorda, mercado dividido"
    elif aligned_pct >= 0.15:
        score = 0.55
        label = f"contrarian — apenas {aligned_pct:.0%} concorda ⚠️"
    else:
        score = 0.40
        label = f"muito contrarian — {aligned_pct:.0%} concorda 🔴 (call solitária)"

    return {"score": score, "consensus_pct": round(aligned_pct, 3),
            "weight": 0.7, "label": label}


# ─────────────────────────────────────────────
# DIMENSÃO 5 — Recência da Call
# ─────────────────────────────────────────────

def score_recency(rec_date: Optional[str]) -> dict:
    """
    Calls mais recentes são mais confiáveis.
    Call de 2 anos atrás sem revisão = contexto mudou muito.
    """
    if not rec_date:
        return {"score": 0.60, "days_old": None, "weight": 0.5,
                "label": "data da call desconhecida"}

    try:
        rec_dt  = datetime.strptime(rec_date, "%Y-%m-%d").date()
        days_old = (date.today() - rec_dt).days
    except ValueError:
        return {"score": 0.60, "days_old": None, "weight": 0.5, "label": "data inválida"}

    if days_old <= 7:
        score = 1.0
        label = f"call de {days_old}d atrás — fresquíssima ✅"
    elif days_old <= RECENT_WINDOW_DAYS:
        score = 0.90
        label = f"call de {days_old}d atrás — recente ✅"
    elif days_old <= 180:
        score = 0.75
        label = f"call de {days_old}d atrás — moderadamente recente"
    elif days_old <= STALE_WINDOW_DAYS:
        score = 0.55
        label = f"call de {days_old}d atrás — envelhecendo ⚠️"
    else:
        score = 0.35
        label = f"call de {days_old}d atrás — desatualizada 🔴"

    return {"score": score, "days_old": days_old, "weight": 1.0, "label": label}


# ─────────────────────────────────────────────
# DIMENSÃO 6 — Fit com Volatilidade
# ─────────────────────────────────────────────

def score_volatility_fit(
    conn: sqlite3.Connection,
    analyst_id: int,
    asset_id: int
) -> dict:
    """
    Verifica se o analista tende a acertar mais em períodos de alta ou baixa volatilidade,
    e compara com a volatilidade atual do ativo (proxy: desvio padrão dos últimos 30 dias).

    Proxy de volatilidade: std de retornos diários dos últimos 30 dias.
    High vol: std > 2.5% | Low vol: std < 1.0%
    """
    cursor = conn.cursor()

    # Volatilidade atual do ativo (últimos 30 dias)
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
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        current_vol = math.sqrt(variance)

        if current_vol > 2.5:
            vol_regime = "high"
        elif current_vol > 1.0:
            vol_regime = "medium"
        else:
            vol_regime = "low"

    if vol_regime == "unknown":
        return {"score": 0.60, "current_vol": None, "weight": 0.3,
                "label": "volatilidade atual desconhecida (sem preços)"}

    # Performance histórica do analista em diferentes regimes
    # Proxy: calls feitas em períodos de maior/menor movimento
    cursor.execute(
        """SELECT
               AVG(perf.direction_score) AS avg_dir,
               COUNT(*) AS total
           FROM performance perf
           JOIN positions pos ON pos.id = perf.position_id
           WHERE pos.analyst_id = ? AND perf.direction_score IS NOT NULL""",
        (analyst_id,)
    )
    overall = cursor.fetchone()

    if not overall or not overall["avg_dir"]:
        return {"score": 0.60, "current_vol": current_vol, "weight": 0.3,
                "label": f"vol atual: {current_vol:.2f}% — sem histórico suficiente"}

    # Sem histórico detalhado de vol, usar um ajuste simples:
    # em alta volatilidade, analistas em geral têm ~15% menos acerto
    base_score = overall["avg_dir"]

    if vol_regime == "high":
        adj_score = base_score * 0.85
        label = f"vol atual alta ({current_vol:.2f}%/dia) — acerto tende a cair ⚠️"
    elif vol_regime == "medium":
        adj_score = base_score * 1.00
        label = f"vol atual moderada ({current_vol:.2f}%/dia) — condições normais ✅"
    else:
        adj_score = base_score * 1.05
        label = f"vol atual baixa ({current_vol:.2f}%/dia) — mercado calmo ✅"

    adj_score = max(0.0, min(1.0, adj_score))
    weight    = 0.7 if overall["total"] >= MIN_CALLS_MODERATE else 0.3

    return {"score": round(adj_score, 3), "current_vol": round(current_vol, 3),
            "weight": weight, "label": label}


# ─────────────────────────────────────────────
# CALIBRAÇÃO FINAL
# ─────────────────────────────────────────────

def calibrate_probability(dimensions: dict) -> float:
    """
    Combina os scores das 6 dimensões em uma probabilidade final calibrada.

    Usa média ponderada dos scores × peso_dimensão × weight_confiança.
    O weight_confiança reflete o quanto de dados reais temos para aquela dimensão.
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for dim_name, dim_weight in WEIGHTS.items():
        dim = dimensions.get(dim_name, {})
        score  = dim.get("score", 0.5)
        conf_w = dim.get("weight", 0.5)   # confiança interna da dimensão

        effective_weight = dim_weight * conf_w
        weighted_sum    += score * effective_weight
        total_weight    += effective_weight

    if total_weight == 0:
        return 0.50

    raw_prob = weighted_sum / total_weight

    # Calibração Platt — suaviza extremos (evita 95%+ ou <10%)
    # Mapeamento: [0, 1] → [0.15, 0.88]
    calibrated = 0.15 + (raw_prob * 0.73)

    return round(calibrated, 4)


def confidence_rating(prob: float, n_calls: int) -> tuple[str, str]:
    """
    Converte probabilidade + volume de dados em rating qualitativo.
    Retorna (rating, emoji).
    """
    # Incerteza alta quando poucos dados
    if n_calls < MIN_CALLS_MODERATE:
        return "INCERTO", "⚪"

    if prob >= 0.75:
        return "ALTA", "🟢"
    elif prob >= 0.62:
        return "MODERADA-ALTA", "🟡"
    elif prob >= 0.50:
        return "MODERADA", "🟠"
    elif prob >= 0.38:
        return "MODERADA-BAIXA", "🔴"
    else:
        return "BAIXA", "🔴"


# ─────────────────────────────────────────────
# INTERFACE PRINCIPAL
# ─────────────────────────────────────────────

def evaluate_call(
    ticker: str,
    analyst_name: str,
    direction: str,
    price_current: float,
    price_target: Optional[float] = None,
    rec_date: Optional[str] = None,
    verbose: bool = True,
    db_path: str = DB_PATH
) -> dict:
    """
    Avalia o risco de uma call específica.
    Retorna dict com probabilidade, breakdown por dimensão e rating.

    Exemplo:
        result = evaluate_call(
            ticker="NVDA",
            analyst_name="Dan Ives",
            direction="buy",
            price_current=182.0,
            price_target=300.0,
            rec_date="2024-03-15"
        )
        print(result["probability_pct"])  # ex: 67.3
        print(result["rating"])           # ex: "MODERADA-ALTA"
    """
    conn   = get_connection(db_path)
    cursor = conn.cursor()

    # Buscar analyst_id
    cursor.execute(
        "SELECT id FROM analysts WHERE name LIKE ?",
        (f"%{analyst_name}%",)
    )
    analyst_row = cursor.fetchone()
    analyst_id  = analyst_row["id"] if analyst_row else None

    # Buscar asset_id + setor
    cursor.execute(
        "SELECT id, sector, country FROM assets WHERE ticker = ?",
        (ticker.upper(),)
    )
    asset_row = cursor.fetchone()
    asset_id  = asset_row["id"]     if asset_row else None
    sector    = asset_row["sector"] if asset_row else None

    # Total de posições históricas do analista
    n_calls_total = 0
    if analyst_id:
        cursor.execute(
            "SELECT COUNT(*) AS n FROM positions WHERE analyst_id = ?",
            (analyst_id,)
        )
        n_calls_total = cursor.fetchone()["n"] or 0

    # ── Calcular cada dimensão ────────────────────────

    dimensions = {}

    # 1. Analista × Ativo
    if analyst_id and asset_id:
        dimensions["analyst_asset"] = score_analyst_asset(conn, analyst_id, asset_id, direction)
    else:
        dimensions["analyst_asset"] = {
            "score": 0.5, "n": 0, "weight": 0.1,
            "label": f"analista '{analyst_name}' ou ticker '{ticker}' não encontrado no banco"
        }

    # 2. Analista × Setor
    if analyst_id and sector:
        dimensions["analyst_sector"] = score_analyst_sector(conn, analyst_id, sector, direction)
    else:
        dimensions["analyst_sector"] = {
            "score": 0.5, "n": 0, "weight": 0.1,
            "label": "sem dados de setor"
        }

    # 3. Magnitude do upside
    dimensions["magnitude"] = score_magnitude(direction, price_current, price_target)

    # 4. Consenso
    if asset_id:
        dimensions["consensus"] = score_consensus(conn, asset_id, direction)
    else:
        dimensions["consensus"] = {"score": 0.65, "weight": 0.2, "label": "ativo não encontrado"}

    # 5. Recência
    dimensions["recency"] = score_recency(rec_date or date.today().isoformat())

    # 6. Volatilidade
    if analyst_id and asset_id:
        dimensions["volatility_fit"] = score_volatility_fit(conn, analyst_id, asset_id)
    else:
        dimensions["volatility_fit"] = {"score": 0.60, "weight": 0.2, "label": "dados insuficientes"}

    # ── Probabilidade final ───────────────────────────

    probability = calibrate_probability(dimensions)
    prob_pct    = round(probability * 100, 1)
    rating, emoji = confidence_rating(probability, n_calls_total)

    # Upside implícito
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
    """Imprime o resultado formatado no terminal."""
    dir_icon = {"buy": "📈 BUY", "sell": "📉 SELL", "hold": "➡️  HOLD"}.get(r["direction"], r["direction"])
    upside   = f"+{r['upside_pct']:.1f}%" if r["upside_pct"] else "sem target"

    print(f"\n{'═'*62}")
    print(f"  🎯  RISK ASSESSMENT — Analyst Tracker")
    print(f"{'═'*62}")
    print(f"  Analista:   {r['analyst']}")
    print(f"  Call:       {r['ticker']} {dir_icon}  |  preço atual: ${r['price_current']:.2f}")
    if r["price_target"]:
        print(f"  Target:     ${r['price_target']:.2f}  ({upside})")
    if r["rec_date"]:
        print(f"  Data:       {r['rec_date']}")
    print(f"  Histórico:  {r['n_calls_history']} calls totais do analista no banco")
    print(f"{'─'*62}")
    print(f"\n  PROBABILIDADE DE ACERTO:  {r['probability_pct']:.1f}%  {r['rating_emoji']} {r['rating']}\n")
    print(f"{'─'*62}")
    print(f"  Breakdown por dimensão:")
    print()

    dim_labels = {
        "analyst_asset":  "Hist. Analista × Ativo",
        "analyst_sector": "Hist. Analista × Setor",
        "magnitude":      "Magnitude do upside",
        "consensus":      "Alinhamento c/ consenso",
        "recency":        "Recência da call",
        "volatility_fit": "Fit com volatilidade",
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

    # Interpretação
    prob = r["probability_pct"]
    if prob >= 75:
        msg = "Sinal forte — histórico e contexto favorecem essa call."
    elif prob >= 62:
        msg = "Sinal moderado-positivo — call razoável, mas monitore de perto."
    elif prob >= 50:
        msg = "Sinal neutro — incerteza considerável, dimensione posição com cuidado."
    elif prob >= 38:
        msg = "Sinal fraco — histórico ou contexto desfavorável. Alta cautela."
    else:
        msg = "Sinal negativo — múltiplos fatores contra. Evite ou hedge."

    print(f"  💬 {msg}")
    print(f"{'═'*62}\n")


# ─────────────────────────────────────────────
# CALCULAR RISK PARA TODAS AS CALLS RECENTES
# ─────────────────────────────────────────────

def calc_all_recent(days: int = 90, db_path: str = DB_PATH):
    """
    Calcula e salva risk score para posições abertas recentes.
    Persiste em tabela risk_assessments (position_id) para uso no dashboard.
    """
    conn = get_connection(db_path)

    # Criar/garantir tabela com position_id
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

    print(f"\n🔢 Calculando risk para {len(positions)} posições abertas (desde {since})...\n")

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
            print(f"  ⚠️  Erro em posição #{pos['id']}: {e}")

    print(f"\n✅ {calculated} risk assessments calculados e salvos.\n")
    conn.close()


# ─────────────────────────────────────────────
# PERFIL DE RISCO DE UM ANALISTA
# ─────────────────────────────────────────────

def analyst_risk_profile(analyst_name: str, db_path: str = DB_PATH):
    """
    Mostra o perfil de risco completo de um analista:
    quais tipos de calls têm maior/menor probabilidade calibrada.
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
        print(f"\n❌ Sem risk assessments para '{analyst_name}'.")
        print("   Rode: python risk_engine.py --calc-all\n")
        return

    df = pd.DataFrame([dict(r) for r in rows])

    print(f"\n{'═'*62}")
    print(f"  🧠  Perfil de Risco — {analyst_name}")
    print(f"{'═'*62}")
    print(f"  Calls avaliadas: {len(df)}")
    print(f"  Prob. média:     {df['probability'].mean()*100:.1f}%")
    print(f"  Prob. mediana:   {df['probability'].median()*100:.1f}%")
    print(f"\n  Por tipo de call:")

    for direction in ["buy", "sell", "hold"]:
        sub = df[df["direction"] == direction]
        if sub.empty:
            continue
        print(f"    {direction.upper():<5}: {sub['probability'].mean()*100:.1f}% prob. média "
              f"({len(sub)} calls)")

    if df["sector"].notna().any():
        print(f"\n  Por setor:")
        sector_stats = (df.groupby("sector")["probability"]
                        .agg(["mean", "count"])
                        .sort_values("mean", ascending=False))
        for sector, row in sector_stats.iterrows():
            print(f"    {sector:<20} {row['mean']*100:.1f}%  ({int(row['count'])} calls)")

    # Top e bottom calls
    print(f"\n  Calls com maior probabilidade:")
    top5 = df.nlargest(5, "probability")
    for _, r in top5.iterrows():
        print(f"    {r['ticker']:<8} {r['direction']:<5} {r['rec_date']}  "
              f"→ {r['probability']*100:.1f}%")

    print(f"\n  Calls com menor probabilidade:")
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
    # Avaliar uma call específica
    parser.add_argument("--ticker",    "-t", type=str,  default=None)
    parser.add_argument("--analyst",   "-a", type=str,  default=None)
    parser.add_argument("--direction", "-d", type=str,  default="buy",
                        choices=["buy", "sell", "hold"])
    parser.add_argument("--price",     "-p", type=float, default=None,
                        help="Preço atual do ativo")
    parser.add_argument("--target",          type=float, default=None,
                        help="Preço-alvo da recomendação")
    parser.add_argument("--date",            type=str,   default=None,
                        help="Data da recomendação (YYYY-MM-DD)")

    # Operações em lote
    parser.add_argument("--calc-all",  action="store_true",
                        help="Calcular risk para todas as calls recentes")
    parser.add_argument("--days",      type=int, default=90,
                        help="Janela de dias para --calc-all (padrão: 90)")
    parser.add_argument("--profile",   type=str, default=None,
                        help="Perfil de risco de um analista")

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
        print("Exemplos:")
        print("  # Avaliar uma call específica:")
        print('  python risk_engine.py --ticker NVDA --analyst "Dan Ives" --direction buy --price 182 --target 300')
        print()
        print("  # Calcular risk para todas as calls recentes:")
        print("  python risk_engine.py --calc-all --days 90")
        print()
        print("  # Perfil de risco de um analista:")
        print('  python risk_engine.py --profile "Dan Ives"')
        print()
