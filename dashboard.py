"""
Analyst Tracker — Dashboard
=============================
Interface visual completa para explorar rankings, performance
e histórico de recomendações de analistas BR e US.

Uso:
    streamlit run dashboard.py

Dependências:
    pip install streamlit plotly pandas
"""

import sqlite3
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DB_PATH = "analyst_tracker.db"

# Importar risk_engine e scoring_engine se disponíveis
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from risk_engine import evaluate_call as _evaluate_call
    RISK_ENGINE_AVAILABLE = True
except ImportError:
    RISK_ENGINE_AVAILABLE = False

try:
    from scoring_engine import compute_yearly_scores, simulate_portfolio
    SCORING_EXTRAS_AVAILABLE = True
except ImportError:
    SCORING_EXTRAS_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG DA PÁGINA
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Analyst Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado — tema escuro refinado, tipografia editorial
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0d0f14;
    --bg2:       #13161e;
    --bg3:       #1a1e2a;
    --border:    #252a38;
    --text:      #e2e6f0;
    --muted:     #6b7490;
    --accent:    #4f9cf9;
    --green:     #3ecf8e;
    --red:       #f96060;
    --amber:     #f9c74f;
    --purple:    #a78bfa;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* Header */
h1 { font-family: 'DM Serif Display', serif; font-size: 2.2rem; color: var(--text); letter-spacing: -0.02em; }
h2 { font-family: 'DM Sans', sans-serif; font-weight: 600; font-size: 1.1rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
h3 { font-family: 'DM Sans', sans-serif; font-weight: 500; color: var(--text); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg2);
    border-right: 1px solid var(--border);
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'DM Mono', monospace !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    color: var(--text);
}

/* Dataframe */
div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }

/* Selectbox / inputs */
div[data-baseweb="select"] > div { background: var(--bg3) !important; border-color: var(--border) !important; }

/* Divider */
hr { border-color: var(--border); margin: 1.5rem 0; }

/* Score badge */
.badge-buy    { background: rgba(62,207,142,0.15); color: #3ecf8e; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; font-family: 'DM Mono', monospace; }
.badge-sell   { background: rgba(249,96,96,0.15);  color: #f96060; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; font-family: 'DM Mono', monospace; }
.badge-hold   { background: rgba(249,199,79,0.15); color: #f9c74f; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; font-family: 'DM Mono', monospace; }
.score-pill   { font-family: 'DM Mono', monospace; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONEXÃO
# ─────────────────────────────────────────────

@st.cache_resource
def get_conn():
    if not Path(DB_PATH).exists():
        st.error(f"Banco '{DB_PATH}' não encontrado. Rode analyst_tracker_setup.py primeiro.")
        st.stop()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=300)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_conn()
    return pd.read_sql_query(sql, conn, params=params)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def composite_score(row) -> float:
    ds  = (row.get("avg_direction_score") or 0) * 100
    ts  = (row.get("avg_target_score")    or 0) / 1.5 * 100
    alp = (row.get("avg_alpha")           or 0)
    con = (row.get("consistency")         or 0.5) * 100
    alp_norm = max(0.0, min(100.0, (alp + 30) / 60 * 100))
    return round(ds * 0.40 + ts * 0.25 + alp_norm * 0.25 + con * 0.10, 1)


def fmt_score(v, suffix="") -> str:
    if v is None:
        return "—"
    return f"{v:.2f}{suffix}"


def fmt_pct(v) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def color_direction(d: str) -> str:
    return {"buy": "🟢", "sell": "🔴", "hold": "🟡"}.get(d, "⚪")


# ─────────────────────────────────────────────
# SIDEBAR — FILTROS GLOBAIS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Analyst Tracker")
    st.markdown("---")

    page = st.radio(
        "Navegação",
        ["🏆 Ranking", "🔍 Analista", "📈 Ativo", "📰 Recomendações", "💼 Portfólio", "🎯 Risk Assessment", "📋 Clipping BR"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Filtros**")

    market_filter = st.multiselect(
        "Mercado",
        ["BR", "US"],
        default=["BR", "US"]
    )

    year_range = st.slider(
        "Período",
        min_value=2022,
        max_value=date.today().year,
        value=(2022, date.today().year)
    )

    min_recs = st.number_input("Mínimo de posições", min_value=1, value=1, step=1)

    st.markdown("---")
    st.caption(f"Banco: `{DB_PATH}`")

    if st.button("🔄 Limpar cache"):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────
# DADOS BASE
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_ranking(markets: list, min_recs: int, year_start: int, year_end: int) -> pd.DataFrame:
    placeholders = ",".join("?" * len(markets))
    df = query(f"""
        SELECT
            sc.analyst_id,
            a.name                 AS analyst,
            s.name                 AS firm,
            s.country,
            sc.hit_rate,
            sc.avg_direction_score,
            sc.avg_target_score,
            sc.avg_alpha,
            sc.consistency,
            sc.total_positions,
            sc.wins,
            sc.losses,
            sc.calc_date
        FROM analyst_scores sc
        JOIN analysts a ON a.id = sc.analyst_id
        JOIN sources  s ON s.id = a.source_id
        WHERE s.country IN ({placeholders})
          AND sc.total_positions >= ?
          AND sc.calc_date = (
              SELECT MAX(calc_date) FROM analyst_scores WHERE analyst_id = sc.analyst_id
          )
    """, tuple(markets) + (min_recs,))

    if df.empty:
        return df

    df["composite"] = df.apply(composite_score, axis=1)
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


@st.cache_data(ttl=300)
def load_recommendations(markets: list, year_start: int, year_end: int) -> pd.DataFrame:
    placeholders = ",".join("?" * len(markets))
    return query(f"""
        SELECT
            pos.id,
            pos.open_date          AS rec_date,
            pos.direction,
            pos.price_at_open      AS price_at_rec,
            pos.final_target       AS price_target,
            pos.horizon_days,
            a.name  AS analyst,
            s.name  AS firm,
            s.country,
            ast.ticker,
            ast.name AS asset_name,
            ast.sector,
            p.return_pct,
            p.direction_score,
            p.target_score,
            p.alpha_vs_spy,
            p.alpha_vs_ibov,
            p.eval_date
        FROM positions pos
        JOIN analysts  a   ON a.id   = pos.analyst_id
        JOIN sources   s   ON s.id   = a.source_id
        JOIN assets    ast ON ast.id = pos.asset_id
        LEFT JOIN performance p ON p.position_id = pos.id
        WHERE s.country IN ({placeholders})
          AND substr(pos.open_date, 1, 4) BETWEEN ? AND ?
        ORDER BY pos.open_date DESC
    """, tuple(markets) + (str(year_start), str(year_end)))


# ─────────────────────────────────────────────
# PAGE: RANKING
# ─────────────────────────────────────────────

if "Ranking" in page:
    st.markdown("# 🏆 Ranking de Analistas")
    st.caption(f"Score composto: Direction×40% + Target×25% + Alpha×25% + Consistency×10%")

    df = load_ranking(market_filter, min_recs, *year_range)

    if df.empty:
        st.warning("Nenhum dado encontrado. Rode o scoring_engine.py primeiro.")
        st.stop()

    # KPIs topo
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analistas rankeados",   len(df))
    c2.metric("Melhor direction score", fmt_score(df["avg_direction_score"].max()))
    c3.metric("Melhor alpha médio",     fmt_pct(df["avg_alpha"].max()))
    c4.metric("Maior consistência",     fmt_score(df["consistency"].max()))

    st.markdown("---")

    # Tabela de ranking
    display = df[[
        "analyst", "firm", "country",
        "avg_direction_score", "avg_target_score",
        "avg_alpha", "consistency",
        "total_positions", "composite"
    ]].copy()

    display.columns = [
        "Analista", "Firma", "País",
        "Dir Score", "Tgt Score",
        "Alpha %", "Consistency",
        "Posições", "Score"
    ]

    display["País"] = display["País"].map({"BR": "🇧🇷", "US": "🇺🇸"}).fillna("🌐")
    display["Dir Score"]   = display["Dir Score"].apply(lambda x: fmt_score(x))
    display["Tgt Score"]   = display["Tgt Score"].apply(lambda x: fmt_score(x))
    display["Alpha %"]     = display["Alpha %"].apply(fmt_pct)
    display["Consistency"] = display["Consistency"].apply(lambda x: fmt_score(x))
    display["Score"]       = display["Score"].apply(lambda x: f"{x:.1f}")

    st.dataframe(display, use_container_width=True, height=500)

    st.markdown("---")

    # Gráfico: scatter direction score vs alpha
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Direction Score vs Alpha")
        fig = px.scatter(
            df.dropna(subset=["avg_direction_score", "avg_alpha"]),
            x="avg_direction_score",
            y="avg_alpha",
            size="total_positions",
            color="country",
            hover_name="analyst",
            hover_data={"firm": True, "composite": True},
            color_discrete_map={"BR": "#3ecf8e", "US": "#4f9cf9"},
            labels={
                "avg_direction_score": "Direction Score",
                "avg_alpha": "Alpha médio (%)",
                "total_positions": "Posições",
            },
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(19,22,30,1)",
            font_color="#e2e6f0",
            showlegend=True,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="#6b7490", line_width=1)
        fig.add_vline(x=0.5, line_dash="dot", line_color="#6b7490", line_width=1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Distribuição de Scores")
        fig2 = px.histogram(
            df.dropna(subset=["composite"]),
            x="composite",
            nbins=20,
            color="country",
            color_discrete_map={"BR": "#3ecf8e", "US": "#4f9cf9"},
            labels={"composite": "Score Composto", "count": "Analistas"},
            barmode="overlay",
            opacity=0.75,
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(19,22,30,1)",
            font_color="#e2e6f0",
        )
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: ANALISTA
# ─────────────────────────────────────────────

elif "Analista" in page:
    st.markdown("# 🔍 Perfil do Analista")

    df_rank = load_ranking(market_filter, min_recs, *year_range)
    if df_rank.empty:
        st.warning("Nenhum analista com dados suficientes.")
        st.stop()

    analyst_list = df_rank["analyst"].tolist()
    selected = st.selectbox("Selecione o analista", analyst_list)

    row = df_rank[df_rank["analyst"] == selected].iloc[0]

    # Header do analista
    flag = "🇧🇷" if row["country"] == "BR" else "🇺🇸"
    st.markdown(f"## {flag} {selected}")
    st.caption(f"{row['firm']} · Score composto: **{row['composite']:.1f}**")

    st.markdown("---")

    # Métricas
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Direction Score",  fmt_score(row["avg_direction_score"]))
    c2.metric("Target Score",     fmt_score(row["avg_target_score"]))
    c3.metric("Alpha médio",      fmt_pct(row["avg_alpha"]))
    c4.metric("Consistency",      fmt_score(row["consistency"]))
    c5.metric("Posições",         int(row["total_positions"]))

    st.markdown("---")

    # Posições do analista
    df_recs = query("""
        SELECT
            pos.open_date AS rec_date, pos.direction,
            pos.price_at_open AS price_at_rec, pos.final_target AS price_target,
            ast.ticker, ast.name AS asset_name,
            p.return_pct, p.direction_score, p.target_score,
            COALESCE(p.alpha_vs_spy, p.alpha_vs_ibov) AS alpha,
            p.eval_date
        FROM positions pos
        JOIN analysts  a   ON a.id   = pos.analyst_id
        JOIN assets    ast ON ast.id = pos.asset_id
        LEFT JOIN performance p ON p.position_id = pos.id
        WHERE a.name = ?
        ORDER BY pos.open_date DESC
    """, (selected,))

    if not df_recs.empty:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Histórico de Recomendações")

            # Formatar para exibição
            disp = df_recs[[
                "rec_date", "ticker", "direction",
                "price_at_rec", "price_target",
                "return_pct", "direction_score", "alpha"
            ]].copy()
            disp.columns = ["Data", "Ticker", "Direção", "Preço entrada", "Target", "Retorno %", "Dir Score", "Alpha %"]
            disp["Direção"]  = disp["Direção"].map({"buy": "🟢 BUY", "sell": "🔴 SELL", "hold": "🟡 HOLD"})
            disp["Retorno %"] = disp["Retorno %"].apply(fmt_pct)
            disp["Alpha %"]   = disp["Alpha %"].apply(fmt_pct)
            disp["Dir Score"] = disp["Dir Score"].apply(fmt_score)

            st.dataframe(disp, use_container_width=True, height=350)

        with col2:
            st.markdown("### Por ticker")
            ticker_stats = df_recs.groupby("ticker").agg(
                recs=("direction_score", "count"),
                avg_dir=("direction_score", "mean"),
                avg_ret=("return_pct", "mean"),
            ).round(3).reset_index()
            ticker_stats = ticker_stats.sort_values("avg_dir", ascending=False)
            ticker_stats.columns = ["Ticker", "Recs", "Dir Score", "Ret% médio"]
            ticker_stats["Dir Score"] = ticker_stats["Dir Score"].apply(fmt_score)
            ticker_stats["Ret% médio"] = ticker_stats["Ret% médio"].apply(fmt_pct)
            st.dataframe(ticker_stats, use_container_width=True, hide_index=True)

        # Evolução do direction score ao longo do tempo
        st.markdown("### Evolução de performance")
        df_time = df_recs.dropna(subset=["direction_score"]).copy()
        df_time["rec_date"] = pd.to_datetime(df_time["rec_date"])
        df_time = df_time.sort_values("rec_date")

        if len(df_time) >= 3:
            # Rolling average (janela 5)
            df_time["rolling_dir"] = df_time["direction_score"].rolling(5, min_periods=1).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_time["rec_date"],
                y=df_time["direction_score"],
                mode="markers",
                marker=dict(color="#4f9cf9", size=6, opacity=0.5),
                name="Direction Score",
            ))
            fig.add_trace(go.Scatter(
                x=df_time["rec_date"],
                y=df_time["rolling_dir"],
                mode="lines",
                line=dict(color="#3ecf8e", width=2),
                name="Média móvel (5)",
            ))
            fig.add_hline(y=0.5, line_dash="dot", line_color="#6b7490", line_width=1,
                          annotation_text="Threshold 0.5")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(19,22,30,1)",
                font_color="#e2e6f0",
                yaxis=dict(range=[0, 1.1]),
                legend=dict(orientation="h"),
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Yearly Score Breakdown ──
        if SCORING_EXTRAS_AVAILABLE:
            st.markdown("---")
            st.markdown("### 📊 Scores Anuais")
            try:
                conn_yearly = get_conn()
                analyst_row = conn_yearly.execute(
                    "SELECT id FROM analysts WHERE name = ?", (selected,)
                ).fetchone()

                if analyst_row:
                    yearly = compute_yearly_scores(conn_yearly, analyst_row["id"])
                    if yearly:
                        trend = yearly[0].get("trend", "stable")
                        trend_icons = {"ascending": "📈 Em ascensão", "declining": "📉 Em declínio", "stable": "➡️ Estável"}
                        st.caption(f"Tendência: **{trend_icons.get(trend, trend)}**")

                        yearly_data = []
                        for yr in yearly:
                            yearly_data.append({
                                "Ano":       yr["year"],
                                "Posições":  yr["positions"],
                                "Hit Rate":  f"{yr['hit_rate']:.0%}" if yr["hit_rate"] is not None else "—",
                                "Direction": fmt_score(yr["avg_direction_score"]),
                                "Target":    fmt_score(yr["avg_target_score"]),
                                "Alpha":     fmt_pct(yr["avg_alpha"]),
                                "W/L":       f"{yr['wins']}/{yr['losses']}",
                            })

                        df_yearly = pd.DataFrame(yearly_data)
                        st.dataframe(df_yearly, use_container_width=True, hide_index=True)

                        # Direction score trend chart
                        if len(yearly) >= 2:
                            dir_scores = [
                                {"Ano": yr["year"], "Direction Score": yr["avg_direction_score"]}
                                for yr in yearly if yr["avg_direction_score"] is not None
                            ]
                            if len(dir_scores) >= 2:
                                df_trend = pd.DataFrame(dir_scores)
                                fig_trend = go.Figure()
                                fig_trend.add_trace(go.Scatter(
                                    x=df_trend["Ano"].astype(str),
                                    y=df_trend["Direction Score"],
                                    mode="lines+markers",
                                    line=dict(color="#3ecf8e", width=2),
                                    marker=dict(size=8),
                                    name="Direction Score",
                                ))
                                fig_trend.add_hline(
                                    y=0.5, line_dash="dot", line_color="#6b7490",
                                    annotation_text="Threshold 0.5"
                                )
                                fig_trend.update_layout(
                                    title="Evolução Direction Score por Ano",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(19,22,30,1)",
                                    font_color="#e2e6f0",
                                    yaxis=dict(range=[0, 1.1]),
                                    height=280,
                                )
                                st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.info("Sem dados anuais disponíveis para este analista.")
            except Exception as e:
                st.warning(f"Erro ao carregar scores anuais: {e}")

    else:
        st.info("Sem recomendações avaliadas para este analista.")




# ─────────────────────────────────────────────
# PAGE: PORTFÓLIO SIMULADO
# ─────────────────────────────────────────────

elif "Portfólio" in page:
    st.markdown("# 💼 Portfólio Simulado")
    st.caption('"Se você tivesse seguido esse analista, quanto teria ganho?"')

    df_rank_port = load_ranking(market_filter, min_recs, *year_range)
    if df_rank_port.empty:
        st.warning("Nenhum analista com dados suficientes.")
        st.stop()

    analyst_list_port = df_rank_port["analyst"].tolist()
    selected_port = st.selectbox("Selecione o analista", analyst_list_port, key="port_analyst")

    if not SCORING_EXTRAS_AVAILABLE:
        st.error("Módulo scoring_engine não disponível. Verifique a instalação.")
        st.stop()

    try:
        conn_port = get_conn()
        analyst_row_port = conn_port.execute(
            "SELECT id FROM analysts WHERE name = ?", (selected_port,)
        ).fetchone()

        if not analyst_row_port:
            st.warning("Analista não encontrado no banco.")
            st.stop()

        result = simulate_portfolio(conn_port, analyst_row_port["id"])

        if not result:
            st.info(f"Sem dados suficientes para simular portfólio de {selected_port}.")
            st.stop()

        # Cumulative metrics
        st.markdown("---")
        st.markdown("### Resultado Acumulado")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Retorno Total", f"{result['cumulative_return']:+.1f}%")
        mc2.metric("Benchmark Total", f"{result['cumulative_bench']:+.1f}%")
        mc3.metric("Alpha Total", f"{result['cumulative_alpha']:+.1f}%")
        mc4.metric("Total Posições", result["total_positions"])

        st.markdown("---")

        # Yearly breakdown table
        st.markdown("### Retornos Anuais")
        yearly_port_data = []
        for yr in result["years"]:
            best_str  = f"{yr['best_call']['ticker']} ({yr['best_call']['return_pct']:+.1f}%)" if yr["best_call"] else "—"
            worst_str = f"{yr['worst_call']['ticker']} ({yr['worst_call']['return_pct']:+.1f}%)" if yr["worst_call"] else "—"
            yearly_port_data.append({
                "Ano":       yr["year"],
                "Posições":  yr["n_positions"],
                "Retorno":   f"{yr['return_pct']:+.1f}%",
                "Benchmark": f"{yr['benchmark_return']:+.1f}%" if yr["benchmark_return"] is not None else "—",
                "Alpha":     f"{yr['alpha']:+.1f}%" if yr["alpha"] is not None else "—",
                "Melhor":    best_str,
                "Pior":      worst_str,
            })

        df_port = pd.DataFrame(yearly_port_data)
        st.dataframe(df_port, use_container_width=True, hide_index=True)

        # Equity curve chart
        st.markdown("---")
        st.markdown("### Curva de Equity")
        all_monthly = []
        carry_equity = 100.0
        for yr in result["years"]:
            year_months = yr.get("monthly_equity", [])
            if year_months:
                scale = carry_equity / 100.0
                for m in year_months:
                    all_monthly.append({"month": m["month"], "equity": round(m["equity"] * scale, 2)})
                carry_equity = all_monthly[-1]["equity"]

        if all_monthly:
            df_equity = pd.DataFrame(all_monthly)
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=df_equity["month"],
                y=df_equity["equity"],
                mode="lines",
                line=dict(color="#4f9cf9", width=2),
                fill="tozeroy",
                fillcolor="rgba(79,156,249,0.15)",
                name="Equity",
            ))
            fig_eq.add_hline(
                y=100, line_dash="dot", line_color="#6b7490",
                annotation_text="Investimento Inicial"
            )
            fig_eq.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(19,22,30,1)",
                font_color="#e2e6f0",
                xaxis_title="Mês",
                yaxis_title="Equity",
                height=350,
            )
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("Sem dados mensais disponíveis para a curva de equity.")

        # Annual returns bar chart
        st.markdown("---")
        st.markdown("### Retorno vs Benchmark por Ano")
        years_list = [yr["year"] for yr in result["years"]]
        returns_list = [yr["return_pct"] for yr in result["years"]]
        bench_list = [yr["benchmark_return"] if yr["benchmark_return"] is not None else 0 for yr in result["years"]]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=[str(y) for y in years_list],
            y=returns_list,
            name="Portfólio",
            marker_color="#3ecf8e",
        ))
        fig_bar.add_trace(go.Bar(
            x=[str(y) for y in years_list],
            y=bench_list,
            name="Benchmark",
            marker_color="#6b7490",
        ))
        fig_bar.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(19,22,30,1)",
            font_color="#e2e6f0",
            xaxis_title="Ano",
            yaxis_title="Retorno (%)",
            height=300,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao simular portfólio: {e}")


# ─────────────────────────────────────────────
# PAGE: ATIVO
# ─────────────────────────────────────────────

elif "Ativo" in page:
    st.markdown("# 📈 Análise por Ativo")

    tickers = query("""
        SELECT DISTINCT ast.ticker, ast.name, ast.country
        FROM positions pos
        JOIN assets ast ON ast.id = pos.asset_id
        WHERE ast.ticker NOT IN ('SPY', '^BVSP')
        ORDER BY ast.country, ast.ticker
    """)

    if tickers.empty:
        st.warning("Nenhum ativo com recomendações no banco.")
        st.stop()

    ticker_opts = tickers["ticker"].tolist()
    sel_ticker  = st.selectbox("Selecione o ativo", ticker_opts)

    asset_info = tickers[tickers["ticker"] == sel_ticker].iloc[0]
    flag = "🇧🇷" if asset_info["country"] == "BR" else "🇺🇸"
    st.markdown(f"## {flag} {sel_ticker} — {asset_info['name']}")
    st.markdown("---")

    # Melhores analistas para esse ativo
    df_best = query("""
        SELECT
            a.name  AS analyst,
            s.name  AS firm,
            s.country,
            COUNT(*) AS total,
            AVG(p.direction_score) AS avg_dir,
            AVG(p.target_score)    AS avg_tgt,
            AVG(p.return_pct)      AS avg_ret,
            AVG(COALESCE(p.alpha_vs_spy, p.alpha_vs_ibov)) AS avg_alpha,
            SUM(CASE WHEN p.direction_score >= 0.5 THEN 1 ELSE 0 END) AS wins
        FROM positions pos
        JOIN analysts  a   ON a.id   = pos.analyst_id
        JOIN sources   s   ON s.id   = a.source_id
        JOIN assets    ast ON ast.id = pos.asset_id
        LEFT JOIN performance p ON p.position_id = pos.id
        WHERE ast.ticker = ?
          AND p.direction_score IS NOT NULL
        GROUP BY a.id
        HAVING total >= 1
        ORDER BY avg_dir DESC
        LIMIT 15
    """, (sel_ticker,))

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(f"### Melhores analistas para {sel_ticker}")
        if not df_best.empty:
            disp = df_best[["analyst", "firm", "total", "avg_dir", "avg_tgt", "avg_ret", "avg_alpha"]].copy()
            disp.columns = ["Analista", "Firma", "Recs", "Dir Score", "Tgt Score", "Ret% médio", "Alpha"]
            disp["Dir Score"]  = disp["Dir Score"].apply(fmt_score)
            disp["Tgt Score"]  = disp["Tgt Score"].apply(fmt_score)
            disp["Ret% médio"] = disp["Ret% médio"].apply(fmt_pct)
            disp["Alpha"]      = disp["Alpha"].apply(fmt_pct)
            st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info("Sem dados de performance para este ativo ainda.")

    with col2:
        st.markdown("### Consenso atual")
        df_consensus = query("""
            SELECT direction, COUNT(*) AS n
            FROM positions pos
            JOIN assets ast ON ast.id = pos.asset_id
            WHERE ast.ticker = ?
              AND pos.open_date >= date('now', '-180 days')
            GROUP BY direction
        """, (sel_ticker,))

        if not df_consensus.empty:
            fig_pie = px.pie(
                df_consensus,
                names="direction",
                values="n",
                color="direction",
                color_discrete_map={"buy": "#3ecf8e", "hold": "#f9c74f", "sell": "#f96060"},
                hole=0.5,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e6f0",
                showlegend=True,
                height=260,
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Sem recomendações recentes (últimos 180 dias).")

    # Histórico de preços + recomendações
    st.markdown("---")
    st.markdown(f"### Preço histórico + recomendações")

    df_price = query("""
        SELECT ph.date, ph.close
        FROM price_history ph
        JOIN assets ast ON ast.id = ph.asset_id
        WHERE ast.ticker = ?
          AND ph.date >= ?
        ORDER BY ph.date
    """, (sel_ticker, f"{year_range[0]}-01-01"))

    df_recs_asset = query("""
        SELECT pos.open_date AS rec_date, pos.direction,
               pos.price_at_open AS price_at_rec, pos.final_target AS price_target,
               a.name AS analyst
        FROM positions pos
        JOIN assets    ast ON ast.id = pos.asset_id
        JOIN analysts  a   ON a.id   = pos.analyst_id
        WHERE ast.ticker = ?
          AND pos.price_at_open IS NOT NULL
          AND pos.open_date >= ?
        ORDER BY pos.open_date
    """, (sel_ticker, f"{year_range[0]}-01-01"))

    if not df_price.empty:
        fig_price = go.Figure()

        # Linha de preço
        fig_price.add_trace(go.Scatter(
            x=df_price["date"],
            y=df_price["close"],
            mode="lines",
            line=dict(color="#4f9cf9", width=1.5),
            name="Preço fechamento",
        ))

        # Marcadores de recomendação
        if not df_recs_asset.empty:
            color_map = {"buy": "#3ecf8e", "sell": "#f96060", "hold": "#f9c74f"}
            symbol_map = {"buy": "triangle-up", "sell": "triangle-down", "hold": "circle"}

            for direction in ["buy", "hold", "sell"]:
                sub = df_recs_asset[df_recs_asset["direction"] == direction]
                if sub.empty:
                    continue
                fig_price.add_trace(go.Scatter(
                    x=sub["rec_date"],
                    y=sub["price_at_rec"],
                    mode="markers",
                    marker=dict(
                        symbol=symbol_map[direction],
                        size=10,
                        color=color_map[direction],
                        line=dict(width=1, color="#0d0f14"),
                    ),
                    name=direction.upper(),
                    text=sub["analyst"],
                    hovertemplate="<b>%{text}</b><br>%{x}<br>$%{y:.2f}<extra></extra>",
                ))

        fig_price.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(19,22,30,1)",
            font_color="#e2e6f0",
            height=380,
            legend=dict(orientation="h", y=1.05),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#1a1e2a"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("Sem histórico de preços. Rode price_fetcher.py.")


# ─────────────────────────────────────────────
# PAGE: RECOMENDAÇÕES
# ─────────────────────────────────────────────

elif "Recomendações" in page:
    st.markdown("# 📰 Feed de Recomendações")

    df_recs = load_recommendations(market_filter, *year_range)

    if df_recs.empty:
        st.warning("Sem recomendações no período selecionado.")
        st.stop()

    # Filtros extras
    col1, col2, col3 = st.columns(3)
    with col1:
        dir_filter = st.multiselect("Direção", ["buy", "sell", "hold"], default=["buy", "sell", "hold"])
    with col2:
        ticker_filter = st.multiselect("Ticker", sorted(df_recs["ticker"].unique().tolist()))
    with col3:
        firm_filter = st.multiselect("Firma", sorted(df_recs["firm"].unique().tolist()))

    filtered = df_recs.copy()
    if dir_filter:
        filtered = filtered[filtered["direction"].isin(dir_filter)]
    if ticker_filter:
        filtered = filtered[filtered["ticker"].isin(ticker_filter)]
    if firm_filter:
        filtered = filtered[filtered["firm"].isin(firm_filter)]

    st.caption(f"{len(filtered)} recomendações encontradas")
    st.markdown("---")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    has_perf = filtered.dropna(subset=["direction_score"])
    c1.metric("Total recomendações",   len(filtered))
    c2.metric("Avaliadas",             len(has_perf))
    c3.metric("Dir Score médio",       fmt_score(has_perf["direction_score"].mean()) if not has_perf.empty else "—")
    c4.metric("Alpha médio",           fmt_pct(has_perf["alpha_vs_spy"].fillna(has_perf["alpha_vs_ibov"]).mean()) if not has_perf.empty else "—")

    st.markdown("---")

    # Tabela
    disp = filtered[[
        "rec_date", "ticker", "analyst", "firm", "country",
        "direction", "price_at_rec", "price_target",
        "return_pct", "direction_score"
    ]].copy()

    disp.columns = [
        "Data", "Ticker", "Analista", "Firma", "País",
        "Direção", "Entrada", "Target",
        "Retorno %", "Dir Score"
    ]
    disp["País"]      = disp["País"].map({"BR": "🇧🇷", "US": "🇺🇸"}).fillna("🌐")
    disp["Direção"]   = disp["Direção"].map({"buy": "🟢 BUY", "sell": "🔴 SELL", "hold": "🟡 HOLD"})
    disp["Retorno %"] = disp["Retorno %"].apply(fmt_pct)
    disp["Dir Score"] = disp["Dir Score"].apply(fmt_score)

    st.dataframe(disp, use_container_width=True, height=480)

    # Gráfico de distribuição de retornos
    if not has_perf.empty:
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Distribuição de retornos")
            fig = px.histogram(
                has_perf.dropna(subset=["return_pct"]),
                x="return_pct",
                color="direction",
                nbins=40,
                color_discrete_map={"buy": "#3ecf8e", "sell": "#f96060", "hold": "#f9c74f"},
                labels={"return_pct": "Retorno %"},
                opacity=0.8,
                barmode="overlay",
            )
            fig.add_vline(x=0, line_dash="dot", line_color="#6b7490")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(19,22,30,1)",
                font_color="#e2e6f0", height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Direction Score por firma")
            firm_perf = has_perf.groupby("firm")["direction_score"].agg(["mean", "count"])
            firm_perf = firm_perf[firm_perf["count"] >= 3].sort_values("mean", ascending=True)
            firm_perf = firm_perf.tail(15)

            fig2 = go.Figure(go.Bar(
                x=firm_perf["mean"],
                y=firm_perf.index,
                orientation="h",
                marker_color="#4f9cf9",
            ))
            fig2.add_vline(x=0.5, line_dash="dot", line_color="#6b7490")
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(19,22,30,1)",
                font_color="#e2e6f0", height=350,
                xaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PAGE: RISK ASSESSMENT
# ─────────────────────────────────────────────

elif "Risk" in page:
    st.markdown("# 🎯 Risk Assessment")
    st.caption("Probabilidade calibrada de uma call estar certa — antes de você seguir ela.")

    if not RISK_ENGINE_AVAILABLE:
        st.error("risk_engine.py não encontrado. Coloque-o na mesma pasta que dashboard.py.")
        st.stop()

    st.markdown("---")

    # ── FORMULÁRIO DE AVALIAÇÃO ──────────────────────
    st.markdown("### Avaliar uma call")

    # Carregar analistas e tickers disponíveis
    df_analysts = query("""
        SELECT DISTINCT a.name, s.country
        FROM analysts a JOIN sources s ON s.id = a.source_id
        ORDER BY a.name
    """)
    df_tickers = query("""
        SELECT DISTINCT ticker FROM assets
        WHERE ticker NOT IN ('SPY', '^BVSP')
        ORDER BY ticker
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_analyst = st.selectbox(
            "Analista",
            options=df_analysts["name"].tolist() if not df_analysts.empty else [""],
        )
        sel_direction = st.selectbox("Direção", ["buy", "sell", "hold"],
                                     format_func=lambda x: {"buy": "📈 BUY", "sell": "📉 SELL", "hold": "➡️ HOLD"}[x])

    with col2:
        sel_ticker = st.selectbox(
            "Ticker",
            options=df_tickers["ticker"].tolist() if not df_tickers.empty else [""],
        )
        sel_price = st.number_input("Preço atual ($)", min_value=0.01, value=100.0, step=0.5)

    with col3:
        sel_target = st.number_input("Preço-alvo ($)", min_value=0.0, value=0.0, step=0.5,
                                     help="Deixe 0 se não houver preço-alvo")
        sel_date   = st.date_input("Data da call", value=date.today())

    run_btn = st.button("🔍 Calcular probabilidade", type="primary", use_container_width=True)

    if run_btn and sel_analyst and sel_ticker and sel_price > 0:
        with st.spinner("Calculando dimensões de risco..."):
            result = _evaluate_call(
                ticker=sel_ticker,
                analyst_name=sel_analyst,
                direction=sel_direction,
                price_current=sel_price,
                price_target=sel_target if sel_target > 0 else None,
                rec_date=sel_date.isoformat(),
                verbose=False,
            )

        prob     = result["probability_pct"]
        rating   = result["rating"]
        emoji    = result["rating_emoji"]
        upside   = result.get("upside_pct")
        n_hist   = result["n_calls_history"]

        # ── RESULTADO PRINCIPAL ──────────────────────
        st.markdown("---")

        # Cor do gauge baseada na probabilidade
        if prob >= 75:
            gauge_color = "#3ecf8e"
        elif prob >= 62:
            gauge_color = "#f9c74f"
        elif prob >= 50:
            gauge_color = "#f9a74f"
        else:
            gauge_color = "#f96060"

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"suffix": "%", "font": {"size": 42, "color": "#e2e6f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6b7490",
                         "tickfont": {"color": "#6b7490"}},
                "bar":  {"color": gauge_color, "thickness": 0.25},
                "bgcolor": "#13161e",
                "bordercolor": "#252a38",
                "steps": [
                    {"range": [0,  38], "color": "rgba(249,96,96,0.12)"},
                    {"range": [38, 62], "color": "rgba(249,199,79,0.10)"},
                    {"range": [62, 100], "color": "rgba(62,207,142,0.10)"},
                ],
                "threshold": {
                    "line": {"color": gauge_color, "width": 3},
                    "thickness": 0.75,
                    "value": prob,
                },
            },
            title={"text": f"{emoji} {rating}", "font": {"size": 16, "color": "#e2e6f0"}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e6f0",
            height=280,
            margin=dict(t=40, b=10, l=20, r=20),
        )

        col_g, col_m = st.columns([1, 1])
        with col_g:
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_m:
            st.markdown(f"### {sel_ticker} {sel_direction.upper()}")
            st.markdown(f"**Analista:** {sel_analyst}")
            st.markdown(f"**Preço atual:** ${sel_price:.2f}")
            if sel_target > 0:
                st.markdown(f"**Target:** ${sel_target:.2f}  {'(+' + str(round(upside,1)) + '%)' if upside else ''}")
            st.markdown(f"**Histórico no banco:** {n_hist} calls")
            st.markdown("---")

            # Interpretação
            if prob >= 75:
                interp = "✅ Sinal forte — histórico e contexto favorecem essa call."
            elif prob >= 62:
                interp = "🟡 Sinal moderado — call razoável, monitore de perto."
            elif prob >= 50:
                interp = "🟠 Sinal neutro — incerteza considerável. Dimensione posição com cuidado."
            elif prob >= 38:
                interp = "🔴 Sinal fraco — histórico ou contexto desfavorável. Alta cautela."
            else:
                interp = "🚨 Sinal negativo — múltiplos fatores contra. Evite ou proteja com hedge."
            st.info(interp)

        # ── BREAKDOWN POR DIMENSÃO ───────────────────
        st.markdown("---")
        st.markdown("### Breakdown por dimensão")

        dim_labels = {
            "analyst_asset":  ("Hist. Analista × Ativo",   "Peso 30% — o mais importante"),
            "analyst_sector": ("Hist. Analista × Setor",   "Peso 20% — fallback quando sem histórico no ativo"),
            "magnitude":      ("Magnitude do upside",       "Peso 20% — calls agressivas têm menor base rate"),
            "consensus":      ("Alinhamento c/ consenso",   "Peso 10% — contrarian ou herd?"),
            "recency":        ("Recência da call",           "Peso 10% — call velha perde força"),
            "volatility_fit": ("Fit com volatilidade",      "Peso 10% — acerto varia com regime de mercado"),
        }

        dims = result["dimensions"]
        for key, (label, desc) in dim_labels.items():
            dim   = dims.get(key, {})
            score = dim.get("score", 0.5)
            text  = dim.get("label", "—")

            pct   = int(score * 100)
            color = "#3ecf8e" if score >= 0.65 else "#f9c74f" if score >= 0.45 else "#f96060"

            col_l, col_b, col_s = st.columns([3, 5, 1])
            with col_l:
                st.markdown(f"**{label}**")
                st.caption(desc)
            with col_b:
                st.progress(score, text=text)
            with col_s:
                st.markdown(f"<span style='color:{color};font-family:monospace;font-size:1.1rem;font-weight:600'>{score:.2f}</span>",
                            unsafe_allow_html=True)

            st.markdown("")  # espaçamento

    elif run_btn:
        st.warning("Preencha analista, ticker e preço atual.")

    # ── HISTÓRICO DE ASSESSMENTS SALVOS ─────────────
    st.markdown("---")
    st.markdown("### Calls já avaliadas")
    st.caption("Rode `python risk_engine.py --calc-all` para calcular em lote.")

    try:
        df_risk = query("""
            SELECT
                ra.probability, ra.rating,
                ra.dim_analyst_asset, ra.dim_magnitude, ra.dim_consensus,
                ra.upside_pct, ra.calc_date,
                pos.direction, pos.open_date AS rec_date,
                a.name  AS analyst,
                ast.ticker
            FROM risk_assessments ra
            JOIN positions   pos ON pos.id    = ra.position_id
            JOIN analysts    an  ON an.id     = pos.analyst_id
            JOIN assets      ast ON ast.id    = pos.asset_id
            LEFT JOIN analysts a ON a.id      = pos.analyst_id
            ORDER BY ra.probability DESC
            LIMIT 100
        """)

        if not df_risk.empty:
            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Calls avaliadas",    len(df_risk))
            c2.metric("Prob. média",        f"{df_risk['probability'].mean()*100:.1f}%")
            c3.metric("Alta confiança (>75%)", len(df_risk[df_risk['probability'] >= 0.75]))
            c4.metric("Baixa confiança (<50%)", len(df_risk[df_risk['probability'] < 0.50]))

            disp = df_risk[[
                "ticker", "analyst", "direction",
                "probability", "rating", "upside_pct",
                "dim_analyst_asset", "dim_magnitude", "rec_date"
            ]].copy()
            disp.columns = [
                "Ticker", "Analista", "Direção",
                "Prob.", "Rating", "Upside%",
                "Dim: Analista×Ativo", "Dim: Magnitude", "Data rec."
            ]
            disp["Prob."]    = disp["Prob."].apply(lambda x: f"{x*100:.1f}%")
            disp["Upside%"]  = disp["Upside%"].apply(lambda x: f"+{x:.1f}%" if x else "—")
            disp["Direção"]  = disp["Direção"].map({"buy": "🟢 BUY", "sell": "🔴 SELL", "hold": "🟡 HOLD"})
            disp["Dim: Analista×Ativo"] = disp["Dim: Analista×Ativo"].apply(fmt_score)
            disp["Dim: Magnitude"]      = disp["Dim: Magnitude"].apply(fmt_score)

            st.dataframe(disp, use_container_width=True, height=400, hide_index=True)

            # Scatter: Prob × Upside
            st.markdown("### Probabilidade vs Upside implícito")
            fig_s = px.scatter(
                df_risk.dropna(subset=["upside_pct", "probability"]),
                x="upside_pct",
                y=df_risk.dropna(subset=["upside_pct", "probability"])["probability"] * 100,
                color="direction",
                hover_name="analyst",
                hover_data={"ticker": True},
                color_discrete_map={"buy": "#3ecf8e", "sell": "#f96060", "hold": "#f9c74f"},
                labels={"x": "Upside implícito (%)", "y": "Probabilidade calibrada (%)"},
            )
            fig_s.add_hline(y=62, line_dash="dot", line_color="#6b7490", line_width=1,
                            annotation_text="threshold moderado")
            fig_s.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(19,22,30,1)",
                font_color="#e2e6f0", height=320,
            )
            st.plotly_chart(fig_s, use_container_width=True)

        else:
            st.info("Nenhum risk assessment calculado ainda. Rode `python risk_engine.py --calc-all`")

    except Exception:
        st.info("Tabela risk_assessments não encontrada. Rode `python risk_engine.py --calc-all`")

    # ── BACKLOG ──────────────────────────────────────
    st.markdown("---")
    with st.expander("📌 Backlog do Produto — Analyst Tracker", expanded=False):
        st.markdown("""
---
### 🟥 Alta prioridade

**Portfólio simulado por analista** *(scoring_engine + nova aba no dashboard)*
> "Se você tivesse seguido esse analista em 2023, quanto teria ganho?"

- Calcular retorno anual simulado por analista (2022, 2023, 2024 separados)
- Equal weight em cada call — cada recomendação aloca 1/N do portfólio
- Regra de posição aberta: não entra nova call no mesmo ativo se já tem posição
- Comparar vs benchmark por ano (SPY para US, Ibovespa para BR)
- Output:
  - Retorno % por ano + alpha vs benchmark
  - Retorno cumulativo 3 anos
  - Melhor e pior call individual
  - Equity curve mês a mês (gráfico de linha)
- Quanto mais período histórico, melhor a calibração — analista com 50 calls
  em 3 anos em regimes diferentes vale muito mais que 5 calls num bull market

**Scores anuais separados** *(scoring_engine)*
> Um analista que acertou 80% em 2022 e errou 70% em 2024 não é igual
> a um consistente nos três anos — o score médio esconde isso

- Calcular hit_rate, direction_score e alpha separado por ano
- Mostrar tabela anual no perfil do analista:
  ```
            2022    2023    2024
  Hit Rate   71%     65%     70%
  Dir Score  0.68    0.61    0.67
  Alpha      +8.2%  +12.1%   +5.4%
  ```
- Detectar analistas em declínio vs analistas em ascensão
- Usar tendência temporal como dimensão extra no risk engine

---
### 🟧 Média prioridade

**Dimensão 7 — Contexto macroeconômico e eventos externos** *(risk_engine)*
> Entender se eventos externos da época podem explicar erros do analista
> e ajustar a probabilidade de calls futuras

- Banco de eventos macroeconômicos indexados por data:
  choques de juros Fed/Copom, crises geopolíticas, crashes setoriais,
  earnings surprises
- Classificar erros em:
  - **Erro endógeno** — analista errou a tese, sem evento externo
  - **Erro exógeno** — evento imprevisível aconteceu após a call
- Analistas com muitos erros exógenos são reavaliados pra cima no score
- Se contexto macro atual é incerto (VIX elevado, guerra, FOMC próximo),
  penalizar probabilidades automaticamente
- Fontes: FRED API, GDELT, Yahoo Finance earnings, BCB Copom

**Cobertura histórica expandida** *(collectors)*
> Quanto mais período temporal, melhor o modelo se calibra

- Expandir coleta para 2019–2021 nas fontes disponíveis
- Capturar comportamento dos analistas em COVID (2020), rally (2021),
  crash de 2022, bull de 2023 — regimes muito diferentes
- Quanto mais regimes cobertos, mais o confidence_weight do risk engine
  vira algo genuinamente preditivo em vez de estimado

---
### 🟨 Baixa prioridade / Futuro

- **Dimensão 8 — Insider trading**: calls alinhadas com insider buying têm maior prob
- **Dimensão 9 — Revisão de estimativas**: analistas que revisam frequentemente
  têm mais confiabilidade que os que nunca revisam (sinal de engajamento ativo)
- **Dimensão 10 — Cross-analyst agreement**: quando 3+ analistas de casas
  diferentes fazem o mesmo call de forma independente, peso aumenta
- **Cobertura Europa e LATAM**: expandir além de BR e US
- **Crowdsourcing**: usuários submetem recomendações que viram, você valida
- **Alertas**: notificação quando analista com 78%+ de acerto faz nova call
- **API pública**: acesso programático aos scores para outros produtos
        """)


# ─────────────────────────────────────────────
# PAGE: CLIPPING BR
# ─────────────────────────────────────────────

elif "Clipping" in page:
    st.markdown("# 📋 Clipping BR — Revisão de Extrações")

    # Verificar se tabela existe
    try:
        df_pending = query("""
            SELECT
                e.id, e.ticker, e.analyst_name, e.source_house,
                e.direction, e.price_target, e.rec_date,
                e.confidence, e.notes, e.status,
                c.source_name, c.article_url, c.title
            FROM clipping_extractions e
            JOIN clipping_raw c ON c.id = e.clipping_id
            WHERE e.status = 'pending'
            ORDER BY e.confidence DESC
        """)
    except Exception:
        st.info("Tabela de clipping não encontrada. Rode collector_br.py primeiro.")
        st.stop()

    # Stats
    df_stats = query("""
        SELECT status, COUNT(*) AS n FROM clipping_extractions GROUP BY status
    """)

    if not df_stats.empty:
        cols = st.columns(len(df_stats))
        for i, (_, row) in enumerate(df_stats.iterrows()):
            label = {"pending": "⏳ Pendentes", "approved": "✅ Aprovadas",
                     "rejected": "❌ Rejeitadas", "imported": "📥 Importadas"}.get(row["status"], row["status"])
            cols[i].metric(label, int(row["n"]))

    st.markdown("---")

    if df_pending.empty:
        st.success("✅ Nenhuma extração pendente de revisão!")
    else:
        st.markdown(f"### {len(df_pending)} extrações aguardando revisão")
        st.caption("Use `python collector_br.py --approve ID` ou `--reject ID` no terminal.")

        for _, row in df_pending.iterrows():
            conf_color = "#3ecf8e" if row["confidence"] >= 0.75 else "#f9c74f" if row["confidence"] >= 0.5 else "#f96060"
            dir_icon   = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}.get(row["direction"], "⚪")
            pt         = f"R$ {row['price_target']:.2f}" if row["price_target"] else "sem target"

            with st.expander(
                f"#{row['id']} | {dir_icon} {row['analyst_name']} → {row['ticker']} {str(row['direction']).upper()} @ {pt} | conf: {row['confidence']:.0%}"
            ):
                c1, c2 = st.columns(2)
                c1.markdown(f"**Fonte:** {row['source_name']}")
                c1.markdown(f"**Data:** {row['rec_date'] or '—'}")
                c1.markdown(f"**Casa:** {row['source_house'] or '—'}")
                c2.markdown(f"**Confiança:** `{row['confidence']:.2f}`")
                c2.markdown(f"**Título:** {row['title'][:80] if row['title'] else '—'}")
                if row["notes"]:
                    st.caption(f"💬 {row['notes'][:200]}")
                st.markdown(f"🔗 [{row['article_url'][:70]}]({row['article_url']})")
                st.code(f"python collector_br.py --approve {row['id']}\npython collector_br.py --reject  {row['id']}")

    # Histórico de importadas
    st.markdown("---")
    st.markdown("### Recomendações importadas via clipping")
    try:
        df_imported = query("""
            SELECT e.ticker, e.analyst_name, e.direction, e.price_target,
                   e.rec_date, e.confidence, c.source_name
            FROM clipping_extractions e
            JOIN clipping_raw c ON c.id = e.clipping_id
            WHERE e.status = 'imported'
            ORDER BY e.rec_date DESC
            LIMIT 50
        """)
        if not df_imported.empty:
            df_imported.columns = ["Ticker", "Analista", "Direção", "Target", "Data", "Confiança", "Fonte"]
            df_imported["Direção"] = df_imported["Direção"].map({"buy": "🟢 BUY", "sell": "🔴 SELL", "hold": "🟡 HOLD"})
            st.dataframe(df_imported, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhuma recomendação importada ainda.")
    except Exception:
        st.info("Sem dados de clipping importados.")
