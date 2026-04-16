"""
Analyst Tracker — Dashboard
=============================
Complete visual interface for exploring rankings, performance
and recommendation history of BR and US analysts.

Usage:
    streamlit run dashboard.py

Dependencies:
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

# Import risk_engine and scoring_engine if available
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
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Analyst Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — refined dark theme, editorial typography
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0d0f14;
    --bg2:       #13161e;
    --bg3:       #1a1e2a;
    --border:    #2d3348;
    --text:      #eef0f6;
    --text-sec:  #c0c6d8;
    --muted:     #8d95b0;
    --accent:    #5ba8ff;
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
h1 { font-family: 'DM Serif Display', serif; font-size: 2.2rem; color: #ffffff; letter-spacing: -0.02em; }
h2 { font-family: 'DM Sans', sans-serif; font-weight: 600; font-size: 1.1rem; color: var(--text-sec); text-transform: uppercase; letter-spacing: 0.08em; }
h3 { font-family: 'DM Sans', sans-serif; font-weight: 500; color: var(--text); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg2);
    border-right: 1px solid var(--border);
}

/* Sidebar radio labels — high contrast */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
    color: var(--text) !important;
    font-size: 0.95rem !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
    color: var(--text) !important;
    font-size: 1rem !important;
    font-weight: 400 !important;
}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
    color: var(--muted) !important;
}

/* Caption text — more readable */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-sec) !important;
    font-size: 0.85rem !important;
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
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'DM Mono', monospace !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace;
    font-size: 1.6rem;
    color: #ffffff;
}

/* Dataframe — dark themed */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}
div[data-testid="stDataFrame"] table { color: var(--text) !important; }
div[data-testid="stDataFrame"] th {
    background-color: var(--bg3) !important;
    color: var(--text-sec) !important;
    border-bottom: 1px solid var(--border) !important;
}
div[data-testid="stDataFrame"] td {
    color: var(--text) !important;
    border-bottom: 1px solid rgba(45, 51, 72, 0.5) !important;
}
/* Glide data grid (Streamlit's default table renderer) */
div[data-testid="stDataFrame"] canvas + div {
    color: var(--text) !important;
}

/* Selectbox / inputs — readable text */
div[data-baseweb="select"] > div {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div[class*="ValueContainer"] {
    color: var(--text) !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="select"] input {
    color: var(--text) !important;
}
/* Dropdown menu */
ul[role="listbox"] {
    background-color: var(--bg3) !important;
}
ul[role="listbox"] li {
    color: var(--text) !important;
}
ul[role="listbox"] li:hover {
    background-color: var(--bg2) !important;
}

/* Number input */
div[data-testid="stNumberInput"] input {
    color: var(--text) !important;
    background-color: var(--bg3) !important;
}

/* Multiselect pills */
span[data-baseweb="tag"] {
    background-color: var(--accent) !important;
    color: #ffffff !important;
}

/* General paragraph & label text */
p, span, label, .stMarkdown {
    color: var(--text);
}

/* Info / warning / error boxes */
div[data-testid="stAlert"] {
    color: var(--text) !important;
}

/* Divider */
hr { border-color: var(--border); margin: 1.5rem 0; }

/* Score badge */
.badge-buy    { background: rgba(62,207,142,0.15); color: #3ecf8e; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; font-family: 'DM Mono', monospace; }
.badge-sell   { background: rgba(249,96,96,0.15);  color: #f96060; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; font-family: 'DM Mono', monospace; }
.badge-hold   { background: rgba(249,199,79,0.15); color: #f9c74f; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; font-family: 'DM Mono', monospace; }
.score-pill   { font-family: 'DM Mono', monospace; font-size: 0.85rem; }

/* Slider text */
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: var(--text-sec) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────

@st.cache_resource
def get_conn():
    if not Path(DB_PATH).exists():
        st.error(f"Database '{DB_PATH}' not found. Run analyst_tracker_setup.py first.")
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
# SIDEBAR — GLOBAL FILTERS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Analyst Tracker")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏆 Ranking", "🔍 Analyst", "📈 Asset", "📰 Recommendations", "💼 Portfolio", "🎯 Risk Assessment", "📋 Clipping BR"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Filters**")

    market_filter = st.multiselect(
        "Market",
        ["BR", "US"],
        default=["BR", "US"]
    )

    year_range = st.slider(
        "Period",
        min_value=2022,
        max_value=date.today().year,
        value=(2022, date.today().year)
    )

    min_recs = st.number_input("Minimum positions", min_value=1, value=1, step=1)

    st.markdown("---")
    st.caption(f"Database: `{DB_PATH}`")

    if st.button("🔄 Clear cache"):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────
# BASE DATA
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
    st.markdown("# 🏆 Analyst Ranking")
    st.caption(f"Composite score: Direction×40% + Target×25% + Alpha×25% + Consistency×10%")

    df = load_ranking(market_filter, min_recs, *year_range)

    if df.empty:
        st.warning("No data found. Run scoring_engine.py first.")
        st.stop()

    # Top KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ranked analysts",        len(df))
    c2.metric("Best direction score",   fmt_score(df["avg_direction_score"].max()))
    c3.metric("Best avg alpha",         fmt_pct(df["avg_alpha"].max()))
    c4.metric("Highest consistency",    fmt_score(df["consistency"].max()))

    st.markdown("---")

    # Ranking table
    display = df[[
        "analyst", "firm", "country",
        "avg_direction_score", "avg_target_score",
        "avg_alpha", "consistency",
        "total_positions", "composite"
    ]].copy()

    display.columns = [
        "Analyst", "Firm", "Country",
        "Dir Score", "Tgt Score",
        "Alpha %", "Consistency",
        "Positions", "Score"
    ]

    display["Country"] = display["Country"].map({"BR": "🇧🇷", "US": "🇺🇸"}).fillna("🌐")
    display["Dir Score"]   = display["Dir Score"].apply(lambda x: fmt_score(x))
    display["Tgt Score"]   = display["Tgt Score"].apply(lambda x: fmt_score(x))
    display["Alpha %"]     = display["Alpha %"].apply(fmt_pct)
    display["Consistency"] = display["Consistency"].apply(lambda x: fmt_score(x))
    display["Score"]       = display["Score"].apply(lambda x: f"{x:.1f}")

    st.dataframe(display, use_container_width=True, height=500)

    st.markdown("---")

    # Chart: scatter direction score vs alpha
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
                "avg_alpha": "Avg Alpha (%)",
                "total_positions": "Positions",
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
        st.markdown("### Score Distribution")
        fig2 = px.histogram(
            df.dropna(subset=["composite"]),
            x="composite",
            nbins=20,
            color="country",
            color_discrete_map={"BR": "#3ecf8e", "US": "#4f9cf9"},
            labels={"composite": "Composite Score", "count": "Analysts"},
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
# PAGE: ANALYST
# ─────────────────────────────────────────────

elif "Analyst" in page:
    st.markdown("# 🔍 Analyst Profile")

    df_rank = load_ranking(market_filter, min_recs, *year_range)
    if df_rank.empty:
        st.warning("No analysts with sufficient data.")
        st.stop()

    analyst_list = df_rank["analyst"].tolist()
    selected = st.selectbox("Select analyst", analyst_list)

    row = df_rank[df_rank["analyst"] == selected].iloc[0]

    # Analyst header
    flag = "🇧🇷" if row["country"] == "BR" else "🇺🇸"
    st.markdown(f"## {flag} {selected}")
    st.caption(f"{row['firm']} · Composite score: **{row['composite']:.1f}**")

    st.markdown("---")

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Direction Score",  fmt_score(row["avg_direction_score"]))
    c2.metric("Target Score",     fmt_score(row["avg_target_score"]))
    c3.metric("Avg Alpha",        fmt_pct(row["avg_alpha"]))
    c4.metric("Consistency",      fmt_score(row["consistency"]))
    c5.metric("Positions",        int(row["total_positions"]))

    st.markdown("---")

    # Analyst positions
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
            st.markdown("### Recommendation History")

            # Format for display
            disp = df_recs[[
                "rec_date", "ticker", "direction",
                "price_at_rec", "price_target",
                "return_pct", "direction_score", "alpha"
            ]].copy()
            disp.columns = ["Date", "Ticker", "Direction", "Entry Price", "Target", "Return %", "Dir Score", "Alpha %"]
            disp["Direction"]  = disp["Direction"].map({"buy": "🟢 BUY", "sell": "🔴 SELL", "hold": "🟡 HOLD"})
            disp["Return %"] = disp["Return %"].apply(fmt_pct)
            disp["Alpha %"]   = disp["Alpha %"].apply(fmt_pct)
            disp["Dir Score"] = disp["Dir Score"].apply(fmt_score)

            st.dataframe(disp, use_container_width=True, height=350)

        with col2:
            st.markdown("### By Ticker")
            ticker_stats = df_recs.groupby("ticker").agg(
                recs=("direction_score", "count"),
                avg_dir=("direction_score", "mean"),
                avg_ret=("return_pct", "mean"),
            ).round(3).reset_index()
            ticker_stats = ticker_stats.sort_values("avg_dir", ascending=False)
            ticker_stats.columns = ["Ticker", "Recs", "Dir Score", "Avg Ret%"]
            ticker_stats["Dir Score"] = ticker_stats["Dir Score"].apply(fmt_score)
            ticker_stats["Avg Ret%"] = ticker_stats["Avg Ret%"].apply(fmt_pct)
            st.dataframe(ticker_stats, use_container_width=True, hide_index=True)

        # Direction score evolution over time
        st.markdown("### Performance Evolution")
        df_time = df_recs.dropna(subset=["direction_score"]).copy()
        df_time["rec_date"] = pd.to_datetime(df_time["rec_date"])
        df_time = df_time.sort_values("rec_date")

        if len(df_time) >= 3:
            # Rolling average (window 5)
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
                name="Moving Avg (5)",
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
            st.markdown("### 📊 Yearly Scores")
            try:
                conn_yearly = get_conn()
                analyst_row = conn_yearly.execute(
                    "SELECT id FROM analysts WHERE name = ?", (selected,)
                ).fetchone()

                if analyst_row:
                    yearly = compute_yearly_scores(conn_yearly, analyst_row["id"])
                    if yearly:
                        trend = yearly[0].get("trend", "stable")
                        trend_icons = {"ascending": "📈 Ascending", "declining": "📉 Declining", "stable": "➡️ Stable"}
                        st.caption(f"Trend: **{trend_icons.get(trend, trend)}**")

                        yearly_data = []
                        for yr in yearly:
                            yearly_data.append({
                                    "Year":      yr["year"],
                                    "Positions": yr["positions"],
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
                                {"Year": yr["year"], "Direction Score": yr["avg_direction_score"]}
                                for yr in yearly if yr["avg_direction_score"] is not None
                            ]
                            if len(dir_scores) >= 2:
                                df_trend = pd.DataFrame(dir_scores)
                                fig_trend = go.Figure()
                                fig_trend.add_trace(go.Scatter(
                                    x=df_trend["Year"].astype(str),
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
                                    title="Direction Score by Year",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(19,22,30,1)",
                                    font_color="#e2e6f0",
                                    yaxis=dict(range=[0, 1.1]),
                                    height=280,
                                )
                                st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.info("No yearly data available for this analyst.")
            except Exception as e:
                st.warning(f"Error loading yearly scores: {e}")

    else:
        st.info("No evaluated recommendations for this analyst.")




# ─────────────────────────────────────────────
# PAGE: SIMULATED PORTFOLIO
# ─────────────────────────────────────────────

elif "Portfolio" in page:
    st.markdown("# 💼 Simulated Portfolio")
    st.caption('"If you had followed this analyst, how much would you have made?"')

    df_rank_port = load_ranking(market_filter, min_recs, *year_range)
    if df_rank_port.empty:
        st.warning("No analysts with sufficient data.")
        st.stop()

    analyst_list_port = df_rank_port["analyst"].tolist()
    selected_port = st.selectbox("Select analyst", analyst_list_port, key="port_analyst")

    if not SCORING_EXTRAS_AVAILABLE:
        st.error("scoring_engine module not available. Check installation.")
        st.stop()

    try:
        conn_port = get_conn()
        analyst_row_port = conn_port.execute(
            "SELECT id FROM analysts WHERE name = ?", (selected_port,)
        ).fetchone()

        if not analyst_row_port:
            st.warning("Analyst not found in database.")
            st.stop()

        result = simulate_portfolio(conn_port, analyst_row_port["id"])

        if not result:
            st.info(f"Not enough data to simulate portfolio for {selected_port}.")
            st.stop()

        # Cumulative metrics
        st.markdown("---")
        st.markdown("### Cumulative Result")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Return", f"{result['cumulative_return']:+.1f}%")
        mc2.metric("Benchmark Total", f"{result['cumulative_bench']:+.1f}%" if result['cumulative_bench'] is not None else "—")
        mc3.metric("Total Alpha", f"{result['cumulative_alpha']:+.1f}%" if result['cumulative_alpha'] is not None else "—")
        mc4.metric("Total Positions", result["total_positions"])

        st.markdown("---")

        # Yearly breakdown table
        st.markdown("### Annual Returns")
        yearly_port_data = []
        for yr in result["years"]:
            best_str  = f"{yr['best_call']['ticker']} ({yr['best_call']['return_pct']:+.1f}%)" if yr["best_call"] else "—"
            worst_str = f"{yr['worst_call']['ticker']} ({yr['worst_call']['return_pct']:+.1f}%)" if yr["worst_call"] else "—"
            yearly_port_data.append({
                "Year":      yr["year"],
                "Positions": yr["n_positions"],
                "Return":    f"{yr['return_pct']:+.1f}%",
                "Benchmark": f"{yr['benchmark_return']:+.1f}%" if yr["benchmark_return"] is not None else "—",
                "Alpha":     f"{yr['alpha']:+.1f}%" if yr["alpha"] is not None else "—",
                "Best":      best_str,
                "Worst":     worst_str,
            })

        df_port = pd.DataFrame(yearly_port_data)
        st.dataframe(df_port, use_container_width=True, hide_index=True)

        # Equity curve chart
        st.markdown("---")
        st.markdown("### Equity Curve")
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
                annotation_text="Initial Investment"
            )
            fig_eq.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(19,22,30,1)",
                font_color="#e2e6f0",
                xaxis_title="Month",
                yaxis_title="Equity",
                height=350,
            )
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("No monthly data available for equity curve.")

        # Annual returns bar chart
        st.markdown("---")
        st.markdown("### Return vs Benchmark by Year")
        years_list = [yr["year"] for yr in result["years"]]
        returns_list = [yr["return_pct"] for yr in result["years"]]
        bench_list = [yr["benchmark_return"] if yr["benchmark_return"] is not None else 0 for yr in result["years"]]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=[str(y) for y in years_list],
            y=returns_list,
            name="Portfolio",
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
            xaxis_title="Year",
            yaxis_title="Return (%)",
            height=300,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"Error simulating portfolio: {e}")


# ─────────────────────────────────────────────
# PAGE: ASSET
# ─────────────────────────────────────────────

elif "Asset" in page:
    st.markdown("# 📈 Asset Analysis")

    tickers = query("""
        SELECT DISTINCT ast.ticker, ast.name, ast.country
        FROM positions pos
        JOIN assets ast ON ast.id = pos.asset_id
        WHERE ast.ticker NOT IN ('SPY', '^BVSP')
        ORDER BY ast.country, ast.ticker
    """)

    if tickers.empty:
        st.warning("No assets with recommendations in database.")
        st.stop()

    ticker_opts = tickers["ticker"].tolist()
    sel_ticker  = st.selectbox("Select asset", ticker_opts)

    asset_info = tickers[tickers["ticker"] == sel_ticker].iloc[0]
    flag = "🇧🇷" if asset_info["country"] == "BR" else "🇺🇸"
    st.markdown(f"## {flag} {sel_ticker} — {asset_info['name']}")
    st.markdown("---")

    # Best analysts for this asset
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
        st.markdown(f"### Best Analysts for {sel_ticker}")
        if not df_best.empty:
            disp = df_best[["analyst", "firm", "total", "avg_dir", "avg_tgt", "avg_ret", "avg_alpha"]].copy()
            disp.columns = ["Analyst", "Firm", "Recs", "Dir Score", "Tgt Score", "Avg Ret%", "Alpha"]
            disp["Dir Score"]  = disp["Dir Score"].apply(fmt_score)
            disp["Tgt Score"]  = disp["Tgt Score"].apply(fmt_score)
            disp["Avg Ret%"] = disp["Avg Ret%"].apply(fmt_pct)
            disp["Alpha"]      = disp["Alpha"].apply(fmt_pct)
            st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info("No performance data for this asset yet.")

    with col2:
        st.markdown("### Current Consensus")
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
            st.info("No recent recommendations (last 180 days).")

    # Price history + recommendations
    st.markdown("---")
    st.markdown(f"### Price History + Recommendations")

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

        # Price line
        fig_price.add_trace(go.Scatter(
            x=df_price["date"],
            y=df_price["close"],
            mode="lines",
            line=dict(color="#4f9cf9", width=1.5),
            name="Closing Price",
        ))

        # Recommendation markers
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
        st.info("No price history. Run price_fetcher.py.")


# ─────────────────────────────────────────────
# PAGE: RECOMMENDATIONS
# ─────────────────────────────────────────────

elif "Recommendations" in page:
    st.markdown("# 📰 Recommendations Feed")

    df_recs = load_recommendations(market_filter, *year_range)

    if df_recs.empty:
        st.warning("No recommendations in the selected period.")
        st.stop()

    # Extra filters
    col1, col2, col3 = st.columns(3)
    with col1:
        dir_filter = st.multiselect("Direction", ["buy", "sell", "hold"], default=["buy", "sell", "hold"])
    with col2:
        ticker_filter = st.multiselect("Ticker", sorted(df_recs["ticker"].unique().tolist()))
    with col3:
        firm_filter = st.multiselect("Firm", sorted(df_recs["firm"].unique().tolist()))

    filtered = df_recs.copy()
    if dir_filter:
        filtered = filtered[filtered["direction"].isin(dir_filter)]
    if ticker_filter:
        filtered = filtered[filtered["ticker"].isin(ticker_filter)]
    if firm_filter:
        filtered = filtered[filtered["firm"].isin(firm_filter)]

    st.caption(f"{len(filtered)} recommendations found")
    st.markdown("---")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    has_perf = filtered.dropna(subset=["direction_score"])
    c1.metric("Total recommendations", len(filtered))
    c2.metric("Evaluated",              len(has_perf))
    c3.metric("Avg Dir Score",          fmt_score(has_perf["direction_score"].mean()) if not has_perf.empty else "—")
    c4.metric("Avg Alpha",              fmt_pct(has_perf["alpha_vs_spy"].fillna(has_perf["alpha_vs_ibov"]).mean()) if not has_perf.empty else "—")

    st.markdown("---")

    # Table
    disp = filtered[[
        "rec_date", "ticker", "analyst", "firm", "country",
        "direction", "price_at_rec", "price_target",
        "return_pct", "direction_score"
    ]].copy()

    disp.columns = [
        "Date", "Ticker", "Analyst", "Firm", "Country",
        "Direction", "Entry", "Target",
        "Return %", "Dir Score"
    ]
    disp["Country"]   = disp["Country"].map({"BR": "🇧🇷", "US": "🇺🇸"}).fillna("🌐")
    disp["Direction"]  = disp["Direction"].map({"buy": "🟢 BUY", "sell": "🔴 SELL", "hold": "🟡 HOLD"})
    disp["Return %"]  = disp["Return %"].apply(fmt_pct)
    disp["Dir Score"] = disp["Dir Score"].apply(fmt_score)

    st.dataframe(disp, use_container_width=True, height=480)

    # Return distribution chart
    if not has_perf.empty:
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Return Distribution")
            fig = px.histogram(
                has_perf.dropna(subset=["return_pct"]),
                x="return_pct",
                color="direction",
                nbins=40,
                color_discrete_map={"buy": "#3ecf8e", "sell": "#f96060", "hold": "#f9c74f"},
                labels={"return_pct": "Return %"},
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
            st.markdown("### Direction Score by Firm")
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
    st.caption("Calibrated probability of a call being right — before you follow it.")

    if not RISK_ENGINE_AVAILABLE:
        st.error("risk_engine.py not found. Place it in the same folder as dashboard.py.")
        st.stop()

    st.markdown("---")

    # ── EVALUATION FORM ─────────────────────────
    st.markdown("### Evaluate a Call")

    # Load available analysts and tickers
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
            "Analyst",
            options=df_analysts["name"].tolist() if not df_analysts.empty else [""],
        )
        sel_direction = st.selectbox("Direction", ["buy", "sell", "hold"],
                                     format_func=lambda x: {"buy": "📈 BUY", "sell": "📉 SELL", "hold": "➡️ HOLD"}[x])

    with col2:
        sel_ticker = st.selectbox(
            "Ticker",
            options=df_tickers["ticker"].tolist() if not df_tickers.empty else [""],
        )
        sel_price = st.number_input("Current price ($)", min_value=0.01, value=100.0, step=0.5)

    with col3:
        sel_target = st.number_input("Price target ($)", min_value=0.0, value=0.0, step=0.5,
                                     help="Leave 0 if there is no price target")
        sel_date   = st.date_input("Call date", value=date.today())

    run_btn = st.button("🔍 Calculate probability", type="primary", use_container_width=True)

    if run_btn and sel_analyst and sel_ticker and sel_price > 0:
        with st.spinner("Calculating risk dimensions..."):
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

        # ── MAIN RESULT ───────────────────────────
        st.markdown("---")

        # Gauge color based on probability
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
            st.markdown(f"**Analyst:** {sel_analyst}")
            st.markdown(f"**Current price:** ${sel_price:.2f}")
            if sel_target > 0:
                st.markdown(f"**Target:** ${sel_target:.2f}  {'(+' + str(round(upside,1)) + '%)' if upside else ''}")
            st.markdown(f"**History in database:** {n_hist} calls")
            st.markdown("---")

            # Interpretation
            if prob >= 75:
                interp = "Strong signal — history and context favor this call."
            elif prob >= 62:
                interp = "🟡 Moderate signal — reasonable call, monitor closely."
            elif prob >= 50:
                interp = "🟠 Neutral signal — considerable uncertainty. Size position carefully."
            elif prob >= 38:
                interp = "🔴 Weak signal — unfavorable history or context. High caution."
            else:
                interp = "🚨 Negative signal — multiple factors against. Avoid or hedge."
            st.info(interp)

        # ── DIMENSION BREAKDOWN ───────────────────────
        st.markdown("---")
        st.markdown("### Dimension Breakdown")

        dim_labels = {
            "analyst_asset":  ("Analyst × Asset History",   "Weight 30% — most important"),
            "analyst_sector": ("Analyst × Sector History",  "Weight 20% — fallback when no asset history"),
            "magnitude":      ("Upside Magnitude",          "Weight 20% — aggressive calls have lower base rate"),
            "consensus":      ("Consensus Alignment",       "Weight 10% — contrarian or herd?"),
            "recency":        ("Call Recency",               "Weight 10% — old calls lose strength"),
            "volatility_fit": ("Volatility Fit",            "Weight 10% — accuracy varies with market regime"),
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

            st.markdown("")  # spacing

    elif run_btn:
        st.warning("Fill in analyst, ticker and current price.")

    # ── SAVED ASSESSMENT HISTORY ─────────────────
    st.markdown("---")
    st.markdown("### Evaluated Calls")
    st.caption("Run `python risk_engine.py --calc-all` to calculate in batch.")

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
            c1.metric("Evaluated calls",     len(df_risk))
            c2.metric("Avg probability",      f"{df_risk['probability'].mean()*100:.1f}%")
            c3.metric("High confidence (>75%)", len(df_risk[df_risk['probability'] >= 0.75]))
            c4.metric("Low confidence (<50%)",  len(df_risk[df_risk['probability'] < 0.50]))

            disp = df_risk[[
                "ticker", "analyst", "direction",
                "probability", "rating", "upside_pct",
                "dim_analyst_asset", "dim_magnitude", "rec_date"
            ]].copy()
            disp.columns = [
                "Ticker", "Analyst", "Direction",
                "Prob.", "Rating", "Upside%",
                "Dim: Analyst×Asset", "Dim: Magnitude", "Rec. Date"
            ]
            disp["Prob."]    = disp["Prob."].apply(lambda x: f"{x*100:.1f}%")
            disp["Upside%"]  = disp["Upside%"].apply(lambda x: f"+{x:.1f}%" if x else "—")
            disp["Direction"]  = disp["Direction"].map({"buy": "🟢 BUY", "sell": "🔴 SELL", "hold": "🟡 HOLD"})
            disp["Dim: Analyst×Asset"] = disp["Dim: Analyst×Asset"].apply(fmt_score)
            disp["Dim: Magnitude"]      = disp["Dim: Magnitude"].apply(fmt_score)

            st.dataframe(disp, use_container_width=True, height=400, hide_index=True)

            # Scatter: Prob × Upside
            st.markdown("### Probability vs Implied Upside")
            fig_s = px.scatter(
                df_risk.dropna(subset=["upside_pct", "probability"]),
                x="upside_pct",
                y=df_risk.dropna(subset=["upside_pct", "probability"])["probability"] * 100,
                color="direction",
                hover_name="analyst",
                hover_data={"ticker": True},
                color_discrete_map={"buy": "#3ecf8e", "sell": "#f96060", "hold": "#f9c74f"},
                labels={"x": "Implied Upside (%)", "y": "Calibrated Probability (%)"},
            )
            fig_s.add_hline(y=62, line_dash="dot", line_color="#6b7490", line_width=1,
                            annotation_text="moderate threshold")
            fig_s.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(19,22,30,1)",
                font_color="#e2e6f0", height=320,
            )
            st.plotly_chart(fig_s, use_container_width=True)

        else:
            st.info("No risk assessment calculated yet. Run `python risk_engine.py --calc-all`")

    except Exception:
        st.info("risk_assessments table not found. Run `python risk_engine.py --calc-all`")

    # ── BACKLOG ──────────────────────────────────────
    st.markdown("---")
    with st.expander("📌 Product Backlog — Analyst Tracker", expanded=False):
        st.markdown("""
---
### 🟥 High Priority

**Simulated portfolio per analyst** *(scoring_engine + new dashboard tab)*
> "If you had followed this analyst in 2023, how much would you have made?"

- Calculate simulated annual return per analyst (2022, 2023, 2024 separately)
- Equal weight per call — each recommendation allocates 1/N of the portfolio
- Open position rule: no new call on same asset if position already exists
- Compare vs benchmark per year (SPY for US, Ibovespa for BR)
- Output:
  - Return % per year + alpha vs benchmark
  - Cumulative return over 3 years
  - Best and worst individual call
  - Month-by-month equity curve (line chart)
- The more historical coverage, the better the calibration — an analyst with 50 calls
  over 3 years in different regimes is worth much more than 5 calls in a bull market

**Separate yearly scores** *(scoring_engine)*
> An analyst who was 80% right in 2022 and 70% wrong in 2024 is not the same
> as one consistent across all three years — the average score hides this

- Calculate hit_rate, direction_score and alpha separately per year
- Show yearly table on analyst profile:
  ```
            2022    2023    2024
  Hit Rate   71%     65%     70%
  Dir Score  0.68    0.61    0.67
  Alpha      +8.2%  +12.1%   +5.4%
  ```
- Detect declining vs ascending analysts
- Use temporal trend as an extra dimension in the risk engine

---
### 🟧 Medium Priority

**Dimension 7 — Macroeconomic context and external events** *(risk_engine)*
> Understand if external events at the time can explain analyst errors
> and adjust probability of future calls

- Database of macroeconomic events indexed by date:
  Fed/Copom rate shocks, geopolitical crises, sector crashes,
  earnings surprises
- Classify errors as:
  - **Endogenous error** — analyst got the thesis wrong, no external event
  - **Exogenous error** — unpredictable event occurred after the call
- Analysts with many exogenous errors are re-evaluated upward in score
- If current macro context is uncertain (high VIX, war, upcoming FOMC),
  automatically penalize probabilities
- Sources: FRED API, GDELT, Yahoo Finance earnings, BCB Copom

**Expanded historical coverage** *(collectors)*
> The more temporal coverage, the better the model calibrates

- Expand collection to 2019–2021 from available sources
- Capture analyst behavior during COVID (2020), rally (2021),
  2022 crash, 2023 bull — very different regimes
- The more regimes covered, the more the confidence_weight in risk engine
  becomes genuinely predictive instead of estimated

---
### 🟨 Low Priority / Future

- **Dimension 8 — Insider trading**: calls aligned with insider buying have higher prob
- **Dimension 9 — Estimate revisions**: analysts who revise frequently
  have more reliability than those who never revise (active engagement signal)
- **Dimension 10 — Cross-analyst agreement**: when 3+ analysts from different
  firms make the same call independently, weight increases
- **Europe and LATAM coverage**: expand beyond BR and US
- **Crowdsourcing**: users submit recommendations they see, you validate
- **Alerts**: notification when analyst with 78%+ accuracy makes a new call
- **Public API**: programmatic access to scores for other products
        """)


# ─────────────────────────────────────────────
# PAGE: CLIPPING BR
# ─────────────────────────────────────────────

elif "Clipping" in page:
    st.markdown("# 📋 Clipping BR — Extraction Review")

    # Check if table exists
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
        st.info("Clipping table not found. Run collector_br.py first.")
        st.stop()

    # Stats
    df_stats = query("""
        SELECT status, COUNT(*) AS n FROM clipping_extractions GROUP BY status
    """)

    if not df_stats.empty:
        cols = st.columns(len(df_stats))
        for i, (_, row) in enumerate(df_stats.iterrows()):
            label = {"pending": "⏳ Pending", "approved": "✅ Approved",
                     "rejected": "❌ Rejected", "imported": "📥 Imported"}.get(row["status"], row["status"])
            cols[i].metric(label, int(row["n"]))

    st.markdown("---")

    if df_pending.empty:
        st.success("No pending extractions to review!")
    else:
        st.markdown(f"### {len(df_pending)} extractions awaiting review")
        st.caption("Use `python collector_br.py --approve ID` or `--reject ID` in the terminal.")

        for _, row in df_pending.iterrows():
            conf_color = "#3ecf8e" if row["confidence"] >= 0.75 else "#f9c74f" if row["confidence"] >= 0.5 else "#f96060"
            dir_icon   = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}.get(row["direction"], "⚪")
            pt         = f"R$ {row['price_target']:.2f}" if row["price_target"] else "no target"

            with st.expander(
                f"#{row['id']} | {dir_icon} {row['analyst_name']} → {row['ticker']} {str(row['direction']).upper()} @ {pt} | conf: {row['confidence']:.0%}"
            ):
                c1, c2 = st.columns(2)
                c1.markdown(f"**Source:** {row['source_name']}")
                c1.markdown(f"**Date:** {row['rec_date'] or '—'}")
                c1.markdown(f"**Firm:** {row['source_house'] or '—'}")
                c2.markdown(f"**Confidence:** `{row['confidence']:.2f}`")
                c2.markdown(f"**Title:** {row['title'][:80] if row['title'] else '—'}")
                if row["notes"]:
                    st.caption(f"💬 {row['notes'][:200]}")
                st.markdown(f"🔗 [{row['article_url'][:70]}]({row['article_url']})")
                st.code(f"python collector_br.py --approve {row['id']}\npython collector_br.py --reject  {row['id']}")

    # Imported history
    st.markdown("---")
    st.markdown("### Recommendations imported via clipping")
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
            df_imported.columns = ["Ticker", "Analyst", "Direction", "Target", "Date", "Confidence", "Source"]
            df_imported["Direction"] = df_imported["Direction"].map({"buy": "🟢 BUY", "sell": "🔴 SELL", "hold": "🟡 HOLD"})
            st.dataframe(df_imported, use_container_width=True, hide_index=True)
        else:
            st.info("No imported recommendations yet.")
    except Exception:
        st.info("No imported clipping data.")
