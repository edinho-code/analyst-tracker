# Testing the Analyst Tracker Dashboard

## Prerequisites
- Python packages: `pip install pandas yfinance streamlit plotly`
- Database: `analyst_tracker.db` (SQLite) must exist with data

## Data Pipeline (run in order)
```bash
python analyst_tracker_setup.py    # Init DB + seed assets
python collector_us.py              # Collect US analyst ratings (Yahoo Finance primary, StockAnalysis.com fallback)
python price_fetcher.py             # Download price history via yfinance
python scoring_engine.py            # Compute scores + rankings
```

## Starting the Dashboard
```bash
streamlit run dashboard.py --server.port 8501 --server.headless true
```
Access at http://localhost:8501

## Key Dashboard Pages to Test
1. **Ranking** (default page) — Shows ranked analysts with KPI metrics (Ranked analysts count, Best direction score, Best avg alpha, Highest consistency)
2. **Recommendations** — Feed of all recommendations with date, ticker, analyst, firm. Can sort by clicking column headers. Supports direction/ticker/firm filters.
3. **Portfolio** — Simulated portfolio with Annual Returns table and Equity Curve chart. Select analyst from dropdown.
4. **Analyst** — Individual analyst profile with yearly scores breakdown and direction score trend chart.
5. **Asset** — Per-ticker view with consensus and price history chart.
6. **Risk Assessment** — Probability calculator for analyst calls.

## Verifying Data Completeness
```bash
python collector_us.py --stats      # Show summary of collected data
python -c "import sqlite3; c=sqlite3.connect('analyst_tracker.db').cursor(); print(c.execute('SELECT COUNT(*) FROM positions').fetchone()[0], 'positions')"
```

## Notes
- The collector uses Yahoo Finance (`yfinance`) as primary data source — provides 500-900+ ratings per ticker going back to ~2012
- StockAnalysis.com is fallback (limited to ~16 recent ratings per ticker on free tier)
- Yahoo Finance data attributes all ratings to "Research Team" per firm (no individual analyst names)
- The year range slider in the sidebar controls which data is displayed (currently 2019-current year)
- No lint/format config or pre-commit hooks in the repo
- No CI pipeline beyond Devin Review
