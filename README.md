# Analyst Tracker

Rastreia recomendações de analistas financeiros (BR e US) e avalia quem realmente acerta.

---

## Estrutura do projeto

```
analyst-tracker/
├── analyst_tracker_setup.py  # Schema do banco + seed data + helpers de posição
├── price_fetcher.py           # Baixa histórico de preços (Yahoo Finance)
├── collector_us.py            # Coleta ratings US (StockAnalysis.com)
├── collector_br.py            # Clipping BR + extração LLM (Claude)
├── scoring_engine.py          # Calcula direction score, target score, ranking
├── risk_engine.py             # Probabilidade calibrada de acerto por call
├── dashboard.py               # Interface Streamlit (6 páginas)
├── requirements.txt
└── analyst_tracker.db         # Criado automaticamente
```

---

## Setup inicial

```bash
# 1. Clonar / criar a pasta
mkdir analyst-tracker && cd analyst-tracker

# 2. Ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Variável de ambiente (necessária para collector_br.py)
export ANTHROPIC_API_KEY='sk-ant-...'

# 5. Inicializar banco + seed data
python analyst_tracker_setup.py
```

---

## Ordem de execução

```bash
# Passo 1 — Baixar preços históricos (2022–hoje)
python price_fetcher.py

# Passo 2 — Coletar ratings US (StockAnalysis.com)
python collector_us.py --since 2022-01-01

# Passo 3 — Clipping BR (requer ANTHROPIC_API_KEY)
python collector_br.py --all --since 2022-01-01
python collector_br.py --review       # revisar extrações pendentes

# Passo 4 — Calcular scores
python scoring_engine.py

# Passo 5 — Calcular risk assessments
python risk_engine.py --calc-all

# Passo 6 — Dashboard
streamlit run dashboard.py
```

---

## Modelo de dados — conceitos chave

### Posição vs Revisão

**Posição** (`positions`) = a tese do analista num ativo.
- Aberta quando analista inicia ou muda de rating
- Fechada quando muda de rating novamente
- É a **unidade de avaliação** — retorno medido do open ao close

**Revisão** (`recommendations`) = updates dentro de uma posição.
- `open` — abre a posição
- `target_up` — eleva preço-alvo (conviction crescendo)
- `target_down` — corta preço-alvo (conviction caindo)
- `reiterate` — mantém sem mudança
- `close` — encerra / muda de rating

### Por que isso importa

```
ERRADO (modelo antigo):
  Dan Ives BUY NVDA Jan → call 1
  Dan Ives BUY NVDA Mar → call 2  ← mesma tese, infla hit rate
  Dan Ives BUY NVDA Jun → call 3  ← mesma tese, infla hit rate

CERTO (modelo novo):
  Posição #1: Dan Ives LONG NVDA (Jan → Ago)
    revisão 1: open      Jan @ $145, target $220
    revisão 2: target_up Mar @ $320, target $350  ← conviction +
    revisão 3: target_up Jun @ $435, target $550  ← conviction +
    revisão 4: close     Ago @ $109, → HOLD
  Performance avaliada UMA vez, na posição inteira
```

### Helpers principais

```python
from analyst_tracker_setup import (
    get_connection, open_position, update_position,
    close_position, get_open_positions, get_position_history
)

conn = get_connection()

# Abrir posição
pos_id = open_position(conn, "Dan Ives", "NVDA", "buy",
                       "2023-01-15", 145.0, price_target=220.0)

# Elevar target (conviction +)
update_position(conn, pos_id, "2023-05-25", 320.0, new_target=350.0)

# Downgrade → fecha posição e abre nova automaticamente
close_position(conn, pos_id, "2024-08-15", 109.0, new_direction="hold")

# Ver histórico completo
hist = get_position_history(conn, pos_id)
```

---

## Scores explicados

### Direction Score (0 → 1)
Quão longe o ativo foi na direção certa, em espectro contínuo.
- `0.0` = errou completamente
- `0.5` = foi na direção certa mas só metade do esperado
- `1.0` = atingiu ou superou o retorno esperado

Substitui o hit rate binário — um analista que recomendou NVDA a $500 com target $650
e o ativo foi a $620 **não errou**, teve direction_score = 0.80.

### Target Score (0 → 1.5)
Quão perto chegou do preço-alvo.
- `0.0` = não saiu do lugar
- `1.0` = atingiu exatamente o target
- `>1.0` = superou o target (bonus, máx 1.5)

### Score Composto (0 → 100)
Ranking final ponderado:
- Direction Score × 40%
- Target Score × 25%
- Alpha vs benchmark × 25%
- Consistency × 10%

### Risk Assessment (0 → 100%)
Probabilidade calibrada de uma call estar certa, calculada em 6 dimensões:
1. Histórico analista × ativo específico (30%)
2. Histórico analista × setor (20%)
3. Magnitude do upside implícito (20%)
4. Alinhamento com consenso de mercado (10%)
5. Recência da call (10%)
6. Fit com regime atual de volatilidade (10%)

---

## Comandos úteis

```bash
# Price fetcher
python price_fetcher.py --ticker NVDA          # só um ativo
python price_fetcher.py --since 2023-01-01     # a partir de uma data
python price_fetcher.py --show PETR4.SA        # ver últimos preços

# Collector US
python collector_us.py --ticker NVDA           # só um ticker
python collector_us.py --stats                 # resumo do que foi coletado
python collector_us.py --analysts              # top analistas

# Collector BR
python collector_br.py --ticker VALE3          # só um ticker
python collector_br.py --review                # revisar extrações pendentes
python collector_br.py --approve 42            # aprovar extração #42
python collector_br.py --reject 43             # rejeitar extração #43
python collector_br.py --stats                 # resumo do clipping

# Scoring Engine
python scoring_engine.py                       # calcular tudo + ranking
python scoring_engine.py --ranking             # só ranking
python scoring_engine.py --ticker NVDA         # melhores analistas para um ativo
python scoring_engine.py --migrate             # migrar banco existente

# Risk Engine
python risk_engine.py --calc-all               # calcular todas as calls recentes
python risk_engine.py --profile "Dan Ives"     # perfil de risco de um analista
python risk_engine.py --ticker NVDA \
  --analyst "Dan Ives" --direction buy \
  --price 182 --target 300                     # avaliar call específica
```

---

## Backlog

### Alta prioridade
- **Portfólio simulado**: retorno anual se tivesse seguido cada analista (2022, 2023, 2024)
- **Scores anuais separados**: detectar analistas em ascensão vs declínio

### Média prioridade
- **Dimensão 7 — Eventos externos**: erros exógenos (macro) vs endógenos (tese errada)
- **Cobertura histórica**: expandir para 2019–2021 (COVID, rally, crash)

### Baixa prioridade / Futuro
- Insider trading como dimensão do risk engine
- Cross-analyst agreement
- Cobertura Europa e LATAM
- Crowdsourcing de recomendações
- Alertas de novas calls de analistas de alta confiança
- API pública

---

## Tickers cobertos

**US:** NVDA, AAPL, MSFT, AMZN, TSLA, META, GOOGL, AMD, NFLX, ORCL

**BR:** PETR4, VALE3, ITUB4, BBDC4, MGLU3, WEGE3, RENT3, RDOR3, PRIO3

**Benchmarks:** SPY (US), ^BVSP (BR)
