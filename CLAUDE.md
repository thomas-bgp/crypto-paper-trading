# Crypto Investment — Project Briefing

> Este arquivo e o ponto de entrada unico para qualquer Claude que trabalhe neste projeto.
> Leia ESTE arquivo primeiro. Ele mapeia TODO o conhecimento, estado atual e decisoes.

---

## O Projeto

Plataforma de pesquisa quantitativa para estrategias de crypto trading com foco em funding rate e momentum cross-sectional. Capital sub-$100k, sem HFT, max 5min candles para execucao.

**Stack:** Python | Binance API (dados) | HMM regime detection | Backtest event-driven

---

## Estado Atual do Projeto

### Implementado (codigo em `src/`)

| Modulo | Arquivo | Status | Descricao |
|--------|---------|--------|-----------|
| Data Fetcher | `src/data_fetcher.py` | OK | Binance klines 4h, funding rates BTC, Fear&Greed |
| Regime Detector | `src/regime_detector.py` | OK | HMM v2 com anti-whipsaw (temperature, hysteresis, min duration) |
| Momentum Signal | `src/strategies.py` | OK | Donchian Channel breakout + cross-sectional ranking |
| Backtester | `src/backtester.py` | OK | Event-driven com regime switch, custos modelados |
| Contrarian | `src/strategy_contrarian.py` | OK | Long-only RSI oversold (CAGR fraco: ~4.3%) |
| Pairs Trading | `src/strategy_pairs.py` | OK | Cointegration z-score (CAGR fraco: ~1.6%) |
| Portfolio Optimizer | `src/portfolio_optimizer.py` | OK | Markowitz + Ledoit-Wolf shrinkage |
| Dashboard | `src/dashboard.py` + `src/app.py` | OK | Streamlit visualization |

### NAO implementado (pesquisado mas sem codigo)

| Item | Pesquisa em | Prioridade |
|------|-------------|------------|
| CTREND signal (elastic net) | `docs/research/strategy3-ctrend-ensemble.md` | ALTA — upgrade do Donchian |
| Funding Rate Delta-Neutral | `docs/research/strategy2-funding-delta-neutral.md` | ALTA — ancora do portfolio |
| Multi-exchange data | `docs/knowledge/fontes-dados-apis.md` | MEDIA |
| Order book (L2) data | `docs/knowledge/metodologia-backtest.md` | BAIXA (fase 2+) |

### Resultados de Backtest (em `results/`)

| Estrategia | CAGR | Sharpe | Max DD | Arquivo |
|------------|------|--------|--------|---------|
| **Momentum Regime v2** | 69.61% | 1.48 | -35.71% | `backtest_result.parquet` |
| Contrarian | ~4.3% | 0.15 | — | `contrarian_result.parquet` |
| Pairs | ~1.6% | 0.55 | — | `pairs_result.parquet` |
| **Portfolio combinado** | ~37% | 1.43 | — | `portfolio_combined.parquet` |

Pesos otimizados: Momentum 60%, Contrarian 10%, Pairs 30%.

### Dados Coletados (em `data/`)

- 21 pares em 4h candles (2022-2026): BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, DOT, MATIC, LINK, LTC, ATOM, NEAR, ARB, OP, APT, SUI, INJ, TIA
- Funding rates: BTCUSDT (2022-2026)
- Fear & Greed Index (~1500 dias)

---

## Mapa de Conhecimento

> Toda pesquisa esta organizada em `docs/`. Cada arquivo e auto-contido.
> Use este mapa para saber ONDE esta o que voce precisa antes de implementar.

### `docs/knowledge/` — Base de conhecimento permanente

| Arquivo | Conteudo | Quando consultar |
|---------|----------|------------------|
| `literatura-academica.md` | 40 papers academicos organizados por tema (precificacao, alpha, microestrutura, risco, dados) | Antes de implementar qualquer estrategia nova |
| `fontes-dados-apis.md` | APIs de exchanges (Binance/Bybit/OKX/dYdX/Hyperliquid), provedores pagos, custos, limites | Ao coletar dados ou adicionar exchanges |
| `frameworks-backtest.md` | Comparativo de frameworks (DolphinDB, hftbacktest, vectorbt, Freqtrade, custom) | Se trocar engine de backtest |
| `repos-github.md` | 7 repos de referencia com analise (ccxt, hftbacktest, funding-rate-arbitrage, etc.) | Para referencia de implementacao |
| `industria-posts.md` | 9 fontes da industria (1Token, Amberdata, BSIC, CryptoQuant, etc.) | Para contexto de mercado |
| `funding-rate-practitioners.md` | Mecanica de calculo por exchange, estado do mercado mar/2026, Ethena, edge operacional | OBRIGATORIO antes de implementar funding rate |
| `metodologia-backtest.md` | Vieses, custos, metricas obrigatorias, checklist de validacao, arquitetura | OBRIGATORIO antes de qualquer backtest |
| `artigos-vertox/` | 26 artigos do VertoxQuant (Substack) | Referencia complementar |

### `docs/strategies/` — Pesquisa de estrategias

| Arquivo | Estrategia | Status | Resumo executivo |
|---------|-----------|--------|------------------|
| `catboost-short-only-baseline.md` | **BASELINE A BATER** — CatBoost Short-Only | AUDITADA | Sharpe 1.36, +933% em 4.5 anos. Shorta losers via ML ranking. Alpha vem de liquidez, nao momentum. TODOS os modelos futuros devem superar |
| `momentum-regime-v2.md` | #1 Momentum + Regime Switch | IMPLEMENTADA | CAGR 69.6%, Sharpe 1.48. Donchian + HMM v2. Funciona em bull, flat em sideways/bear |
| `funding-delta-neutral.md` | #2 Funding Rate Delta-Neutral | PESQUISADA | 12-18% aa, Sharpe 2-4, DD <5%. Unica market-neutral. Precisa de implementacao |
| `ctrend-ensemble.md` | Upgrade signal → CTREND | PESQUISADA | Elastic net JFQA 2024. Substitui Donchian, academicamente mais robusto |
| `contrarian-long-only.md` | Contrarian RSI | DESCARTADA | CAGR ~4.3%. Sem alpha |
| `pairs-cointegration.md` | Pairs Trading | MARGINAL | CAGR ~1.6%. Cointegracao instavel em crypto |
| `funding-rate-como-enhancement.md` | FR como overlay em outras estrategias | REFERENCIA | Como usar FR como filtro, sinal, carry overlay e cascade warning em momentum, mean rev, grid, portfolio alloc. Guia para futuros squads |
| `descartadas.md` | Grid, vol selling, DeFi LP, recursive lending, cross-exchange arb | DESCARTADAS | Com justificativas e dados |

### `docs/decisions/` — Registro de decisoes do comite

| Arquivo | Decisao |
|---------|---------|
| `2026-03-15-escopo-funding-rate-research.md` | Fase 0 aprovada, confianca 5/10 |
| `2026-03-15-estrategia-momentum-regime-v2.md` | Strategy #1 aprovada com metricas |
| `2026-03-15-estrategias-20pct-aprovadas.md` | Portfolio 3 camadas + regime switch |
| `2026-03-15-portfolio-markowitz-strategies.md` | Pesos otimizados + correlacoes |
| `2026-03-15-strategy-revision-post-research.md` | Revisao final: 17-18% aa maximo honesto |

### `docs/research/` — Pesquisas ativas e concluidas

| Arquivo | Topico | Status | Conclusao-chave |
|---------|--------|--------|-----------------|
| `momentum-factor-research-2026-03-16.md` | Momentum & Momentum Factor L/S | COMPLETO | Alpha real (CTREND t-stat 4.22). 30 papers, cost model, risk analysis. Live Sharpe 0.8-1.2 com solucoes tecnicas |
| `momentum-solucoes-tecnicas.md` | Solucoes para cada problema de momentum | COMPLETO | 12 problemas com solucao tecnica. Net esperado 14-23% aa pos-haircut ($100k+) |
| `factor-model-features-complete.md` | 112+ fatores para modelo cross-sectional ML | COMPLETO | CTREND 28 TA + momentum + vol + liquidity + on-chain + derivada polinomial + sentiment. Pipeline: LGBMRanker + ElasticNet + RF ensemble |

Artigos e posts sobre funding rate sendo varridos por outro prompt. Novos achados integrar em `docs/knowledge/`.

---

## Conclusoes-Chave (para decisoes rapidas)

1. **20% aa homogeneo NAO existe** em crypto sem alavancagem. Maximo honesto: 17-18% aa medio.
2. **Momentum + regime switch** e o melhor alpha generator (CAGR 69.6%), mas e direcional e flat em bear/sideways.
3. **Funding rate delta-neutral** e a unica estrategia com Sharpe >2 e market-neutral, mas alpha esta em compressao (Ethena, ETFs).
4. **CTREND (elastic net)** deve substituir Donchian — evidencia academica superior (JFQA 2024).
5. **Backtests superestimam**: live = 50-60% do backtestado. Sharpe 2.0 backtest → 1.0-1.5 live.
6. **Haircut obrigatorio**: 40-50% em qualquer resultado de backtest.
7. **Contrarian e Pairs** tem alpha insuficiente como standalone. Contrarian mantido a 10% por descorrelacao.

---

## Regras de Implementacao

### Custos (SEMPRE modelar)
```
Maker fee:  0.02% (Binance com BNB)
Taker fee:  0.075%
Slippage:   0.05% (base) + f(order_size / book_depth)
Total/lado: ~0.07% (maker + slippage)
```

### Metricas obrigatorias em todo backtest
- CAGR net de custos
- Sharpe + Sortino + Calmar
- Max Drawdown (esperar 1.5-2x em live)
- Decomposicao de PnL (por fonte de retorno)
- Walk-forward validation (treino 60% / teste 40%)
- Deflated Sharpe se testou >3 variantes

### Vieses a evitar
- Survivorship bias (incluir delistados)
- Look-ahead bias (usar indicative funding, NAO a final)
- Overfitting (DSR, MinBTL)
- Crowdedness (monitorar OI)

---

## Como Usar Este Projeto

### "Implementar estrategia X"
1. Leia `docs/strategies/[estrategia].md` para o research completo
2. Leia `docs/knowledge/metodologia-backtest.md` para o checklist
3. Veja `src/backtester.py` como template de implementacao
4. Siga os custos e metricas acima

### "Pesquisar topico Y"
1. Consulte `docs/knowledge/literatura-academica.md` para papers existentes
2. Consulte `docs/knowledge/industria-posts.md` para fontes da industria
3. Registre novos achados no arquivo de conhecimento relevante

### "Tomar decisao Z"
1. Consulte `docs/decisions/` para decisoes anteriores
2. Use o protocolo do Comite Quant (abaixo) para decisoes novas

---

## Protocolo do Comite Quant

Para decisoes que envolvem alocacao, nova estrategia ou mudanca de arquitetura:

### Roles
- **OPUS (CIO)**: Sintetiza, desafia, decide. Combate groupthink
- **SONNET-A (Alpha)**: Sinais, estatistica, modelagem. "Se parece bom demais, provavelmente e"
- **SONNET-R (Risk)**: Advogado do diabo. "Como isso morre?" Kill switch se confianca ≤ 3
- **SONNET-M (Microstructure)**: Execucao, slippage, capacity. "Funciona no mundo real?"
- **SONNET-D (Data/Infra)**: Viabilidade tecnica, qualidade de dados. "Os dados existem e sao confiaveis?"

### Processo (quando necessario)
1. **Round 1**: 4 Sonnets analisam em paralelo (independente)
2. **Round 2**: Cross-examination (cada um critica os outros)
3. **Round 3**: Opus sintetiza e decide
4. **Round 4** (opcional): Follow-up em gaps criticos

### Regras de ouro
- Dissent e obrigatorio
- Pre-mortem em toda estrategia: "como isso morre?"
- Numeros ou nao aconteceu
- Haircut 40-50% em backtests
- Kill switch: Risk confianca ≤ 3 = VETO
