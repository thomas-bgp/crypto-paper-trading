# Pesquisa: Momentum & Momentum Factor Long/Short em Crypto

> **Data:** 2026-03-16
> **Comite:** OPUS (CIO) + 8 Sonnets (4 squads × 2), Round 1 + Round 2 cross-examination
> **Escopo:** Estrategias SOTA de momentum, viabilidade para $1k-$1M, sem dados pagos
> **Status:** COMPLETO — Round 3 (Sintese CIO)

---

## DECISAO DO CIO

### Veredito: MOMENTUM CROSS-SECTIONAL E REAL MAS NAO VIAVEL COMO PROPOSTO PARA RETAIL

O alpha de momentum em crypto e academicamente robusto (t-stat 4.22, JFQA 2024). Porem, apos custos reais, haircut de publicacao, e ajuste de viabilidade operacional, o retorno liquido esperado para um portfolio de $100k e **~4-12% aa** — marginalmente melhor que BTC com filtro SMA que custa quase nada.

### Recomendacao Final: ESTRATEGIA SIMPLIFICADA

| Para quem | Estrategia recomendada |
|-----------|----------------------|
| $1k-$25k | NAO IMPLEMENTAR momentum cross-sectional. Usar BTC+ETH (70/30) com filtro 200d SMA |
| $25k-$100k | BTC+ETH+SOL regime-filtered. Opcional: top 5 momentum mensal como overlay leve |
| $100k-$500k | Momentum cross-sectional mensal simplificado (ver spec abaixo) |
| $500k-$1M | Full spec com TWAP execution, inv-vol sizing, composite regime filter |

### Confianca do Comite

| Membro | R1 | R2 | R3 (pos-solucoes) |
|--------|----|----|-------------------|
| ALPHA-A | 5.5/10 | — | — |
| ALPHA-B | 5.5/10 | — | — |
| ALPHA (R2 consolidado) | — | 4.5/10 | 6/10 |
| CONSTRUCTION (R2) | — | 5/10 | 7/10 |
| FEASIBILITY (R2) | — | 2/10 | 5/10 |
| **OPUS (CIO)** | — | — | **6.5/10** |

**Media ponderada pos-solucoes: ~6/10** — Alpha real, implementavel com engenharia correta.
Ver `momentum-solucoes-tecnicas.md` para solucao tecnica de cada problema.

---

## PARTE I — ACHADOS ACADEMICOS CONSOLIDADOS

### Papers Fundamentais (por relevancia)

| # | Paper | Achado-chave | Sharpe | Caveat |
|---|-------|-------------|--------|--------|
| 1 | **CTREND** (Fieberg et al., JFQA 2024) | Elastic net sobre 28 indicadores, alpha semanal 2.62%, t-stat 4.22. Melhor fator publicado | Mediana 1.34 em 2000+ variantes | Sample ends May 2022. Overfit risk com 28 features |
| 2 | **Catching Crypto Trends** (Zarattini et al., SSRN 2025) | Donchian ensemble (5-360 dias). BTC Sharpe 1.58, diversificado 1.57. 10 anos de backtest | 1.57 | Survivorship bias em altcoins. DD 11% implausivel |
| 3 | **TSMOM Crypto** (Han, Kang, Ryu, 2024) | TSMOM > CSMOM. Otimo: 28d lookback, 5d hold. Sharpe 1.51 | 1.51 | Cross-sectional quase zero apos custos realistas |
| 4 | **Risk-Managed Momentum** (FRL 2025) | Variance scaling: Sharpe 1.12→1.42. +27% melhoria | 1.42 | Melhora modesta em termos absolutos |
| 5 | **Momentum is an Illusion?** (Grobys, 2025) | Power-law α<3 → variancia INFINITA. Sharpe ratios matematicamente indefinidos | N/A | CRITICO: invalida toda a framework estatistica se verdadeiro |
| 6 | **Survivorship Bias** (Ammann et al., 2023) | Equal-weight bias: +62% aa. Value-weight: +0.93% | — | Invalida maioria dos backtests equal-weight |
| 7 | **Stop-Loss + Momentum** (Sadaqat & Butt, 2023) | Raw CSMOM Sharpe = **-0.235** sem stop-loss. DESTROI valor | -0.235 | Stop-loss e OBRIGATORIO, nao opcional |
| 8 | **Common Risk Factors** (Liu, Tsyvinski, Wu, JF 2022) | Momentum funciona 4.2x melhor em large-caps (4.2%/semana vs 0.6%) | — | 2-week lookback, high turnover |
| 9 | **DS3 Lasso** (2026) | Apenas 13 de 49 anomalias crypto sobrevivem 2014-2023 | — | Momentum (MOM2) e uma das 13 |
| 10 | **Man Group "In Crypto We Trend"** (2024) | Otimo: 10-15 coins. Crypto pairwise corr ~0.6 | — | Practitioner grade |

### Fontes de Industria

| Fonte | Achado | Status |
|-------|--------|--------|
| XBTO Trend Strategy | Sharpe 1.62 live (2020-2025) | Nao replicavel (proprietario) |
| Hashdex HAMO ETP | Momentum factor ETP, inv-vol sizing | Acessivel mas dados nao publicados |
| Bitwise Trendwise | Momentum-based rotation | ETF acessivel |
| Freqtrade ADXMomentum | ADX>25 + MOM(14) no 1h | Template util, sem backtest longo |
| NostalgiaForInfinity | 2.29MB black-box, 3000 stars | Regime-sensitive, nao transparente |

---

## PARTE II — O PROBLEMA DO SHORT LEG

### Custos anuais do short book via perpetuals (Binance)

| Componente | Custo anual |
|-----------|------------|
| Funding rate (bull market, alts) | 20-55% |
| Funding rate (neutro, BTC) | ~11% |
| Futures taker fee | 1.5% |
| Spread + impact | ~2% |
| **Total short book (bull)** | **25-60%** |

### Veredito do Comite

**Long-short e INVIAVEL para capital < $1M.** O short leg custa mais que o alpha que gera. A evidencia academica mostra que o alpha vem "largely from short positions" (Fieberg 2024), mas o custo de implementar shorts em crypto e proibitivo.

**Long-only + cash e a unica opcao viavel para retail.**

---

## PARTE III — MODELO DE CUSTOS COMPLETO

### $100k, long-only, 20 assets, Binance spot

| Cenario | Custo anual | % AUM |
|---------|------------|-------|
| Monthly rebalance | $2,730 | 2.73% |
| Weekly rebalance | $5,270 | 5.27% |
| Daily rebalance | ~$51,000 | ~51% |

### Break-even alpha

| Tipo | Custo anual | Alpha minimo |
|------|------------|-------------|
| Long-only mensal | 2.73% | 2.73% |
| Long-only semanal | 5.27% | 5.27% |
| Long-short mensal | 12.60% | 12.60% |

### Haircut backtest → live

| Fonte de degradacao | Haircut |
|--------------------|---------|
| Overfitting / data snooping | 30-40% |
| Execucao (slippage vs mid-price) | 15-25% |
| Decay de parametros | 10-20% |
| Survivorship / backfill | 10-15% |
| **Cumulativo** | **55-70%** |

**Backtest Sharpe 1.0-1.5 → Live Sharpe 0.35-0.65**

---

## PARTE IV — ANALISE DE VIABILIDADE ($100k)

```
Retorno bruto esperado (Sharpe 0.45 × vol 35%):  +15.75%  = +$15,750
Custos (weekly rebalance):                         -5.27%  = -$5,270
Cash drag do regime filter (~25% do ano):          -2.81%  = -$2,810
Slippage alem do backtest:                         -1.50%  = -$1,500
Decay de parametros:                               -2.00%  = -$2,000
                                                  --------
RETORNO LIQUIDO ESPERADO:                          ~4.17%  = ~$4,170
```

**$4,170 de retorno esperado com banda de 1-sigma de -$25,000 a +$55,000.**

### Comparacao com BTC+200d SMA (quase gratis)

| Metrica | CS Momentum (live est.) | BTC 200d SMA |
|---------|----------------------|-------------|
| Retorno anual esperado | 4-12% net | 25-45% (em bull) |
| Custo anual | 2.7-5.3% | 0.04-0.08% |
| Complexidade | Alta | Minima |
| Sharpe live estimado | 0.35-0.65 | 0.8-1.3 |
| Max DD | -40 a -70% | -45 a -55% |

---

## PARTE V — RISCOS DOCUMENTADOS

### Top 5 por expected loss

| # | Risco | Prob anual | Magnitude | EL anual |
|---|-------|-----------|-----------|----------|
| 1 | Cascade/flash crash (Oct 2025-type) | 50-70% | -20 a -70% | ~25-35% |
| 2 | Regime failure (whipsaw 12-18 meses) | 25-35% | -40 a -65% | ~15-20% |
| 3 | Exchange failure/freeze | 5-10% | -5 a -100% | ~10-15% |
| 4 | Cross-asset contagion (carry unwind) | 30-40% | -20 a -40% | ~10-12% |
| 5 | Stablecoin depeg + API failure | 15-20% | -10 a -35% | ~8-12% |

### Evento de referencia: October 10-11, 2025

- $19.13B liquidados em 24h (1.6M traders)
- $3.21B em 60 segundos (21:15 UTC)
- Order book BTC: $103M → $0.17M (-98%)
- Spreads: 1,321x mais largos
- ADL destruiu posicoes corretas de short
- Recuperacao de liquidez: ainda -33% dois meses depois

### Pre-mortem: Como $100k vira $30k em 6 meses

1. Over-leverage (3x) amplifica perdas
2. Correlacao converge a 0.90+ → diversificacao = ficcao
3. Rate limiting da API perde 3 stops criticos
4. ADL fecha short correto no pior preco
5. Slippage de saida: 8-15% em altcoins pos-crash
6. Whipsaw: 22 sinais falsos em 2 meses = -$8,800
7. Short squeeze de single token DeFi: -$4,500
8. Erro psicologico: concentracao em 3 posicoes correlacionadas

---

## PARTE VI — CRITICAS DO ROUND 2

### Consenso REAL (nao groupthink)

- **Long-only domina long-short** — 3 squads independentes, metodologias diferentes
- **Weekly rebalance e o sweet spot academico** — mas custa 4x mais que mensal
- **Large-cap only** — momentum 4.2x mais forte, micro-caps revertem
- **BTC regime filter e necessario** — mas tem lag de 4-8 semanas

### Groupthink identificado

- **"Sharpe 1.5+ e atingivel"** — nenhum squad reconciliou os numeros de alpha com custos e riscos
- **Lookback de 28 dias** — derivado de um unico paper, nao testado em 2025

### Divergencias criticas

| Topico | Posicao A | Posicao B | Vencedor |
|--------|----------|----------|---------|
| Signal: ensemble vs simples | Construction: 7+14+28 ensemble | Alpha R2: 20d simples | **Empate** — ensemble nao tem validacao OOS |
| Rebalance: weekly vs monthly | Construction: weekly | Feasibility: mensal domina net of costs | **Mensal** — 2.5% aa de economia |
| Regime filter: binario vs gradual | Construction: 3 estados | Alpha R2: 2 estados | **2 estados** — simplicidade, menos falsos positivos |
| Stop: -25% vs relativo | Construction: -25% flat | Construction R2: alpha-relative | **Alpha-relative** — -25% flat e calibrado para equities |
| Vol sizing: inv-vol vs equal-weight | Construction: inv-vol | Alpha R2: equal-weight (Grobys) | **Disputado** — variancia pode ser infinita |

### Perguntas NAO respondidas

1. **Performance 2025 do fator momentum?** Nenhum squad apresentou dados
2. **Half-life do alpha?** Necessario para justificar weekly vs monthly
3. **Grobys: aplica a top-20 ou so a micro-caps?** Muda toda a framework
4. **AUM minimo viavel?** Provavelmente $25-30k minimo

---

## PARTE VII — SPEC RECOMENDADA PELO CIO

### Para $100k-$500k (unico tier que faz sentido)

```
=== CRYPTO MOMENTUM SIMPLIFICADO v1.0 ===
Exchange:      Binance Spot
Capital:       $100k-$500k (nao viavel abaixo de $25k)
Dados:         Binance API (gratis) + CoinGecko (gratis)

--- UNIVERSO ---
Coins:         Top 20 por market cap (90d media)
Floor:         $500M market cap, $10M ADV
Seasoning:     180 dias minimo (nao 90)
Excluir:       Stablecoins, wrapped, leverage tokens

--- SINAL ---
Primario:      28-day return (skip t-1), rank no universo
Secundario:    Blend 50/50 de 28d e 60d returns (multi-horizonte real)
Alternativa:   Se implementar CTREND simplificado, usar elastic net
               sobre RSI(14), MACD(12,26,9), EMA(20/50), volume ratio

--- PORTFOLIO ---
Posicoes:      Top quintile = 4 coins
Sizing:        Equal-weight (nao inv-vol — Grobys warning)
Cap:           20% max por posicao
Direcao:       LONG-ONLY. Zero shorts.

--- REBALANCE ---
Frequencia:    MENSAL (1o domingo do mes, 23:30 UTC)
Buffer:        Apenas trocar se rank muda >5 posicoes
Urgencia:      Se qualquer posicao cai >30% em 1 semana, sair imediatamente

--- REGIME (2 estados, nao 3) ---
RISK-ON:       BTC > 200d SMA → 100% alocado (4 posicoes)
RISK-OFF:      BTC < 200d SMA → 100% USDC/USDT (0 posicoes)
Sem estado intermediario.

--- DRAWDOWN ---
Primario:      Alpha relativo (portfolio return - BTC return) < -20% rolling 90d
               → Reduzir para 2 posicoes (top 2 do ranking)
Secundario:    Alpha relativo < -30% → 100% cash
Hard floor:    -40% absoluto do peak → full stop, revisao obrigatoria
Recovery:      Re-entrar quando alpha rolling 90d > -5% E BTC > 200d SMA

--- CUSTOS ESTIMADOS ---
Mensal rebalance: ~2.73% aa ($2,730 em $100k)
Retorno liquido esperado: 8-12% aa (central), range -35% a +60%
```

### Para $1k-$100k: Alternativa simplificada

```
=== BTC+ETH REGIME FILTERED ===
Long BTC (70%) + ETH (30%) quando BTC > 200d SMA
USDC quando BTC < 200d SMA

Custo: ~0.05% aa
Sharpe esperado: 0.8-1.3
Complexidade: 15 min/semana
```

---

## PARTE VIII — REPOSITORIOS E FERRAMENTAS

### Top 5 para implementacao

| # | Repo | Stars | Uso |
|---|------|-------|-----|
| 1 | polakowo/vectorbt | 6,900 | Parameter sweep, backtest rapido |
| 2 | freqtrade/freqtrade-strategies | 4,900 | Templates (ADXMomentum, Supertrend) |
| 3 | jesse-ai/jesse | 7,500 | Framework crypto-nativo, multi-timeframe |
| 4 | kernc/backtesting.py | 8,100 | Mais simples para prototipar |
| 5 | marketcalls/vectorbt-backtesting-skills | 97 | Donchian, Dual Momentum prontos |

### Bibliotecas de indicadores

| Biblioteca | Stars | Notas |
|-----------|-------|-------|
| ta-lib-python | 11,800 | C backend, mais rapido |
| pandas-ta | ~4,500 | Pandas-nativo, facil integracao |
| quantstats | 6,800 | Melhor para reporting pos-backtest |
| tuneta | 457 | Anti-overfitting (Optuna + K-means) |

---

## PARTE IX — REFERENCIAS COMPLETAS

### Academicas (20 papers)
1. Fieberg et al. — CTREND (JFQA 2024)
2. Zarattini, Pagani, Barbon — Catching Crypto Trends (SSRN 5209907)
3. Han, Kang, Ryu — TSMOM vs CSMOM (SSRN 4675565)
4. Grobys & Shahzad — Momentum Illusion / Power Law (Springer 2025, Wiley 2025)
5. Ammann et al. — Survivorship Bias (SSRN 4287573)
6. Sadaqat & Butt — Stop-Loss + Momentum (JBEF 2023)
7. Liu, Tsyvinski, Wu — Common Risk Factors (JF 2022, NBER w25882)
8. Fieberg, Liedtke, Zaremba — Crypto Anomalies (IRFA 2024)
9. DS3 Lasso Factor Model (NAJEF 2026)
10. Cakici et al. — ML Cross-Section (IRFA 2024)
11. Fieberg et al. — Crypto Factor Momentum (Quant Finance 2023)
12. Mercik et al. — Cross-Sectional Interactions (IRFA 2025)
13. Huang, Sangiorgi, Urquhart — Volume-Weighted TSMOM (SSRN 4825389)
14. Risk-Managed Momentum (Finance Research Letters 2025)
15. Nguyen — Adaptive Trend-Following (arXiv 2602.11708)
16. Borgards — Dynamic TSMOM (ScienceDirect 2021)
17. Karassavidis et al. — Volatility-Adaptive (SSRN 5821842)
18. Liu & Tsyvinski — Risks and Returns (NBER 24877, RFS 2021)
19. Tan & Tao — Trend-Based Forecast (Economic Modelling 2023)
20. Ficura — Size/Volume/Momentum Interaction (FFA 2023)

### Industria
21. Man Group — "In Crypto We Trend"
22. Starkiller Capital — Cross-Sectional Momentum blog
23. Artemis Analytics — Crypto Factor Model
24. Hashdex HAMO — Momentum Factor ETP
25. Grayscale — "Trend Is Your Friend" research

### Dados de custos
26. Binance Fee Schedule (spot + futures)
27. Talos — Market Impact Model
28. Kaiko — Bid-Ask Spread Cheatsheet
29. CoinGecko — State of Perpetuals 2025
30. CoinGlass — Funding Rate Dashboard
