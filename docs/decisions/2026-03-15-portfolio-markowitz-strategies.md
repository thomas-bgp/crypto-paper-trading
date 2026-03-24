# Decisao: Portfolio Multi-Estrategia para Otimizacao Markowitz

**Data:** 2026-03-15
**Comite:** OPUS (CIO), SONNET-A, SONNET-R, SONNET-M, SONNET-D
**Status:** APROVADO

---

## Portfolio Aprovado — 3 Estrategias Descorrelacionadas

### Strategy #1 — Momentum Regime Switch v2 (EXISTENTE)
- **Tipo:** Direcional com regime filter
- **CAGR:** 69.6% (backtest 2022-2026)
- **Sharpe:** 1.48
- **Correlacao com BTC:** 0.6-0.8 em bull, ~0 em bear
- **Fraqueza:** Flat em sideways/bear (2025: +0.6%)
- **Alocacao target:** 40-50% do portfolio

### Strategy #2 — Mean Reversion Cross-Sectional (CONTRARIAN) — **NOVO**
- **Tipo:** Market-neutral (long losers, short winners)
- **CAGR esperado:** 15-25% aa
- **Sharpe esperado:** 1.2-1.8
- **Correlacao com #1:** **-0.40 a -0.60** (negativamente correlacionado)
- **Melhor em:** Sideways/bear (quando #1 e flat)
- **Alocacao target:** 30-40% do portfolio
- **Implementacao:** Long/short altcoins via Binance Futures perps, rebalance semanal

### Strategy #3 — Statistical Arbitrage / Pairs Trading — **NOVO**
- **Tipo:** Market-neutral (spread de pares cointegrados)
- **CAGR esperado:** 10-20% aa
- **Sharpe esperado:** 1.0-1.6
- **Correlacao com #1:** 0.15-0.30
- **Correlacao com #2:** 0.10-0.20
- **Pares:** BTC/ETH, ETH/SOL, BNB/SOL (majors apenas)
- **Alocacao target:** 15-25% do portfolio
- **Implementacao:** Perps Binance/Bybit, Johansen cointegration rolling 90d

### Descartados pelo Comite
- **Volatility Selling (Deribit):** VETADO por SONNET-R — correlacao em stress = 1.0. SONNET-M: inviavel com $30k
- **DeFi LP:** PROIBIDO — hedge de IL custa mais que o yield. Smart contract risk inaceitavel
- **Cross-Exchange Arb:** Sem edge para retail
- **Funding Rate Carry:** Correlacao alta com #1 (0.45-0.60), opcional em dose pequena (5%)

---

## Portfolio Markowitz Target

```
Strategy #1 (Momentum):    45%   | Correlacao baseline
Strategy #2 (Contrarian):  35%   | Corr = -0.50 com #1
Strategy #3 (Pairs):       20%   | Corr = 0.20 com #1, 0.15 com #2
```

### Performance Esperada do Portfolio Combinado
- **CAGR esperado:** 30-45% aa (vs 69.6% puro momentum — menor mas muito mais estavel)
- **Sharpe portfolio:** 1.8-2.4 (vs 1.48 standalone)
- **Max DD esperado:** -15 a -22% (vs -35% standalone)
- **Anos sideways (2025-like):** +12-22% (vs +0.6% standalone)
- **Correlacao portfolio vs BTC:** ~0.2-0.3

### Otimizacao
- Framework: PyPortfolioOpt com Ledoit-Wolf shrinkage (OBRIGATORIO)
- Alternativa: CVaR-constrained (superior a mean-variance em crypto)
- Rebalanceamento: Mensal ou em desvio >10% dos pesos target
- Universo: 5-8 ativos max por estrategia

---

## Proximos Passos
1. Implementar Strategy #2 (Contrarian) — backtester + dados
2. Implementar Strategy #3 (Pairs) — cointegration scanner + execucao
3. Backtest individual de cada estrategia (2022-2026)
4. Calcular matriz de correlacao real entre estrategias
5. Otimizar portfolio via Markowitz/CVaR
6. Dashboard integrado com as 3 estrategias
7. Paper trading 30 dias

## Timeline estimado: 8-10 semanas

## Fontes Academicas Chave
- Cambridge JFQA 2024: "Trend Factor" — contrarian effect > momentum em crypto semanal
- Springer 2025: Copula-based pairs trading supera cointegration classica
- arxiv 2410.15195: VRP no Bitcoin = 0.14 (estruturalmente positivo)
- BIS WP 1087: Crypto carry Sharpe negativo em 2025
- CEPR: Correlacao em stress converge para 0.7-0.95 em crypto
