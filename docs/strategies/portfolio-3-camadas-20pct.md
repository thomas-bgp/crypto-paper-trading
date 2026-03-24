# Estrategias para Target 20%+ aa — Capital Baixo, Sem HFT

> Decisao do Comite Quant em 2026-03-15.
> Restricoes: capital sub-$100k, sem HFT, max 5min candles.
> Confianca do comite: 6/10 (atingivel em horizonte de 3+ anos, nao garantido todo ano).

---

## Arquitetura do Portfolio: 3 Camadas + Regime Switch

```
┌─────────────────────────────────────────────────────────────┐
│                    REGIME DETECTOR                           │
│  BTC vs SMA200 + Fear&Greed EMA + Funding Rate medio        │
├────────────┬────────────────┬───────────────────────────────┤
│  BULL      │  SIDEWAYS      │  BEAR                         │
│  BTC>SMA200│  BTC~SMA200    │  BTC<SMA200                   │
│  F&G > 50  │  F&G 30-50     │  F&G < 30                     │
│  FR > 0.03%│  FR 0.01-0.03% │  FR < 0.01% ou negativo       │
├────────────┼────────────────┼───────────────────────────────┤
│ CAMADA 1   │ CAMADA 2       │ CAMADA 3                      │
│ Trend      │ Mean Rev +     │ Cash + Stablecoin             │
│ Following  │ Grid Trading   │ Yield                         │
│ 70% capital│ 60% capital    │ 90% em stables                │
│            │                │ 10% contrarian                │
└────────────┴────────────────┴───────────────────────────────┘
```

---

## CAMADA 1 — Core Alpha: Cross-Sectional Momentum + Trend Following

**Alocacao:** 60-70% do capital em regime bull

### Estrategia
- **Universo:** Top 20-40 altcoins por liquidez (volume diario >$50M)
- **Sinal:** Ensemble de Donchian Channels (20, 50, 100 periodos) em candles de 4h
- **Execucao:** 5min candles para timing de entrada/saida (limit orders SEMPRE)
- **Position sizing:** 1/volatilidade (ATR-based). Mais volatil = menor posicao
- **Rebalanceamento:** Semanal
- **Macro filter:** BTC > SMA200 E DXY < SMA50 (se disponivel)
- **Funding overlay:** Se funding >0.05%/8h em um ativo, reduzir tamanho (mercado sobreaquecido)

### Evidencia
| Fonte | Resultado | Periodo |
|-------|-----------|---------|
| Zarattini, Pagani, Barbon (SSRN 2025) | Sharpe >1.5, alpha 10.8% vs BTC | 2015-2025 |
| XBTO Trend Strategy (live) | Sharpe 1.62, max DD -15.5% | 2020-2025 |
| Risk-managed momentum (ScienceDirect) | Sharpe 1.42 com vol filter | 2018-2025 |
| Man Group "In Crypto We Trend" | Trend following funciona melhor em crypto que em TradFi | 2015-2024 |

### Retorno esperado
- Bull market: **25-50% aa**
- Neutro: **5-15% aa**
- Bear: **-10 a -20%** (stop loss limita, mas whipsaws custam)

### Implementacao
- **Framework:** Freqtrade com estrategia custom
- **Timeframe de sinal:** 4h (menor ruido que 5min)
- **Timeframe de execucao:** 5min (precisao de entrada)
- **Exchange:** Binance (BNB discount: maker 0.02%, taker 0.075%)

---

## CAMADA 2 — Complementar: Mean Reversion + Grid + Stat Arb

**Alocacao:** 20-30% do capital (60% em regime sideways)

### 2A. Mean Reversion (regime lateral)
- **Sinal:** RSI <30 + preco na banda inferior Bollinger (2σ) em 1h
- **Exit:** RSI >50 ou preco retorna a media
- **Filtro obrigatorio:** ADX <25 (confirma lateralidade). Se ADX >25, desligar MR
- **Pares:** BTC/USDT, ETH/USDT, SOL/USDT (apenas alta liquidez)

### 2B. Grid Trading (regime lateral confirmado)
- **Range:** ±15-20% do preco atual
- **Grid levels:** 10-20 niveis
- **Capital por grid:** minimo $1k por ativo
- **Exchange:** Binance/Bybit grid nativo (sem custo extra) ou Freqtrade
- **STOP obrigatorio:** Se preco sai do range, fechar grid e reavaliar

### 2C. Pairs Trading (opcional, capital >$20k)
- **Pares:** BTC/ETH (mais estavel), ETH/SOL, LTC/BCH
- **Sinal:** Z-score do spread via Kalman filter >2σ
- **Retreinamento:** A cada 30 dias (cointegration em crypto e instavel)
- **Retorno esperado:** 10-20% aa com Sharpe 0.6-1.2

### Retorno esperado da Camada 2
- Sideways: **15-25% aa**
- Bull: **5-10% aa** (menos oportunidades de reversao)
- Bear: **0-10% aa** (grid pode funcionar em ranges de consolidacao)

---

## CAMADA 3 — Preservacao + Yield Base

**Alocacao:** 10-15% do capital (90% em regime bear)

### Opcoes
1. **Stablecoin LP em L2** (Uniswap v3/Aerodrome no Base/Arbitrum)
   - USDC/USDT ou USDC/DAI
   - APY: 5-8% com risco minimo de IL
   - Gas em L2: ~$0.01/tx

2. **Lending (Aave/Compound)**
   - USDC supply: 4-8% APY
   - Risco: smart contract (mitigado por track record de 4+ anos)

3. **ETH Staking (se aplicavel)**
   - Via Lido (stETH): 3-5% APY
   - Adicional ao portfolio se ETH ja e posicao core

### Retorno esperado da Camada 3
- **5-8% aa** consistente, independente de regime

---

## Retornos Combinados por Regime

| Regime | Prob. | C1 (60-70%) | C2 (20-30%) | C3 (10-15%) | **Portfolio** |
|--------|-------|-------------|-------------|-------------|---------------|
| **Bull** | 30% | 25-50% | 5-10% | 5-8% | **20-38%** |
| **Neutro** | 40% | 5-15% | 15-25% | 5-8% | **8-16%** |
| **Bear** | 30% | -10-20% | 0-10% | 5-8% | **-3 a +5%** |
| **Media ponderada** | — | — | — | — | **~12-20% aa** |

**Para atingir 20%+ consistente:** O regime switch e o que faz a diferenca. Sem ele, a media cai para ~10-12%. Com ele, captura-se mais upside em bull e limita-se downside em bear.

---

## Infraestrutura

### Stack Minima ($5-25/mes)

```
Hardware:   Hetzner CX22 (2 vCPU, 4GB RAM, 40GB SSD) — €4.35/mes
Framework:  Freqtrade (Docker container)
Exchange:   Binance (principal) + Bybit (backup)
Dados:      Binance Data Vision (gratuito, 5min desde 2017)
Monitor:    Telegram bot (nativo Freqtrade) + UptimeRobot
Dashboard:  FreqUI (incluido)
DB:         SQLite (default Freqtrade)
```

### Custos Operacionais

| Item | Custo mensal |
|------|-------------|
| VPS | $5-16 |
| Exchange fees (com BNB) | ~0.02-0.075% por trade |
| Dados | $0 (Binance Data Vision) |
| Monitoramento | $0 |
| **Total infra** | **$5-16/mes** |

---

## Regras de Ouro (Inegociaveis)

1. **SEMPRE limit orders.** Maker fee 0.02% vs taker 0.075% — em 10 trades/dia, isso e 1.6% aa de diferenca
2. **Regime detection automatizado.** BTC vs SMA200 + ADX como filtros minimos. Sem isso, trend following em bear e suicidio
3. **Backtesting com fees + slippage.** Adicionar 0.1% de fee total + 0.05% slippage. Se nao for lucrativo assim, descartar
4. **Walk-forward validation.** Treinar em 2022-2023, testar em 2024. Se Sharpe cai >40%, e overfitting
5. **Freqtrade lookahead-analysis** antes de aceitar qualquer resultado
6. **Dry-run por 30+ dias** antes de capital real
7. **Maximo 3x leverage.** Sem excecoes. Outubro 2025 liquidou $19B em horas
8. **Nao scalpar agressivamente em 5min.** O sinal e gerado em 1h-4h, execucao em 5min. 5min puro = fee machine
9. **Nenhuma altcoin com volume <$50M/dia** para trend following
10. **Stop de sistema:** Se portfolio cai -15% do pico, pausar tudo por 7 dias e reavaliar

---

## Plano de Implementacao

### Fase 0 — Setup (Semanas 1-2, custo $5)
- [ ] VPS Hetzner + Docker + Freqtrade instalado
- [ ] Download 3 anos de dados 5m/1h/4h do Binance Data Vision (BTC, ETH, SOL, BNB + top 20)
- [ ] Primeiro backtest: EMA(21/55) crossover simples em BTC 4h
- [ ] Rodar lookahead-analysis para validar dados

### Fase 1 — Research (Semanas 3-6, custo $5)
- [ ] Implementar regime detector (BTC vs SMA200 + ADX)
- [ ] Backtest Camada 1 (momentum cross-sectional) em 2022-2024
- [ ] Walk-forward: treinar 2022-2023, testar 2024
- [ ] Se Sharpe out-of-sample >0.8: prosseguir. Senao: ajustar ou pivotar
- [ ] Backtest Camada 2 (mean reversion + grid) em periodos sideways

### Fase 2 — Dry Run (Semanas 7-12, custo $5)
- [ ] Deploy em Binance dry-run mode (paper trading real-time)
- [ ] Monitorar 30-60 dias: Sharpe, max DD, win rate, fees reais
- [ ] Comparar performance vs backtest. Se gap >30%: investigar

### Fase 3 — Live (Mes 4+, custo $5 + capital)
- [ ] Comecar com 10-20% do capital total
- [ ] Escalar a cada 30 dias se metrics positivos
- [ ] Review trimestral do comite (regime, performance, riscos)

---

## Fontes Principais

### Papers Academicos
1. Zarattini, Pagani, Barbon — "Catching Crypto Trends" (SSRN 2025)
2. Man Group — "In Crypto We Trend" (2024)
3. Cambridge Core — "Trend Factor for Cross Section of Crypto Returns" (2024)
4. ScienceDirect — "Risk-Managed Momentum Strategies in Crypto" (2025)
5. Palazzi — "Trading Games: Beating Passive in Bullish Crypto" (J. Futures Markets, 2025)
6. Springer — "Copula-Based Pairs Trading of Cointegrated Crypto" (2025)
7. Frontiers — "Deep Learning Pairs Trading" (2026)

### Live Performance
8. XBTO Trend Strategy — Sharpe 1.62, live jan/2020-set/2025
9. Bitwise Trendwise ETF — Momentum-based crypto rotation
10. Grayscale — "The Trend is Your Friend: Bitcoin Momentum Signals"

### Implementacao
11. Freqtrade — github.com/freqtrade/freqtrade
12. NostalgiaForInfinity — github.com/iterativv/NostalgiaForInfinity
13. Hummingbot — hummingbot.org
14. Binance Data Vision — data.binance.vision (dados gratuitos)
15. CryptoDataDownload — cryptodatadownload.com

### Risco
16. FTI Consulting — "Crypto Crash Oct 2025: Leverage Meets Liquidity"
17. BIS Working Paper #1087 — "Crypto Carry" (compressao de yields)
18. CryptoSlate — "10 Biggest Crypto Failures 2025"
19. Man Group — Systematic Trend-Following with Adaptive Portfolio Construction (arXiv:2602.11708)
