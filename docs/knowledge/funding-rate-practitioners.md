# Funding Rate: Conhecimento de Practitioners e Alpha SOTA

> Compilado pelo Comite Quant em 2026-03-15.
> Fontes: Forums (Reddit, EliteTrader, QuantConnect), blogs de practitioners, GitHub repos, reports de fundos reais, Substacks quant, documentacao de exchanges.
> Revisao CIO com cross-examination.

---

## 1. ESTADO DO MERCADO (Marco 2026)

### Estrutura de Taxas Atual
- BTC/ETH funding medio: 0.0057%-0.0126%/8h (Binance vs Hyperliquid)
- Positivas em >92% do tempo em Q3 2025
- Hyperliquid BTC: 0.0120%/8h (mais alta que Binance 0.0057%)
- BitMEX: mais estavel, exato 0.01% em 78% do tempo (floor da formula)

### O Efeito ETF (BIS, 2024)
- ETFs spot BTC (jan 2024) reduziram crypto carry em ~3pp geral e ~5pp no CME
- Reducao de 36% (geral) e 97% (CME) do carry medio historico
- Basis trade CME: ~2% aa — **estruturalmente morto**
- Fonte: BIS Working Papers No 1087

### Ethena como Teto de Yields
- Ethena USDe: ~$7.83B de capital em shorts delta-neutros (set 2025)
- Representa ~12% do OI total dos principais CEXs ($65.7B)
- Funciona como arbitrageur automatico: quando funding sobe, capital entra e comprime
- sUSDe atual: 6-7% aa (down de 55.9% pico em mar 2024)
- Fonte: Ethena docs, DeFiLlama, CoinMetrics

---

## 2. MECANICA DE CALCULO POR EXCHANGE

### Binance (USDT-M)
```
Funding Rate = Avg Premium Index (P) + clamp(Interest Rate - P, -0.05%, +0.05%)
Premium Index = [Max(0, Impact Bid - Spot Index) - Max(0, Spot Index - Impact Ask)] / Spot Index
```
- Impact Bid/Ask: preco medio de execucao para ordem do tamanho do Impact Margin Notional ($200k-$600k)
- TWAP: a cada **5 segundos**, 5.760 pontos em 8h — nao da pra manipular no final
- Interest Rate fixo: 0.01%/8h (=0.03%/dia) — cria **structural positive bias** que beneficia shorts
- Settlement: 00:00, 08:00, 16:00 UTC (janela de 1 min de desvio)
- **CRITICO**: ~60% dos contratos USDT-M mudaram de 8h para 4h ou 1h — quebra backtests que assumem 8h fixo

### Bybit
- Formula identica na estrutura ao Binance
- **Diferenca critica**: weighting scheme **crescente** — leituras recentes pesam mais
- Implicacao: movimentos nos ultimos 30-60 min antes do settlement tem impacto desproporcional
- Edge: front-running do order book pre-funding e mais impactante que em Binance
- **Predicted vs realized diverge 20-50% durante volatilidade**

### OKX
- Otimizacao em marco 2024: mudou de lookback parcial para N horas completas anteriores
- Reduziu volatilidade da predicted funding e eliminou edge de janela recente
- Qualquer backtest pre-marco 2024 com dados OKX usa metodologia descontinuada

### Hyperliquid
- **Settlement: A CADA HORA** (vs 8h CEXs) — 8x mais frequent
- Premium sampling: a cada 5 segundos, medias horarias
- Oracle: mediana ponderada dos CEX spot prices
- **Cap: 4%/hora** — muito mais alto que CEXs, permite premios extremos persistentes
- Baseline exato (0.0125%/h) apenas 39% do tempo vs 78% BitMEX — mais volatilidade = mais oportunidades
- **Sem cross-margining spot-perp**: capital efetivo = 2x

### dYdX v4 (on-chain)
- Block proposers submetem FundingPremiumVote a cada bloco
- Funding-sample epoch: 1 min; Funding-tick epoch: 1h
- Caps por tier: Large-Cap=12% (8h), Mid-Cap=30%, Long-Tail=60%
- **Latencia estrutural** CEX → dYdX: minutos a horas para funding refletir novo preco

### Drift Protocol (Solana)
- Formula: (1/24) x (mark_TWAP - oracle_TWAP) / oracle_TWAP
- Updates **lazy** — so ocorre quando ha trade, deposito ou saque
- Em baixa atividade: funding pode estar **desatualizado** por longos periodos = info assimetrica

---

## 3. ESTRATEGIAS RANQUEADAS POR ALPHA REAL

### Cross-Examination CIO: Numeros Brutos vs Liquidos

O yield bruto (CoinGlass, screeners) tipicamente sofre haircut de **40-68%** apos custos reais:
- Fees round-trip: 0.11% maker, 0.23% taker (Hyperliquid)
- Collateral drift correction: ~6 rebalancings extras/ano
- Funding inversions: ~18% dos periodos 8h com funding negativo
- One-sided fill risk: 10% das entradas parciais
- ADL expected value: ~0.3% aa
- API throttle em volatilidade
- Custo de oportunidade: capital imobilizado vs T-bill 4.2%

---

### #1 — SPOT-PERP DELTA-NEUTRO (Altcoins Liquidas)

**Mecanica**: Long spot + Short perpetuo, mesmo ativo, delta zero
**Melhor implementacao**: SOL, AVAX, ARB no Binance (maker orders)

**Retornos verificados por fundos reais:**
| Fonte | Periodo | Retorno | Sharpe | MDD |
|---|---|---|---|---|
| ANB Investments | 2021 (bull peak) | ~100% aa | N/D | <1% |
| ANB Investments | Nov2023-Feb2024 | 34% | N/D | <1% |
| 1Token (9 funds, $4B AUM) | 2024 | 14.39% aa | 5-10 (Calmar) | <2% |
| 1Token | 2025 | 19.26% aa | 5-10 (Calmar) | <2% |
| ScienceDirect (peer-reviewed) | 2020-2024 | ate 115.9%/6m | N/D | 1.92% |

**CROSS-EXAM VERDICT (CIO)**:
- 19.26% (1Token): **provavelmente bruto ou com alavancagem nao declarada**
- P&L walk para $100K: yield bruto $19,260 - custos reais $13,080 = **liquido $6,180 (6.18% aa)**
- Sharpe 5-10 inflado por autocorrelacao de carry payments; **ajustado: 2.5-4**
- MDD <2% ignora exchange counterparty risk (FTX happened)

**Parametros de implementacao (repos open source):**
```python
min_funding_rate = 0.0001    # 0.01% por periodo
min_annual_yield = 0.08      # 8% aa apos fees
max_basis = 0.001            # 0.1% max divergencia spot-perp
single_asset_cap = 0.15      # 15% max por ativo
total_utilization = 0.60     # 60% max do capital
global_drawdown_stop = 0.02  # 2% stop loss
delta_threshold = 0.005      # 0.5% para rehedge
scoring = "(rate * 3 * 365) - (fees * 4)"  # Simplificada
```

**Formula de scoring CORRIGIDA (pos cross-exam):**
```python
score = ((rate * P_positive * payments_per_day * 365)
         - (fees * 4.5)           # inclui spread assimetrico
         - (rebal_cost * freq)    # custo de rebalancing
         - drift_adjustment)      # collateral drift
        * capital_efficiency      # 0.5 sem cross-margin, 1.0 com
```

**Alpha nao-obvio:**
1. Colateral nao-USDT tem haircuts: BTC = 5%, altcoins ate 90% (Binance)
2. ADL e o risco real, nao price vol — exchange fecha sua posicao mais lucrativa a mercado
3. Perp-spot divergence em crises: drawdowns temporarios devastadores se alavancado
4. Immediate sell de funding payments >>> hold em BTC (Sharpe muito superior)

---

### #2 — CROSS-EXCHANGE DIFFERENTIAL (BitMEX/Hyperliquid)

**Mecanica**: Short perp no exchange com funding alto + Long perp no exchange com funding baixo

**Edge documentado H1 2025:**
- SOL: 15.6% aa unlevered (gap medio 0.00841%/8h BitMEX vs Hyperliquid)
- AVAX: 15.7% aa unlevered (gap medio 0.00891%/8h)
- Com 2-3x leverage: 25-30%+

**CROSS-EXAM VERDICT (CIO)**:
- Capital executavel SEM mover mercado: <$50K por par
- $50-250K: 5-9% liquido (borderline)
- $250K+: custo de execucao consome spread
- A estrategia e mais lucrativa quando menos executavel (alta vol = alta oportunidade + alta friction)
- Custo cross-exchange: 72-165% mais caro que same-exchange (transfers, buffer, contraparte)

**Execucao**: Simultaneidade OBRIGATORIA — atraso >1-2s entre legs expoe a risco direcional.
**Rebalancing**: Delta bound de 3%, so rebalancear quando exceder.

---

### #3 — FUNDING RATE COMO PREDITOR (Mean-Reversion/Direcional)

**Evidencia quantificada:**
- ML approach: 31% aa Sharpe 2.3 (backtest)
- Basis overlay simples: 94.1% retorno Sharpe 1.51
- CF Benchmarks: "long spot quando basis > SOFR+300bps"
- Presto Labs: R²=12.5% FR vs preco (mesmo periodo), **R²≈0% next period**

**CROSS-EXAM VERDICT (CIO)**:
- **ML 31% Sharpe 2.3: 90% probabilidade de overfitting**
- R²≈0% out-of-sample destroi a base da previsao por ML
- Predicao de funding via DAR models (SSRN, Emre Inan 2025) funciona mas e time-varying
- Implementavel com disciplina: 8-15% aa, Sharpe 1.2-1.8 (muito abaixo do backtest)

**Sinais de entrada/saida:**
```
Entrada short (fade longs): FR > +0.05%/8h + OI crescente + momentum desacelerando
Entrada long (fade shorts): FR < -0.02%/8h + OI crescente → short squeeze
Confirmar com Fear & Greed: entrada com F&G > 0.8; fechar quando F&G < 0.2
```

**Signal composite (FR + OI + Liquidations):**
| Combinacao | Interpretacao | Acao |
|---|---|---|
| High FR + Rising OI | Longs overcrowded | Short ou reduzir long |
| Low/Neg FR + Rising OI | Shorts overcrowded | Long / short squeeze |
| Stable FR + Declining OI | Desalavancagem | Neutralizar |
| FR spike extremo + OI plateau | Cascade iminente | Reduzir exposicao |

---

### #4 — CROSS-FUNDING CEX vs DEX (Latencia)

**Edge estrutural**: DEXs respondem mais lentamente que CEXs a mudancas de preco

**Venues e latencias:**
| Venue | Funding Freq | Cap | Edge Type |
|---|---|---|---|
| Hyperliquid | 1h | 4%/h | Cap + freq arb vs CEX |
| dYdX v4 | 1h (2 epochs) | 12-60% (8h) | Lag exploitation |
| Drift (Solana) | 1h (lazy) | 0.125-0.417% | Lazy update arb |

**Dados empiricos (nov 2025, 26 exchanges, 35.7M observacoes):**
- CEXs dominam price discovery: 61% maior integracao que DEXs
- Todo fluxo significativo: CEX → DEX (zero causalidade reversa)
- 17% das observacoes tem spread >=20bps entre CEX e DEX
- **Apenas 40% dessas geram retorno positivo apos custos**

---

## 4. INSIGHTS DE MICROESTRUTURA NAO-OBVIOS

### 4.1 Floor de 0.01% e Armadilha de Sinal
78% do tempo BTC na BitMEX esta exatamente em 0.01% (piso da formula). Nao e leitura de sentimento — e ancora matematica. O sinal real esta na **variacao em torno do piso**, nao no nivel absoluto.

### 4.2 Hyperliquid como "Clean Price" Venue
Dentro dos perps, Hyperliquid agora lidera price discovery. Inverter logica de arb: Hyperliquid como referencia, nao Binance.

### 4.3 WLFI como Leading Indicator (Out 2025)
Evento documentado: WLFI colapsou 5h ANTES do BTC reagir. Volume WLFI: 21.7x baseline ($474M). Resultado: $6.93B liquidados em 40 min. **Alpha**: monitorar tokens usados como colateral cruzado como leading indicators.

### 4.4 Leverage Stacking Oculto
Um deposito BTC pode carregar 5.5x leverage efetivo via cross-margining. Mapear leverage oculto = saber onde cascatas acontecem.

### 4.5 Bybit Weighting Asymmetry Edge
Aumento subito de OI nos ultimos 60 min antes do settlement inflaciona funding realizada acima do TWAP simples. Predicted funding subestima o realizado.

### 4.6 Predicted vs Realized Gap
Em Bybit e OKX, predicted funding na interface pode divergir da realizada em 20-50% durante volatilidade. Nao confiar apenas em predicted rate.

---

## 5. PROBLEMAS PRATICOS QUE SO QUEM OPERA DESCOBRE

### 5.1 One-Sided Fill
Se spot long executa mas perp short e rejeitado → exposicao direcional sem hedge. Bot precisa de logica de rollback: detectar fill parcial e fechar leg executado.

### 5.2 Collateral Drift
Apos 5-10 funding periods, PnL assimetrico acumula: 60% capital em um exchange, 40% no outro. Rebalancing manual = 30-60 min de exposicao nao-neutra.

### 5.3 Low-Volume Execution
Madrugada UTC (02:00-06:00): order book raso. Funding pode parecer atrativa mas execucao move preco. Impact Bid/Ask calculado com liquidez de pico, sua ordem executa com liquidez de minimo.

### 5.4 Funding Rate Inversion Risk
Durante cascade: FR pode ir de +0.1%/h para -0.05%/h em minutos. Short perp + long spot passa de receber para pagar funding, ENQUANTO spot cai — double loss.

### 5.5 Binance Variable Intervals (60% dos contratos)
Bots que assumem 8h fixo para annualizacao estao com backtests imprecisos. Pipeline precisa inferir intervalo real por coin e periodo.

### 5.6 Hyperliquid: Falta de Cross-Margining
Conta spot e perp separadas. Trade spot-perp puro requer capital em ambas = 2x capital necessario.

---

## 6. PERFORMANCE REAL DE FUNDOS (2025)

| Fundo/Estrategia | Retorno | Sharpe/Calmar | Observacao |
|---|---|---|---|
| Market-neutral arb (1Token, 9 funds $4B) | 19.26% aa (bruto provavel) | 5-10 (Calmar) | MDD <2% |
| Sigil Fund Stable | +11.26% (out 2025) | N/D | +37pp vs direcional |
| 319 Capital (cross-exchange) | +12.2% YTD out/2025 | N/D | +1.5% no mes da crise |
| Amphibian (fund of funds) | Mid-to-high teens | 5+ | 60-65% market neutral |
| ANB Investments | 34% (nov23-fev24) | N/D | MDD <1% |
| VisionTrack Mkt Neutral Index | +18.5% (2024) | N/D | vs BTC +120% |

---

## 7. GAP BACKTEST → EXECUCAO REAL (Cross-Exam)

| Estrategia | Backtest | Real 2026 | Gap | Razao |
|---|---|---|---|---|
| Ethena sUSDe | 6-7% | 5-6% | ~15% | Execucao on-chain, sem slippage |
| BTC Spot-Perp (Binance) | 12-15% | 6-10% | ~30% | Liquidez suficiente |
| Top-5 Spot-Perp | 15-19% | 9-13% | ~35% | Slippage controlavel |
| Cross-exchange | 15-28% | 3-9% | ~55% | Transfer + timing risk |
| ML approach | 31% S=2.3 | 8-15% | ~60% | Overfitting + regime change |

---

## 8. DADOS, FERRAMENTAS E INFRA

### APIs e Dados Historicos

**Tier 1 — Gratis:**
- Binance USDS-M: `GET /fapi/v1/fundingRate` (8h, completo desde 2020, 1000 req/min)
- Binance COIN-M: `GET /dapi/v1/fundingRate`
- Bybit: `GET /v5/market/funding/history` (200 registros/call)
- OKX: docs v5 (desde mar 2022, download CSV)
- Hyperliquid: REST API nativa (1h, sem auth)

**Tier 2 — Pagas:**
- **Tardis.dev**: tick-level, formato nativo + normalizado. Melhor para HFT/quant. Python: `pip install tardis-dev`
- **Crypto Lake**: funding a cada 3s (Binance). Python: `pip install lakeapi`. Mais barato que Tardis.
- **CoinGlass API**: funding OI-weighted, heatmaps. Hobbyist $29/mes, Startup $79/mes.
- **Kaiko**: institucional ($50k+/ano). 200k+ contratos derivativos.

**Datasets publicos:**
- Kaggle "Crypto Perpetuals Funding Rates": BTC, ETH, LTC, BNB, altcoins (338 downloads, escopo limitado)
- OKX Historical Data: CSV gratuito desde mar 2022
- Binance Funding History: pagina web publica

### Ferramentas de Visualizacao
| Ferramenta | Features Chave |
|---|---|
| CoinGlass | Heatmap, OI-weighted FR, arb APR screener, countdown |
| Coinalyze | FR + predicted + OI + liquidations agregados |
| CryptoQuant | FR + OI + dados on-chain correlacionados |
| Hyperliquid native | Comparador funding HL vs outros |
| FundingView.app | 12+ exchanges, normalizacao de intervals |
| ArbitrageScanner.io | Scanner multi-exchange |
| Loris.tools | Screener de oportunidades |

### Repos Open Source

**Bots de Arbitragem (CEX-CEX):**
- `ynhy513/funding-rate-arbitrage` — Mais completo. Binance+OKX via CCXT. State machine, scoring, risk limits.
- `kir1l/Funding-Arbitrage-Screener` — Screener Binance/Bybit/OKX/MEXC. Top 20 por exchange.
- `hamood1337/CryptoFundingArb` — Binance+KuCoin+OKX+Bybit+Kraken+**Hyperliquid**. Min spread customizavel.
- `aoki-h-jp/funding-rate-arbitrage` — 6 exchanges. Revenue por 100 USDT apos comissoes.

**Bots DEX-DEX / CEX-DEX:**
- `50shadesofgwei/funding-rate-arbitrage` — Synthetix v3+GMX+Bybit. Funding velocity model: `dr/dt = c * skew`.
- `ksmit323/funding-rate-arbitrage` — Orderly+Hyperliquid+ApexPro.

**Frameworks com suporte a funding:**
- **Hummingbot**: `v2_funding_rate_arb.py` (desde v1.27.0). 30+ connectors. Bug conhecido: double-fill em HL (#7295).
- **PassivBot**: `funding_fee_collect_mode=True` — filtra direcao por predicted FR.
- **NautilusTrader**: Rust core + Python. Subscribe `BinanceFuturesMarkPriceUpdate`.
- **Freqtrade**: `@informative('8h', candle_type="funding_rate")`. Valor na coluna `open`.

### Sinais Alternativos para Feature Engineering
- Open Interest (OI): OI crescente + FR positivo = overleveraged, reversao provavel
- Predicted Funding Rate: WebSocket Binance `@markPrice` (5s updates)
- Long/Short Ratio: complemento ao FR para sentimento
- Liquidation Data: correlacionada com picos de FR (CoinGlass, Coinalyze)
- Order Book Imbalance: Crypto Lake (100ms snapshots)
- Basis (Futures-Spot): max aceitavel 0.1% para entrada

---

## 9. DECISAO CIO — SINTESE FINAL

### Consenso Real do Comite
1. Alpha de funding rate **existe mas esta em compressao acelerada** (ETFs + Ethena + capital institucional)
2. BTC/ETH basis: alpha residual minimo. **Edge migrou para altcoins tier-2 e cross-venue DEX/CEX**
3. ML preditivo de FR: **nao implementavel com confianca** (R²≈0% out-of-sample)
4. **Gap backtest-realidade: 35-68%** — qualquer decisao baseada em backtest puro sera decepcionante

### Divergencias Registradas
- SONNET-A vs SONNET-M: SONNET-A projeta 15-28% cross-exchange; SONNET-M demonstra que custos reais reduzem para 3-9%. **CIO concorda com SONNET-M** — custos de execucao sao subestimados sistematicamente.
- SONNET-A vs SONNET-D: ML 31% Sharpe 2.3 (SONNET-A) contradiz R²≈0% (SONNET-D/Presto). **CIO considera ML overfitted ate prova em contrario**.

### Estimativa CIO de Alpha Liquido (Marco 2026)
```
Operador profissional, sem alavancagem:
  Altcoins liquidas spot-perp:  8-14% aa bruto → 5-9% aa liquido
  BTC/ETH spot-perp:            10-14% aa bruto → 3-6% aa liquido
  Cross-exchange:               8-12% aa (capital <$50K)

  Sharpe real ajustado:         2-3 (nao 5-10)
  MDD real (inc. tail risk):    3-8% (nao <2%)

  Vs T-bill 4.2%: premio de ~1-5% aa com risco operacional consideravel
```

### Condicoes de Revisao
- Se Ethena colapsar ou reduzir <$3B: yields voltam a subir materialmente
- Se novo bull extremo (BTC >$200K): funding rates 2021-level retornam
- Se regulacao forcar fechamento de exchanges offshore: spreads cross-venue explodem

---

## FONTES PRINCIPAIS (Nao-Academicas)

- 1Token Blog: Crypto Quant Strategy Index VII-VIII (Oct-Nov 2025)
- BitMEX Blog: Q3 2025 Derivatives Report; Harvest Funding Payments on Hyperliquid
- CF Benchmarks: Revisiting the Bitcoin Basis (2025)
- CME Group: Spot ETFs Give Rise to Crypto Basis Trading
- Navnoor Bawa Substack: October 2025 Liquidation Cascade Analysis
- QuantJourney Substack: Funding Rates — The Hidden Cost, Sentiment Signal
- Presto Labs Research: Can Funding Rate Predict Price Change?; Optimizing Funding Fee Arb
- The Hedge Fund Journal: ANB Investments; Amphibian Quant
- Ethena Docs: Protocol Revenue, Funding Risk
- Hummingbot Blog: Funding Rate Arbitrage on Hyperliquid
- SSRN (Emre Inan 2025): Predictability of Funding Rates (DAR models)
- Chainspot: Basis, Funding & Cross-Venue Arbitrage
- Pi2 Network: Arbitrage Opportunities in Perpetual DEXs
- Gate.io Learn: Perpetual Contract Funding Rate Arbitrage 2025
- CoinCryptoRank: Funding Rate Arbitrage Complete Guide 2025
- Amberdata: Ultimate Guide to Funding Rate Arbitrage
- FMZ Quant Blog: Automated Implementation with AI
