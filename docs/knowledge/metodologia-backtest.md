# Metodologia de Backtest para Estrategias de Funding Rate

> Documento de referencia metodologica. Toda implementacao de backtest DEVE seguir estas diretrizes para garantir resultados fidedignos.

---

## 1. Dados Necessarios

### 1.1 Streams Obrigatorios

| Dado | Granularidade | Motivo |
|------|---------------|--------|
| **Funding rate historico** | Por settlement (8h/1h) com timestamp exato | Source of alpha |
| **Preco spot OHLCV** | 1min (minimo) | Modelar entrada/saida realista |
| **Mark price do perpetual** | 1min (minimo) | Calculo de margem e liquidacao |
| **Index price** | 1min | Referencia para premium index |
| **Order book (Level 2)** | Snapshots nos momentos de entrada/saida | Modelar slippage |
| **Open interest** | Horario | Avaliar crowdedness da estrategia |

### 1.2 Streams Opcionais (recomendados)

- **Liquidation data**: modelar tail risk e eventos de ADL
- **Volume por periodo**: confirmar liquidez disponivel
- **Indicative/predicted funding rate**: a taxa publicada antes do settlement

### 1.3 Fontes de Dados

| Fonte | Tipo | Custo |
|-------|------|-------|
| **Binance API** (`/fapi/v1/fundingRate`) | Funding rates, OHLCV, mark price desde ~2019 | Gratis |
| **Bybit API** | Similar a Binance | Gratis |
| **Tardis.dev** | Tick-level, order book, normalizado entre 20+ exchanges | Pago |
| **CoinGlass** | Funding rate cross-exchange, open interest | Freemium |
| **CoinAPI** | Funding rates normalizados (Binance, Bybit, OKX, Deribit) | Pago |

**Regra**: Sempre usar dados com timestamps exatos de settlement, NUNCA medias diarias.

---

## 2. Vieses e Armadilhas Criticas

### 2.1 Survivorship Bias
- Incluir tokens que foram delistados e exchanges que falharam (ex: FTX)
- Testar apenas BTC/ETH em Binance superestima retornos
- Impacto medido: estrategias esperadas de 20% caem para ~8% apos correcao

### 2.2 Look-Ahead Bias
- **NUNCA** usar a funding rate final (settled) para decidir entrada ANTES do settlement
- Usar apenas o **indicative/predicted funding rate** (publicado ~8h antes) para decisoes
- A taxa final so e conhecida no momento do settlement

### 2.3 Vies de Execucao no Timestamp
- Nao e possivel abrir posicao exatamente no momento do settlement e coletar o pagamento
- A posicao deve estar aberta ANTES do settlement
- Modelar lead time realista (entrada 1-5 min antes do settlement)

### 2.4 Autocorrelacao do Funding Rate
- Funding rates tem **baixa autocorrelacao para lags > 3 periodos**
- Seguem processo de Ornstein-Uhlenbeck (mean-reverting para ~0.01%)
- Estrategias que assumem persistencia por muitos dias provavelmente estao overfitting a regimes de bull market

### 2.5 Overfitting / Data Snooping
- Testar 7 variacoes pode produzir pelo menos 1 backtest com Sharpe > 1.0 mesmo sem edge real
- Usar **Deflated Sharpe Ratio (DSR)** para ajustar para multiplos testes
- Calcular **Minimum Backtest Length (MinBTL)**

### 2.6 Crowdedness
- Quando muito capital persegue funding rate arb, as taxas comprimem
- Backtests com taxas de 2021 (bull market) superestimam retornos futuros
- Monitorar open interest como proxy de crowdedness

---

## 3. Modelagem de Custos

### 3.1 Tabela de Custos

| Componente | Range Tipico | Notas |
|------------|-------------|-------|
| **Spot taker fee** | 0.04-0.10% | Na entrada e saida |
| **Perp taker fee** | 0.04-0.06% | Por leg |
| **Perp maker fee** | 0.00-0.02% | Pode ser negativa (rebate) |
| **Slippage spot** | 0.01-0.20% | Depende do tamanho vs book depth |
| **Slippage perp** | 0.01-0.20% | Pior em small-caps |
| **Margin borrowing** | 5-15% APR | Para spot alavancado ou USDT |
| **Collateral haircut** | 5-90% | Colateral nao-stable vale 10-95% |
| **Transfer/withdrawal** | Variavel | Critico para arb cross-exchange |

### 3.2 Formula de Lucro por Periodo de Funding

```
Net_Profit = (Funding_Rate * Notional)
             - Spot_Entry_Fee
             - Perp_Entry_Fee
             - Spot_Slippage
             - Perp_Slippage
             - Margin_Interest_Accrued
             - (Exit_Fees ao fechar)
```

**Referencia**: Apos todos os custos, yields reais tipicamente ficam abaixo de 10% (fonte: 1Token Tech).

### 3.3 Modelo de Slippage

```
Effective_Slippage = Base_Slippage + (Order_Size / Avg_Book_Depth) * Impact_Factor
```

`Impact_Factor`: 0.5 a 2.0 dependendo da volatilidade. Em periodos volateis (quando funding spikes), book depth diminui dramaticamente.

---

## 4. Logica de Entrada e Saida

### 4.1 Entrada (Funding Positivo - Cash & Carry)

1. **Indicative funding rate** > threshold (sugerido: > 0.03% por 8h = ~32.85% anualizado bruto)
2. **Simultaneamente**: Compra spot + Short perpetual (mesmo notional)
3. **Execucao**: Uma leg em maker, outra em taker para reduzir custos
4. **Timing**: Entrar bem antes do settlement, nao no momento exato

### 4.2 Saida

1. Funding rate cai abaixo do threshold minimo de lucratividade (ex: < 0.005% apos custos)
2. Funding rate vira negativo (voce passa a pagar)
3. Ranking de ADL sobe acima do threshold de seguranca
4. Maintenance margin ratio se aproxima de liquidacao
5. Basis (premium do perp sobre spot) comprime o suficiente para take profit

### 4.3 Funding Negativo (Trade Reverso)

- Long perp + Short spot (via margin borrowing)
- So viavel se custo de borrowing < |funding rate|
- Risco maior por borrowing costs e short-squeeze

### 4.4 Estrategia Ativa vs Passiva

- **Ativa** (30-50% turnover diario): Rota entre instrumentos com maior funding a cada horas
- **Passiva** (<10% turnover diario): Mantem posicoes por dias/semanas em regimes persistentes

---

## 5. Metricas de Performance

### 5.1 Metricas Obrigatorias

| Metrica | Benchmark | Notas |
|---------|-----------|-------|
| **Retorno Anualizado (net)** | 8-15% institutional grade | Deve ser NET, nao bruto |
| **Sharpe Ratio** | > 1.0 solido; > 2.0 forte; > 3.0 suspeito | Crypto Sharpe 1.5 ~ equity Sharpe 2.0 |
| **Sortino Ratio** | Maior = melhor | Penaliza apenas downside vol |
| **Calmar Ratio** | 5-10 para bom funding arb | Retorno / Max Drawdown |
| **Max Drawdown** | < 5% (tolerancia institucional) | Esperar DD live = 1.5-2x backtested |
| **Win Rate por periodo** | % de periodos onde net funding > custos | |
| **Deflated Sharpe Ratio** | Ajusta para multiplos testes | Previne falsos positivos |
| **Tempo no mercado** | % do tempo com posicoes ativas | Eficiencia de capital |

### 5.2 Decomposicao de PnL (OBRIGATORIO)

Todo backtest DEVE separar o PnL em:

1. **Receita de funding** (alpha source)
2. **Basis PnL** (mudancas mark-to-market no spread perp-spot)
3. **Custos de trading** (fees + slippage)
4. **Custos de margem/borrowing**
5. **Custos de rebalanceamento** (ajustes no hedge ratio)

---

## 6. Position Sizing e Risco

### 6.1 Sizing

```
Max_Position = Account_Equity * Max_Leverage * Max_Concentration_Pct
Margin_Buffer = Position * (1/Leverage - Maintenance_Margin_Rate) * Safety_Factor
Capital_Necessario = Position + Margin_Buffer
```

- **Large caps (BTC/ETH)**: Posicoes maiores OK (liquidez profunda, funding mais estavel)
- **Small caps**: Limites estritos. Funding extremo (>0.1% por 8h) frequentemente precede squeezes violentos

### 6.2 Controles de Risco a Modelar

1. **Buffer de liquidacao**: Manter colateral bem acima de maintenance margin
2. **Risco ADL**: Modelar como evento adverso aleatorio (exchange deleverage sua posicao lucrativa)
3. **Monitoramento de delta**: Posicoes "delta-neutral" driftam com fills parciais e tempos diferentes de execucao. Definir tolerancia maxima de delta
4. **Risco de transferencia cross-exchange**: Congestao de rede e freezes de withdrawal
5. **Risco de contraparte**: Probabilidade de falha da exchange (diversificar entre venues)

---

## 7. Realismo de Execucao

### 7.1 O Problema do Timestamp

- NAO modelar entrada no preco exato do funding settlement
- Indicative rate e publicado ~8h antes do settlement
- Outros arbitradores competem pelo mesmo trade, impactando o premium
- **Modelar delay realista**: entrada 1-5 min antes do settlement usando precos daquele momento

### 7.2 Modelo de Execucao de Ordens

- **Market orders**: Assumir execucao no bid/ask + slippage baseado em book depth
- **Limit orders**: Modelar fills parciais e tempo para fill (pode perder o funding period)
- **Ambas as legs**: Sempre ha gap de timing entre spot e perp. Modelar 1-10 seg de delta exposure

### 7.3 Gap Live vs Backtest

Pesquisas mostram consistentemente:
- **Drawdown live = 1.5x a 2x** do backtested
- **Sharpe backtested 2.0 vira 1.0-1.5** em live
- **Retorno backtested 20% entrega 8-12%** net em live

---

## 8. Consideracoes Multi-Exchange

### 8.1 Intervalos de Funding

| Exchange | Intervalo | Notas |
|----------|-----------|-------|
| **Binance** | 8h (00:00, 08:00, 16:00 UTC) | Alguns pares tem 4h |
| **Bybit** | 8h | TWAP minute-by-minute do premium index |
| **OKX** | 8h | Mudou de variavel em Jan 2024 |
| **Hyperliquid** | 1h | 0.01%/h = 0.03%/8h equivalente |
| **dYdX** | 1h | DEX, settlement on-chain |

### 8.2 Armadilha de Anualizacao

```
0.01% em exchange 8h = 10.95% anualizado
0.01% em exchange 1h = 87.6% anualizado
```

**OBRIGATORIO**: Normalizar taxas para timeframe comum antes de comparar entre exchanges.

### 8.3 Diferencas no Calculo

- **Binance**: `FR = Premium_Index + clamp(Interest_Rate - Premium_Index, -0.05%, 0.05%)`. Interest rate default = 0.01% por 8h
- **Bybit**: Similar mas usa TWAP continuo minute-by-minute
- **DEXes (Synthetix)**: Modelo de funding velocity baseado em skew. SUA posicao impacta o funding rate

---

## 9. Arquitetura do Backtester

### 9.1 Event-Driven, NAO Vectorizado

Usar backtester event-driven (nao calculos vetorizados simples em pandas) que modele:

- Contas separadas para spot e futures
- Order placement -> fill simulation com latencia realista
- Eventos de funding settlement nos timestamps corretos
- Calculos de margem atualizados a cada tick de preco
- Engine de liquidacao que verifica maintenance margin continuamente

### 9.2 Walk-Forward Validation (OBRIGATORIO)

1. **In-sample**: Otimizar parametros nos primeiros 60% dos dados
2. **Out-of-sample**: Validar nos 40% restantes
3. **Walk-forward**: Janela rolante de otimizacao para detectar mudancas de regime
4. **NUNCA** otimizar no dataset completo e reportar esses resultados

### 9.3 Ferramentas Recomendadas

| Ferramenta | Uso | Destaque |
|-----------|-----|----------|
| **Python custom + Tardis.dev** | Dados normalizados + engine customizado | Maior controle |
| **hftbacktest** (GitHub) | Considera queue position e latencia | Suporta Binance/Bybit |
| **DolphinDB** | Event-driven completo para crypto | Multi-account, slippage configuravel |
| **btrccts** (GitHub) | Interface ccxt para backtest e live | Mesmo codigo backtest/live |

---

## 10. Checklist de Validacao

Antes de confiar em qualquer resultado de backtest, verificar:

- [ ] Dados com timestamps exatos de settlement (nao medias diarias)
- [ ] Mark price usado para margem/liquidacao (nao last trade price)
- [ ] Ambas as legs modeladas separadamente com fee e slippage independentes
- [ ] Entrada ANTES do settlement, nao no timestamp exato
- [ ] TODOS os custos incluidos: fees, slippage, margin interest, collateral haircuts
- [ ] Risco de liquidacao modelado com maintenance margin check por tick
- [ ] Correcao de survivorship bias aplicada
- [ ] Walk-forward validation (nao in-sample optimization)
- [ ] Deflated Sharpe Ratio reportado se testou multiplos parametros
- [ ] Funding rates normalizados entre exchanges
- [ ] Expectativa de performance live = 50-60% do backtested
- [ ] PnL decomposto em: funding income, basis PnL, custos

---

## 11. Referencias

### Academicas
1. **BIS Working Paper 1087** - "Crypto Carry" (Schmeling, Schrimpf, Todorov, 2023)
2. **"Predictability of Funding Rates"** (Emre Inan, SSRN #5576424)
3. **"Perpetual Futures Pricing"** (Ackerer, Hugonnier, Jermann - Wharton)
4. **"Designing Funding Rates for Perpetual Futures"** (arXiv:2506.08573)
5. **"Risk and Return Profiles of Funding Rate Arbitrage"** (ScienceDirect, 2025)
6. **"The Crypto Carry Trade"** (Carnegie Mellon)

### Industria
7. **1Token Tech** - "Crypto Fund 101: Funding Fee Arbitrage" (Sharpe/Calmar 5-10, $20B+ deployados)
8. **Amberdata** - "Ultimate Guide to Funding Rate Arbitrage"
9. **CoinGlass** - "What is Funding Rate Arbitrage"

### Dados e Ferramentas
10. **Tardis.dev** - Dados historicos tick-level normalizados
11. **CoinAPI** - Funding rates historicos normalizados
12. **hftbacktest** (GitHub: nkaz001/hftbacktest) - Backtester com order book
13. **funding-rate-arbitrage** (GitHub: 50shadesofgwei) - Bot de referencia
14. **fundingrate** (PyPI) - Package Python para consulta de funding rates
