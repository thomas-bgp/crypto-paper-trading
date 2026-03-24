# Estrategia Aprovada: Momentum Cross-Sectional + Regime Switch v2

**Data:** 2026-03-15
**Status:** APROVADA — Portfolio Strategy #1
**Codename:** MOMENTUM-REGIME-v2

---

## Performance (Backtest 2022-03 a 2026-03, ~4 anos)

| Metrica | Valor |
|---------|-------|
| CAGR | 69.61% |
| Sharpe | 1.48 |
| Sortino | 1.96 |
| Calmar | 1.95 |
| Max DD | -35.71% |
| $10k → | $81,586 |
| BTC CAGR (benchmark) | 12.43% |

## Performance por Ano

| Ano | Return | BTC |
|-----|--------|-----|
| 2022 | +63.9% | -65% |
| 2023 | +214.9% | +155% |
| 2024 | +69.5% | +129% |
| 2025 | +0.6% | -6.6% |

## Descricao

### Regime Detection (HMM v2 com anti-whipsaw)
- GaussianHMM 3 estados, features [log_ret, rvol_14, vol_ratio_20]
- Temperature scaling (t=2.0) para evitar confidence=1.0
- Hysteresis filter (margem 15% para trocar regime)
- Minimum duration filter (5 candles = 20h confirmacao)
- Funding rate + Fear & Greed overrides

### Alocacao (continua, nao binaria)
- mom_weight = blend proporcional: P(BULL)*0.80 + P(SIDE)*0.40 + P(BEAR)*0.15
- Floor: minimo 15% momentum em qualquer regime
- Teto: maximo 85%
- Cash remainder: stablecoin yield ~5% aa

### Momentum Signal
- Donchian Channel breakout (fast=20, slow=50) em 4h candles
- Cross-sectional ranking: top 20% por retorno de 24 candles (4 dias)
- Universo: top 20 altcoins por liquidez
- Rebalanceamento: a cada 6 candles (24h)

### Custos modelados
- Maker fee: 0.02% (Binance BNB)
- Slippage: 0.05%
- Turnover cost proporcional a mudanca de alocacao

## Correlacao esperada
- Alta correlacao com BTC em bull markets
- Baixa correlacao em bear (cash)
- **Precisa de estrategia descorrelacionada para suavizar anos flat (2025)**

## Codigo
- `src/regime_detector.py` — HMM v2
- `src/strategies.py` — Momentum + MR signals
- `src/backtester.py` — Backtester event-driven
- `src/dashboard.py` — Streamlit dashboard
