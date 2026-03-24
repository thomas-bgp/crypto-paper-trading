# Strategy #1 — Momentum Cross-Sectional + Regime Switch v2

**Status:** IMPLEMENTADA
**Codename:** MOMENTUM-REGIME-v2
**Codigo:** `src/backtester.py` + `src/strategies.py` + `src/regime_detector.py`

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

### Por Ano

| Ano | Return | BTC |
|-----|--------|-----|
| 2022 | +63.9% | -65% |
| 2023 | +214.9% | +155% |
| 2024 | +69.5% | +129% |
| 2025 | +0.6% | -6.6% |

---

## Arquitetura

### Regime Detection (HMM v2)
- GaussianHMM 3 estados, features: [log_ret, rvol_14, vol_ratio_20]
- Temperature scaling (t=2.0) — evita confidence=1.0
- Hysteresis filter (margem 15% para trocar regime)
- Minimum duration filter (5 candles = 20h confirmacao)
- Funding rate + Fear & Greed overrides
- Walk-forward: refit a cada 42 candles (7 dias)

### Alocacao (continua, nao binaria)
- mom_weight = P(BULL)*0.80 + P(SIDE)*0.40 + P(BEAR)*0.15
- Floor: 15% momentum | Teto: 85%
- Cash = stablecoin yield ~5% aa

### Momentum Signal
- Donchian Channel breakout (fast=20, slow=50) em 4h candles
- Cross-sectional ranking: top 20% por retorno de 24 candles (4 dias)
- Universo: 20 altcoins
- Rebalanceamento: a cada 6 candles (24h)

### Custos modelados
- Maker: 0.02% | Slippage: 0.05% | Total/lado: 0.07%
- Turnover cost proporcional a mudanca de alocacao

---

## Upgrade pendente: CTREND

O signal Donchian deve ser substituido por CTREND (elastic net, JFQA 2024).
Ver `docs/strategies/ctrend-ensemble.md` para detalhes.

## Fraquezas conhecidas
- Flat em sideways/bear (2025: +0.6%)
- Alta correlacao com BTC em bull (0.6-0.8)
- Max DD de -35.71% e alto para estrategia com regime filter
