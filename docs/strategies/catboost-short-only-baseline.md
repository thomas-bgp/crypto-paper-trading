# Strategy BASELINE: CatBoost Short-Only (Modelo a Bater)

> **Status:** BENCHMARK INTERNO — todos os modelos futuros devem superar este
> **Data:** 2026-03-16
> **Codigo:** `src/ml_v3_catboost.py` (perna short isolada)

---

## Performance (Auditado, 2021-09 a 2026-02)

| Metrica | Short-Only | Combinado L/S | Long-Only |
|---------|-----------|---------------|-----------|
| **$100k →** | **$1,033,756** | $189,360 | $33,646 |
| **Retorno total** | **+933.8%** | +89.4% | -66.4% |
| **Sharpe** | **1.36** | 0.80 | -0.28 |
| **Avg ret/periodo** | **+2.37%** | +0.62% | -0.50% |
| **Win Rate** | **51%** | 52% | 42% |
| **Max DD** | **-37.9%** | -22.9% | -78.1% |

### Por Ano

| Ano | Short-Only | Long-Only |
|-----|-----------|-----------|
| 2022 | **+106.6%** | -44.3% |
| 2023 | **+30.3%** | +8.6% |
| 2024 | **+48.3%** | -33.1% |
| 2025 | **+109.8%** | +11.8% |
| 2026 (2mo) | **+23.5%** | -10.1% |

---

## Por que funciona

O modelo e excelente em identificar LOSERS (coins que vao underperformar).

### Mecanismo
1. Features dominantes sao de LIQUIDEZ (spread_28: 14.1%, amihud: 13.0%, turnover: 8.6%)
2. Coins iliquidas, com spread alto e momentum negativo sao alvos faceis de short
3. Em QUALQUER regime (bull, bear, sideways), as piores coins continuam caindo
4. O CatBoost YetiRank com ordered boosting consegue rankear losers de forma robusta

### Feature Importance

| # | Feature | Import. | Categoria |
|---|---------|---------|-----------|
| 1 | spread_28 | 14.09% | Liquidez |
| 2 | amihud | 12.98% | Liquidez |
| 3 | rvol_28_csrank | 9.06% | Vol cross-sect |
| 4 | turnover_28 | 8.57% | Liquidez |
| 5 | min_ret_28 | 7.29% | Tail risk |
| 6 | mom_14_skip1 | 6.92% | Momentum |
| 7 | poly_slope_56 | 5.97% | Derivada |
| 8 | mom_56 | 4.67% | Momentum |
| 9 | mom_14_csrank | 4.59% | Mom CS |
| 10 | max_ret_28 | 4.51% | Tail risk |
| 11 | poly_curve_56 | 4.21% | Derivada |

### ML Diagnostics
- Rank IC medio: 0.1064 (forte para finance)
- Rank IC positivo em 74% dos periodos
- Correlacao Long-Short: -0.59 (negativamente correlacionados)

---

## Configuracao Tecnica

```python
# CatBoost params
loss_function = 'YetiRank'
iterations = 300
depth = 4
learning_rate = 0.05
l2_leaf_reg = 5.0
random_strength = 2.0
bagging_temperature = 1.0
boosting_type = 'Ordered'  # anti-overfit critico

# Portfolio
TOP_N = 5              # short bottom 5
UNIVERSE_TOP = 40      # top 40 por volume
HOLDING_DAYS = 14      # rebalance bi-semanal
STOP_PCT = 0.15        # trailing stop 15%
COST_PER_SIDE = 0.002  # 0.20% realista
FUNDING = real Binance  # nao flat

# Target
target = market-neutralized ranked return (quintiles 0-4)
purge = 16 dias (14d holding + 2d buffer)
train_window = 18 meses rolling
retrain = a cada ~2 meses
ensemble = 3 modelos em janelas staggered

# Features: 20 curadas
momentum: mom_14, mom_28, mom_56, mom_14_skip1
derivada: poly_slope_28, poly_curve_28, poly_slope_56, poly_curve_56
volatilidade: rvol_28, vol_ratio, max_ret_28, min_ret_28
liquidez: amihud, spread_28, turnover_28
tecnico: rsi_14, macd_hist, donchian_pos
cross-sectional: mom_14_csrank, rvol_28_csrank
```

---

## Custos ja incluidos no backtest

- Trailing stop path-dependent (intraday high/low)
- Funding rate real da Binance (nao flat)
- Transaction cost: 0.20% por lado
- Purge de 16 dias no treino
- Sem look-ahead bias (auditado por 3 equipes independentes)

---

## Implicacoes para proximos modelos

1. **Qualquer novo modelo deve comparar contra este short-only como baseline**
2. A perna LONG precisa de um modelo separado ou abordagem diferente (BTC trend, equal-weight)
3. O alpha esta em identificar LOSERS, nao WINNERS
4. Liquidez (spread, amihud, turnover) sao os features mais preditivos — mais que momentum
5. Derivadas polinomiais (poly_slope, poly_curve) sao o 7o e 11o feature — contribuem mas nao dominam

---

## Proximos passos sugeridos

- [ ] Implementar short-only + BTC trend na perna long
- [ ] Treinar modelo separado para identificar winners (features diferentes?)
- [ ] Testar com UNIVERSE_TOP = 50, 60 (mais coins para shortar)
- [ ] Adicionar features on-chain (NVT, active addresses) se dados disponíveis
- [ ] Paper trading 30 dias antes de capital real
