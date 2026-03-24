# Factor Model: Feature Specification Complete

> Compilado em 2026-03-16. 112+ fatores da literatura + fatores de derivada polinomial.
> Para implementação detalhada, ver o research report completo em memória do comitê.

## Resumo de Categorias

| Categoria | Qtd | Top Preditores (ML) | Fonte Principal |
|-----------|-----|---------------------|----------------|
| Momentum (retornos passados) | 12 | RET1W, RET4W, RET52W_high | Liu et al. JF 2022 |
| Price / Size | 4 | LOGPRC (dominante), LOGMCAP | Liu et al. 2022 |
| Liquidez | 7 | Amihud ILLIQ (dominante) | Cakici et al. 2024 |
| Volatilidade & Momentos | 12 | IVOL, RVOL4W, RSJ (signed jump) | Lee & Wang JFQA 2024 |
| Downside / Tail Risk | 6 | DOWNBETA (+62-78% aa premium) | Multiple |
| Technical (TA) - CTREND | 28 | Aggregate > qualquer TA individual | Fieberg et al. JFQA 2024 |
| Volume / Microestrutura | 8 | TURNOVER, VWAP deviation | Multiple |
| On-chain | 14 | Active addresses, MVRV, exchange flow | Sakkas & Urquhart 2024 |
| Derivativos / Futures | 8 | Funding rate, OI change | BIS WP 1087 |
| Sentimento / Atencao | 9 | GOOGLE, Fear&Greed beta | Liu & Tsyvinski 2021 |
| **Derivada polinomial (novo)** | **4** | **Poly1st, Poly2nd (curvatura)** | **Proposta do usuario** |
| **TOTAL** | **~112** | | |

## Fatores de Derivada Polinomial (User-Proposed)

Regressao polinomial em log(price) sobre janela de T dias:
```
log(P_t) = β₀ + β₁·t + β₂·t² + ε
```

| Fator | Formula | Interpretacao |
|-------|---------|--------------|
| Poly1st (slope) | β̂₁ | Direcao e magnitude da tendencia |
| Poly2nd (curvature) | β̂₂ | Positivo = acelerando, Negativo = desacelerando/pico |
| Poly_velocity | β̂₁ + 2·β̂₂·T | Velocidade instantanea no ultimo ponto |
| Poly_R² | R² do fit quadratico | Quanto da variacao e explicada pela tendencia |

Janelas: 14d, 28d, 56d, 90d (4 variantes × 4 = 16 features no total)

### Hipotese do usuario (a testar):
- Alto retorno + alta vol + Poly2nd negativo (concavo) = **pullback iminente**
- Poly2nd positivo (convexo) + vol estavel = **aceleracao saudavel**
- Pico (Poly1st alto + Poly2nd muito negativo) = **reversao**

## Modelo ML Recomendado

### Pipeline

```
1. Feature preprocessing:
   - Cross-sectional rank normalization por periodo
   - Winsorize returns 1st/99th percentile
   - Rolling Z-score (12-month window)

2. Feature selection:
   - Stability selection (LASSO bootstrap 100x, keep >50%)
   - Distance correlation filter (dCor > 0.05)
   - SHAP post-hoc pruning

3. Primary model: LGBMRanker (LambdaMART)
   - Objective: lambdarank (otimiza ranking direto)
   - Target: ranked return percentile [0-4] quintiles
   - Query group = cada mes (cross-section)

4. Baseline: Elastic Net (Fama-MacBeth mensal) + Random Forest

5. Ensemble: media dos ranks dos 3 modelos (Borda count)

6. Validacao: CPCV (N=6, k=2, 15 combinacoes) + DSR + PBO
```

### Referências de Implementação

| Biblioteca | Uso |
|-----------|-----|
| lightgbm.LGBMRanker | Modelo principal (learning-to-rank) |
| sklearn.linear_model.ElasticNetCV | Baseline linear |
| sklearn.ensemble.RandomForestRegressor | Baseline tree |
| mlfinlab | Purged K-Fold, CPCV, DSR, PBO |
| shap | Feature importance |
| dcor | Distance correlation |
| numpy.polyfit | Derivadas polinomiais |
