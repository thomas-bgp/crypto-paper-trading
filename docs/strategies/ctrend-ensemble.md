# Strategy #3 Research: CTREND Ensemble (Cross-Sectional Trend Factor)

> Pesquisa intensiva 2026-03-15. Baseada no paper JFQA 2024.
> Conclusao: melhor alpha generator mas NAO homogeneo — depende de regime.

---

## O que e o CTREND

Paper publicado no **Journal of Financial and Quantitative Analysis (2024)** por Fieberg, Liedtke, Poddig, Walker, Zaremba. Usa **elastic net** (ML regularizado) para agregar 20+ indicadores tecnicos em multiplos timeframes, gerando ranking semanal de coins.

**Resultado do paper:**
- 3.87% por semana no long-short bruto (universo de 3000+ coins)
- Sharpe >1.5
- Alpha de 10.8% aa vs BTC (apos custos)
- Funciona em subperiodos 2015-2018 E 2018-2022
- Supera TODOS os outros fatores crypto conhecidos

## Por que NAO e Strategy #2 (e sim #3)

O CTREND e **direcional** (long-heavy). Em 2022, qualquer estrategia long crypto perdeu 30-50%. NAO e market-neutral. Por isso e complementar, nao substituto do funding rate.

## Diferenca do nosso momentum atual

Nosso momentum usa Donchian Channel simples (fast=20, slow=50). O CTREND usa:
- Elastic net sobre 20+ features (MAs em 7d/14d/30d/60d, RSI, MACD, volume, volatilidade)
- Regularizacao que evita overfitting
- Cross-sectional ranking (nao time-series)
- **E academicamente mais robusto que Donchian**

## Implementacao Pratica

```python
# Simplified CTREND implementation
from sklearn.linear_model import ElasticNet
import pandas_ta as ta

def compute_ctrend_features(df):
    """Compute features for CTREND factor."""
    feats = pd.DataFrame(index=df.index)
    # Multi-timeframe momentum
    for p in [7, 14, 30, 60]:
        feats[f'ret_{p}'] = df['close'].pct_change(p)
    # Technical indicators
    feats['rsi_14'] = ta.rsi(df['close'], 14)
    feats['rsi_7'] = ta.rsi(df['close'], 7)
    macd = ta.macd(df['close'])
    feats['macd_hist'] = macd['MACDh_12_26_9']
    feats['ema_ratio'] = ta.ema(df['close'], 20) / ta.ema(df['close'], 50)
    feats['vol_14'] = df['close'].pct_change().rolling(14).std()
    feats['vol_ratio'] = feats['vol_14'] / feats['vol_14'].rolling(60).mean()
    feats['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    return feats.dropna()

# Weekly: rank coins by CTREND score → long top quintile
```

## Performance Esperada (conservadora, long-only top 20 coins)

| Ano | Estimativa | Notas |
|---|---|---|
| Bull | +30-60% | Momentum works |
| Sideways | +5-15% | Rotacao captura dispersao |
| Bear | -15 a -30% | **Problema: long-only perde** |

**Com regime filter (HMM):**
- BULL: 100% alocado em CTREND
- SIDEWAYS: 50% CTREND, 50% cash
- BEAR: 0% CTREND, 100% cash

**Resultado ponderado com filter: ~20-35% aa medio, mas volatil entre anos**

## Correlacao com Strategy #1 (Momentum Regime)

**Alta (~0.6-0.8)** — ambos sao momentum. A diversificacao vem do METODO (elastic net vs Donchian), nao da direcao. CTREND pode ser usado como UPGRADE do nosso momentum, nao como estrategia independente.

## Recomendacao do Squad

**Usar CTREND como upgrade do signal de momentum na Strategy #1**, nao como Strategy #3 independente. A correlacao e alta demais para Markowitz.

---

## Alternativas para Strategy #3 (baixa correlacao)

### Opcao A: Intraday Seasonality (21h-23h UTC)
- QuantPedia documenta CAGR 40.6%, Calmar 1.79
- Comprar BTC as 20h UTC, vender as 23h UTC
- **Completamente descorrelacionado** com momentum cross-sectional
- Problema: dados limitados, precisa de validacao fora da amostra

### Opcao B: Funding Rate Sentiment Signal
- Usar funding rate NAO como carry, mas como SINAL CONTRARIAN
- Funding > 0.1%/8h = mercado sobreaquecido = REDUZIR long
- Funding negativo persistente = oversold = ENTRAR long
- Descorrelacionado com momentum puro

### Opcao C: Calendar Effects + Token Unlock
- Token unlocks causam queda media de -2.4% no dia
- Shortar tokens pre-unlock = alpha episodico
- Completamente descorrelacionado mas low frequency

---

## Fontes
- JFQA 2024: Fieberg et al. "A Trend Factor for the Cross-Section of Cryptocurrency Returns"
- SSRN 2025: Zarattini, Pagani, Barbon "Catching Crypto Trends"
- SSRN 2025: Mann "Quantitative Alpha in Crypto Markets"
- SSRN 2024: Han, Kang, Ryu "Time-Series and Cross-Sectional Momentum in Crypto"
- QuantPedia: "Intraday Seasonality in Bitcoin"
