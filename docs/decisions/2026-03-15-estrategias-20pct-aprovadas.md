# Decisao: Estrategias para Target 20%+ aa

**Data:** 2026-03-15
**Comite:** OPUS (CIO), SONNET-A, SONNET-R, SONNET-M, SONNET-D
**Status:** APROVADO

---

## Decisao

Portfolio de 3 camadas com regime switch aprovado para implementacao.

## Core da estrategia

1. **Trend Following cross-sectional** (Camada 1, 60-70%) — Donchian ensemble em top 20-40 altcoins
2. **Mean Reversion + Grid** (Camada 2, 20-30%) — RSI/Bollinger + grid nativo em sideways
3. **Stablecoin Yield** (Camada 3, 10-15%) — LP em L2 ou lending Aave

## Regime switch (OBRIGATORIO)
- Bull: BTC > SMA200, F&G > 50 → Camada 1 dominante
- Sideways: BTC ~ SMA200, ADX < 25 → Camada 2 dominante
- Bear: BTC < SMA200, F&G < 30 → 90% Camada 3 (cash/stables)

## Confianca

| Membro | Confianca |
|--------|-----------|
| SONNET-A | 6.5/10 |
| SONNET-R | 5/10 |
| SONNET-M | 6/10 |
| SONNET-D | 6/10 |
| OPUS (CIO) | 6/10 |

**Media: 5.9/10** — atingivel em horizonte de 3+ anos, nao garantido todo ano.

## Retorno esperado ponderado: 12-20% aa
- Bull (30% do tempo): 20-38%
- Neutro (40%): 8-16%
- Bear (30%): -3 a +5%

## Infra aprovada
- Hetzner CX22 ($5/mes) + Freqtrade + Binance
- Custo total: $5-16/mes

## Dissenting opinions
- SONNET-R: "20% nao e garantido todo ano. O investidor que PRECISA de 20% vai assumir risco excessivo."
- SONNET-M: "LP concentrada tem APY atraente mas 63% das posicoes non-stable terminam em prejuizo."

## Proximos passos
1. Setup VPS + Freqtrade (1-2 semanas)
2. Backtest momentum cross-sectional com walk-forward (3-4 semanas)
3. Dry-run 30-60 dias
4. Live com 10-20% do capital (mes 4+)
