# Decisao: Revisao de Estrategias Pos-Pesquisa Intensiva

**Data:** 2026-03-15
**Comite:** OPUS (CIO) + 4 Squads de pesquisa
**Status:** PORTFOLIO REVISADO

---

## Conclusao Principal

**20% aa homogeneo em crypto sem alavancagem NAO EXISTE documentado.**

Nenhum paper academico, nenhum hedge fund publico, nenhum track record verificavel prova retorno de 20%+ TODOS os anos de 2020 a 2025. O ano de 2022 (-65% BTC) destroi qualquer estrategia direcional, e estrategias market-neutral entregam 12-18% na melhor das hipoteses.

## Portfolio Revisado

### Strategy #1 — Momentum Regime Switch v2 (MANTIDA, UPGRADE SIGNAL)
- **Alocacao:** 50%
- **Mudanca:** Upgrade do signal Donchian para CTREND (elastic net, JFQA 2024)
- **CAGR esperado:** 40-60% aa medio (nao homogeneo: +200% em bull, flat em bear)
- **Funcao:** Alpha generator direcional com regime filter

### Strategy #2 — Funding Rate Delta-Neutral + Regime Filter (NOVA)
- **Alocacao:** 40%
- **Tipo:** Market-neutral (delta zero)
- **CAGR esperado:** 12-18% aa (o mais homogeneo disponivel)
- **Sharpe esperado:** 2-4
- **Max DD:** <5% em operacao normal
- **Funcao:** Gerador de retorno estavel, ancora do portfolio
- **Correlacao com #1:** ~0.10-0.25 (baixa — funding nao depende de qual coin lidera)

### Cash/Stablecoin Yield (BUFFER)
- **Alocacao:** 10%
- **Yield:** 5% aa (Aave/Compound USDC)
- **Funcao:** Buffer de margem + emergency reserve

### Descartadas definitivamente
- Contrarian long/short: CAGR -22%
- Pairs trading classico: CAGR 1.6%
- Grid trading: DD -40% em crash
- Vol selling Deribit: corr stress = 1.0, capital insuficiente
- DeFi LP: 54% dos LPs perdem
- Recursive lending: bomba-relogio

## Expectativa Honesta do Portfolio

| Cenario | Prob | #1 (50%) | #2 (40%) | Cash (10%) | **Portfolio** |
|---|---|---|---|---|---|
| Bull forte | 20% | +60% | +25% | +5% | **+41%** |
| Bull moderado | 25% | +30% | +18% | +5% | **+22%** |
| Sideways | 30% | +5% | +12% | +5% | **+8%** |
| Bear | 25% | -5% | +5% | +5% | **-1%** |
| **Media ponderada** | — | — | — | — | **~17-18%** |

**Nota:** 17-18% medio e o MAXIMO HONESTO. Para atingir 20%+, necessario ou aceitar mais risco ou usar leverage moderada (1.5x) na Strategy #2.

## Proximos Passos
1. Implementar Strategy #2 (Funding Delta-Neutral) no backtester
2. Upgrade signal de momentum para CTREND
3. Re-rodar backtests
4. Dashboard atualizado
5. Paper trading 30 dias
