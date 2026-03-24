# Estrategias Descartadas — Com Justificativas

> Estas estrategias foram avaliadas e REJEITADAS pelo Comite Quant.
> Nao reimplementar sem evidencia nova substancial.

---

## Descartadas com Backtest

| Estrategia | CAGR | Motivo | Fonte |
|---|---|---|---|
| **Contrarian long/short** | -22% | Short altcoins = squeezes destrutivos | Backtest proprio |
| **Pairs trading classico** | 1.6% | Cointegration quebra em stress, universo pequeno | `src/strategy_pairs.py` |
| **Contrarian long-only RSI** | ~4.3% | Alpha insuficiente (mantido a 10% por descorrelacao) | `src/strategy_contrarian.py` |
| **Grid trading** | -40% DD em 2022 | Preco sai do grid em crash, sem recovery | Simulacao |

## Descartadas com Pesquisa

| Estrategia | Motivo | Fonte |
|---|---|---|
| **Vol selling (Deribit)** | Correlacao em stress = 1.0. Capital insuficiente para sub-$100k | SONNET-R veto |
| **DeFi LP** | 54% dos LPs perdem. IL > fees. Hedge de IL custa mais que yield | Research academico |
| **Recursive lending** | Bomba-relogio alavancada. Max DD -50% possivel | SONNET-R veto |
| **Cross-exchange arb** | Apenas 40% das oportunidades geram retorno positivo pos-custos | [P7] Zhivkov 2026 |
| **Cross-chain arb** | Spread eliminado por custos de bridge (0-2% net) | SONNET-M analise |
| **Cross-exchange funding arb** | Sem edge para retail. CEX domina price discovery 61% | [P7] Zhivkov 2026 |
