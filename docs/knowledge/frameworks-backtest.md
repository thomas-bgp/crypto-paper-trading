# Frameworks de Backtesting — Comparativo

> Decisao do comite: usar **Custom Python**. Maximo controle sobre funding, custos e regime.

| Framework | Funding Nativo? | Linguagem | Notas |
|-----------|----------------|-----------|-------|
| **DolphinDB** | SIM | DolphinDB script | Unico com `fundingSettlementMode`. Nao suporta posicoes iniciais |
| **hftbacktest** | Nao | Python+Rust | Melhor para HFT. Adaptavel para funding como custo periodico |
| **vectorbt** | Nao | Python | Rapido para research. Nao event-driven |
| **Freqtrade** | Bugado | Python | Issues #9106, #7291, #10888, #12583. **NAO USAR para funding** |
| **Custom Python** | Implementar | Python | **RECOMENDADO**. Usado em `src/backtester.py` |

## Requisitos do backtester (de `docs/knowledge/metodologia-backtest.md`)
- Event-driven, NAO vectorizado
- Contas separadas spot/futures
- Order placement com latencia realista
- Funding settlement nos timestamps corretos
- Engine de liquidacao com maintenance margin check
