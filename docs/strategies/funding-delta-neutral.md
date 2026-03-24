# Strategy #2 Research: Funding Rate Delta-Neutral com Regime Filter

> Pesquisa intensiva realizada em 2026-03-15 por 4 squads independentes.
> Conclusao: unica estrategia market-neutral com evidencia de 12-20% aa e DD <5%.

---

## Por que esta estrategia (e nao as outras)

### Descartadas com evidencia:

| Estrategia | Resultado | Motivo da rejeicao |
|---|---|---|
| Contrarian long/short | CAGR -22% | Short altcoins = suicidio em crypto (squeezes) |
| Pairs trading classico | CAGR 1.6% | Universo pequeno, cointegration quebra em stress |
| Contrarian long-only RSI | CAGR -0.9% | Comprar oversold funciona pouco, nao gera alpha |
| Grid trading | -40% em 2022 | Preco sai do grid em crash, sem recovery |
| Vol selling (Deribit) | Corr stress = 1.0 | Capital insuficiente + tail risk destrutivo |
| DeFi LP | 54% dos LPs perdem | IL > fees, hedge custa mais que yield |
| Recursive lending | -50% DD possivel | Bomba-relogio alavancada |
| Cross-chain arb | 0-2% net | Spread eliminado por custos de bridge |

### Por que Funding Rate Delta-Neutral vence:

1. **Sharpe historico: 3-6** (o mais alto de qualquer estrategia crypto documentada)
2. **Max DD: <5%** em operacao normal (market-neutral por construcao)
3. **Mecanismo estrutural**: varejo paga para ser long alavancado → shorts recebem
4. **Funciona em bull E sideways** (funding positivo 85%+ do tempo)
5. **Problema resolvivel**: em bear, funding fica negativo → nosso HMM detecta e sai para cash

---

## Mecanismo

```
LONG spot (BTC/ETH) + SHORT perpetual (mesmo notional)
= Delta zero (nao importa se preco sobe ou desce)
= Recebe funding rate a cada 8h (quando funding > 0)
```

## Performance Historica Documentada

| Ano | Funding Rate Medio | Retorno Estimado | Fonte |
|---|---|---|---|
| 2020 | Alto (bull) | 20-40% | BIS WP 1087 |
| 2021 | Muito alto | 25-45% | BIS WP 1087 |
| 2022 | Baixo/negativo | 0-5% (com regime filter: sai para cash) | BIS: Sharpe cai |
| 2023 | Moderado | 10-15% | 1Token Quant Index |
| 2024 | Alto (ETF rally) | 14-22% | ScienceDirect 2025 |
| 2025 | Comprimido (Ethena) | 5-10% (com regime: 8-12%) | Bitget/Gate research |

**Com regime filter HMM:**
- Evita periodos de funding negativo (2022 bear, crashes)
- Aloca para cash/stablecoin yield (5% aa) quando HMM = BEAR
- Retorno estimado ponderado: **12-18% aa** homogeneo

## Custos Reais com $30k

```
Capital: $15k spot + $15k margem para short perp
Fee entrada (2 legs): 2 x 0.02% x $15k = $6
Fee saida: $6
Funding recebido (0.01%/8h medio): $15k x 0.01% x 3/dia = $4.50/dia
Funding anual bruto: ~$1,640 (10.9% sobre $15k notional, 5.5% sobre $30k)

Em bull (0.03%/8h): ~$4,920/ano = 16.4% sobre $30k
Em bear: SAI para cash (HMM filter) → 5% stablecoin yield
```

## Implementacao

### Selecao de pares
- Monitorar funding rate de top 20 perps (Binance/Bybit)
- Entrar nos 5-8 pares com funding > 0.01%/8h
- Diversificar: nunca >25% em um unico par

### Regime Filter (HMM v2)
- BULL/SIDEWAYS: carry ativo
- BEAR: fechar tudo, mover para USDC lending (Aave 5%)

### Rebalanceamento
- Checar funding rates a cada 8h (3x/dia)
- Rotacionar pares quando funding de um par cai abaixo de 0.005%
- Custo de rotacao: ~$12 por par (2 legs x entry+exit)

### Riscos e Mitigacoes

| Risco | Probabilidade | Mitigacao |
|---|---|---|
| Funding negativo prolongado | 15-25%/ano | HMM filter sai para cash |
| Exchange insolvency | 5-10%/5 anos | Split capital 50/50 entre 2 exchanges |
| ADL (auto-deleveraging) | 10-20%/ano | Monitorar ADL ranking via API |
| Flash crash (basis diverge) | 5%/ano | Stop loss se basis > 3% |
| Ethena crowding (comprime yields) | Estrutural | Diversificar em altcoin perps (funding mais alto) |

---

## Expectativa Honesta

| Cenario | Retorno aa | Probabilidade |
|---|---|---|
| Bull forte | 18-30% | 25% |
| Bull moderado | 14-20% | 25% |
| Sideways | 8-14% | 30% |
| Bear (com filter) | 4-6% (cash) | 20% |
| **Media ponderada** | **~12-16%** | — |

**Para atingir 20%+:** necessario aceitar mais risco em altcoin perps (funding mais alto mas mais volatil) ou usar leverage moderada (1.5-2x).

---

## Fontes Principais
- BIS Working Paper 1087 — Crypto Carry (2023/2025)
- ScienceDirect 2025 — Risk and Return of Funding Rate Arbitrage
- arxiv 2510.14435 — Cryptocurrency as Investable Asset Class
- 1Token Quant Strategy Index (2025)
- SSRN 5292305 — Chan, Leveraged BTC Funding Carry Algorithm
