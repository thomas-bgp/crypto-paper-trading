# Fontes de Dados e APIs

> Referencia para coleta de dados. Consultar ao adicionar exchanges ou dados novos.

---

## APIs de Exchanges (Gratuitas)

| Exchange | Endpoint | Limite | Historico | Notas |
|----------|----------|--------|-----------|-------|
| **Binance** | `GET /fapi/v1/fundingRate` | 1000/req, 500 req/5min | ~2019+ | Base obrigatoria. Funding nao e mais 8h universal desde 2023 |
| **Bybit** | `GET /v5/market/funding/history` | 200/req | ~2020+ | `startTime` requer `endTime`. Intervalo variavel por simbolo |
| **OKX** | `/api/v5/public/funding-rate-history` | 100/req | ~2020+ | Intervalos 8h ou 4h por par |
| **dYdX v4** | Get Historical Funding (indexer API) | — | ~2023+ | Funding horario. V3 descontinuada |
| **Hyperliquid** | POST `fundingHistory` + S3 archive | — | ~2023+ | "No guarantee of timely updates" — admitem gaps |

## Provedores Pagos

| Provedor | Cobertura | Custo | Destaque |
|----------|-----------|-------|----------|
| **Tardis.dev** | 35+ exchanges, desde 2019 | ~$100-300/mes | Tick-level normalizado. Padrao de facto para research |
| **CoinGlass** | Multi-exchange | Freemium | OI-weighted funding rates. Cross-check |
| **CoinAPI** | Binance, OKX, Bybit, Bitget | ~$79+/mes | Flat Files para ETL. Normalizacao cross-exchange |
| **Kaiko** | 100+ exchanges, 200k+ contratos | $9,500-55,000/ano | Padrao institucional. Parceria LSEG |

## Documentacao Tecnica de Exchanges

- **Binance**: `F = P + clamp(0.01% - P, +/-0.05%)`. Interest rate fixo: 0.01%/8h. Intervals: 00:00, 08:00, 16:00 UTC
- **BitMEX**: Funding = Interest Rate Component + Premium/Discount Component
- **Bybit**: TWAP minute-by-minute do premium index

## Intervalos de Funding por Exchange

| Exchange | Intervalo | Equivalencia |
|----------|-----------|-------------|
| Binance | 8h (00:00, 08:00, 16:00 UTC) | Alguns pares tem 4h |
| Bybit | 8h | TWAP minute-by-minute |
| OKX | 8h | Mudou de variavel em Jan 2024 |
| Hyperliquid | 1h | 0.01%/h = 0.03%/8h equivalente |
| dYdX | 1h | DEX, settlement on-chain |

**ARMADILHA**: 0.01% em exchange 8h = 10.95% aa. 0.01% em exchange 1h = 87.6% aa. NORMALIZAR antes de comparar.

## Dados Gratuitos

- **Binance Data Vision**: data.binance.vision — OHLCV desde 2017, gratuito
- **CryptoDataDownload**: cryptodatadownload.com
- **alternative.me**: Fear & Greed Index (`api.alternative.me/fng/`)
