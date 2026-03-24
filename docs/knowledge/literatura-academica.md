# Literatura: Estrategias de Funding Rate em Criptomoedas

> Documento compilado pelo Comite Quant em 2026-03-15.
> Revisado por: OPUS (CIO), SONNET-A (Alpha), SONNET-R (Risk), SONNET-M (Microstructure), SONNET-D (Data/Infra).

---

## Sumario Executivo do Comite

**Consenso:** O alpha de funding rate existe mas esta em compressao estrutural (ETFs spot + Ethena). Retornos pos-2024 sao materialmente menores. Estrategia requer regime filter, tem capacity limitada ($50-200M), e nenhum framework faz backtest correto out-of-the-box.

**Decisao:** Aprovacao para Fase 0 (pesquisa e validacao). Sem alocacao de capital ate validacao empirica com dados 2024-2025.

**Confianca do comite:** 5/10 (media ponderada).

---

## PARTE I — PAPERS ACADEMICOS E WORKING PAPERS

---

### 1. Precificacao e Mecanismo de Perpetual Futures

**[P1] "Fundamentals of Perpetual Futures"**
- **Autores:** Songrun He, Asaf Manela, Omri Ross, Victor von Wachter
- **Ano:** 2022 (v6: agosto 2024)
- **Fonte:** arXiv:2212.06888
- **Citacoes:** 13
- **Resumo:** Paper fundacional. Deriva precos de no-arbitrage para perpetuais. Documenta que desvios do preco teorico sao maiores em crypto que em moedas tradicionais, covariam entre ativos, e diminuem ao longo do tempo. "An implied arbitrage strategy yields high Sharpe ratios."
- **Relevancia:** ALTA — referencia teorica central para funding rate strategies

**[P2] "Perpetual Futures Pricing"**
- **Autores:** Damien Ackerer, Julien Hugonnier, Urban Jermann (Wharton)
- **Ano:** 2023 (revisado set/2024; publicado Mathematical Finance, Wiley, 2025)
- **Fonte:** arXiv:2310.11771 | NBER Working Paper #32936
- **URL:** https://finance.wharton.upenn.edu/~jermann/AHJ-main-10.pdf
- **Resumo:** Expressoes explicitas de no-arbitrage para contratos perpetuos (linear, inverso, quanto). O preco do futuro e a expectativa risk-neutral do spot amostrado em tempo aleatorio refletindo a intensidade do price anchoring. Demonstra condicoes de replicacao perfeita. Sob stress de liquidez, ancoragem quebra temporariamente — criando basis risk.
- **Relevancia:** ALTA — formalizacao do mecanismo que governa premium/discount

**[P3] "A Primer on Perpetuals"**
- **Autores:** Guillermo Angeris, Tarun Chitra, Alex Evans, Matthew Lorig
- **Ano:** 2022
- **Fonte:** arXiv:2209.03307
- **Resumo:** Analisa perpetuais com funding rate payments e com fator de desconto variavel. Formulas model-free e estrategias de replicacao, incluindo casos com saltos (jumps). Conecta perpetuais a variance swaps e leveraged ETFs.
- **Relevancia:** ALTA — explica limites teoricos do alpha de arbitragem pura

**[P4] "Designing Funding Rates for Perpetual Futures in Cryptocurrency Markets"**
- **Autores:** Jaehyun Kim, Hyungbin Park
- **Ano:** 2025
- **Fonte:** arXiv:2506.08573
- **Resumo:** Usa BSDEs path-dependent de horizonte infinito para projetar funding rates otimos. Demonstra existencia e unicidade de solucao. Compara mecanismos alternativos entre exchanges.
- **Relevancia:** MEDIA — mais relevante para design de protocolos que para trading

---

### 2. Evidencia Empirica de Alpha

**[P5] "The Recurrent Reinforcement Learning Crypto Agent"**
- **Autores:** Gabriel Borrageiro, Nick Firoozye, Paolo Barucca
- **Ano:** 2022
- **Fonte:** arXiv:2201.04699 | IEEE Access, vol. 10, pp. 38590-38599
- **Resumo:** Agente RL operando XBTUSD perpetual no BitMEX intraday. Retorno total 350% em ~5 anos (liquido de custos). **71% do retorno total atribuido a funding profit.** Information Ratio anualizado: 1.46.
- **Relevancia:** ALTISSIMA — prova empirica de que funding e a componente dominante do alpha
- **CAVEAT (SONNET-R):** Periodo 2019-2024 inclui bull market extremo. IR 1.46 e upper bound, nao estimativa conservadora.

**[P6] "Crypto Carry: Market Segmentation and Price Distortions" (BIS Working Paper #1087)**
- **Autores:** Maik Schmeling, Andreas Schrimpf, Karamfil Todorov (BIS)
- **Ano:** 2023/2025
- **URL:** https://www.bis.org/publ/work1087.pdf | https://cepr.org/voxeu/columns/crypto-carry-market-segmentation-and-price-distortions-digital-asset-markets
- **Resumo:** Paper definitivo do BIS. Carry medio anualizado: 7-8%, picos >40%. Sharpe do crypto carry: 6.45 (amostra completa) → 4.06 (2024) → **NEGATIVO em 2025**. ETFs spot (jan/2024) reduziram carry em 3-5pp. Aumento de 10% no carry prediz 22% mais liquidacoes short.
- **Relevancia:** ALTISSIMA — documenta compressao estrutural do alpha
- **IMPACTO NA DECISAO:** Este paper e a razao principal para o comite nao aprovar alocacao imediata.

**[P7] "The Two-Tiered Structure of Cryptocurrency Funding Rate Markets"**
- **Autores:** Peter Zhivkov
- **Ano:** 2026
- **Fonte:** Semantic Scholar / MDPI Mathematics, vol. 14, n. 2
- **URL:** https://www.mdpi.com/2227-7390/14/2/346
- **Resumo:** 35.7 milhoes de observacoes, 26 exchanges, 749 simbolos. CEX domina price discovery (61% maior integracao que DEX). Fluxo informacional CEX→DEX exclusivamente. **Apenas 40% das oportunidades de arbitragem geram retorno positivo apos custos.**
- **Relevancia:** ALTISSIMA — mata a tese de cross-exchange arb em escala

**[P8] "Leveraged BTC Funding Carry Algorithm: A Delta-Neutral Strategy"**
- **Autores:** Skyler Chan
- **Ano:** Junho 2025
- **Fonte:** SSRN: 5292305
- **URL:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5292305
- **Resumo:** Backtest completo de delta-neutral com 3x leverage. Retorno anualizado 16.0%, Sharpe 6.1, max drawdown <2%.
- **Relevancia:** ALTA
- **CAVEAT (SONNET-M):** Sharpe 6.1 provavelmente superestimado — sem market impact real, sem custos de borrowing para leverage. Max drawdown de 2% ignora risco de liquidacao em flash crashes.

**[P9] "The Risk and Return of Cryptocurrency Carry Trade"**
- **Autores:** Zhenzhen Fan, Feng Jiao, Lei Lu, Xin Tong
- **Ano:** Novembro 2024
- **Fonte:** SSRN: 4666425
- **Resumo:** Carry cross-sectional (long high-funding, short low-funding) gera 43.4% anualizado com Sharpe 0.74. Retornos parcialmente sao premio por risco de volatilidade de equities — nao alpha puro.
- **Relevancia:** MEDIA — confirma que carry tem componente de risco sistematico

**[P10] "Exploring Risk and Return Profiles of Funding Rate Arbitrage on CEX and DEX"**
- **Ano:** 2025
- **Fonte:** ScienceDirect / International Review of Financial Analysis
- **URL:** https://www.sciencedirect.com/science/article/pii/S2096720925000818
- **Resumo:** 60 cenarios em BTC, ETH, XRP, BNB, SOL (Binance, BitMEX, ApolloX, Drift). Retornos ate 115.9% em 6 meses, perda maxima 1.92%.
- **Relevancia:** ALTA — mas periodo historico favoravel. Nao estressou funding negativo, ADL ou falencia de exchange.

---

### 3. Funding Rate como Preditor / Sinal

**[P11] "BitMEX Funding Correlation with Bitcoin Exchange Rate"**
- **Autores:** Sai Srikar Nimmagadda, Pawan Sasanka Ammanamanchi
- **Ano:** 2019
- **Fonte:** arXiv:1912.03270
- **Resumo:** Primeiro paper sistematico sobre relacao funding-preco. Causalidade Granger bidirecional confirmada. Modelos GARCH para previsao.
- **Relevancia:** ALTA — prova poder preditivo do funding rate

**[P12] "Predictability of Funding Rates"**
- **Autores:** Emre Inan
- **Ano:** 2025
- **Fonte:** SSRN: 5576424
- **Resumo:** Modelos double autoregressive superam no-change models para previsao de funding rate. Previsibilidade e time-varying.
- **Relevancia:** ALTA — input para regime filter

**[P13] "Who Sets the Range? Funding Mechanics and 4h Context in Crypto Markets"**
- **Autores:** Habib Badawi, Mohamed Hani, T. Taufikin
- **Ano:** Janeiro 2026
- **Fonte:** arXiv:2601.06084
- **Resumo:** Alinhamento funding + contexto 4H → expansao de preco. Divergencia → range-bound. 32 paginas, 14 tabelas.
- **Relevancia:** MEDIA — interessante como framework, evidencia estatistica nao confirmada rigorosamente

**[P14] "HODL Strategy or Fantasy? 480M Crypto Market Simulations"**
- **Autores:** Weikang Zhang, Alison Watts
- **Ano:** 2025
- **Fonte:** arXiv:2512.02029
- **Resumo:** 480M simulacoes Monte Carlo, 378 ativos. Median excess return 2-3 anos: -28.4%. EMA 24 semanas do Fear & Greed Index domina retornos futuros. Um desvio padrao de sentiment shock reduz retornos em 15-22pp.
- **Relevancia:** ALTA — fundamenta o uso de regime filter baseado em sentimento

**[P15] "Crypto Pricing with Hidden Factors"**
- **Autores:** Matthew Brigida
- **Ano:** 2026
- **Fonte:** arXiv:2601.07664
- **Resumo:** Estima premios de risco via Giglio-Xiu three-pass. Identifica fatores crypto-especificos + equity-industry. Inclui Fear & Greed Index como variavel de estado.
- **Relevancia:** ALTA — funding rate captura leverage/sentiment implicitamente

---

### 4. Microestrutura e Execucao

**[P16] "Microstructure and Market Dynamics in Crypto Markets"**
- **Autores:** David Easley, Maureen O'Hara, Songshan Yang, Zhibai Zhang (Cornell)
- **Ano:** Abril 2024
- **Fonte:** SSRN: 4814346
- **Resumo:** VPIN (Volume-Synchronized Probability of Informed Trading) tem poder preditivo para dinamicas de preco (AUC >0.55). Resultados estaveis durante crypto winter.
- **Relevancia:** ALTA — VPIN como sinal complementar para timing de entrada

**[P17] "Perpetual Futures Contracts and Cryptocurrency Market Quality"**
- **Autores:** Qihong Ruan, Artem Streltsov (Cornell)
- **Ano:** 2022/2025
- **Fonte:** SSRN: 4218907
- **Resumo:** Padrao **U-shaped** em ciclos de 8h: volume e spreads altos nas bordas, baixos no meio. Perpetuais aumentam volume spot mas alargam spreads por informed trading durante settlement.
- **Relevancia:** ALTA — **insight mais acionavel para execucao**. Entrar nas horas 3-5 do ciclo, NUNCA nos 30-60 min antes do settlement.

**[P18] "Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books"**
- **Autores:** Haochuan (Kevin) Wang (U. Chicago)
- **Ano:** Maio 2025
- **Fonte:** arXiv:2506.05764
- **Resumo:** LOB de BTC/USDT a 100ms no Bybit. Order imbalance multi-nivel e metricas de supply-demand sao os features mais preditivos (acuracia 0.53-0.73 em 1s).
- **Relevancia:** MEDIA — complementa sinais de funding com order flow

**[P19] "Explainable Patterns in Cryptocurrency Microstructure"**
- **Autores:** Bartosz Bieganowski, Robert Slepaczuk
- **Ano:** 2026
- **Fonte:** arXiv:2602.00776
- **Resumo:** Padroes cross-asset estaveis em LOB (BTC, LTC, ETC, ENJ, ROSE) no Binance Futures. CatBoost + SHAP confirma order flow imbalance como preditor robusto.
- **Relevancia:** ALTA — valida robustez de sinais de microestrutura

**[P20] "Slippage-at-Risk (SaR)"**
- **Autores:** Otar Sepper
- **Ano:** 2026
- **Fonte:** arXiv:2603.09164
- **Resumo:** Framework de risco de liquidez para exchanges de perpetuais. Metricas para stress sistemico via order book data.
- **Relevancia:** MEDIA — SaR quantifica slippage como metrica de risco

**[P21] "High-frequency Dynamics of Bitcoin Futures"**
- **Ano:** 2025
- **Fonte:** ScienceDirect / Journal of Financial Markets
- **URL:** https://www.sciencedirect.com/science/article/pii/S2214845025001188
- **Resumo:** Testa MDH vs ITIH em BTC/ETH perps na Binance (2020-2024). ITIH implica market impact proporcional a raiz quadrada do volume — dobrar ordem = +41% de custo, nao +100%.
- **Relevancia:** MEDIA-ALTA — critico para sizing de ordens

**[P22] "Optimal Liquidation of Perpetual Contracts"**
- **Autores:** Ryan Donnelly, Junhan Lin, Matthew Lorig
- **Ano:** 2026
- **Fonte:** arXiv:2601.10812
- **Resumo:** Solucao otima de controle estocastico para saida de posicao em perpetual. Funding rate entra diretamente na funcao objetivo. Solucao closed-form para payoff linear.
- **Relevancia:** ALTA — otimizacao de timing de saida considerando funding

---

### 5. Risco, Tail Events e Mecanismos de Falha

**[P23] "Autodeleveraging: Impossibilities and Optimization"**
- **Autores:** Tarun Chitra
- **Ano:** Dezembro 2025
- **Fonte:** arXiv:2512.01112
- **Resumo:** Primeiro modelo formal de ADL. Prova **trilemma fundamental**: nenhuma politica de ADL garante simultaneamente solvencia, receita e fairness. Moral hazard assintotico torna socializacao "zero-loss" impossivel.
- **Relevancia:** ALTA CRITICA — ADL pode fechar o lado short (hedge) de estrategia delta-neutro

**[P24] "Autodeleveraging as Online Learning"**
- **Autores:** Tarun Chitra, Nagu Thogiti, et al.
- **Ano:** Fevereiro 2026
- **Fonte:** arXiv:2602.15182
- **Resumo:** Formaliza ADL como online learning. Documenta evento Hyperliquid de outubro 2025. ~$51.7M em overliquidation evitavel. ADL nao reconhece hedges.
- **Relevancia:** ALTA — evidencia empirica de ADL destruindo hedges

**[P25] "Carry Trades and Currency Crashes"**
- **Autores:** Brunnermeier, Nagel, Pedersen
- **Ano:** 2008
- **Fonte:** NBER Macroeconomics Annual, Vol. 23, pp. 313-347
- **URL:** https://www.nber.org/papers/w14473
- **Resumo:** Paper seminal. Carry trades exibem **negative skewness** sistematica — "picking up nickels in front of a steamroller". Unwind em periodos de reducao de risk appetite. Crashes sem news fundamental.
- **Relevancia:** ALTA CRITICA — paralelo estrutural direto com funding rate strategies

**[P26] "BIS Bulletin No. 90 — Market Turbulence and Carry Trade Unwind of August 2024"**
- **Ano:** 2024
- **URL:** https://www.bis.org/publ/bisbull90.pdf
- **Resumo:** Unwind do yen carry trade impactou crypto diretamente. Bitcoin -30% do pico. Open interest em futures proximo de maximas historicas.
- **Relevancia:** ALTA — evento recente, documentado pelo BIS

**[P27] "Anatomy of a Stablecoin's Failure: The Terra-Luna Case"**
- **Ano:** 2022
- **Fonte:** Finance Research Letters | arXiv:2207.13914
- **Resumo:** Death spiral de $50B. Spillover para liquidacoes em cascata em BTC e ETH perps. Semanas de funding negativo.
- **Relevancia:** ALTA — evento que destruiu estrategias long-spot/short-perp

**[P28] "FTX Collapse and Systemic Risk Spillovers"**
- **Ano:** 2023
- **Fonte:** ScienceDirect
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S1544612323004713
- **Resumo:** Quantifica spillover sistemico. Fundos com pernas em FTX perderam 100% do capital naquela venue + delta exposure residual.
- **Relevancia:** ALTA CRITICA — evento paradigmatico de counterparty risk

**[P29] "A Retrospective on the Crypto Runs of 2022"**
- **Ano:** 2023
- **Fonte:** Federal Reserve Bank of Chicago
- **URL:** https://www.chicagofed.org/publications/chicago-fed-letter/2023/479
- **Resumo:** "Corridas" em crypto seguem dinamicas classicas de bank runs.
- **Relevancia:** MEDIA-ALTA

**[P30] "Modelling Extreme Tail Risk of Bitcoin Returns"**
- **Ano:** 2024
- **URL:** https://www.intechopen.com/chapters/1173468
- **Resumo:** GPD mostra VaR e Expected Shortfall substancialmente maiores que modelos gaussianos. Fat tails + volatility clustering severo.
- **Relevancia:** ALTA — backtests com VaR gaussiano subestimam tail

**[P31] "Regime- and Tail-Dependent CVaR Portfolio Strategies in Crypto"**
- **Ano:** 2025
- **URL:** https://www.mdpi.com/2227-7072/14/3/53
- **Resumo:** Performance de CVaR e fortemente regime-dependente. Em stress, correlacoes mudam e premissas de "market neutral" quebram.
- **Relevancia:** ALTA

**[P32] "Assessing Macrofinancial Risks from Crypto Assets" (IMF WP 2023/214)**
- **URL:** https://www.imf.org/-/media/Files/Publications/WP/2023/English/wpiea2023214-print-pdf.ashx
- **Resumo:** Queda do BTC pode acionar >$10B em liquidacoes. Contagion via liquidity shortages.
- **Relevancia:** MEDIA-ALTA

**[P33] "Crowded Spaces and Anomalies"**
- **Ano:** 2025
- **URL:** https://www.sciencedirect.com/science/article/pii/S0378426625001992
- **Resumo:** Capacidade de estrategias de trading e compartilhada entre todos os participantes. Multiplos pequenos traders = impacto de um grande trader. Estrategia popular demais se auto-destroi.
- **Relevancia:** ALTA

---

### 6. Qualidade de Dados e Metodologia

**[P34] "AutoQuant: Auditable Expert-System Framework for Crypto Perpetual Futures"**
- **Autores:** Kaihong Deng
- **Ano:** 2025
- **Fonte:** arXiv:2512.22476
- **Resumo:** Prova que backtests sem funding, fees e slippage "materially overestimate annualized returns". Bayesian optimization, CSCV/PBO diagnostics para overfitting.
- **Relevancia:** ALTISSIMA (metodologica) — qualquer backtest sem estes custos e invalido

**[P35] "Aggregate Confusion in Crypto Market Data"**
- **Autores:** Gustavo Schwenkler, Aakash Shah, Darren Yang
- **Ano:** Dezembro 2023
- **Fonte:** SSRN: 4673905
- **Resumo:** Primeiro audit sistematico de qualidade de dados crypto entre provedores. Mislabeling generalizado, instabilidade de identificadores, discrepancias cross-provider.
- **Relevancia:** ALTA — framework para avaliar qualidade de dados

**[P36] "Reconciling Open Interest with Traded Volume in Perpetual Swaps"**
- **Autores:** Ioannis Giagkiozis, Emilio Said
- **Ano:** 2023 (Ledger, Volume 9, 2024)
- **Fonte:** arXiv:2310.14973
- **Resumo:** "Open interest in Bitcoin perpetual swaps is systematically misquoted" por grandes exchanges. Dados de OI corrompidos invalidam sinais.
- **Relevancia:** ALTA — dados de OI como input para estrategias podem ser espurios

---

### 7. Outros Papers Relevantes

**[P37] "The Impact of Derivatives on Spot Markets"**
- **Autores:** Patrick Augustin, A. Rubtsov, Donghwa Shin
- **Ano:** 2023 (26 citacoes)
- **Resumo:** Introducao de futuros melhora eficiencia de mercado e reduz alpha de arbitragem ao longo do tempo.

**[P38] "Perpetual Demand Lending Pools"**
- **Autores:** Tarun Chitra, Theo Diamandis, et al.
- **Ano:** 2025
- **Fonte:** arXiv:2502.06028
- **Resumo:** PDLPs (Jupiter, Hyperliquid, GMX) reduzem borrowing costs de traders de perpetuais. Impacto direto em funding rate dynamics.

**[P39] "Agent-Based Simulation of a Perpetual Futures Market"**
- **Autores:** Ramshreyas Rao
- **Ano:** 2025
- **Fonte:** arXiv:2501.09404
- **Resumo:** Reproduz pegging de preco perp→spot via agentes heterogeneos.

**[P40] "A Low-Volatility Strategy Based on Hedging a Quanto Perpetual Swap"**
- **Autores:** Daniel Atzberger et al.
- **Ano:** 2024
- **Resumo:** Quanto perpetual hedge com retornos positivos em alta e volatilidade reduzida.

---

## PARTE II — REPOSITORIOS GITHUB E IMPLEMENTACOES

**[G1] matthias-wyss/crypto-carry-trade-strategies**
- **Stars:** 8 | **Ano:** Atualizado maio 2025
- **URL:** https://github.com/matthias-wyss/crypto-carry-trade-strategies
- **Descricao:** 3 estrategias delta-neutras (Classical Carry, Staking-Enhanced, Pendle-Based). Dados 2019-2024. Funding positivo >85% do tempo em BTC/ETH. Staking-enhanced +3.9%/ano vs pure funding.
- **Relevancia:** ALTA — implementacao com dados reais de multiplos regimes

**[G2] 50shadesofgwei/funding-rate-arbitrage**
- **Stars:** 168 | **Forks:** 45 | **Licenca:** MIT
- **URL:** https://github.com/50shadesofgwei/funding-rate-arbitrage
- **Descricao:** Delta-neutral CEX/DEX (Synthetix v3, GMX, Binance, HMX). Python. Conceito de "funding velocity" — quanto tempo funding favoravel persiste.
- **Relevancia:** ALTA — template funcional com conceito unico (velocity)

**[G3] PietroC21/Crypto-PerpetualFutures**
- **Stars:** 1 | **Ano:** Atualizado marco 2026
- **URL:** https://github.com/PietroC21/Crypto-PerpetualFutures
- **Descricao:** Z-score signal por ativo (desvio vs historico proprio). Dados Binance + Bybit OI + variaveis macro (VIX, SPY, risk-free).
- **Relevancia:** ALTA — z-score cross-asset e abordagem mais sofisticada que threshold fixo

**[G4] nkaz001/hftbacktest**
- **Stars:** ~3,800 | **Ano:** Ativo
- **URL:** https://github.com/nkaz001/hftbacktest
- **Descricao:** Full order book reconstruction (L2/L3), latency modeling, queue position. Python (Numba JIT) + Rust. Binance/Bybit.
- **Relevancia:** MEDIA para funding carry (overkill), ALTA para HFT/market making

**[G5] ccxt/ccxt**
- **Stars:** 30,000+ | **Ano:** Ativo
- **URL:** https://github.com/ccxt/ccxt
- **Descricao:** Abstracacao unificada para 100+ exchanges. `fetchFundingRateHistory()` disponivel. Bugs conhecidos em Bybit (#17854) e Hyperliquid (#24822).
- **Relevancia:** ALTA como utilitario de coleta

**[G6] Aidasvenc/funding-rate-trading**
- **URL:** https://github.com/Aidasvenc/funding-rate-trading
- **Descricao:** Carry trade via perpetual futures. Notebooks basicos.
- **Relevancia:** MEDIA — sem metricas documentadas

**[G7] QuNoSleep/Binance-Crypto-Backtester**
- **URL:** https://github.com/QuNoSleep/Binance-Crypto-Backtester
- **Descricao:** Engine de backtesting Binance com funding rate + slippage.
- **Relevancia:** MEDIA

---

## PARTE III — FONTES DE DADOS E APIs

### APIs de Exchanges (Gratuitas)

| Exchange | Endpoint | Limite | Historico | Notas |
|----------|----------|--------|-----------|-------|
| **Binance** | `GET /fapi/v1/fundingRate` | 1000/req, 500 req/5min | ~2019+ | Base obrigatoria. Funding nao e mais 8h universal desde 2023 |
| **Bybit** | `GET /v5/market/funding/history` | 200/req | ~2020+ | `startTime` requer `endTime`. Intervalo variavel por simbolo |
| **OKX** | `/api/v5/public/funding-rate-history` | 100/req | ~2020+ | Intervalos 8h ou 4h por par |
| **dYdX v4** | Get Historical Funding (indexer API) | — | ~2023+ | Funding horario. V3 descontinuada |
| **Hyperliquid** | POST `fundingHistory` + S3 archive | — | ~2023+ | "No guarantee of timely updates" — admitem gaps |

### Provedores Pagos

| Provedor | Cobertura | Custo | Destaque |
|----------|-----------|-------|----------|
| **Tardis.dev** | 35+ exchanges, desde 2019 | ~$100-300/mes | Tick-level normalizado. Padrao de facto para research |
| **CoinGlass** | Multi-exchange | Freemium | OI-weighted funding rates. Cross-check |
| **CoinAPI** | Binance, OKX, Bybit, Bitget | ~$79+/mes | Flat Files para ETL. Normalizacao cross-exchange |
| **Kaiko** | 100+ exchanges, 200k+ contratos | $9,500-55,000/ano | Padrao institucional. Parceria LSEG |

### Documentacao Tecnica de Exchanges

**[D1] Binance — Introduction to Funding Rates**
- **URL:** https://www.binance.com/en/support/faq/introduction-to-binance-futures-funding-rates-360033525031
- Formula: `F = P + clamp(0.01% - P, ±0.05%)`. Interest rate fixo: 0.01%/8h. Intervals: 00:00, 08:00, 16:00 UTC.

**[D2] BitMEX — Perpetual Contract Guide**
- **URL:** https://www.bitmex.com/app/perpetualContractsGuide
- Funding = Interest Rate Component + Premium/Discount Component.

**[D3] Bybit — Introduction to Funding Rate**
- **URL:** https://www.bybit.com/en/help-center/article/Introduction-to-Funding-Rate
- TWAP minute-by-minute do premium index.

---

## PARTE IV — FRAMEWORKS DE BACKTESTING

| Framework | Funding Nativo? | Linguagem | Notas |
|-----------|----------------|-----------|-------|
| **DolphinDB** | SIM | DolphinDB script | Unico com `fundingSettlementMode`. Limitacao: nao suporta posicoes iniciais |
| **hftbacktest** | Nao | Python+Rust | Melhor para HFT. Adaptavel para funding como custo periodico |
| **vectorbt** | Nao | Python | Rapido para research. Nao event-driven |
| **Freqtrade** | Bugado | Python | Issues #9106, #7291, #10888, #12583. **NAO USAR para funding** |
| **Custom Python** | Implementar | Python | Recomendado pelo comite. Maximo controle |

---

## PARTE V — INDUSTRIA E POSTS TECNICOS

**[I1] 1Token Tech — "Crypto Fund 101: Funding Fee Arbitrage"**
- $20B+ deployados globalmente em funding arb. Sharpe/Calmar 5-10 historico.

**[I2] Amberdata — "The Ultimate Guide to Funding Rate Arbitrage"**
- **URL:** https://blog.amberdata.io/the-ultimate-guide-to-funding-rate-arbitrage-amberdata
- Guia pratico completo.

**[I3] BSIC (Bocconi) — "Perpetual Complexity: Arbitrage Mechanics"**
- **URL:** https://bsic.it/perpetual-complexity-an-introduction-to-perpetual-future-arbitrage-mechanics-part-1/
- Framework entry/exit com z-score threshold >=2. Alerta para mismatch de settlement intervals.

**[I4] CF Benchmarks — "Revisiting the Bitcoin Basis"**
- **URL:** https://www.cfbenchmarks.com/blog/revisiting-the-bitcoin-basis-how-momentum-sentiment-impact-the-structural-drivers-of-basis-activity
- Basis driven por momentum e sentimento. Contango 15-30% anualizado em bull.

**[I5] 21Shares — "Perps Explained: Hyperliquid and dYdX"**
- **URL:** https://www.21shares.com/en-us/research/perps-explained-how-hyperliquid-and-dydx-are-powering-the-next-phase-of-crypto-trading
- Hyperliquid: 70-80% market share DEX perps, $350B+/mes, latencia <1s.

**[I6] CryptoQuant — "Cross-Exchange Study of Perpetual Futures 2024"**
- **URL:** https://cryptoquant.com/insights/quicktake/675da3993def7560598bcaa4
- Binance+OKX+Bybit = ~85% do OI global. OKX mudou funding em jan/2024.

**[I7] Inside the $19B Flash Crash (Outubro 2025)**
- **URL:** https://insights4vc.substack.com/p/inside-the-19b-flash-crash
- Market depth colapsou 98%. 35,000 ADL events em minutos. ADL nao reconhece hedges.

**[I8] 7 Unwinding Carry Trades That Crashed the Markets**
- **URL:** https://www.alt21.com/hedging-insights/7-unwinding-carry-trades-that-crashed-the-markets/
- 1997, 1998 (LTCM), 2008, 2015 (CHF). Padrao: retornos estaveis por anos → liquidacao violenta em dias.

---

## PARTE VI — ANALISE DO COMITE POR DIMENSAO

### Tipos de Alpha Identificados

| Alpha | Evidencia | Sharpe Estimado | Status do Comite |
|-------|-----------|-----------------|------------------|
| **Funding Harvest Delta-Neutral** | FORTE (IEEE, BIS, repos) | 0.6-1.5 (pos-Ethena) | INVESTIGAR (Fase 0) |
| **Enhanced Carry (Staking+FR)** | FORTE | +0.3-0.5 adicional | INVESTIGAR em paralelo |
| **FR como Preditor de Reversal** | MODERADA (Granger causal) | Instavel entre regimes | BACKTEST necessario |
| **Cross-Exchange Arb** | FRACA (40% win rate) | Negativo pos-custos | DESCARTADO |

### Cenarios de Risco

| Cenario | Probabilidade (5 anos) | Impacto | Mitigavel? |
|---------|----------------------|---------|------------|
| Funding negativo prolongado | 40-60% | -15% a -40% | Regime filter (parcial) |
| Exchange insolvency | 10-20% | -30% a -100% da venue | Diversificacao (parcial) |
| ADL destroi hedge | 20-30% | -20% a -50% | Monitoramento (limitado) |
| Carry unwind macro | 25-35% | -10% a -30% | Sem hedge efetivo |
| Combinacao (pior caso) | 5-15% | -40% a -80% | Stop de sistema |

### Plano de Fases Aprovado

```
FASE 0 — Validacao (4 semanas, custo $0)
  → Coletar funding historico gratuito (Binance/Bybit/OKX)
  → Calcular IR net de custos em dados 2024-2025
  → Stress test: nov/2022, mai/2021, ago/2024
  → GO/NO-GO baseado em IR > 0.7

FASE 1 — Piloto (3 meses, ~$50k capital de risco)
  → Backtester event-driven customizado
  → Paper trading + $50-100k real em Hyperliquid
  → Validar slippage real vs simulado

FASE 2 — Escala (condicional a Fase 1)
  → Dados Tardis/Kaiko (~$500-2000/mes)
  → Multi-exchange (Binance + Bybit + Hyperliquid)
  → $500k-5M com controles de risco completos

FASE 3 — Producao (condicional a 12 meses de track record)
  → Decisao de escala final baseada em Sharpe live
```

---

## PARTE VII — REFERENCIAS CONSOLIDADAS

### Academicas (por relevancia)
1. BIS Working Paper #1087 — Crypto Carry (Schmeling, Schrimpf, Todorov, 2023/2025)
2. Borrageiro et al. — RL Crypto Agent, IEEE Access 2022 (arXiv:2201.04699)
3. Brunnermeier, Nagel, Pedersen — Carry Trades and Currency Crashes, NBER 2008
4. He, Manela, Ross, von Wachter — Fundamentals of Perpetual Futures (arXiv:2212.06888)
5. Ackerer, Hugonnier, Jermann — Perpetual Futures Pricing (Mathematical Finance, 2025)
6. Zhivkov — Two-Tiered Structure of Funding Rate Markets (MDPI, 2026)
7. Chitra — Autodeleveraging: Impossibilities (arXiv:2512.01112)
8. Deng — AutoQuant (arXiv:2512.22476)
9. Ruan, Streltsov — Perpetual Futures and Market Quality (Cornell, SSRN:4218907)
10. Fan, Jiao, Lu, Tong — Risk and Return of Crypto Carry (SSRN:4666425)
11. Zhang, Watts — HODL Strategy or Fantasy (arXiv:2512.02029)
12. Schwenkler, Shah, Yang — Aggregate Confusion in Crypto Data (SSRN:4673905)
13. Easley, O'Hara et al. — Microstructure and Market Dynamics (Cornell, SSRN:4814346)
14. BIS Bulletin #90 — Carry Trade Unwind August 2024
15. IMF WP 2023/214 — Macrofinancial Risks from Crypto Assets
16. Nimmagadda, Ammanamanchi — BitMEX Funding Correlation (arXiv:1912.03270)
17. Inan — Predictability of Funding Rates (SSRN:5576424)
18. Chan — Leveraged BTC Funding Carry (SSRN:5292305)
19. Kim, Park — Designing Funding Rates (arXiv:2506.08573)
20. Chitra et al. — Autodeleveraging as Online Learning (arXiv:2602.15182)

### Industria
21. 1Token Tech — Crypto Fund 101: Funding Fee Arbitrage
22. Amberdata — Ultimate Guide to Funding Rate Arbitrage
23. BSIC Bocconi — Perpetual Complexity: Arbitrage Mechanics
24. CoinGlass — What is Funding Rate Arbitrage
25. CF Benchmarks — Revisiting the Bitcoin Basis
26. 21Shares — Perps Explained: Hyperliquid and dYdX
27. CryptoQuant — Cross-Exchange Study 2024
28. Inside the $19B Flash Crash (Oct 2025 — insights4vc)
29. ALT21 — 7 Unwinding Carry Trades That Crashed the Markets
30. Fed Chicago — Retrospective on Crypto Runs of 2022
