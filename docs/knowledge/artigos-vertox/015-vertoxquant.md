# VertoxQuant

> Pricing Options on Altcoins using BTC and ETH.
**URL:** https://www.vertoxquant.com/p/pricing-illiquid-options-using-proxy
**Nota:** Artigo com paywall - conteúdo parcial

---

In the previous article, we talked about how you can figure out the fair price of a polymarket bet using the options market of the underlying.

How to Price Touch-Style Options (Polymarket)
VERTOX
·
16 DE NOVEMBRO DE 2025

“What price will Bitcoin hit in November?”. This is one of many markets on Polymarket that allows you to speculate on the price of Cryptocurrencies.

Read full story

This is fine if you are working with BTC and ETH, but what do you do if you want to bet on a coin like SOL, XRP, or DOGE that doesn’t have a liquid options market?

We will be covering the answer to that in this article!

Table of Contents

Project Structure

Pricing Model

Data Gathering

SSVI Model

Cross-Asset Statistics

Baseline Proxy Model

Behaviour-Aware Proxy Model

Conclusion

Project Structure

The project structure looks as follows:

project/
│
├── notebook.ipynb
│
└── src/
    ├── analysis/
    │   └── factors.py
    │
    ├── data/
    │   └── loader.py
    │
    ├── models/
    │   └── proxy.py
    │
    └── numerics/
        ├── black.py
        └── ssvi.py
factors.py:

Contains all the functions necessary to compute betas of returns and factors of a volatility surface.

loader.py:

Handles data gathering from Deribit using CCXT.

proxy.py:

Contains the proxy model that allows us to map the IV surface of a liquid coin like BTC or ETH, to an altcoin.

black.py:

Contains the pricing model and functions for computing Greeks and implied volatilities.

ssvi.py:

Contains all the functions for representing and fitting volatility surfaces using SSVI as well as checking for no-arbitrage conditions.

notebook.ipynb

Wraps everything together in a structured way, following this article.

Pricing Model