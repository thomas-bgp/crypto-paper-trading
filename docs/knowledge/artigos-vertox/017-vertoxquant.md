# VertoxQuant

> Trading Funding Arbs with only Perps
**URL:** https://www.vertoxquant.com/p/beta-neutral-funding-arbitrage
**Nota:** Artigo com paywall - conteúdo parcial

---

Funding Rate Arbitrage is the trade I’ve always recommended the most to new traders, as it has a few nice properties:

No need for automation:
Of course, you can get fancy with good execution and really large portfolios across many exchanges, but the reality is that you can get started with just an Excel spreadsheet and a manual trade once a day.

Simple:
It’s not a complex trade. You short coins with highly positive funding rates and long coins with highly negative funding rates (A Positive funding rate means that longs pay shorts).

Money comes in around the clock:
Funding payments usually happen every 8 hours, and you typically make money on most of those. This means once you’re up and running, you’ll have a mostly consistent source of returns.

Low capital requirements:
Funding arbs can be done with both little and a lot of capital. The less capital you have, the more illiquid coins you can trade, which have much wilder funding rates.

Easy to build on top of:
Really simple models already work for funding rate arbs. That doesn’t mean you can’t improve your system, though! In fact, there are many, many areas where you can improve your system; Better forecasts of funding rates, better estimates of beta, volatility targeting, all kinds of safety features like safety scores for illiquid coins, etc.

I’ve talked about spot-perp funding arbitrage before in the following article:

Funding Arbitrage - How to Start
VERTOX
·
10 DE OUTUBRO DE 2024

There is a unique type of futures contract in the crypto world, perpetual futures.

Read full story

Now we will focus on a more sophisticated type of funding rate arbitrage. Instead of hedging your exposure entirely with spot, we are gonna hedge perps with other perps!

This has 2 major advantages:

You are able to earn funding on both legs!

Perps are easy to short, so you are able to profit from negative funding rates easily.

But it also has 1 major difficulty: Staying market neutral is more difficult.
You are no longer perfectly hedged with spot, so your P&L is gonna be a lot more volatile.

Let’s dive into how to forecast funding rates and how to hedge your market exposure properly!

Table of Contents

A Quick Recap of Funding Rates

Building a Forecasting Model

Estimating Betas

Constructing the Portfolio

Final Remarks