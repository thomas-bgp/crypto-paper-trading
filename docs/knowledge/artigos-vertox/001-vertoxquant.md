# VertoxQuant

> Learning to Rank for Cross-Sectional Strategies
**URL:** https://www.vertoxquant.com/p/learning-to-rank
**Nota:** Artigo com paywall - conteúdo parcial

---

Introduction

Most models are trained to predict individual asset returns.

That seems reasonable. But for cross-sectional strategies, prediction accuracy is often the wrong objective.

Cross-sectional portfolios turn model outputs into trades by ranking assets at each rebalance date and going long the top names while shorting the bottom. Once trades are determined by rank, the absolute level of forecasts often becomes irrelevant: any monotone transformation of model scores produces the same portfolio.

In other words, getting the ordering right matters more than predicting returns precisely.

This creates a subtle but important mismatch. A model can achieve excellent pointwise accuracy (low RMSE) while producing a weak, or even negative, long–short spread. Conversely, a model with poor regression metrics may generate strong cross-sectional performance.

However, ranking is not always all that matters. When position sizing, risk budgeting, transaction cost models, or portfolio optimizers depend on forecast magnitudes and calibration, pointwise accuracy becomes important again.

In this article, we develop Learning to Rank (LTR) as a principled framework for cross-sectional strategies.

The argument proceeds in three steps:

We formalize the cross-sectional decision problem and show how the score-to-weight mapping determines which properties of model output actually affect PnL.

We demonstrate with numerical examples how low RMSE can coexist with negative long–short spreads, and vice versa.

We derive practical LTR objectives for statistical arbitrage, including label engineering, pair weighting, neutralization, and cost-aware evaluation.

I write about quantitative trading the way it’s actually practiced:
Robust models and portfolios, combining signals and strategies, understanding the assumptions behind your models.

More broadly, I write about:

Statistical and cross-sectional arbitrage

Managing multiple strategies and signals

Risk and capital allocation

Research tooling and methodology

In-depth model assumptions and derivations

If this way of thinking resonates, you’ll probably like what I publish.