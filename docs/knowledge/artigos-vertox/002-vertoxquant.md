# VertoxQuant

> What every research paper ignores
**URL:** https://www.vertoxquant.com/p/discrete-market-making
**Nota:** Artigo com paywall - conteúdo parcial

---

Classical market-making theory is often developed in a continuous-price setting where quotes can be adjusted by arbitrarily small amounts and fill intensities depend smoothly on the distance to a reference price.

Real limit order books are discrete: all displayed prices lie on a grid with tick size > 0, and, especially in “large-tick” assets, the bid–ask spread is frequently pinned at one tick.
This changes the economics of liquidity provision (where rents come from, how they are competed away, and how adverse selection manifests) and it changes the mathematics of optimal control (from smooth optimization to discrete choice, hybrid/impulse control, and state augmentation for queue position).

This article provides a rigorous, pedagogical deep-dive that starts from the canonical continuous-time market-making framework and then explains, in detail, why and how tick size and queue priority alter optimal quoting.
We then survey the spectrum of modeling and control approaches used in practice: exact discrete-state dynamic programming, queue-aware Markov models, continuous relaxations with discrete corrections, and pragmatic heuristics used when the exact discrete control problem is computationally intractable.

VertoxQuant is a reader-supported publication. To receive new posts and support my work, consider becoming a free or paid subscriber.

Subscribe

I write about quantitative trading the way it’s actually practiced:
Robust models and portfolios, combining signals and strategies, understanding the assumptions behind your models.

More broadly, I write about:

Statistical and cross-sectional arbitrage

Managing multiple strategies and signals

Risk and capital allocation

Research tooling and methodology

In-depth model assumptions and derivations

If this way of thinking resonates, you’ll probably like what I publish.