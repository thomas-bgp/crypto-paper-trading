# VertoxQuant

> Reducing an infeasible problem to something you can actually run
**URL:** https://www.vertoxquant.com/p/fitting-regime-switching-models
**Nota:** Artigo com paywall - conteúdo parcial

---

Introduction

In the previous article, we built all of the theory behind HMMs for microstructural data. This article is where it finally gets real, and we fit those models to actual market data efficiently.

We are going to take something that is completely infeasible in practice, on the order of 10^430 computations. To put that in perspective, if you took every atom in our universe, put a new universe inside each atom, and repeated that process five times, you would reach a number around that size. Then we collapse it down to roughly 9000 computations that you can actually run.

By the end of this article, you will be able to take real market data, fit an HMM to it, and identify which market regime you are most likely in