# Fat-Tail Risk Management for Cross-Sectional Crypto L/S
# From Arithmetic Alpha to Compoundable Returns

> **Date:** 2026-03-17
> **Status:** SPECIFICATION — ready for implementation
> **Problem:** Decile monotonicity in arithmetic returns (D1=-2.59%, D10=+0.62%/28d) does not survive geometric compounding. The short leg blows up from +50-100% upside moves in "loser" coins.
> **Context:** CatBoost Short-Only baseline (Sharpe 1.36, +933%) already uses 15% trailing stop. This document specifies the FULL risk framework.

---

## Table of Contents

1. [The Core Problem: Arithmetic vs Geometric](#1-the-core-problem)
2. [Position Sizing for Fat-Tailed Cross-Sectional Strategies](#2-position-sizing)
3. [Stop-Losses and Risk Limits](#3-stop-losses-and-risk-limits)
4. [Short-Side Risk Management](#4-short-side-risk-management)
5. [Crypto-Specific Solutions](#5-crypto-specific-solutions)
6. [Portfolio Construction That Preserves Signal](#6-portfolio-construction)
7. [Correct Backtest Methodology](#7-backtest-methodology)
8. [Concrete Implementation Specification](#8-implementation-specification)

---

## 1. The Core Problem: Arithmetic vs Geometric {#1-the-core-problem}

### The Volatility Drag Formula

```
G ≈ A - 0.5 * σ²
```

Where G = geometric mean return, A = arithmetic mean return, σ² = variance of returns.

For our short leg:
- Arithmetic mean per period: +2.37% (from baseline doc)
- But individual short positions can lose -50% to -100% when a "loser" coin rallies
- A single position losing -80% needs +400% to recover
- The variance of the short leg is MUCH higher than the mean suggests

**KEY INSIGHT:** The arithmetic-geometric gap grows with the SQUARE of volatility. In crypto, where individual coin volatility is 80-150% annualized, the drag is catastrophic for concentrated positions. A position with 100% annualized vol has a geometric drag of ~50 percentage points — meaning arithmetic +10% becomes geometric -40%.

**Reference:** Estrada (2010) "Geometric Mean Maximization: An Overlooked Portfolio Approach?" — Journal of Investing 19(4), 134-147. Shows that maximizing geometric mean (terminal wealth) produces fundamentally different portfolios than maximizing arithmetic mean (Sharpe ratio). The GMM approach naturally penalizes high-variance assets.

**Reference:** Chambers & Koulaouzidis (2020) explore how the arithmetic-geometric gap varies across asset classes and is most severe in high-volatility, fat-tailed environments — exactly crypto.

### Why This Kills the Short Leg Specifically

The return distribution of shorts is **asymmetric by construction:**
- Maximum gain on a short: +100% (coin goes to zero)
- Maximum loss on a short: **unlimited** (coin goes to infinity)
- A coin doing +200% costs you 2x your position. Five coins doing -40% each only makes you 5 * 0.4 * (position_size/5) = 0.4x.

The CONVEXITY of losses means: even if the arithmetic expected return of shorting bottom-decile coins is positive, the geometric expected return can be deeply negative because rare large losses compound away capital permanently.

---

## 2. Position Sizing for Fat-Tailed Cross-Sectional Strategies {#2-position-sizing}

### 2.1 How Production Quant Funds Handle This

**AQR (2023, "Key Design Choices in Long/Short Equity"):**
- Individual position sizes in systematic L/S are typically **10-50 basis points** (0.1-0.5% of portfolio)
- The natural diversification of holding hundreds of positions is the primary risk control
- Single-name concentration limits are strictly enforced

**The Core Principle:** Position sizes so small that even a +200% move in a single name is a manageable portfolio-level event.

**For our case with 5 shorts:** We are MASSIVELY concentrated. 5 positions = 20% each. A +100% move in one name = -20% portfolio hit. This is the root cause of the blowup.

### 2.2 Kelly Criterion Adjustments for Fat Tails

**Full Kelly** maximizes log-wealth growth but is KNOWN to be catastrophic with estimation error and fat tails.

**Practical consensus (Thorp 2008, Ziemba):**
- Use **quarter-Kelly to half-Kelly** as maximum
- Half-Kelly captures ~75% of optimal growth with ~50% less drawdown
- In fat-tailed environments, even half-Kelly is too aggressive

**For crypto cross-sectional:**
```python
# Kelly fraction for a single short position
# f* = (p * b - q) / b  where p=win_rate, q=1-p, b=avg_win/avg_loss
# But with fat tails, use fractional Kelly:
kelly_fraction = 0.25  # quarter-Kelly
max_position = kelly_fraction * edge / variance_of_position
```

**KEY INSIGHT:** With crypto-level variance (~1.5-4.0 annualized for individual alts), Kelly fractions for individual positions come out to **1-3%** of capital. This is consistent with what production funds use.

**Reference:** "A Prospect-Theory Approach to the Kelly Criterion for Fat-Tail Portfolios" (Busseti, Ryu, Boyd) — shows Kelly fractions must be reduced by 50-75% when returns follow Student-t distributions with df < 5 (which crypto empirically does).

### 2.3 Inverse-Volatility Weighting Within Each Leg

**The Method:**
```python
# For each position i in the short leg:
w_i = (1 / σ_i) / Σ(1 / σ_j)  # for all j in short leg
# Then scale to target total short exposure
w_i_final = w_i * target_short_exposure
```

**Why it works for our problem:**
- Coins with high recent volatility (the ones most likely to do +100%) get SMALLER positions
- Coins with low volatility (grinding down steadily) get LARGER positions
- This naturally protects against the exact tail events that kill the short leg
- Barroso & Santa-Clara (2015) showed volatility scaling **nearly doubled** the Sharpe ratio of momentum strategies and **virtually eliminated crashes**

**Reference:** Barroso, P. & Santa-Clara, P. (2015) "Momentum has its moments" — Journal of Financial Economics 116(1), 111-120.

**Reference:** Daniel, K. & Moskowitz, T. (2016) "Momentum crashes" — Journal of Financial Economics 122(2), 221-247. Shows dynamic volatility scaling using predicted momentum return + realized vol.

### 2.4 Equal Risk Contribution (ERC)

**The Method:** Each position contributes equally to portfolio variance.

```python
# For N positions, each position's risk contribution = σ_portfolio² / N
# Solved iteratively:
w_i * (Σ * w)_i = σ²_p / N  # for all i
```

**Comparison to inverse-vol:**
- ERC accounts for correlations between positions
- Inverse-vol is simpler and works well when correlations are moderate
- For 5-10 crypto positions with high correlation, ERC is marginally better
- **Recommendation:** Start with inverse-vol (simpler), upgrade to ERC if needed

---

## 3. Stop-Losses and Risk Limits {#3-stop-losses-and-risk-limits}

### 3.1 Individual Position Stop-Losses

**Current baseline:** 15% trailing stop. This is reasonable but needs refinement.

**Academic evidence on stops in systematic strategies:**
- Stop-losses HELP trend-following and momentum strategies (protect against reversals)
- Stop-losses HURT mean-reversion strategies (trigger at worst time)
- Our short-loser strategy is closer to momentum → stops are beneficial

**Production practice:**
- **ATR-based stops** adapt to each coin's volatility (better than fixed percentage)
- Typical: 2-3x ATR(14) from entry for shorts
- For crypto: 2x ATR(14) on daily bars is roughly 10-25% depending on the coin

**Triple Barrier Method (Lopez de Prado, "Advances in Financial Machine Learning"):**
- Profit-take barrier (short covered at target)
- Stop-loss barrier (short covered at max loss)
- Time barrier (covered at rebalance date regardless)
- This is the CORRECT framework for backtesting with stops

**RECOMMENDATION for our model:**
```python
# Per-position stop-loss:
stop_loss_pct = max(0.15, 2.0 * ATR_14_daily / entry_price)
# Minimum 15%, but wider for more volatile coins
# This prevents the stop from being too tight on volatile coins
# while capping loss on calm coins

# Trailing: YES, but only ratchet every EOD (not intraday noise)
trailing = True
trailing_ratchet = 'daily_close'  # not intraday
```

### 3.2 Portfolio-Level Drawdown Limits

**Production practice:**
- Hard drawdown limit: typically 10-15% from peak for systematic L/S
- At limit: reduce all positions by 50% (not full liquidation)
- Resume full sizing after recovery to within 5% of peak

```python
# Portfolio drawdown circuit breaker:
MAX_DRAWDOWN_SOFT = 0.10   # reduce to 50% sizing
MAX_DRAWDOWN_HARD = 0.20   # reduce to 25% sizing
MAX_DRAWDOWN_KILL = 0.30   # flatten everything, pause 1 month
```

### 3.3 Single-Name Concentration Limits

**AQR-style for systematic L/S:** 10-50bps per name in large portfolios.

**For our concentrated 5-name short portfolio with $100k capital:**
```python
MAX_SINGLE_SHORT = 0.05    # 5% of portfolio per short position
# With 5 positions: total short exposure = 25% (not 100%)
# This is the CRITICAL change: reduce total short exposure dramatically
```

**KEY INSIGHT:** The existing backtest assumes 100% short exposure (5 names at 20% each). This must drop to 20-30% total short exposure, with 4-6% per name maximum.

### 3.4 Interaction with Rebalancing Frequency

Current: 14-day rebalance. Between rebalances, positions drift.

**The problem:** A 5% position that goes against you by 50% becomes a 7.5% position (due to portfolio shrinkage + position growth). This is ANTI-risk-management.

**Solution: Continuous position capping**
```python
# Between rebalances, if any position exceeds max:
if abs(position_value / portfolio_value) > MAX_SINGLE_SHORT * 1.5:
    trim_to(MAX_SINGLE_SHORT)  # trim, don't close
```

---

## 4. Short-Side Risk Management {#4-short-side-risk-management}

### 4.1 Maximum Position Size on Shorts

**Industry standard for systematic short-selling:**
- Large quant funds: 10-50bps per name (AQR, Two Sigma, DE Shaw)
- Smaller funds ($10M-$100M AUM): 1-2% per name maximum
- Crypto-specific: even tighter due to higher volatility and thinner liquidity

**For $100k crypto portfolio:**
```python
MAX_SHORT_PER_NAME = 0.04      # 4% of portfolio ($4,000)
MAX_TOTAL_SHORT_EXPOSURE = 0.25 # 25% of portfolio ($25,000)
N_SHORT_POSITIONS = 6-8         # more diversification, smaller each
```

### 4.2 Short Squeeze Protection

**The mechanism:** "Loser" coins in crypto can squeeze +50-200% in days due to:
- Low float + high short interest on perps
- Narrative shifts (partnership announcements, listings)
- Coordinated buying (social media, whale accumulation)

**Protection layers:**
1. **Position size caps** (primary defense — already covered)
2. **Trailing stops** (secondary — already covered)
3. **Open Interest monitoring:** If OI on a coin spikes >2x in 48h, reduce position by 50%
4. **Funding rate signal:** Extremely negative funding = crowded short. Reduce position
5. **Diversification:** 6-8 names instead of 5 reduces single-name squeeze impact

```python
# Squeeze detection heuristic:
if funding_rate < -0.05:  # very negative = crowded short
    reduce_position(0.5)  # halve exposure
if oi_change_48h > 2.0:   # OI doubled
    reduce_position(0.5)
```

### 4.3 Funding Rate Budgeting

On perpetual futures, the short side RECEIVES funding when funding is positive (normal contango) and PAYS when negative (backwardation/squeeze).

**Current crypto market (March 2026):**
- Median funding: +0.01% per 8h ≈ +10.95% annualized (shorts RECEIVE this)
- But during squeezes: funding can hit -0.5% per 8h = shorts PAY 547% annualized
- Budget for negative funding events in the backtest

```python
# Funding rate cost model (already using Binance real data — good)
# But add a funding rate cap for risk:
MAX_NEGATIVE_FUNDING_TOLERANCE = -0.10  # per 8h
# If funding more negative than this, close the position
# (you're paying >365% annualized to hold the short)
```

### 4.4 The Asymmetric Pain of Shorts

**Mathematical reality:**
- Long $100: max loss = $100 (coin goes to 0)
- Short $100: max loss = UNLIMITED (coin goes to infinity)
- In practice with stops at 15-20%: max loss per position = $15-20

**This means the short leg MUST be:**
1. More diversified than the long leg
2. Smaller per-position than the long leg
3. More frequently monitored
4. Subject to tighter stops

---

## 5. Crypto-Specific Solutions {#5-crypto-specific-solutions}

### 5.1 Perpetual Futures vs Spot Shorting

**Perpetual futures are the ONLY viable instrument for crypto shorting at our scale.**

Advantages:
- Available for top 40+ coins on Binance
- No borrow needed (vs CEX margin shorting with limited availability)
- Funding rate usually positive (shorts get paid)
- Leverage available (but we should use 1-2x only)

Risks:
- **Auto-Deleveraging (ADL):** When the insurance fund is insufficient, exchanges forcibly close profitable positions. This is a TAIL RISK that cannot be hedged.
- **Exchange risk:** Binance could freeze withdrawals, change margin requirements, etc.
- **Liquidation cascades:** In extreme moves, your position can be liquidated even before your stop triggers if using leverage

**Reference:** Bybit ADL mechanism — positions are ranked by profit and leverage; most profitable, most leveraged positions are closed first. October 2025 ADL event showed this is a real, not theoretical, risk.

### 5.2 ADL Risk Mitigation

```python
# ADL protection:
MAX_LEVERAGE = 2.0          # never exceed 2x (most ADL hits >5x)
MARGIN_BUFFER = 0.50        # maintain 50% excess margin over requirement
# Use isolated margin per position (not cross-margin)
# This prevents one bad position from cascading to others
MARGIN_MODE = 'isolated'    # NOT cross
```

### 5.3 Exchange Diversification

For production (not backtest, but note for live):
- Split across Binance + Bybit + OKX
- No more than 40% of capital on any single exchange
- This protects against exchange-specific ADL events and operational risk

### 5.4 Stablecoin Considerations

- Collateral in USDC (not USDT) for lower depegging risk
- Alternative: collateral in BTC for natural hedge against crypto-wide squeezes
- Budget 0.5-1% annual for stablecoin risk

---

## 6. Portfolio Construction That Preserves Signal {#6-portfolio-construction}

### 6.1 The Critical Distinction: Cap POSITION SIZE vs Cap RETURNS

**Cap returns (winsorization):** You clip returns at e.g. +/-50%. This changes the BACKTEST but not reality. In live trading, the coin still moves +200% and you still lose. Winsorizing returns is **statistical fraud in backtesting.**

**Cap position sizes:** You limit each position to e.g. 4% of portfolio. The coin can still move +200%, but your portfolio loss is capped at 4% * 200% = 8% maximum per name (with no stop) or 4% * 15% = 0.6% (with 15% stop). This changes REALITY, not just statistics.

**KEY INSIGHT:** NEVER winsorize returns in backtesting. ALWAYS cap position sizes. The former is deceptive; the latter is implementable.

### 6.2 Rank-Weighted vs Quantile (Equal-Weight Within Decile)

**Current approach:** Equal-weight the bottom 5 coins → short 20% each.

**Rank-weighted approach (VertoxQuant style):**
```python
# Instead of equal weight within the short decile:
# Weight proportional to rank (worst coin gets most weight)
ranks = model.predict_rank(features)  # 1 = best, N = worst
# For short leg, use bottom 10 coins:
short_candidates = bottom_10_by_rank
# Weight inversely to rank (worst = highest weight, but still capped)
raw_weights = [rank_i / sum(ranks) for rank_i in short_ranks]
# Apply volatility adjustment:
vol_adjusted = [w_i / vol_i for w_i, vol_i in zip(raw_weights, vols)]
# Normalize and cap:
final_weights = normalize(vol_adjusted)
final_weights = cap_each(final_weights, MAX_SINGLE_SHORT)
final_weights = normalize(final_weights)  # re-normalize after capping
```

**Why rank-weighting is superior:**
1. **Naturally reduces tail exposure:** The 10th-worst coin (borderline) gets much less weight than the worst coin, so a surprise rally in a borderline coin has minimal impact
2. **Better signal utilization:** The model's strongest conviction positions get more weight
3. **More granular:** Uses continuous rank information instead of discrete quantile cutoffs
4. **Still cap at 4-5% per name:** Hard cap prevents any single conviction from dominating

### 6.3 Combined: Rank + Inverse-Vol Weighting

**The recommended approach combines both:**

```python
def compute_short_weights(ranks, volatilities, max_per_name=0.05, total_short=0.25):
    """
    ranks: array of model-predicted ranks (higher = worse = more short conviction)
    volatilities: array of trailing 28d realized volatilities
    """
    # Step 1: Rank-based raw weight (linear in rank)
    rank_weight = ranks / ranks.sum()

    # Step 2: Inverse-vol adjustment
    inv_vol = 1.0 / volatilities
    inv_vol_weight = inv_vol / inv_vol.sum()

    # Step 3: Combine (geometric mean of the two)
    combined = np.sqrt(rank_weight * inv_vol_weight)
    combined = combined / combined.sum()

    # Step 4: Hard cap per name
    combined = np.minimum(combined, max_per_name / total_short)
    combined = combined / combined.sum()

    # Step 5: Scale to target total short exposure
    final = combined * total_short

    return final
```

This gives: **More weight to high-conviction, low-volatility shorts. Less weight to low-conviction, high-volatility shorts.** Exactly the right thing.

---

## 7. Correct Backtest Methodology {#7-backtest-methodology}

### 7.1 Mark-to-Market with Intra-Period Stops: YES

**The question:** Should we use period-end returns (current approach: 14d returns) or mark-to-market daily with stops checked daily?

**Answer: Mark-to-market daily.** Period-end returns HIDE the path. A position that goes -40% intraperiod and recovers to -5% at period end looks fine in period-end accounting, but in reality:
- Your stop would have triggered at -15%
- You would have been OUT of the position
- The "recovery" is irrelevant

**Lopez de Prado's Triple Barrier Method** is the gold standard:
```python
# For each position entered at time t:
for each day d in [t, t + holding_period]:
    if price[d] < entry * (1 - stop_loss):
        exit at stop_loss price  # STOP barrier hit
        break
    elif price[d] > entry * (1 + take_profit):
        exit at take_profit price  # PROFIT barrier hit (for shorts: reversed)
        break
else:
    exit at period end price  # TIME barrier hit
```

**Reference:** Lopez de Prado, M. (2018) "Advances in Financial Machine Learning" — Chapter 3: Triple Barrier Method.

### 7.2 Position Sizing Changes: THE Primary Fix

**The #1 change that will make the backtest realistic and compoundable:**

Reduce position sizes and increase diversification. The current 5 x 20% = 100% short exposure is the problem. Moving to 6-8 x 3-4% = 25% total short exposure makes the short leg survivable even WITHOUT stops.

The stop-loss is a SECONDARY defense. Position sizing is PRIMARY.

### 7.3 Industry Standard: What Good Backtests Include

**Harvey, Liu, Zhu (2016) "...and the Cross-Section of Expected Returns":**
- Multiple testing adjustment (deflated Sharpe ratio)
- Report number of backtests attempted
- Out-of-sample validation mandatory
- t-stat > 3.0 for new factors (not 2.0)

**Bailey, Borwein, Lopez de Prado, Zhu (2014) "Pseudo-Mathematics and Financial Charlatanism":**
- Probability of Backtest Overfitting (PBO) metric
- Combinatorial Symmetric Cross-Validation (CSCV)
- If PBO > 0.5, strategy is likely overfit

**Our backtest already does well:**
- Walk-forward validation (18-month rolling window)
- Purge period (16 days)
- Real funding rates and transaction costs
- Ordered boosting (anti-overfit)
- Ensemble of 3 staggered models

**What we MUST add:**
- Mark-to-market daily with triple barrier stops
- Realistic position sizing (not 20% per name)
- Inverse-vol weighting
- Portfolio-level drawdown circuit breakers
- Report deflated Sharpe ratio

### 7.4 Cost Model (Already Good, Minor Updates)

```python
# Current (good):
COST_PER_SIDE = 0.002  # 0.20% = taker + slippage for small alts

# Recommended update:
COST_MODEL = {
    'maker_fee': 0.0002,       # Binance with BNB
    'taker_fee': 0.00075,      # Binance with BNB
    'slippage_base': 0.0005,   # base slippage
    'slippage_impact': 0.001,  # additional for position size / book depth
    'funding_rate': 'real',    # use actual Binance funding (already doing this)
    'borrow_premium': 0.0,     # zero for perps (no borrow)
}
# Total per side ≈ 0.07-0.15% for maker, 0.13-0.20% for taker
# Use 0.15% per side as conservative average (LOWER than current 0.20%)
# But for less liquid alts in bottom decile: keep 0.20-0.30%
```

---

## 8. Concrete Implementation Specification {#8-implementation-specification}

### 8.1 Position Sizing

```python
# BEFORE (current baseline):
N_SHORTS = 5
WEIGHT_PER_SHORT = 0.20          # 20% each
TOTAL_SHORT_EXPOSURE = 1.00      # 100%

# AFTER (new specification):
N_SHORTS = 8                     # more diversification
MAX_WEIGHT_PER_SHORT = 0.04      # 4% hard cap per name
TOTAL_SHORT_EXPOSURE = 0.25      # 25% total
WEIGHTING = 'rank_x_invvol'      # rank * inverse-volatility
VOL_LOOKBACK = 28                # 28-day trailing realized vol
VOL_FLOOR = 0.005                # daily vol floor to prevent division explosion
```

### 8.2 Stop-Loss Rules

```python
# Per-position stop-loss:
STOP_TYPE = 'trailing'
STOP_BASE_PCT = 0.15             # minimum 15% trailing stop
STOP_ATR_MULT = 2.0              # OR 2x ATR(14) on daily, whichever is WIDER
STOP_RATCHET = 'daily_close'     # update trailing level at daily close only
STOP_CHECK = 'daily_close'       # check stop at daily close (not intraday)

# Portfolio-level drawdown limits:
DD_SOFT = 0.10                   # at -10% from peak: reduce all positions to 50% size
DD_HARD = 0.20                   # at -20% from peak: reduce to 25% size
DD_KILL = 0.30                   # at -30% from peak: flatten all, pause 30 days
```

### 8.3 Maximum Position Sizes

```python
MAX_SINGLE_SHORT = 0.04          # 4% of portfolio
MAX_SINGLE_LONG = 0.06           # 6% (longs have bounded downside)
MAX_TOTAL_SHORT = 0.25           # 25% total short
MAX_TOTAL_LONG = 0.25            # 25% total long (if running L/S)
MAX_GROSS = 0.50                 # 50% gross exposure (conservative)
MAX_NET = 0.10                   # 10% net exposure (near market-neutral)
```

### 8.4 Rebalancing Protocol

```python
REBALANCE_PERIOD = 14            # days (keep current)
REBALANCE_TYPE = 'rolling'       # stagger entries (not all at once)
STAGGER_DAYS = 3                 # spread entries over 3 days
INTRA_PERIOD_TRIM = True         # trim positions that exceed 1.5x max weight
INTRA_PERIOD_CHECK = 'daily'     # check daily

# Between rebalances:
# 1. Check stops daily → exit stopped positions
# 2. Check position weight drift → trim if >1.5x target
# 3. Check portfolio drawdown → apply DD limits
# 4. Do NOT add new positions between rebalances (wait for next rebalance)
# 5. Stopped-out capital stays in cash until next rebalance
```

### 8.5 Cost Model

```python
COST_PER_SIDE = 0.0015           # 0.15% base (maker + slippage)
COST_ILLIQUIDITY_ADD = 0.0010    # add 0.10% for bottom-quartile liquidity coins
FUNDING_RATE = 'real_binance'    # use actual historical funding rates
FUNDING_MAX_NEG = -0.001         # -0.1% per 8h → close position if exceeded
```

### 8.6 Backtest Engine Changes Required

```python
# REQUIRED CHANGES to src/backtester.py or new src/backtester_v2.py:

# 1. Daily mark-to-market (not period-end only)
# 2. Triple barrier exit logic per position
# 3. Inverse-vol weighting at entry
# 4. Rank-based weighting at entry
# 5. Position weight drift monitoring
# 6. Portfolio-level drawdown circuit breakers
# 7. Isolated position tracking (each position is independent)
# 8. Cash management (stopped-out capital earns 0 until rebalance)
# 9. Staggered entry across 3 days
# 10. Funding rate applied per position per 8h period
```

---

## Expected Impact on Performance

### What will happen to the numbers:

| Metric | Current (100% short) | Expected (25% short) | Ratio |
|--------|---------------------|----------------------|-------|
| Arithmetic return/period | +2.37% | +0.59% | 0.25x |
| Volatility | ~15% ann | ~4-5% ann | 0.30x |
| Sharpe ratio | 1.36 | **1.2-1.5** | ~same or better |
| Max drawdown | -37.9% | **-10 to -15%** | 0.3-0.4x |
| Terminal wealth ($100k, 4.5yr) | $1,033k | **$125-160k** | 0.12-0.15x |
| COMPOUNDABLE? | **NO** (blows up in some paths) | **YES** | -- |
| Realistic live performance | Unknown (likely negative) | **+5-12% ann** | -- |

### The Key Tradeoff:

The headline return drops from +933% to +25-60% over 4.5 years. But the CURRENT number is **not achievable in reality** because it does not survive compounding with realistic position sizes. The new number IS achievable.

**Sharpe ratio stays similar or IMPROVES** because:
- Vol drops proportionally more than return (vol drag removed)
- Stops prevent tail losses
- Inverse-vol weighting concentrates in low-vol shorts (better Sharpe per unit risk)

### Adding Leverage (Carefully):

Once the unlevered strategy is validated at 25% short exposure with Sharpe >1.0:
```python
# Cautious leverage to scale returns:
LEVERAGE = 2.0                    # 2x gross → 50% short exposure
# Expected: returns double, vol doubles, Sharpe unchanged
# Max DD: ~20-30% (vs 10-15% unlevered)
# This is the CORRECT way to increase returns — not larger positions
```

At 2x leverage: ~$100k → $200-300k over 4.5 years, with -20-30% max DD. This is realistic and compoundable.

---

## Summary: The 5 Changes That Matter (Priority Order)

1. **REDUCE POSITION SIZES** from 20% to 3-4% per name (the single most important change)
2. **INCREASE DIVERSIFICATION** from 5 to 8 short positions
3. **ADD INVERSE-VOL WEIGHTING** within the short leg
4. **IMPLEMENT DAILY MARK-TO-MARKET** with triple barrier exits
5. **ADD PORTFOLIO DRAWDOWN LIMITS** as circuit breakers

Everything else (rank-weighting, funding rate caps, ADL protection, staggered entry) is incremental improvement on top of these five.

---

## References

### Academic Papers
- Estrada, J. (2010) "Geometric Mean Maximization: An Overlooked Portfolio Approach?" — Journal of Investing 19(4), 134-147
- Barroso, P. & Santa-Clara, P. (2015) "Momentum has its moments" — Journal of Financial Economics 116(1), 111-120
- Daniel, K. & Moskowitz, T. (2016) "Momentum crashes" — Journal of Financial Economics 122(2), 221-247
- Harvey, C., Liu, Y. & Zhu, H. (2016) "...and the Cross-Section of Expected Returns" — Review of Financial Studies 29(1), 5-68
- Bailey, D., Borwein, J., Lopez de Prado, M. & Zhu, Q. (2014) "Pseudo-Mathematics and Financial Charlatanism" — Notices of the AMS 61(5)
- Lopez de Prado, M. (2018) "Advances in Financial Machine Learning" — Wiley (Triple Barrier Method, Chapter 3)
- Thorp, E. (2008) "The Kelly Criterion in Blackjack, Sports Betting and the Stock Market" — Handbook of Asset and Liability Management
- Ang, A., Hodrick, R., Xing, Y. & Zhang, X. (2006) "The Cross-Section of Volatility and Expected Returns" — Journal of Finance 61(1)
- Moreira, A. & Muir, T. (2017) "Volatility-Managed Portfolios" — Journal of Finance 72(4), 1611-1644

### Practitioner References
- AQR (2023) "Key Design Choices in Long/Short Equity" — Alternative Thinking Q4 2023
- AQR "Building a Better Long-Short Equity Portfolio" — White Paper
- Kitces, M. "Volatility Drag: How Variance Drains Investment Returns"
- QuantPedia "How Does Weighting Scheme Impact Systematic Equity Portfolios?"

### Crypto-Specific
- Bybit: Auto-Deleveraging (ADL) Mechanism documentation
- Binance: ADL and Risk Limit documentation
- CoinDesk (2025) "How ADL on Crypto Perp Trading Platforms Can Cut Winning Trades"
