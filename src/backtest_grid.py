"""
Vectorized Parameter Grid Backtester.
Sweeps across: direction (long/long-short), holding period, lookback,
top-N, rebalance freq, trailing stop, regime filter.
Uses joblib for parallel execution.
"""
import numpy as np
import pandas as pd
import os
import json
import time
from itertools import product
from joblib import Parallel, delayed

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

COST_PER_SIDE = 0.00125  # 0.075% fee + 0.05% slippage
STABLE_YIELD_DAILY = 0.045 / 365


# ─── Parameter Grid ───
PARAM_GRID = {
    'direction': ['long_only', 'long_short'],
    'lookback_days': [7, 14, 21, 28, 56],
    'holding_days': [7, 14, 28],
    'top_n': [3, 5, 8],
    'universe_top': [20, 30, 50],
    'trailing_stop_pct': [0.0, 0.15, 0.25],  # 0 = no trailing stop
    'regime_filter': [True, False],
}


def load_panel():
    """Load precomputed feature panel."""
    path = os.path.join(DATA_DIR, 'feature_panel.parquet')
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def build_simple_panel():
    """Build a minimal panel for fast grid search (daily OHLCV only)."""
    from feature_engine import load_all_data, build_panel
    data = load_all_data(min_candles=250)
    panel = build_panel(data, resample='1D')

    # Compute only essential features for speed
    grouped = panel.groupby(level='symbol')

    # Returns at various lookbacks
    for lb in [7, 14, 21, 28, 56]:
        panel[f'ret_{lb}d'] = grouped['close'].pct_change(lb)

    # Volume for universe filtering
    if 'quote_vol' in panel.columns:
        panel['daily_volume'] = panel['quote_vol']
    else:
        panel['daily_volume'] = panel['volume'] * panel['close']

    panel['avg_vol_28d'] = grouped['daily_volume'].transform(
        lambda x: x.rolling(28).mean()
    )

    # Volatility for ATR-based stop
    log_ret = grouped['close'].transform(lambda x: np.log(x / x.shift(1)))
    panel['rvol_28d'] = log_ret.groupby(level='symbol').transform(
        lambda x: x.rolling(28).std() * np.sqrt(365)
    )

    # Regime: BTC SMA
    if 'BTCUSDT' in panel.index.get_level_values('symbol').unique():
        btc = panel.xs('BTCUSDT', level='symbol')['close']
        btc_sma100 = btc.rolling(100).mean()
        regime = (btc > btc_sma100).astype(float)
        regime.name = 'regime_bull'
        panel = panel.join(regime, on='date')
    else:
        panel['regime_bull'] = 1.0

    # Forward returns for target
    for h in [7, 14, 28]:
        panel[f'fwd_ret_{h}d'] = grouped['close'].transform(
            lambda x: x.pct_change(h).shift(-h)
        )

    return panel


# ─── Vectorized Backtest Core ───

def run_single_backtest(panel, params):
    """
    Run a single backtest configuration. Fully vectorized where possible.
    Returns metrics dict.
    """
    direction = params['direction']
    lookback = params['lookback_days']
    holding = params['holding_days']
    top_n = params['top_n']
    universe_top = params['universe_top']
    stop_pct = params['trailing_stop_pct']
    use_regime = params['regime_filter']

    ret_col = f'ret_{lookback}d'
    fwd_col = f'fwd_ret_{holding}d'

    if ret_col not in panel.columns or fwd_col not in panel.columns:
        return None

    # Get all unique dates
    dates = panel.index.get_level_values('date').unique().sort_values()

    # Rebalance dates (every `holding` days)
    rebal_dates = dates[::holding]

    equity = 100_000.0
    equity_curve = []
    btc_prices = []

    for rebal_date in rebal_dates:
        if rebal_date not in panel.index.get_level_values('date'):
            continue

        cross = panel.loc[rebal_date].copy()

        if len(cross) < 10:
            equity_curve.append({'date': rebal_date, 'equity': equity})
            continue

        # Regime filter
        if use_regime and 'regime_bull' in cross.columns:
            regime_val = cross['regime_bull'].iloc[0] if 'regime_bull' in cross.columns else 1.0
            if regime_val < 0.5:
                # Bear regime: stay in cash
                cash_yield = equity * STABLE_YIELD_DAILY * holding
                equity += cash_yield
                equity_curve.append({'date': rebal_date, 'equity': equity})
                continue

        # Filter universe by volume (top N by avg volume)
        cross = cross.dropna(subset=[ret_col, fwd_col, 'avg_vol_28d'])
        cross = cross.nlargest(universe_top, 'avg_vol_28d')

        if len(cross) < top_n * 2:
            equity_curve.append({'date': rebal_date, 'equity': equity})
            continue

        # Rank by momentum
        cross['mom_rank'] = cross[ret_col].rank(ascending=False)

        # Long positions (top N)
        longs = cross.nsmallest(top_n, 'mom_rank')
        long_ret = longs[fwd_col].mean()

        # Apply trailing stop approximation (reduce returns by stop effect)
        if stop_pct > 0:
            # Approximate: if fwd return < -stop_pct, cap loss at -stop_pct
            # This is an approximation of trailing stop effect
            long_rets_individual = longs[fwd_col].clip(lower=-stop_pct)
            long_ret = long_rets_individual.mean()

        # Short positions (bottom N) for long-short
        short_ret = 0.0
        if direction == 'long_short':
            shorts = cross.nlargest(top_n, 'mom_rank')
            short_rets_individual = shorts[fwd_col]
            if stop_pct > 0:
                short_rets_individual = short_rets_individual.clip(upper=stop_pct)
            short_ret = -short_rets_individual.mean()  # profit from short

            # Funding cost for shorts (approximate 0.01% per 8h = ~0.037% per day)
            funding_cost = 0.00037 * holding * top_n / (2 * top_n)  # per unit
            short_ret -= funding_cost

        # Total return
        if direction == 'long_short':
            total_ret = (long_ret + short_ret) / 2  # 50/50 allocation
        else:
            total_ret = long_ret

        # Transaction costs (entry + exit for each position)
        n_positions = top_n * (2 if direction == 'long_short' else 1)
        cost = n_positions * 2 * COST_PER_SIDE / n_positions  # per-unit cost
        total_ret -= cost

        # Cash yield on uninvested portion
        if direction == 'long_only':
            cash_yield = equity * 0.3 * STABLE_YIELD_DAILY * holding  # ~30% cash buffer
            equity += cash_yield

        # Update equity
        equity *= (1 + total_ret)

        equity_curve.append({
            'date': rebal_date,
            'equity': equity,
        })

    if not equity_curve:
        return None

    result = pd.DataFrame(equity_curve).set_index('date')
    return compute_fast_metrics(result, params)


def compute_fast_metrics(result, params):
    """Fast metrics computation."""
    eq = result['equity']
    if len(eq) < 10:
        return None

    returns = eq.pct_change().dropna()
    if len(returns) < 5 or returns.std() == 0:
        return None

    n_days = (result.index[-1] - result.index[0]).days
    n_years = n_days / 365.25
    if n_years <= 0:
        return None

    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    if eq.iloc[-1] <= 0 or eq.iloc[0] <= 0:
        return None
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1

    # Annualize based on holding period
    periods_per_year = 365 / params['holding_days']
    sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0

    downside = returns[returns < 0].std()
    sortino = returns.mean() / downside * np.sqrt(periods_per_year) if downside and downside > 0 else 0

    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Yearly returns
    yearly = {}
    for year in sorted(result.index.year.unique()):
        yr = result[result.index.year == year]
        if len(yr) > 1:
            yearly[year] = yr['equity'].iloc[-1] / yr['equity'].iloc[0] - 1

    return {
        **params,
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'total_return': round(total_return * 100, 2),
        'n_periods': len(returns),
        'final_equity': round(eq.iloc[-1], 0),
        **{f'yr_{k}': round(v * 100, 1) for k, v in yearly.items()},
    }


def run_grid_search(panel=None, n_jobs=-1, max_configs=None):
    """Run all parameter combinations in parallel."""
    if panel is None:
        print("Building panel...")
        panel = build_simple_panel()
        print(f"Panel: {panel.shape}")

    # Generate all parameter combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    all_combos = [dict(zip(keys, v)) for v in product(*values)]

    # Filter impossible combos
    valid_combos = []
    for c in all_combos:
        # Holding period should be <= lookback
        if c['holding_days'] > c['lookback_days'] * 2:
            continue
        # Top N should be <= universe_top / 3
        if c['top_n'] > c['universe_top'] // 3:
            continue
        valid_combos.append(c)

    if max_configs:
        valid_combos = valid_combos[:max_configs]

    print(f"Running {len(valid_combos)} configurations with {n_jobs} workers...")
    t0 = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_single_backtest)(panel, params)
        for params in valid_combos
    )

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s ({len(valid_combos) / elapsed:.1f} configs/sec)")

    # Filter None results
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)

    if df.empty:
        print("No valid results!")
        return df

    # Sort by Sharpe
    df = df.sort_values('sharpe', ascending=False)

    # Save
    df.to_csv(os.path.join(RESULTS_DIR, 'grid_search_results.csv'), index=False)
    df.to_parquet(os.path.join(RESULTS_DIR, 'grid_search_results.parquet'))

    print(f"\n{'=' * 80}")
    print(f"  GRID SEARCH RESULTS — {len(df)} valid configurations")
    print(f"{'=' * 80}")

    # Top 10 by Sharpe
    print("\n  TOP 10 BY SHARPE:")
    cols = ['direction', 'lookback_days', 'holding_days', 'top_n',
            'universe_top', 'trailing_stop_pct', 'regime_filter',
            'cagr', 'sharpe', 'sortino', 'max_dd', 'calmar']
    print(df[cols].head(10).to_string(index=False))

    # Top 10 by CAGR
    print("\n  TOP 10 BY CAGR:")
    print(df.sort_values('cagr', ascending=False)[cols].head(10).to_string(index=False))

    # Top 10 by Calmar (risk-adjusted)
    print("\n  TOP 10 BY CALMAR (CAGR/MaxDD):")
    print(df.sort_values('calmar', ascending=False)[cols].head(10).to_string(index=False))

    # Best per direction
    print("\n  BEST LONG-ONLY:")
    lo = df[df['direction'] == 'long_only'].head(5)
    print(lo[cols].to_string(index=False))

    print("\n  BEST LONG-SHORT:")
    ls = df[df['direction'] == 'long_short'].head(5)
    print(ls[cols].to_string(index=False))

    # Statistics
    print(f"\n  STATISTICS:")
    print(f"    Mean Sharpe (all):       {df['sharpe'].mean():.3f}")
    print(f"    Mean Sharpe (long-only): {df[df['direction']=='long_only']['sharpe'].mean():.3f}")
    print(f"    Mean Sharpe (long-short):{df[df['direction']=='long_short']['sharpe'].mean():.3f}")
    print(f"    % with Sharpe > 0.5:     {(df['sharpe'] > 0.5).mean()*100:.1f}%")
    print(f"    % with Sharpe > 1.0:     {(df['sharpe'] > 1.0).mean()*100:.1f}%")
    print(f"    Best Sharpe:             {df['sharpe'].max():.3f}")
    print(f"    Best CAGR:               {df['cagr'].max():.1f}%")
    print(f"    Best MaxDD:              {df['max_dd'].max():.1f}%")

    # Regime filter effect
    print(f"\n  REGIME FILTER EFFECT:")
    for direction in ['long_only', 'long_short']:
        sub = df[df['direction'] == direction]
        with_regime = sub[sub['regime_filter'] == True]['sharpe'].mean()
        no_regime = sub[sub['regime_filter'] == False]['sharpe'].mean()
        print(f"    {direction}: with regime={with_regime:.3f}, without={no_regime:.3f}, delta={with_regime-no_regime:+.3f}")

    print(f"\n  Results saved to {RESULTS_DIR}/grid_search_results.csv")

    return df


if __name__ == '__main__':
    import sys
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    run_grid_search(n_jobs=n_jobs)
