"""
Portfolio Optimizer — Markowitz with Ledoit-Wolf shrinkage.
Combines Strategy #1 (Momentum), #2 (Contrarian), #3 (Pairs).
"""
import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import minimize

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')


def load_strategy_returns():
    """Load equity curves and compute returns for each strategy."""
    strats = {}

    # Strategy #1 — Momentum Regime Switch
    p = os.path.join(RESULTS_DIR, 'backtest_result.parquet')
    if os.path.exists(p):
        df = pd.read_parquet(p)
        strats['Momentum'] = df['equity'].pct_change().dropna()

    # Strategy #2 — Contrarian
    p = os.path.join(RESULTS_DIR, 'contrarian_result.parquet')
    if os.path.exists(p):
        df = pd.read_parquet(p)
        strats['Contrarian'] = df['equity'].pct_change().dropna()

    # Strategy #3 — Pairs
    p = os.path.join(RESULTS_DIR, 'pairs_result.parquet')
    if os.path.exists(p):
        df = pd.read_parquet(p)
        strats['Pairs'] = df['equity'].pct_change().dropna()

    # Align on common dates
    combined = pd.DataFrame(strats)
    combined.dropna(inplace=True)
    return combined


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> np.ndarray:
    """Shrink covariance matrix using Ledoit-Wolf constant-correlation model."""
    T, N = returns.shape
    X = returns.values - returns.values.mean(axis=0)
    S = X.T @ X / T  # sample covariance

    # Target: constant correlation matrix
    var = np.diag(S)
    std = np.sqrt(var)
    rho_bar = 0.0
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            rho_bar += S[i, j] / (std[i] * std[j])
            count += 1
    rho_bar /= max(count, 1)

    F = np.outer(std, std) * rho_bar
    np.fill_diagonal(F, var)

    # Optimal shrinkage intensity (simplified)
    delta = np.sum((S - F) ** 2) / T
    kappa = delta / max(np.sum((S - F) ** 2), 1e-10)
    shrinkage = max(0, min(1, kappa))

    return shrinkage * F + (1 - shrinkage) * S


def optimize_portfolio(returns: pd.DataFrame, risk_free: float = 0.0) -> dict:
    """Mean-variance optimization with constraints."""
    mu = returns.mean() * 6 * 365.25  # annualize (6 candles/day)
    cov = ledoit_wolf_shrinkage(returns) * 6 * 365.25
    n = len(mu)

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        return -(port_ret - risk_free) / max(port_vol, 1e-10)

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # fully invested
    ]
    bounds = [(0.10, 0.60)] * n  # min 10%, max 60% per strategy

    x0 = np.ones(n) / n
    result = minimize(neg_sharpe, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    weights = result.x
    port_ret = weights @ mu
    port_vol = np.sqrt(weights @ cov @ weights)
    port_sharpe = (port_ret - risk_free) / port_vol

    # Correlation matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    return {
        'weights': dict(zip(returns.columns, np.round(weights, 4))),
        'expected_return': round(port_ret * 100, 2),
        'expected_vol': round(port_vol * 100, 2),
        'sharpe': round(port_sharpe, 2),
        'correlation_matrix': pd.DataFrame(corr, index=returns.columns,
                                           columns=returns.columns).round(3).to_dict(),
        'individual_returns': dict(zip(returns.columns,
                                       (mu * 100).round(2))),
        'individual_sharpe': dict(zip(returns.columns,
                                      (mu / np.sqrt(np.diag(cov))).round(2))),
    }


def build_combined_equity(returns: pd.DataFrame, weights: dict) -> pd.Series:
    """Build combined portfolio equity curve from strategy returns and weights."""
    w = np.array([weights.get(col, 0) for col in returns.columns])
    port_returns = (returns.values * w).sum(axis=1)
    equity = 10000 * (1 + pd.Series(port_returns, index=returns.index)).cumprod()
    return equity


def run_optimization():
    """Main optimization pipeline."""
    print("Loading strategy returns...")
    returns = load_strategy_returns()
    print(f"Common period: {returns.index[0]} to {returns.index[-1]} ({len(returns)} candles)")
    print(f"Strategies: {list(returns.columns)}")

    print("\nOptimizing portfolio...")
    result = optimize_portfolio(returns)

    print("\n" + "=" * 60)
    print("  PORTFOLIO OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"  Weights: {result['weights']}")
    print(f"  Expected Return: {result['expected_return']}%")
    print(f"  Expected Volatility: {result['expected_vol']}%")
    print(f"  Sharpe Ratio: {result['sharpe']}")
    print(f"\n  Individual Returns: {result['individual_returns']}")
    print(f"  Individual Sharpe: {result['individual_sharpe']}")
    print(f"\n  Correlation Matrix:")
    for k, v in result['correlation_matrix'].items():
        print(f"    {k}: {v}")
    print("=" * 60)

    # Build combined equity
    combined_eq = build_combined_equity(returns, result['weights'])
    combined_eq.name = 'equity'
    combined_eq.to_frame().to_parquet(os.path.join(RESULTS_DIR, 'portfolio_combined.parquet'))

    # Save optimization results
    with open(os.path.join(RESULTS_DIR, 'portfolio_optimization.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nCombined portfolio: ${combined_eq.iloc[0]:,.0f} -> ${combined_eq.iloc[-1]:,.0f}")
    n_years = (combined_eq.index[-1] - combined_eq.index[0]).days / 365.25
    cagr = ((combined_eq.iloc[-1] / combined_eq.iloc[0]) ** (1/n_years) - 1) * 100
    rets = combined_eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(6*365.25)
    peak = combined_eq.expanding().max()
    max_dd = ((combined_eq - peak) / peak).min() * 100
    print(f"CAGR: {cagr:.2f}%, Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2f}%")

    return result


if __name__ == '__main__':
    run_optimization()
