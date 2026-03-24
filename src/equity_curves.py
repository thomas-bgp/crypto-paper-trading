"""
Generate full equity curves for the best strategies + BTC benchmark.
Produces interactive Plotly chart with all curves overlaid.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Reuse the grid search infrastructure
from backtest_grid import build_simple_panel, COST_PER_SIDE, STABLE_YIELD_DAILY

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

# ─── Strategies to test ───
STRATEGIES = {
    'L/S Champion (Sharpe 3.58)': {
        'direction': 'long_short', 'lookback_days': 14, 'holding_days': 14,
        'top_n': 8, 'universe_top': 50, 'trailing_stop_pct': 0.15,
        'regime_filter': False,
    },
    'L/S Best Calmar (DD -5%)': {
        'direction': 'long_short', 'lookback_days': 56, 'holding_days': 28,
        'top_n': 8, 'universe_top': 50, 'trailing_stop_pct': 0.15,
        'regime_filter': False,
    },
    'L/S Realistic (Regime ON)': {
        'direction': 'long_short', 'lookback_days': 14, 'holding_days': 7,
        'top_n': 8, 'universe_top': 30, 'trailing_stop_pct': 0.15,
        'regime_filter': True,
    },
    'Long-Only Best Sharpe': {
        'direction': 'long_only', 'lookback_days': 14, 'holding_days': 14,
        'top_n': 5, 'universe_top': 50, 'trailing_stop_pct': 0.15,
        'regime_filter': False,
    },
    'Long-Only + Regime': {
        'direction': 'long_only', 'lookback_days': 14, 'holding_days': 7,
        'top_n': 8, 'universe_top': 30, 'trailing_stop_pct': 0.15,
        'regime_filter': True,
    },
    'L/S Conservative (lb56 h28)': {
        'direction': 'long_short', 'lookback_days': 56, 'holding_days': 28,
        'top_n': 5, 'universe_top': 50, 'trailing_stop_pct': 0.15,
        'regime_filter': True,
    },
}

COLORS = {
    'L/S Champion (Sharpe 3.58)': '#FF5722',
    'L/S Best Calmar (DD -5%)': '#FF9800',
    'L/S Realistic (Regime ON)': '#4CAF50',
    'Long-Only Best Sharpe': '#2196F3',
    'Long-Only + Regime': '#03A9F4',
    'L/S Conservative (lb56 h28)': '#9C27B0',
    'BTC Buy & Hold': '#FFD700',
    'BTC + SMA100 Filter': '#BDBDBD',
}


def run_equity_curve(panel, params):
    """Run backtest returning full daily equity curve."""
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

    dates = panel.index.get_level_values('date').unique().sort_values()
    rebal_dates = dates[::holding]

    equity = 100_000.0
    curve = []

    for rebal_date in rebal_dates:
        if rebal_date not in panel.index.get_level_values('date'):
            continue

        cross = panel.loc[rebal_date].copy()
        if len(cross) < 10:
            curve.append({'date': rebal_date, 'equity': equity})
            continue

        # Regime
        if use_regime and 'regime_bull' in cross.columns:
            regime_val = cross['regime_bull'].iloc[0]
            if regime_val < 0.5:
                cash_yield = equity * STABLE_YIELD_DAILY * holding
                equity += cash_yield
                curve.append({'date': rebal_date, 'equity': equity})
                continue

        cross = cross.dropna(subset=[ret_col, fwd_col, 'avg_vol_28d'])
        cross = cross.nlargest(universe_top, 'avg_vol_28d')
        if len(cross) < top_n * 2:
            curve.append({'date': rebal_date, 'equity': equity})
            continue

        cross['mom_rank'] = cross[ret_col].rank(ascending=False)

        # Long
        longs = cross.nsmallest(top_n, 'mom_rank')
        long_rets = longs[fwd_col]
        if stop_pct > 0:
            long_rets = long_rets.clip(lower=-stop_pct)
        long_ret = long_rets.mean()

        # Short
        short_ret = 0.0
        if direction == 'long_short':
            shorts = cross.nlargest(top_n, 'mom_rank')
            short_rets = shorts[fwd_col]
            if stop_pct > 0:
                short_rets = short_rets.clip(upper=stop_pct)
            short_ret = -short_rets.mean()
            short_ret -= 0.00037 * holding  # funding cost

        if direction == 'long_short':
            total_ret = (long_ret + short_ret) / 2
        else:
            total_ret = long_ret

        # Costs
        n_pos = top_n * (2 if direction == 'long_short' else 1)
        total_ret -= n_pos * 2 * COST_PER_SIDE / n_pos

        # Cash yield
        if direction == 'long_only':
            equity += equity * 0.3 * STABLE_YIELD_DAILY * holding

        equity *= (1 + total_ret)
        if equity <= 0:
            equity = 1  # prevent negative
        curve.append({'date': rebal_date, 'equity': equity})

    return pd.DataFrame(curve).set_index('date')


def btc_buy_and_hold(panel):
    """BTC buy and hold equity curve."""
    if 'BTCUSDT' not in panel.index.get_level_values('symbol').unique():
        return None
    btc = panel.xs('BTCUSDT', level='symbol')['close'].dropna()
    equity = 100_000 * btc / btc.iloc[0]
    return pd.DataFrame({'equity': equity.values}, index=btc.index)


def btc_sma_filter(panel):
    """BTC with SMA100 trend filter."""
    if 'BTCUSDT' not in panel.index.get_level_values('symbol').unique():
        return None
    btc = panel.xs('BTCUSDT', level='symbol')[['close']].dropna()
    sma100 = btc['close'].rolling(100).mean()
    signal = (btc['close'] > sma100).astype(float)

    returns = btc['close'].pct_change()
    strat_returns = returns * signal.shift(1)  # no look-ahead
    equity = 100_000 * (1 + strat_returns).cumprod()
    return pd.DataFrame({'equity': equity.values}, index=btc.index)


def build_dashboard(curves, metrics_list):
    """Build comprehensive equity curve dashboard."""

    # ─── FIGURE 1: All equity curves (log scale) ───
    fig1 = go.Figure()
    for name, curve in curves.items():
        if curve is None or curve.empty:
            continue
        fig1.add_trace(go.Scatter(
            x=curve.index, y=curve['equity'],
            name=name, line=dict(color=COLORS.get(name, '#888'), width=2.5),
            hovertemplate=f'{name}<br>Date: %{{x}}<br>Equity: $%{{y:,.0f}}<extra></extra>',
        ))
    fig1.update_layout(
        title='Evolução de Patrimônio — Melhores Estratégias vs BTC ($100k inicial)',
        yaxis_title='Equity ($)', yaxis_type='log',
        xaxis_title='', height=700, template='plotly_dark',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)'),
        hovermode='x unified',
    )
    fig1.add_hline(y=100_000, line_dash="dot", line_color="gray",
                   annotation_text="$100k inicial")

    # ─── FIGURE 2: Drawdown curves ───
    fig2 = go.Figure()
    for name, curve in curves.items():
        if curve is None or curve.empty:
            continue
        eq = curve['equity']
        peak = eq.expanding().max()
        dd = (eq - peak) / peak * 100
        fig2.add_trace(go.Scatter(
            x=curve.index, y=dd,
            name=name, line=dict(color=COLORS.get(name, '#888'), width=1.5),
            fill='tozeroy',
            hovertemplate=f'{name}<br>DD: %{{y:.1f}}%<extra></extra>',
        ))
    fig2.update_layout(
        title='Drawdown — Todas as Estratégias',
        yaxis_title='Drawdown (%)', xaxis_title='',
        height=400, template='plotly_dark',
        legend=dict(x=0.01, y=-0.3, orientation='h'),
    )

    # ─── FIGURE 3: Annual returns grouped bar ───
    yr_data = []
    for name, curve in curves.items():
        if curve is None or curve.empty:
            continue
        eq = curve['equity']
        for year in sorted(eq.index.year.unique()):
            yr = eq[eq.index.year == year]
            if len(yr) > 1:
                ret = (yr.iloc[-1] / yr.iloc[0] - 1) * 100
                yr_data.append({'strategy': name, 'year': str(year), 'return': ret})

    yr_df = pd.DataFrame(yr_data)
    fig3 = go.Figure()
    for name in curves.keys():
        sub = yr_df[yr_df['strategy'] == name]
        if sub.empty:
            continue
        fig3.add_trace(go.Bar(
            x=sub['year'], y=sub['return'],
            name=name, marker_color=COLORS.get(name, '#888'),
            text=sub['return'].apply(lambda x: f'{x:+.0f}%'),
            textposition='outside', textfont_size=9,
        ))
    fig3.update_layout(
        title='Retornos Anuais por Estratégia',
        yaxis_title='Return (%)', barmode='group',
        height=500, template='plotly_dark',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)'),
    )
    fig3.add_hline(y=0, line_color="gray")

    # ─── FIGURE 4: Summary metrics table ───
    table_data = []
    for name, curve in curves.items():
        if curve is None or curve.empty:
            continue
        eq = curve['equity']
        rets = eq.pct_change().dropna()
        n_years = (eq.index[-1] - eq.index[0]).days / 365.25
        if n_years <= 0 or eq.iloc[-1] <= 0:
            continue
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/n_years) - 1
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        downside = rets[rets < 0].std()
        sortino = rets.mean() / downside * np.sqrt(252) if downside and downside > 0 else 0
        peak = eq.expanding().max()
        max_dd = ((eq - peak) / peak).min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        table_data.append({
            'Strategy': name,
            'Final ($)': f'${eq.iloc[-1]:,.0f}',
            'CAGR': f'{cagr*100:.0f}%',
            'Sharpe': f'{sharpe:.2f}',
            'Sortino': f'{sortino:.2f}',
            'Max DD': f'{max_dd*100:.1f}%',
            'Calmar': f'{calmar:.1f}',
        })

    tdf = pd.DataFrame(table_data)
    fig4 = go.Figure(go.Table(
        header=dict(
            values=list(tdf.columns),
            fill_color='#1a1a2e', font=dict(color='white', size=13),
            align='center', line_color='#333',
        ),
        cells=dict(
            values=[tdf[c] for c in tdf.columns],
            fill_color=[
                ['#0d47a1' if 'Long' in s else '#bf360c' if 'L/S' in s else '#333'
                 for s in tdf['Strategy']]
            ],
            font=dict(color='white', size=12),
            align='center', line_color='#333', height=30,
        ),
    ))
    fig4.update_layout(height=max(250, 50 + 35 * len(tdf)), template='plotly_dark',
                       title='Métricas Consolidadas')

    # ─── FIGURE 5: Rolling 90-day Sharpe ───
    fig5 = go.Figure()
    for name, curve in curves.items():
        if curve is None or curve.empty:
            continue
        eq = curve['equity']
        rets = eq.pct_change().dropna()
        rolling_sharpe = rets.rolling(90).mean() / rets.rolling(90).std() * np.sqrt(252)
        fig5.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe,
            name=name, line=dict(color=COLORS.get(name, '#888'), width=1.5),
        ))
    fig5.update_layout(
        title='Rolling 90-Day Sharpe Ratio',
        yaxis_title='Sharpe', height=400, template='plotly_dark',
    )
    fig5.add_hline(y=0, line_color="gray")
    fig5.add_hline(y=1, line_dash="dash", line_color="yellow",
                   annotation_text="Sharpe=1")

    # ─── BUILD HTML ───
    html = [
        '<html><head>',
        '<title>Equity Curves Dashboard</title>',
        '<style>',
        'body { background: #0a0a0a; color: #e0e0e0; font-family: monospace; margin: 20px; }',
        'h1 { color: #ff9800; text-align: center; font-size: 28px; }',
        'p.sub { text-align: center; color: #888; margin-bottom: 30px; }',
        '</style></head><body>',
        '<h1>CRYPTO MOMENTUM — EQUITY CURVES</h1>',
        '<p class="sub">$100k initial capital | 2020-2026 | Survivorship-bias-free (534 coins)</p>',
        fig4.to_html(full_html=False, include_plotlyjs='cdn'),
        fig1.to_html(full_html=False, include_plotlyjs=False),
        fig2.to_html(full_html=False, include_plotlyjs=False),
        fig3.to_html(full_html=False, include_plotlyjs=False),
        fig5.to_html(full_html=False, include_plotlyjs=False),
        '</body></html>',
    ]

    path = os.path.join(RESULTS_DIR, 'equity_curves_dashboard.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))

    print(f"Dashboard saved to: {path}")
    return path


def main():
    print("Building panel...")
    panel = build_simple_panel()
    print(f"Panel: {panel.shape}")

    curves = {}
    metrics = {}

    # Run each strategy
    for name, params in STRATEGIES.items():
        print(f"Running: {name}...")
        curve = run_equity_curve(panel, params)
        if curve is not None and len(curve) > 10:
            curves[name] = curve
            print(f"  -> ${curve['equity'].iloc[-1]:,.0f} final")
        else:
            print(f"  -> FAILED")

    # BTC benchmarks
    print("Computing BTC benchmarks...")
    btc_bh = btc_buy_and_hold(panel)
    if btc_bh is not None:
        curves['BTC Buy & Hold'] = btc_bh
        print(f"  BTC B&H -> ${btc_bh['equity'].iloc[-1]:,.0f}")

    btc_sma = btc_sma_filter(panel)
    if btc_sma is not None:
        curves['BTC + SMA100 Filter'] = btc_sma
        print(f"  BTC+SMA -> ${btc_sma['equity'].iloc[-1]:,.0f}")

    # Build dashboard
    path = build_dashboard(curves, metrics)

    # Open
    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(path)}')


if __name__ == '__main__':
    main()
