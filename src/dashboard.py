"""
Streamlit Dashboard — Backtest Results & Regime Analysis
Run: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

st.set_page_config(page_title="Quant Committee — Crypto Backtest", layout="wide",
                   page_icon="📊")


@st.cache_data
def load_results():
    result = pd.read_parquet(os.path.join(RESULTS_DIR, 'backtest_result.parquet'))
    regimes = pd.read_parquet(os.path.join(RESULTS_DIR, 'regimes.parquet'))
    with open(os.path.join(RESULTS_DIR, 'metrics.json')) as f:
        metrics = json.load(f)
    return result, regimes, metrics


def regime_color(regime):
    colors = {
        'BULL': 'rgba(0, 200, 83, 0.15)',
        'SIDEWAYS': 'rgba(255, 193, 7, 0.15)',
        'BEAR': 'rgba(244, 67, 54, 0.15)',
    }
    return colors.get(regime, 'rgba(128,128,128,0.1)')


def main():
    st.title("COMITE QUANT — Backtest Dashboard")
    st.markdown("**Portfolio 3 Camadas + Regime Switch (HMM + Funding Rate + FNG)**")

    try:
        result, regimes, metrics = load_results()
    except FileNotFoundError:
        st.error("Results not found. Run `python backtester.py` first.")
        return

    # ─── Header Metrics ───
    st.markdown("---")
    cols = st.columns(6)
    cols[0].metric("CAGR", metrics.get('CAGR', 'N/A'))
    cols[1].metric("Sharpe", metrics.get('Sharpe Ratio', 'N/A'))
    cols[2].metric("Max Drawdown", metrics.get('Max Drawdown', 'N/A'))
    cols[3].metric("Sortino", metrics.get('Sortino Ratio', 'N/A'))
    cols[4].metric("Final Equity", metrics.get('Final Equity', 'N/A'))
    cols[5].metric("BTC CAGR", metrics.get('BTC CAGR', 'N/A'))

    # ─── Equity Curve with Regime Background ───
    st.markdown("### Equity Curve vs BTC Buy & Hold")

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.45, 0.20, 0.15, 0.20],
                        subplot_titles=('Equity Curve', 'Regime', 'Drawdown', 'Funding Rate & FNG'))

    # Normalize BTC for comparison
    btc_norm = result['btc_close'] / result['btc_close'].iloc[0] * float(metrics['Initial Capital'].replace('$', '').replace(',', ''))

    # Add regime background shading
    regime_changes = result['regime'].ne(result['regime'].shift())
    regime_starts = result.index[regime_changes]

    for i in range(len(regime_starts)):
        start = regime_starts[i]
        end = regime_starts[i + 1] if i + 1 < len(regime_starts) else result.index[-1]
        r = result.loc[start, 'regime']
        color = regime_color(r)
        fig.add_vrect(x0=start, x1=end, fillcolor=color, layer='below',
                      line_width=0, row=1, col=1)

    # Equity curve
    fig.add_trace(go.Scatter(x=result.index, y=result['equity'],
                             name='Portfolio', line=dict(color='#2196F3', width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=btc_norm,
                             name='BTC Buy & Hold', line=dict(color='#FF9800', width=1, dash='dash')),
                  row=1, col=1)

    # Regime timeline
    regime_num = result['regime'].map({'BULL': 1, 'SIDEWAYS': 0, 'BEAR': -1}).fillna(0)
    colors_regime = result['regime'].map({
        'BULL': '#4CAF50', 'SIDEWAYS': '#FFC107', 'BEAR': '#F44336'
    }).fillna('#9E9E9E')
    fig.add_trace(go.Bar(x=result.index, y=regime_num, marker_color=colors_regime,
                         name='Regime', showlegend=False),
                  row=2, col=1)

    # Drawdown
    peak = result['equity'].expanding().max()
    dd = (result['equity'] - peak) / peak * 100
    fig.add_trace(go.Scatter(x=result.index, y=dd, fill='tozeroy',
                             name='Drawdown %', line=dict(color='#F44336', width=1)),
                  row=3, col=1)

    # Funding Rate & FNG
    fig.add_trace(go.Scatter(x=result.index, y=result['funding_rate'] * 100,
                             name='Funding Rate %', line=dict(color='#9C27B0', width=1)),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=result['fng'],
                             name='Fear & Greed', line=dict(color='#00BCD4', width=1),
                             yaxis='y8'),
                  row=4, col=1)

    fig.update_layout(height=900, template='plotly_dark',
                      legend=dict(orientation='h', y=1.02),
                      margin=dict(l=60, r=20, t=80, b=40))
    fig.update_yaxes(title_text='$', row=1, col=1)
    fig.update_yaxes(title_text='Regime', row=2, col=1)
    fig.update_yaxes(title_text='DD %', row=3, col=1)
    fig.update_yaxes(title_text='FR% / FNG', row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ─── PnL Decomposition ───
    st.markdown("### PnL Decomposition")
    col1, col2 = st.columns(2)

    with col1:
        # Cumulative PnL by layer
        cum_mom = result['mom_pnl'].cumsum()
        cum_mr = result['mr_pnl'].cumsum()
        cum_cash = result['cash_yield'].cumsum()

        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(x=result.index, y=cum_mom, name='Momentum',
                                     stackgroup='one', line=dict(color='#2196F3')))
        fig_pnl.add_trace(go.Scatter(x=result.index, y=cum_mr, name='Mean Reversion',
                                     stackgroup='one', line=dict(color='#FF9800')))
        fig_pnl.add_trace(go.Scatter(x=result.index, y=cum_cash, name='Cash Yield',
                                     stackgroup='one', line=dict(color='#4CAF50')))
        fig_pnl.update_layout(title='Cumulative PnL by Layer', template='plotly_dark',
                              height=400, yaxis_title='$')
        st.plotly_chart(fig_pnl, use_container_width=True)

    with col2:
        # Pie chart of PnL attribution
        total_mom = result['mom_pnl'].sum()
        total_mr = result['mr_pnl'].sum()
        total_cash = result['cash_yield'].sum()

        fig_pie = go.Figure(data=[go.Pie(
            labels=['Momentum', 'Mean Reversion', 'Cash Yield'],
            values=[max(total_mom, 0), max(total_mr, 0), max(total_cash, 0)],
            marker_colors=['#2196F3', '#FF9800', '#4CAF50'],
            textinfo='label+percent'
        )])
        fig_pie.update_layout(title='PnL Attribution', template='plotly_dark', height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ─── Monthly Returns Heatmap ───
    st.markdown("### Monthly Returns")
    monthly = result['equity'].resample('ME').last().pct_change() * 100
    monthly_df = pd.DataFrame({
        'Year': monthly.index.year,
        'Month': monthly.index.month,
        'Return': monthly.values
    }).dropna()

    if not monthly_df.empty:
        pivot = monthly_df.pivot_table(values='Return', index='Year', columns='Month')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index.astype(str),
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate='%{text:.1f}%',
            textfont=dict(size=12),
        ))
        fig_heat.update_layout(title='Monthly Returns (%)', template='plotly_dark',
                               height=300, yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_heat, use_container_width=True)

    # ─── Regime Statistics ───
    st.markdown("### Regime Statistics")
    col1, col2, col3 = st.columns(3)

    for regime, col in zip(['BULL', 'SIDEWAYS', 'BEAR'], [col1, col2, col3]):
        mask = result['regime'] == regime
        if mask.any():
            regime_returns = result.loc[mask, 'equity'].pct_change().dropna()
            avg_ret = regime_returns.mean() * 6 * 365.25 * 100  # annualized
            n_candles = mask.sum()
            pct_time = mask.mean() * 100
            col.markdown(f"**{regime}**")
            col.markdown(f"- Time: {pct_time:.1f}%")
            col.markdown(f"- Candles: {n_candles:,}")
            col.markdown(f"- Ann. Return: {avg_ret:.1f}%")

    # ─── Full Metrics Table ───
    st.markdown("### Full Metrics")
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


if __name__ == '__main__':
    main()
