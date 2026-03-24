"""
Flask Dashboard — Portfolio Multi-Strategy.
Replaces Streamlit (faster, better control).
"""
from flask import Flask, render_template_string
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
app = Flask(__name__)

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quant Committee — Portfolio Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:'Segoe UI',system-ui,sans-serif}
.header{background:#161b22;border-bottom:1px solid #30363d;padding:16px 24px;display:flex;align-items:center;gap:16px}
.header h1{font-size:18px;color:#58a6ff}
.header .tag{background:#238636;color:#fff;padding:2px 8px;border-radius:12px;font-size:12px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;padding:16px 24px}
.metric{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;text-align:center}
.metric .label{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px}
.metric .value{font-size:22px;font-weight:700;margin-top:4px}
.metric .value.green{color:#3fb950}
.metric .value.red{color:#f85149}
.metric .value.blue{color:#58a6ff}
.section{padding:8px 24px}
.section h2{font-size:14px;color:#8b949e;margin:12px 0 8px;text-transform:uppercase;letter-spacing:1px}
.chart{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:4px;margin-bottom:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
.table-wrap{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:8px;color:#8b949e;border-bottom:1px solid #30363d;font-weight:500}
td{padding:8px;border-bottom:1px solid #21262d}
tr:hover{background:#1c2128}
.pos{color:#3fb950}.neg{color:#f85149}
@media(max-width:900px){.grid2,.grid3{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">
  <h1>COMITE QUANT — Portfolio Dashboard</h1>
  <span class="tag">LIVE</span>
  <span style="color:#8b949e;font-size:12px;margin-left:auto">{{ period }}</span>
</div>

<div class="metrics">
  {% for m in top_metrics %}
  <div class="metric">
    <div class="label">{{ m.label }}</div>
    <div class="value {{ m.cls }}">{{ m.value }}</div>
  </div>
  {% endfor %}
</div>

<div class="section">
  <h2>Equity Curves — All Strategies vs Combined</h2>
  <div class="chart">{{ equity_chart | safe }}</div>
</div>

<div class="section grid2">
  <div>
    <h2>Drawdown</h2>
    <div class="chart">{{ dd_chart | safe }}</div>
  </div>
  <div>
    <h2>Correlation Matrix</h2>
    <div class="chart">{{ corr_chart | safe }}</div>
  </div>
</div>

<div class="section">
  <h2>Monthly Returns — Combined Portfolio</h2>
  <div class="chart">{{ monthly_chart | safe }}</div>
</div>

<div class="section grid2">
  <div>
    <h2>Portfolio Weights (Markowitz)</h2>
    <div class="chart">{{ weights_chart | safe }}</div>
  </div>
  <div>
    <h2>Yearly Returns</h2>
    <div class="table-wrap">{{ yearly_table | safe }}</div>
  </div>
</div>

<div class="section">
  <h2>Strategy Comparison</h2>
  <div class="table-wrap">{{ strategy_table | safe }}</div>
</div>
</body>
</html>"""


def load_data():
    data = {}
    for name, fname in [('momentum', 'backtest_result.parquet'),
                         ('contrarian', 'contrarian_result.parquet'),
                         ('pairs', 'pairs_result.parquet'),
                         ('combined', 'portfolio_combined.parquet')]:
        p = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(p):
            data[name] = pd.read_parquet(p)

    opt_path = os.path.join(RESULTS_DIR, 'portfolio_optimization.json')
    if os.path.exists(opt_path):
        with open(opt_path) as f:
            data['optimization'] = json.load(f)

    return data


def compute_stats(equity: pd.Series) -> dict:
    n_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 0.01)
    ret = equity.pct_change().dropna()
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1/n_years) - 1) * 100
    sharpe = ret.mean() / ret.std() * np.sqrt(6*365.25) if ret.std() > 0 else 0
    sortino_d = ret[ret < 0].std()
    sortino = ret.mean() / sortino_d * np.sqrt(6*365.25) if sortino_d > 0 else 0
    peak = equity.expanding().max()
    max_dd = ((equity - peak) / peak).min() * 100
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {'cagr': cagr, 'sharpe': sharpe, 'sortino': sortino,
            'max_dd': max_dd, 'calmar': calmar, 'final': equity.iloc[-1]}


def fig_to_html(fig, height=350):
    fig.update_layout(template='plotly_dark', height=height,
                      margin=dict(l=50, r=20, t=30, b=30),
                      paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                      font=dict(size=11, color='#c9d1d9'),
                      legend=dict(orientation='h', y=-0.15))
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


@app.route('/')
def dashboard():
    data = load_data()
    if not data:
        return "<h1>No data. Run backtests first.</h1>"

    # Compute stats for each strategy
    strats = {}
    colors = {'momentum': '#58a6ff', 'contrarian': '#f0883e', 'pairs': '#a371f7', 'combined': '#3fb950'}
    labels = {'momentum': 'Momentum', 'contrarian': 'Contrarian', 'pairs': 'Pairs', 'combined': 'Combined'}

    for key in ['momentum', 'contrarian', 'pairs', 'combined']:
        if key in data:
            eq = data[key]['equity']
            strats[key] = compute_stats(eq)

    # Top metrics (combined portfolio)
    c = strats.get('combined', strats.get('momentum', {}))
    opt = data.get('optimization', {})
    top_metrics = [
        {'label': 'Combined CAGR', 'value': f"{c.get('cagr',0):.1f}%", 'cls': 'green' if c.get('cagr',0) > 0 else 'red'},
        {'label': 'Sharpe', 'value': f"{c.get('sharpe',0):.2f}", 'cls': 'blue'},
        {'label': 'Sortino', 'value': f"{c.get('sortino',0):.2f}", 'cls': 'blue'},
        {'label': 'Max Drawdown', 'value': f"{c.get('max_dd',0):.1f}%", 'cls': 'red'},
        {'label': 'Calmar', 'value': f"{c.get('calmar',0):.2f}", 'cls': 'blue'},
        {'label': 'Final Equity', 'value': f"${c.get('final',0):,.0f}", 'cls': 'green'},
        {'label': 'Opt Sharpe', 'value': f"{opt.get('sharpe','N/A')}", 'cls': 'blue'},
        {'label': 'Opt Return', 'value': f"{opt.get('expected_return','N/A')}%", 'cls': 'green'},
    ]

    # Period
    if 'combined' in data:
        idx = data['combined'].index
        period = f"{idx[0].strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')}"
    else:
        period = "N/A"

    # === EQUITY CHART ===
    fig_eq = go.Figure()
    for key in ['momentum', 'contrarian', 'pairs', 'combined']:
        if key in data:
            eq = data[key]['equity']
            # Normalize to $10k start
            eq_norm = eq / eq.iloc[0] * 10000
            fig_eq.add_trace(go.Scatter(
                x=eq_norm.index, y=eq_norm, name=labels[key],
                line=dict(color=colors[key], width=3 if key == 'combined' else 1.5)
            ))
    # BTC benchmark
    if 'momentum' in data and 'btc_close' in data['momentum'].columns:
        btc = data['momentum']['btc_close']
        btc_norm = btc / btc.iloc[0] * 10000
        fig_eq.add_trace(go.Scatter(
            x=btc_norm.index, y=btc_norm, name='BTC Buy&Hold',
            line=dict(color='#6e7681', width=1, dash='dot')
        ))
    equity_chart = fig_to_html(fig_eq, 420)

    # === DRAWDOWN CHART ===
    fig_dd = go.Figure()
    for key in ['momentum', 'contrarian', 'pairs', 'combined']:
        if key in data:
            eq = data[key]['equity']
            peak = eq.expanding().max()
            dd = (eq - peak) / peak * 100
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd, name=labels[key], fill='tozeroy' if key == 'combined' else None,
                line=dict(color=colors[key], width=2 if key == 'combined' else 1)
            ))
    dd_chart = fig_to_html(fig_dd, 300)

    # === CORRELATION CHART ===
    corr_data = opt.get('correlation_matrix', {})
    if corr_data:
        names = list(corr_data.keys())
        z = [[corr_data[r].get(c, 0) for c in names] for r in names]
        fig_corr = go.Figure(data=go.Heatmap(
            z=z, x=names, y=names, colorscale='RdBu_r', zmid=0,
            text=np.round(z, 2), texttemplate='%{text}', textfont=dict(size=14)
        ))
    else:
        fig_corr = go.Figure()
        fig_corr.add_annotation(text="Run portfolio_optimizer.py first", showarrow=False)
    corr_chart = fig_to_html(fig_corr, 300)

    # === MONTHLY RETURNS HEATMAP ===
    fig_monthly = go.Figure()
    if 'combined' in data:
        eq = data['combined']['equity']
        monthly = eq.resample('ME').last().pct_change().dropna() * 100
        mdf = pd.DataFrame({'Year': monthly.index.year, 'Month': monthly.index.month, 'Return': monthly.values}).dropna()
        if not mdf.empty:
            pivot = mdf.pivot_table(values='Return', index='Year', columns='Month')
            month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            pivot.columns = [month_names[int(c)-1] for c in pivot.columns]
            fig_monthly = go.Figure(data=go.Heatmap(
                z=pivot.values, x=pivot.columns, y=pivot.index.astype(str),
                colorscale='RdYlGn', zmid=0,
                text=np.round(pivot.values, 1), texttemplate='%{text:.1f}%',
                textfont=dict(size=11)
            ))
            fig_monthly.update_yaxes(autorange='reversed')
    monthly_chart = fig_to_html(fig_monthly, 250)

    # === WEIGHTS PIE ===
    weights = opt.get('weights', {})
    if weights:
        fig_w = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            marker_colors=[colors.get(k.lower(), '#888') for k in weights.keys()],
            textinfo='label+percent', textfont_size=13
        )])
    else:
        fig_w = go.Figure()
        fig_w.add_annotation(text="Run optimizer first", showarrow=False)
    weights_chart = fig_to_html(fig_w, 300)

    # === YEARLY RETURNS TABLE ===
    rows = []
    if 'combined' in data:
        eq = data['combined']['equity']
        for yr in sorted(eq.index.year.unique()):
            yr_eq = eq[eq.index.year == yr]
            if len(yr_eq) > 1:
                ret = (yr_eq.iloc[-1] / yr_eq.iloc[0] - 1) * 100
                cls = 'pos' if ret > 0 else 'neg'
                rows.append(f"<tr><td>{yr}</td><td class='{cls}'>{ret:+.1f}%</td></tr>")
    yearly_table = f"<table><tr><th>Year</th><th>Return</th></tr>{''.join(rows)}</table>"

    # === STRATEGY COMPARISON TABLE ===
    strat_rows = []
    for key, label in labels.items():
        if key in strats:
            s = strats[key]
            c_cls = 'pos' if s['cagr'] > 0 else 'neg'
            strat_rows.append(
                f"<tr><td><b>{label}</b></td>"
                f"<td class='{c_cls}'>{s['cagr']:.1f}%</td>"
                f"<td>{s['sharpe']:.2f}</td>"
                f"<td>{s['sortino']:.2f}</td>"
                f"<td class='neg'>{s['max_dd']:.1f}%</td>"
                f"<td>{s['calmar']:.2f}</td>"
                f"<td>${s['final']:,.0f}</td></tr>"
            )
    strategy_table = (
        "<table><tr><th>Strategy</th><th>CAGR</th><th>Sharpe</th>"
        "<th>Sortino</th><th>Max DD</th><th>Calmar</th><th>Final ($10k)</th></tr>"
        + ''.join(strat_rows) + "</table>"
    )

    return render_template_string(TEMPLATE,
        period=period, top_metrics=top_metrics,
        equity_chart=equity_chart, dd_chart=dd_chart,
        corr_chart=corr_chart, monthly_chart=monthly_chart,
        weights_chart=weights_chart, yearly_table=yearly_table,
        strategy_table=strategy_table
    )


if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')
