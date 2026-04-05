"""
Paper Trading Dashboard — Live monitoring of CatBoost Short + Long paper test.
Self-contained HTML dashboard served via Flask. Auto-refreshes every 5 minutes.

Usage:
    python paper_dashboard.py                # Serve on http://localhost:5001
    python paper_dashboard.py --export       # Export static HTML to results/
"""
from flask import Flask, Response
import pandas as pd
import numpy as np
import json
import os
import argparse
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
PAPER_DIR = PROJECT_DIR / 'paper_trading'
RESULTS_DIR = PROJECT_DIR / 'results'

app = Flask(__name__)


def load_json(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def build_html():
    state = load_json(PAPER_DIR / 'state.json') or {
        'capital': 10000, 'equity': 5000, 'equity_short': 5000,
        'equity_long': 5000, 'positions': [], 'positions_long': [],
        'last_rebalance': None, 'last_rebalance_long': None,
        'total_trades': 0, 'total_trades_long': 0, 'started_at': None,
        'regime_skip_count': 0,
    }
    trades = load_json(PAPER_DIR / 'trades.json') or []
    trades_long = load_json(PAPER_DIR / 'trades_long.json') or []
    equity_hist = load_json(PAPER_DIR / 'equity.json') or []
    equity_hist_long = load_json(PAPER_DIR / 'equity_long.json') or []
    model_meta = load_json(PAPER_DIR / 'model_meta.json') or {}
    model_meta_long = load_json(PAPER_DIR / 'model_meta_long.json') or {}
    feat_imp = load_json(PAPER_DIR / 'feature_importance.json') or {}
    feat_imp_long = load_json(PAPER_DIR / 'feature_importance_long.json') or {}
    log_lines = []
    log_path = PAPER_DIR / 'log.jsonl'
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                try:
                    log_lines.append(json.loads(line.strip()))
                except Exception:
                    pass

    # Load ablation for comparison
    ablation_path = RESULTS_DIR / 'ablation_results.csv'
    ablation = None
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)

    initial = state.get('capital', 10000) or 10000
    initial_short = initial * 0.5
    initial_long = initial * 0.5
    equity = state.get('equity', state.get('equity_short', initial_short))
    equity_long = state.get('equity_long', initial_long)
    equity_total = equity + equity_long
    total_return = (equity / initial_short - 1) * 100
    total_return_long = (equity_long / initial_long - 1) * 100
    total_return_combined = (equity_total / initial - 1) * 100
    positions = state.get('positions', [])
    positions_long = state.get('positions_long', [])
    cumulative_yield = state.get('cumulative_yield', 0.0)
    cumulative_yield_long = state.get('cumulative_yield_long', 0.0)
    regime_skip_count = state.get('regime_skip_count', 0)

    # SHORT trade stats
    n_trades = len(trades)
    wins = sum(1 for t in trades if t.get('return', 0) > 0)
    win_rate = (wins / n_trades * 100) if n_trades > 0 else 0
    avg_ret = np.mean([t.get('return', 0) for t in trades]) * 100 if trades else 0
    avg_win = np.mean([t['return'] for t in trades if t.get('return', 0) > 0]) * 100 if wins > 0 else 0
    avg_loss = np.mean([t['return'] for t in trades if t.get('return', 0) <= 0]) * 100 if (n_trades - wins) > 0 else 0
    best_trade = max([t.get('return', 0) for t in trades], default=0) * 100
    worst_trade = min([t.get('return', 0) for t in trades], default=0) * 100

    # LONG trade stats
    n_trades_long = len(trades_long)
    wins_long = sum(1 for t in trades_long if t.get('return', 0) > 0)
    win_rate_long = (wins_long / n_trades_long * 100) if n_trades_long > 0 else 0
    avg_ret_long = np.mean([t.get('return', 0) for t in trades_long]) * 100 if trades_long else 0
    avg_win_long = np.mean([t['return'] for t in trades_long if t.get('return', 0) > 0]) * 100 if wins_long > 0 else 0
    avg_loss_long = np.mean([t['return'] for t in trades_long if t.get('return', 0) <= 0]) * 100 if (n_trades_long - wins_long) > 0 else 0
    best_trade_long = max([t.get('return', 0) for t in trades_long], default=0) * 100
    worst_trade_long = min([t.get('return', 0) for t in trades_long], default=0) * 100

    # SHORT equity series for chart
    eq_dates = []
    eq_values = []
    eq_mark = []
    for snap in equity_hist:
        eq_dates.append(snap['timestamp'][:19])
        eq_values.append(snap['equity'])
        eq_mark.append(snap.get('mark_equity', snap['equity']))

    # LONG equity series for chart
    eq_dates_long = []
    eq_values_long = []
    eq_mark_long = []
    for snap in equity_hist_long:
        eq_dates_long.append(snap['timestamp'][:19])
        eq_values_long.append(snap['equity'])
        eq_mark_long.append(snap.get('mark_equity', snap['equity']))

    # Compute running Sharpe from equity snapshots (SHORT)
    running_sharpe = 0
    if len(eq_values) > 5:
        eq_s = pd.Series(eq_values)
        rets = eq_s.pct_change().dropna()
        if rets.std() > 0:
            snapshots_per_year = 365 * 6
            running_sharpe = rets.mean() / rets.std() * np.sqrt(snapshots_per_year)

    # Compute running Sharpe (LONG)
    running_sharpe_long = 0
    if len(eq_values_long) > 5:
        eq_s_l = pd.Series(eq_values_long)
        rets_l = eq_s_l.pct_change().dropna()
        if rets_l.std() > 0:
            snapshots_per_year = 365 * 6
            running_sharpe_long = rets_l.mean() / rets_l.std() * np.sqrt(snapshots_per_year)

    # Max drawdown from equity history (SHORT)
    max_dd = 0
    if eq_values:
        peak = eq_values[0]
        for v in eq_values:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd

    # Max drawdown (LONG)
    max_dd_long = 0
    if eq_values_long:
        peak_l = eq_values_long[0]
        for v in eq_values_long:
            if v > peak_l:
                peak_l = v
            dd = (v - peak_l) / peak_l * 100
            if dd < max_dd_long:
                max_dd_long = dd

    # Days running
    started = state.get('started_at')
    days_running = 0
    if started:
        try:
            days_running = (datetime.now(timezone.utc) - pd.Timestamp(started)).days
        except Exception:
            pass

    # Unrealized PnL (SHORT)
    n_at_entry = state.get('n_positions_at_entry', 5)
    total_unreal = sum(p.get('unrealized_pnl', 0) for p in positions) * 100 / n_at_entry if positions else 0

    # Unrealized PnL (LONG)
    n_at_entry_long = state.get('n_positions_at_entry_long', 5)
    total_unreal_long = sum(p.get('unrealized_pnl', 0) for p in positions_long) * 100 / n_at_entry_long if positions_long else 0

    # ─── Build SHORT positions table rows ───
    pos_rows = ""
    for p in positions:
        pnl = p.get('unrealized_pnl', 0) * 100
        pnl_cls = 'pos' if pnl > 0 else 'neg' if pnl < 0 else ''
        days = p.get('days_held', 0)
        stop_dist = 0
        if p.get('trough') and p.get('current_price'):
            stop_level = p['trough'] * (1 + 0.15)
            stop_dist = (stop_level / p['current_price'] - 1) * 100
        funding = p.get('funding_rate', 0) * 100
        pos_rows += f"""<tr>
            <td><b>{p['symbol']}</b></td>
            <td class="mono">${p['entry_price']:,.4f}</td>
            <td class="mono">${p.get('current_price', 0):,.4f}</td>
            <td class="mono {pnl_cls}">{pnl:+.2f}%</td>
            <td>{days}d / {5}d</td>
            <td class="mono">{stop_dist:+.1f}%</td>
            <td class="mono">{funding:.4f}%</td>
            <td class="mono">{p.get('score', 0):.3f}</td>
        </tr>"""

    if not pos_rows:
        pos_rows = '<tr><td colspan="8" style="text-align:center;color:#8b949e">No open short positions</td></tr>'

    # ─── Build LONG positions table rows ───
    pos_rows_long = ""
    for p in positions_long:
        pnl = p.get('unrealized_pnl', 0) * 100
        pnl_cls = 'pos' if pnl > 0 else 'neg' if pnl < 0 else ''
        days = p.get('days_held', 0)
        stop_dist = 0
        if p.get('peak') and p.get('current_price') and p['peak'] > 0:
            trail_stop = p['peak'] * (1 - 0.05)
            stop_dist = (p['current_price'] / trail_stop - 1) * 100
        hard_stop_dist = 0
        if p.get('entry_price') and p.get('current_price'):
            hard_level = p['entry_price'] * (1 - 0.08)
            hard_stop_dist = (p['current_price'] / hard_level - 1) * 100
        funding = p.get('funding_rate', 0) * 100
        pos_rows_long += f"""<tr>
            <td><b>{p['symbol']}</b></td>
            <td class="mono">${p['entry_price']:,.4f}</td>
            <td class="mono">${p.get('current_price', 0):,.4f}</td>
            <td class="mono {pnl_cls}">{pnl:+.2f}%</td>
            <td>{days}d / {2}d</td>
            <td class="mono">{stop_dist:+.1f}%</td>
            <td class="mono">{funding:.4f}%</td>
            <td class="mono">{p.get('score', 0):.3f}</td>
        </tr>"""

    if not pos_rows_long:
        pos_rows_long = '<tr><td colspan="8" style="text-align:center;color:#8b949e">No open long positions</td></tr>'

    # ─── Build SHORT trades table rows ───
    trade_rows = ""
    for t in reversed(trades[-30:]):
        ret = t.get('return', 0) * 100
        ret_cls = 'pos' if ret > 0 else 'neg'
        entry_t = t.get('entry_time', '')[:10]
        exit_t = t.get('exit_time', '')[:10]
        trade_rows += f"""<tr>
            <td><b>{t['symbol']}</b></td>
            <td>{entry_t}</td>
            <td class="mono">${t['entry_price']:,.4f}</td>
            <td>{exit_t}</td>
            <td class="mono">${t.get('exit_price', 0):,.4f}</td>
            <td class="mono {ret_cls}">{ret:+.2f}%</td>
            <td><span class="tag tag-{t.get('exit_reason', 'unknown')}">{t.get('exit_reason', '?')}</span></td>
        </tr>"""

    if not trade_rows:
        trade_rows = '<tr><td colspan="7" style="text-align:center;color:#8b949e">No short trades yet</td></tr>'

    # ─── Build LONG trades table rows ───
    trade_rows_long = ""
    for t in reversed(trades_long[-30:]):
        ret = t.get('return', 0) * 100
        ret_cls = 'pos' if ret > 0 else 'neg'
        entry_t = t.get('entry_time', '')[:10]
        exit_t = t.get('exit_time', '')[:10]
        trade_rows_long += f"""<tr>
            <td><b>{t['symbol']}</b></td>
            <td>{entry_t}</td>
            <td class="mono">${t['entry_price']:,.4f}</td>
            <td>{exit_t}</td>
            <td class="mono">${t.get('exit_price', 0):,.4f}</td>
            <td class="mono {ret_cls}">{ret:+.2f}%</td>
            <td><span class="tag tag-{t.get('exit_reason', 'unknown')}">{t.get('exit_reason', '?')}</span></td>
        </tr>"""

    if not trade_rows_long:
        trade_rows_long = '<tr><td colspan="7" style="text-align:center;color:#8b949e">No long trades yet</td></tr>'

    # ─── Build log rows ───
    log_rows = ""
    for entry in reversed(log_lines[-50:]):
        ts = entry.get('timestamp', '')[:19]
        etype = entry.get('type', '')
        msg = entry.get('message', '')
        tag_cls = {'open': 'open', 'close': 'close', 'stop': 'stop',
                   'rebalance': 'rebalance', 'train': 'train', 'error': 'error'}.get(etype, '')
        log_rows += f"""<tr>
            <td class="mono" style="color:#8b949e">{ts}</td>
            <td><span class="tag tag-{tag_cls}">{etype}</span></td>
            <td>{msg}</td>
        </tr>"""

    # ─── Feature importance bars ───
    feat_bars = ""
    if feat_imp:
        sorted_feats = sorted(feat_imp.items(), key=lambda x: -x[1])[:15]
        max_imp = sorted_feats[0][1] if sorted_feats else 1
        for fname, fval in sorted_feats:
            width = (fval / max_imp) * 100
            cat_cls = 'liq' if fname in ('amihud', 'spread_28', 'turnover_28') else \
                      'mom' if 'mom' in fname else \
                      'poly' if 'poly' in fname else \
                      'vol' if 'rvol' in fname or 'vol' in fname else 'tech'
            feat_bars += f"""<div class="feat-row">
                <span class="feat-name">{fname}</span>
                <div class="feat-bar-bg"><div class="feat-bar feat-{cat_cls}" style="width:{width:.0f}%"></div></div>
                <span class="feat-val">{fval*100:.1f}%</span>
            </div>"""

    # ─── Ablation comparison rows ───
    ablation_rows = ""
    if ablation is not None:
        for _, row in ablation.head(8).iterrows():
            s = row.get('sharpe', 0)
            c = row.get('cagr', 0)
            ablation_rows += f"""<tr>
                <td>{row['name']}</td>
                <td class="mono">{s:.2f}</td>
                <td class="mono">{c:.0f}%</td>
                <td class="mono neg">{row.get('max_dd', 0):.0f}%</td>
            </tr>"""

    # ─── Model info (SHORT) ───
    model_trained = model_meta.get('trained_at', 'Never')[:10]
    model_rows = model_meta.get('n_rows', 0)
    model_n = model_meta.get('n_models', 0)
    model_period = f"{model_meta.get('train_start', '?')} to {model_meta.get('train_end', '?')}"

    # ─── Model info (LONG) ───
    model_trained_long = model_meta_long.get('trained_at', 'Never')[:10] if model_meta_long else 'Never'
    model_rows_long = model_meta_long.get('n_rows', 0) if model_meta_long else 0
    model_n_long = model_meta_long.get('n_models', 0) if model_meta_long else 0
    model_period_long = f"{model_meta_long.get('train_start', '?')} to {model_meta_long.get('train_end', '?')}" if model_meta_long else '?'

    # Status indicator
    if positions:
        status_cls = 'live'
        status_text = 'ACTIVE'
    elif state.get('last_rebalance'):
        status_cls = 'waiting'
        status_text = 'WAITING'
    else:
        status_cls = 'idle'
        status_text = 'NOT STARTED'

    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>Paper Trading — CatBoost Short + Long</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e; --green: #3fb950;
    --red: #f85149; --blue: #58a6ff; --orange: #f0883e;
    --purple: #a371f7; --yellow: #d29922;
}}
* {{ margin:0; padding:0; box-sizing:border-box }}
body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,-apple-system,sans-serif; font-size:14px }}
.mono {{ font-family:'Cascadia Code','Fira Code','Consolas',monospace }}

/* Header */
.header {{ background:var(--surface); border-bottom:1px solid var(--border); padding:16px 24px; display:flex; align-items:center; gap:16px; flex-wrap:wrap }}
.header h1 {{ font-size:18px; color:var(--blue) }}
.status {{ padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px }}
.status.live {{ background:#238636; color:#fff }}
.status.waiting {{ background:var(--yellow); color:#000 }}
.status.idle {{ background:var(--border); color:var(--muted) }}
.header-right {{ margin-left:auto; font-size:12px; color:var(--muted) }}

/* Metrics Grid */
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; padding:16px 24px }}
.metric {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px; text-align:center }}
.metric .label {{ font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px }}
.metric .value {{ font-size:22px; font-weight:700; margin-top:4px }}
.pos {{ color:var(--green) }} .neg {{ color:var(--red) }}

/* Sections */
.section {{ padding:8px 24px }}
.section h2 {{ font-size:13px; color:var(--muted); margin:14px 0 8px; text-transform:uppercase; letter-spacing:1px }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:12px }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:12px }}
.grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px }}

/* Tables */
table {{ width:100%; border-collapse:collapse; font-size:13px }}
th {{ text-align:left; padding:8px; color:var(--muted); border-bottom:1px solid var(--border); font-weight:500; white-space:nowrap }}
td {{ padding:8px; border-bottom:1px solid #21262d; white-space:nowrap }}
tr:hover {{ background:#1c2128 }}

/* Tags */
.tag {{ padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600 }}
.tag-rebalance {{ background:#1f6feb33; color:var(--blue) }}
.tag-open {{ background:#23863633; color:var(--green) }}
.tag-close,.tag-holding_expired {{ background:#f0883e33; color:var(--orange) }}
.tag-stop,.tag-trailing_stop {{ background:#f8514933; color:var(--red) }}
.tag-train {{ background:#a371f733; color:var(--purple) }}
.tag-error {{ background:#f8514933; color:var(--red) }}

/* Feature importance bars */
.feat-row {{ display:flex; align-items:center; gap:8px; margin:3px 0 }}
.feat-name {{ width:140px; font-size:12px; color:var(--muted); text-align:right; font-family:monospace }}
.feat-bar-bg {{ flex:1; height:16px; background:#21262d; border-radius:3px; overflow:hidden }}
.feat-bar {{ height:100%; border-radius:3px }}
.feat-liq {{ background:var(--orange) }}
.feat-mom {{ background:var(--blue) }}
.feat-poly {{ background:var(--red) }}
.feat-vol {{ background:var(--green) }}
.feat-tech {{ background:var(--purple) }}
.feat-val {{ width:45px; font-size:11px; color:var(--muted); font-family:monospace }}

/* Progress bar */
.progress-wrap {{ background:#21262d; border-radius:6px; height:8px; margin:8px 0; overflow:hidden }}
.progress-bar {{ height:100%; background:var(--blue); border-radius:6px; transition:width 0.3s }}

/* Responsive */
@media(max-width:900px) {{ .grid2,.grid3 {{ grid-template-columns:1fr }} }}
</style>
</head>
<body>

<div class="header">
    <h1>PAPER TRADING — CatBoost Short + Long</h1>
    <span class="status {status_cls}">{status_text}</span>
    <span class="tag" style="background:var(--surface);color:var(--muted);border:1px solid var(--border)">Short Baseline 1.36</span>
    <span class="header-right">Updated: {now_str} &middot; Auto-refresh 5min</span>
</div>

<!-- COMBINED TOP METRICS -->
<div class="metrics">
    <div class="metric">
        <div class="label">Total Equity</div>
        <div class="value {'pos' if equity_total >= initial else 'neg'}">${equity_total:,.2f}</div>
    </div>
    <div class="metric">
        <div class="label">Combined Return</div>
        <div class="value {'pos' if total_return_combined >= 0 else 'neg'}">{total_return_combined:+.2f}%</div>
    </div>
    <div class="metric">
        <div class="label">Short Equity</div>
        <div class="value {'pos' if equity >= initial_short else 'neg'}">${equity:,.2f}</div>
    </div>
    <div class="metric">
        <div class="label">Short Return</div>
        <div class="value {'pos' if total_return >= 0 else 'neg'}">{total_return:+.2f}%</div>
    </div>
    <div class="metric">
        <div class="label">Long Equity</div>
        <div class="value {'pos' if equity_long >= initial_long else 'neg'}">${equity_long:,.2f}</div>
    </div>
    <div class="metric">
        <div class="label">Long Return</div>
        <div class="value {'pos' if total_return_long >= 0 else 'neg'}">{total_return_long:+.2f}%</div>
    </div>
    <div class="metric">
        <div class="label">Days Running</div>
        <div class="value" style="color:var(--blue)">{days_running}</div>
    </div>
    <div class="metric">
        <div class="label">Regime Skips</div>
        <div class="value" style="color:var(--orange)">{regime_skip_count}</div>
    </div>
    <div class="metric">
        <div class="label">Stable Yield</div>
        <div class="value pos">${cumulative_yield + cumulative_yield_long:,.2f}</div>
    </div>
</div>

<!-- 30-DAY PROGRESS BAR -->
<div class="section">
    <div class="card" style="padding:12px 16px">
        <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--muted)">
            <span>Paper test progress: {days_running}/30 days</span>
            <span>{'READY FOR LIVE REVIEW' if days_running >= 30 else f'{30-days_running} days remaining'}</span>
        </div>
        <div class="progress-wrap">
            <div class="progress-bar" style="width:{min(days_running/30*100, 100):.0f}%"></div>
        </div>
    </div>
</div>

<!-- EQUITY CHARTS -->
<div class="section grid2">
    <div>
        <h2>Short Equity Curve</h2>
        <div class="card" id="equity-chart-short" style="height:300px"></div>
    </div>
    <div>
        <h2>Long Equity Curve</h2>
        <div class="card" id="equity-chart-long" style="height:300px"></div>
    </div>
</div>

<!-- SHORT: POSITIONS + TRADE STATS -->
<div class="section">
    <h2 style="color:var(--red)">SHORT STRATEGY</h2>
</div>
<div class="section grid2">
    <div>
        <h2>Short Positions ({len(positions)})</h2>
        <div class="card" style="overflow-x:auto">
            <table>
                <tr>
                    <th>Symbol</th><th>Entry</th><th>Current</th><th>PnL</th>
                    <th>Held</th><th>To Stop</th><th>Funding</th><th>Score</th>
                </tr>
                {pos_rows}
            </table>
        </div>
    </div>
    <div>
        <h2>Short Trade Statistics</h2>
        <div class="card">
            <table>
                <tr><td style="color:var(--muted)">Total Trades</td><td class="mono"><b>{n_trades}</b></td></tr>
                <tr><td style="color:var(--muted)">Wins / Losses</td><td class="mono"><span class="pos">{wins}</span> / <span class="neg">{n_trades - wins}</span></td></tr>
                <tr><td style="color:var(--muted)">Win Rate</td><td class="mono">{win_rate:.1f}%</td></tr>
                <tr><td style="color:var(--muted)">Avg Win</td><td class="mono pos">{avg_win:+.2f}%</td></tr>
                <tr><td style="color:var(--muted)">Avg Loss</td><td class="mono neg">{avg_loss:+.2f}%</td></tr>
                <tr><td style="color:var(--muted)">Best Trade</td><td class="mono pos">{best_trade:+.2f}%</td></tr>
                <tr><td style="color:var(--muted)">Worst Trade</td><td class="mono neg">{worst_trade:+.2f}%</td></tr>
                <tr><td style="color:var(--muted)">Running Sharpe</td><td class="mono" style="color:var(--blue)">{running_sharpe:.2f}</td></tr>
                <tr><td style="color:var(--muted)">Unrealized</td><td class="mono {'pos' if total_unreal >= 0 else 'neg'}">{total_unreal:+.2f}%</td></tr>
                <tr><td style="color:var(--muted)">Max Drawdown</td><td class="mono neg">{max_dd:.1f}%</td></tr>
            </table>
        </div>
    </div>
</div>

<!-- LONG: POSITIONS + TRADE STATS -->
<div class="section">
    <h2 style="color:var(--green)">LONG STRATEGY</h2>
</div>
<div class="section grid2">
    <div>
        <h2>Long Positions ({len(positions_long)})</h2>
        <div class="card" style="overflow-x:auto">
            <table>
                <tr>
                    <th>Symbol</th><th>Entry</th><th>Current</th><th>PnL</th>
                    <th>Held</th><th>To Trail</th><th>Funding</th><th>Score</th>
                </tr>
                {pos_rows_long}
            </table>
        </div>
    </div>
    <div>
        <h2>Long Trade Statistics</h2>
        <div class="card">
            <table>
                <tr><td style="color:var(--muted)">Total Trades</td><td class="mono"><b>{n_trades_long}</b></td></tr>
                <tr><td style="color:var(--muted)">Wins / Losses</td><td class="mono"><span class="pos">{wins_long}</span> / <span class="neg">{n_trades_long - wins_long}</span></td></tr>
                <tr><td style="color:var(--muted)">Win Rate</td><td class="mono">{win_rate_long:.1f}%</td></tr>
                <tr><td style="color:var(--muted)">Avg Win</td><td class="mono pos">{avg_win_long:+.2f}%</td></tr>
                <tr><td style="color:var(--muted)">Avg Loss</td><td class="mono neg">{avg_loss_long:+.2f}%</td></tr>
                <tr><td style="color:var(--muted)">Best Trade</td><td class="mono pos">{best_trade_long:+.2f}%</td></tr>
                <tr><td style="color:var(--muted)">Worst Trade</td><td class="mono neg">{worst_trade_long:+.2f}%</td></tr>
                <tr><td style="color:var(--muted)">Running Sharpe</td><td class="mono" style="color:var(--blue)">{running_sharpe_long:.2f}</td></tr>
                <tr><td style="color:var(--muted)">Unrealized</td><td class="mono {'pos' if total_unreal_long >= 0 else 'neg'}">{total_unreal_long:+.2f}%</td></tr>
                <tr><td style="color:var(--muted)">Max Drawdown</td><td class="mono neg">{max_dd_long:.1f}%</td></tr>
                <tr><td style="color:var(--muted)">Regime Skips</td><td class="mono" style="color:var(--orange)">{regime_skip_count}</td></tr>
            </table>
        </div>
    </div>
</div>

<!-- MODEL INFO -->
<div class="section grid2">
    <div>
        <h2>Short Model Info</h2>
        <div class="card">
            <table>
                <tr><td style="color:var(--muted)">Target</td><td class="mono">fwd_min (losers)</td></tr>
                <tr><td style="color:var(--muted)">Trained</td><td class="mono">{model_trained}</td></tr>
                <tr><td style="color:var(--muted)">Training Period</td><td class="mono">{model_period}</td></tr>
                <tr><td style="color:var(--muted)">Ensemble Size</td><td class="mono">{model_n} models</td></tr>
                <tr><td style="color:var(--muted)">Training Rows</td><td class="mono">{model_rows:,}</td></tr>
                <tr><td style="color:var(--muted)">Loss Function</td><td class="mono">YetiRank (pairwise)</td></tr>
                <tr><td style="color:var(--muted)">Rebalance</td><td class="mono">Every {5} days</td></tr>
                <tr><td style="color:var(--muted)">Trailing Stop</td><td class="mono">{15}% (trough)</td></tr>
            </table>
        </div>
    </div>
    <div>
        <h2>Long Model Info</h2>
        <div class="card">
            <table>
                <tr><td style="color:var(--muted)">Target</td><td class="mono">fwd_max (winners)</td></tr>
                <tr><td style="color:var(--muted)">Trained</td><td class="mono">{model_trained_long}</td></tr>
                <tr><td style="color:var(--muted)">Training Period</td><td class="mono">{model_period_long}</td></tr>
                <tr><td style="color:var(--muted)">Ensemble Size</td><td class="mono">{model_n_long} models</td></tr>
                <tr><td style="color:var(--muted)">Training Rows</td><td class="mono">{model_rows_long:,}</td></tr>
                <tr><td style="color:var(--muted)">Loss Function</td><td class="mono">YetiRank (pairwise)</td></tr>
                <tr><td style="color:var(--muted)">Rebalance</td><td class="mono">Every {5} days</td></tr>
                <tr><td style="color:var(--muted)">Trail/Hard Stop</td><td class="mono">5% / 8%</td></tr>
                <tr><td style="color:var(--muted)">Regime Filter</td><td class="mono">med 7d ret &gt; 0</td></tr>
                <tr><td style="color:var(--muted)">Holding</td><td class="mono">2 days max</td></tr>
            </table>
        </div>
    </div>
</div>

<!-- FEATURE IMPORTANCE -->
<div class="section">
    <h2>Feature Importance (Short Model)</h2>
    <div class="card">
        {feat_bars if feat_bars else '<p style="color:var(--muted);text-align:center">Train model first</p>'}
        <div style="margin-top:10px;font-size:11px;color:var(--muted)">
            <span style="color:var(--orange)">&#9632;</span> Liquidity &nbsp;
            <span style="color:var(--blue)">&#9632;</span> Momentum &nbsp;
            <span style="color:var(--red)">&#9632;</span> Polynomial &nbsp;
            <span style="color:var(--green)">&#9632;</span> Volatility &nbsp;
            <span style="color:var(--purple)">&#9632;</span> Technical
        </div>
    </div>
</div>

<!-- BACKTEST COMPARISON -->
<div class="section">
    <h2>Backtest Baseline Comparison (Ablation Study)</h2>
    <div class="card" style="overflow-x:auto">
        <table>
            <tr><th>Strategy</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th></tr>
            <tr style="background:#23863622">
                <td><b>PAPER TRADING (live)</b></td>
                <td class="mono"><b>{running_sharpe:.2f}</b></td>
                <td class="mono"><b>{total_return:+.1f}% ({days_running}d)</b></td>
                <td class="mono neg"><b>{max_dd:.1f}%</b></td>
            </tr>
            {ablation_rows}
        </table>
        <p style="margin-top:8px;font-size:11px;color:var(--muted)">
            Expected live performance: 50-60% of backtest (Sharpe ~0.7-0.8). Backtest baseline: Sharpe 1.36.
        </p>
    </div>
</div>

<!-- SHORT TRADE HISTORY -->
<div class="section">
    <h2>Short Trade History (last 30)</h2>
    <div class="card" style="overflow-x:auto;max-height:400px;overflow-y:auto">
        <table>
            <tr><th>Symbol</th><th>Entry Date</th><th>Entry Price</th><th>Exit Date</th><th>Exit Price</th><th>Return</th><th>Reason</th></tr>
            {trade_rows}
        </table>
    </div>
</div>

<!-- LONG TRADE HISTORY -->
<div class="section">
    <h2>Long Trade History (last 30)</h2>
    <div class="card" style="overflow-x:auto;max-height:400px;overflow-y:auto">
        <table>
            <tr><th>Symbol</th><th>Entry Date</th><th>Entry Price</th><th>Exit Date</th><th>Exit Price</th><th>Return</th><th>Reason</th></tr>
            {trade_rows_long}
        </table>
    </div>
</div>

<!-- EVENT LOG -->
<div class="section">
    <h2>Event Log (last 50)</h2>
    <div class="card" style="overflow-x:auto;max-height:350px;overflow-y:auto">
        <table>
            <tr><th>Timestamp</th><th>Type</th><th>Message</th></tr>
            {log_rows if log_rows else '<tr><td colspan="3" style="text-align:center;color:var(--muted)">No events yet</td></tr>'}
        </table>
    </div>
</div>

<!-- GO/NO-GO CRITERIA -->
<div class="section">
    <h2>Go-Live Criteria (30-day checklist)</h2>
    <div class="card">
        <table>
            <tr>
                <td>{'&#9989;' if days_running >= 30 else '&#11036;'}</td>
                <td>30 days of paper trading completed</td>
                <td class="mono" style="color:var(--muted)">{days_running}/30 days</td>
            </tr>
            <tr>
                <td>{'&#9989;' if n_trades >= 10 else '&#11036;'}</td>
                <td>Minimum 10 trades executed</td>
                <td class="mono" style="color:var(--muted)">{n_trades}/10 trades</td>
            </tr>
            <tr>
                <td>{'&#9989;' if running_sharpe > 0.5 else '&#11036;'}</td>
                <td>Running Sharpe > 0.50 (50% of backtest baseline)</td>
                <td class="mono" style="color:var(--muted)">{running_sharpe:.2f}</td>
            </tr>
            <tr>
                <td>{'&#9989;' if max_dd > -30 else '&#11036;'}</td>
                <td>Max drawdown < 30%</td>
                <td class="mono" style="color:var(--muted)">{max_dd:.1f}%</td>
            </tr>
            <tr>
                <td>{'&#9989;' if win_rate > 40 else '&#11036;'}</td>
                <td>Win rate > 40%</td>
                <td class="mono" style="color:var(--muted)">{win_rate:.0f}%</td>
            </tr>
            <tr>
                <td>{'&#9989;' if total_return > -10 else '&#11036;'}</td>
                <td>Total return > -10% (not catastrophic)</td>
                <td class="mono" style="color:var(--muted)">{total_return:+.1f}%</td>
            </tr>
        </table>
    </div>
</div>

<div style="text-align:center;padding:20px;color:var(--muted);font-size:11px">
    CatBoost Short + Long Paper Trading &middot; Comite Quant &middot; Capital: $10k (50/50 split) &middot;
    <a href="/" style="color:var(--blue)">Refresh</a>
</div>

<!-- EQUITY CHART SCRIPTS -->
<script>
// SHORT equity chart
const dates_s = {json.dumps(eq_dates)};
const equity_s = {json.dumps(eq_values)};
const mark_s = {json.dumps(eq_mark)};
const initial_s = {initial_short};

if (dates_s.length > 0) {{
    const traces_s = [
        {{
            x: dates_s, y: equity_s, name: 'Realized Equity',
            line: {{ color: '#f85149', width: 2.5 }},
            fill: 'tozeroy', fillcolor: 'rgba(248,81,73,0.05)'
        }},
        {{
            x: dates_s, y: mark_s, name: 'Mark-to-Market',
            line: {{ color: '#58a6ff', width: 1.5, dash: 'dot' }}
        }},
        {{
            x: [dates_s[0], dates_s[dates_s.length-1]], y: [initial_s, initial_s],
            name: 'Initial Capital', line: {{ color: '#8b949e', width: 1, dash: 'dash' }},
            showlegend: true
        }}
    ];
    Plotly.newPlot('equity-chart-short', traces_s, {{
        template: 'plotly_dark',
        paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
        font: {{ color: '#c9d1d9', size: 11 }},
        margin: {{ l: 60, r: 20, t: 10, b: 40 }},
        legend: {{ orientation: 'h', y: -0.15 }},
        yaxis: {{ title: 'Equity ($)', gridcolor: '#21262d' }},
        xaxis: {{ gridcolor: '#21262d' }}
    }}, {{ responsive: true }});
}} else {{
    document.getElementById('equity-chart-short').innerHTML =
        '<p style="text-align:center;padding:80px;color:#8b949e">No short equity data yet.</p>';
}}

// LONG equity chart
const dates_l = {json.dumps(eq_dates_long)};
const equity_l = {json.dumps(eq_values_long)};
const mark_l = {json.dumps(eq_mark_long)};
const initial_l = {initial_long};

if (dates_l.length > 0) {{
    const traces_l = [
        {{
            x: dates_l, y: equity_l, name: 'Realized Equity',
            line: {{ color: '#3fb950', width: 2.5 }},
            fill: 'tozeroy', fillcolor: 'rgba(63,185,80,0.05)'
        }},
        {{
            x: dates_l, y: mark_l, name: 'Mark-to-Market',
            line: {{ color: '#58a6ff', width: 1.5, dash: 'dot' }}
        }},
        {{
            x: [dates_l[0], dates_l[dates_l.length-1]], y: [initial_l, initial_l],
            name: 'Initial Capital', line: {{ color: '#8b949e', width: 1, dash: 'dash' }},
            showlegend: true
        }}
    ];
    Plotly.newPlot('equity-chart-long', traces_l, {{
        template: 'plotly_dark',
        paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
        font: {{ color: '#c9d1d9', size: 11 }},
        margin: {{ l: 60, r: 20, t: 10, b: 40 }},
        legend: {{ orientation: 'h', y: -0.15 }},
        yaxis: {{ title: 'Equity ($)', gridcolor: '#21262d' }},
        xaxis: {{ gridcolor: '#21262d' }}
    }}, {{ responsive: true }});
}} else {{
    document.getElementById('equity-chart-long').innerHTML =
        '<p style="text-align:center;padding:80px;color:#8b949e">No long equity data yet.</p>';
}}
</script>

</body>
</html>"""
    return html


@app.route('/')
def dashboard():
    return Response(build_html(), mimetype='text/html')


@app.route('/api/state')
def api_state():
    state = load_json(PAPER_DIR / 'state.json') or {}
    return Response(json.dumps(state, indent=2, default=str), mimetype='application/json')


@app.route('/api/trades')
def api_trades():
    trades = load_json(PAPER_DIR / 'trades.json') or []
    return Response(json.dumps(trades, indent=2, default=str), mimetype='application/json')


@app.route('/api/equity')
def api_equity():
    eq = load_json(PAPER_DIR / 'equity.json') or []
    return Response(json.dumps(eq, indent=2, default=str), mimetype='application/json')


@app.route('/api/trades_long')
def api_trades_long():
    trades = load_json(PAPER_DIR / 'trades_long.json') or []
    return Response(json.dumps(trades, indent=2, default=str), mimetype='application/json')


@app.route('/api/equity_long')
def api_equity_long():
    eq = load_json(PAPER_DIR / 'equity_long.json') or []
    return Response(json.dumps(eq, indent=2, default=str), mimetype='application/json')


def export_html():
    """Export static HTML to results/."""
    html = build_html()
    out = RESULTS_DIR / 'paper_trading_dashboard.html'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Exported to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', action='store_true', help='Export static HTML')
    parser.add_argument('--port', type=int, default=5001, help='Port (default 5001)')
    args = parser.parse_args()

    if args.export:
        export_html()
    else:
        print(f"Paper Trading Dashboard: http://localhost:{args.port}")
        app.run(debug=False, port=args.port, host='0.0.0.0')
