"""
Adaptive Strategy Dashboard — Paper Trading Monitor
Single-page dashboard: regime status, equity curve, positions, trades.
"""
from flask import Flask, Response, jsonify
import json, os, argparse
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
PAPER_DIR = PROJECT_DIR / 'paper_trading'
RESULTS_DIR = PROJECT_DIR / 'results'
app = Flask(__name__)

def load_json(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

@app.route('/api/state')
def api_state():
    return jsonify(load_json(PAPER_DIR / 'state.json') or {})

@app.route('/api/trades')
def api_trades():
    return jsonify(load_json(PAPER_DIR / 'trades.json') or [])

@app.route('/api/trades_long')
def api_trades_long():
    return jsonify(load_json(PAPER_DIR / 'trades_long.json') or [])

@app.route('/api/equity')
def api_equity():
    return jsonify(load_json(PAPER_DIR / 'equity.json') or [])

@app.route('/api/equity_long')
def api_equity_long():
    return jsonify(load_json(PAPER_DIR / 'equity_long.json') or [])

def build_html():
    state = load_json(PAPER_DIR / 'state.json') or {}
    trades_short = load_json(PAPER_DIR / 'trades.json') or []
    trades_long = load_json(PAPER_DIR / 'trades_long.json') or []
    equity_short = load_json(PAPER_DIR / 'equity.json') or []
    equity_long = load_json(PAPER_DIR / 'equity_long.json') or []
    log_lines = []
    log_path = PAPER_DIR / 'log.jsonl'
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                try: log_lines.append(json.loads(line.strip()))
                except: pass
    log_lines = log_lines[-30:]

    # Extract state
    cap = state.get('capital', 10000)
    eq_s = state.get('equity_short', state.get('equity', cap/2))
    eq_l = state.get('equity_long', cap/2)
    eq_total = eq_s + eq_l
    ret_total = (eq_total / cap - 1) * 100 if cap > 0 else 0
    ret_s = (eq_s / (cap/2) - 1) * 100 if cap > 0 else 0
    ret_l = (eq_l / (cap/2) - 1) * 100 if cap > 0 else 0
    pos_short = state.get('positions', [])
    pos_long = state.get('positions_long', [])
    regime = state.get('regime_direction', 'unknown')
    regime_skips = state.get('regime_skip_count', 0)
    started = state.get('started_at', '')
    days_running = 0
    if started:
        try:
            days_running = (datetime.now(timezone.utc) - datetime.fromisoformat(started)).days
        except: pass

    # Regime color
    regime_color = '#2ecc71' if regime == 'long' else ('#e74c3c' if regime == 'short' else '#f1c40f')
    regime_icon = '📈' if regime == 'long' else ('📉' if regime == 'short' else '⏸️')

    # Equity chart data
    eq_s_data = json.dumps([{'x': e.get('timestamp','')[:10], 'y': e.get('equity',0)} for e in equity_short[-60:]])
    eq_l_data = json.dumps([{'x': e.get('timestamp','')[:10], 'y': e.get('equity',0)} for e in equity_long[-60:]])

    # Trades tables
    def trades_rows(trades, direction):
        rows = ''
        for t in reversed(trades[-15:]):
            sym = t.get('symbol','?')
            ret = t.get('return', 0) * 100
            entry = t.get('entry_price', 0)
            exit_p = t.get('exit_price', 0)
            reason = t.get('exit_reason', '?')
            color = '#2ecc71' if ret > 0 else '#e74c3c'
            rows += f'<tr><td>{sym}</td><td>{direction}</td><td>{entry:.4f}</td><td>{exit_p:.4f}</td><td style="color:{color}">{ret:+.2f}%</td><td>{reason}</td></tr>'
        return rows

    short_rows = trades_rows(trades_short, 'SHORT')
    long_rows = trades_rows(trades_long, 'LONG')

    # Positions tables
    def pos_rows(positions, direction):
        rows = ''
        for p in positions:
            sym = p.get('symbol','?')
            entry = p.get('entry_price', 0)
            current = p.get('current_price', entry)
            if direction == 'SHORT':
                pnl = -(current/entry - 1) * 100 if entry > 0 else 0
            else:
                pnl = (current/entry - 1) * 100 if entry > 0 else 0
            days = p.get('days_held', 0)
            color = '#2ecc71' if pnl > 0 else '#e74c3c'
            rows += f'<tr><td>{sym}</td><td>{direction}</td><td>{entry:.4f}</td><td>{current:.4f}</td><td style="color:{color}">{pnl:+.2f}%</td><td>{days}d</td></tr>'
        return rows

    open_short = pos_rows(pos_short, 'SHORT')
    open_long = pos_rows(pos_long, 'LONG')

    # Log
    log_html = ''
    for l in reversed(log_lines):
        ts = l.get('timestamp','')[:19]
        evt = l.get('event','')
        msg = l.get('message','')
        color = '#8b949e'
        if 'error' in evt: color = '#e74c3c'
        elif 'regime' in evt: color = '#f1c40f'
        elif 'open' in evt: color = '#2ecc71'
        elif 'stop' in evt or 'close' in evt: color = '#e67e22'
        log_html += f'<div style="color:{color};font-size:12px;margin:2px 0">[{ts}] <b>{evt}</b> {msg}</div>'

    # Win rates
    def calc_wr(trades):
        if not trades: return 0, 0, 0
        rets = [t.get('return',0) for t in trades]
        wins = sum(1 for r in rets if r > 0)
        return len(trades), wins/len(trades)*100 if trades else 0, sum(rets)/len(rets)*100

    n_s, wr_s, avg_s = calc_wr(trades_short)
    n_l, wr_l, avg_l = calc_wr(trades_long)

    return f'''<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="300">
<title>Adaptive Strategy — Paper Trading</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;padding:20px}}
h1{{color:#58a6ff;margin-bottom:5px}} h2{{color:#79c0ff;margin:20px 0 10px}}
.grid{{display:grid;gap:12px;margin:10px 0}}
.g4{{grid-template-columns:repeat(4,1fr)}} .g3{{grid-template-columns:repeat(3,1fr)}} .g2{{grid-template-columns:1fr 1fr}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:15px}}
.metric{{text-align:center}} .metric .val{{font-size:2em;font-weight:bold}} .metric .label{{font-size:0.8em;color:#8b949e}}
.pos{{color:#2ecc71}} .neg{{color:#e74c3c}} .neu{{color:#f1c40f}}
table{{width:100%;border-collapse:collapse;margin:5px 0}}
th,td{{border:1px solid #30363d;padding:6px 10px;text-align:right;font-size:13px}}
th{{background:#161b22;color:#58a6ff}} td:first-child{{text-align:left;font-weight:bold}}
.regime-badge{{display:inline-block;padding:8px 20px;border-radius:20px;font-size:1.2em;font-weight:bold}}
canvas{{max-height:300px}}
</style></head><body>

<h1>{regime_icon} Adaptive Strategy — Paper Trading</h1>
<p style="color:#8b949e">Running {days_running} days | Auto-refresh 5min | <span class="regime-badge" style="background:{regime_color}22;color:{regime_color};border:2px solid {regime_color}">REGIME: {regime.upper()}</span></p>

<div class="grid g4" style="margin-top:15px">
<div class="card metric"><div class="val {'pos' if ret_total>=0 else 'neg'}">${eq_total:,.0f}</div><div class="label">Total Equity ({ret_total:+.1f}%)</div></div>
<div class="card metric"><div class="val" style="color:{regime_color}">{regime.upper()}</div><div class="label">Current Regime</div></div>
<div class="card metric"><div class="val">{len(pos_short)}S / {len(pos_long)}L</div><div class="label">Open Positions</div></div>
<div class="card metric"><div class="val">{n_s + n_l}</div><div class="label">Total Trades ({regime_skips} skips)</div></div>
</div>

<div class="grid g2">
<div class="card metric"><div class="val {'pos' if ret_s>=0 else 'neg'}">${eq_s:,.0f} <small>({ret_s:+.1f}%)</small></div><div class="label">Short Equity (bear legs)</div></div>
<div class="card metric"><div class="val {'pos' if ret_l>=0 else 'neg'}">${eq_l:,.0f} <small>({ret_l:+.1f}%)</small></div><div class="label">Long Equity (bull legs)</div></div>
</div>

<h2>Equity Curves</h2>
<div class="card" style="height:320px"><canvas id="eqChart"></canvas></div>
<script>
const eqS = {eq_s_data};
const eqL = {eq_l_data};
new Chart(document.getElementById('eqChart'),{{type:'line',data:{{
datasets:[
{{label:'Short',data:eqS,borderColor:'#e74c3c',borderWidth:2,fill:false,pointRadius:0,tension:0.1}},
{{label:'Long',data:eqL,borderColor:'#2ecc71',borderWidth:2,fill:false,pointRadius:0,tension:0.1}}
]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:'#8b949e'}}}}}},scales:{{x:{{type:'category',ticks:{{color:'#8b949e',maxTicksLimit:10}},grid:{{color:'#21262d'}}}},y:{{ticks:{{color:'#8b949e',callback:v=>'$'+v.toLocaleString()}},grid:{{color:'#21262d'}}}}}}}}}});
</script>

<h2>Open Positions</h2>
<div class="card">
<table>
<tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Current</th><th>PnL</th><th>Held</th></tr>
{open_short}{open_long}
</table>
{f'<p style="color:#8b949e;margin-top:5px">No open positions</p>' if not pos_short and not pos_long else ''}
</div>

<h2>Trade Stats</h2>
<div class="grid g2">
<div class="card">
<h3 style="color:#e74c3c">Short Leg</h3>
<p>Trades: {n_s} | WR: {wr_s:.0f}% | Avg: {avg_s:+.2f}%</p>
</div>
<div class="card">
<h3 style="color:#2ecc71">Long Leg</h3>
<p>Trades: {n_l} | WR: {wr_l:.0f}% | Avg: {avg_l:+.2f}%</p>
</div>
</div>

<h2>Recent Trades</h2>
<div class="card">
<table>
<tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>Return</th><th>Reason</th></tr>
{short_rows}{long_rows}
</table>
</div>

<h2>Strategy Logic</h2>
<div class="card" style="font-size:13px;line-height:1.6">
<b>Regime R3 (Breadth + Momentum):</b><br>
Breadth &gt; 55% or rising fast → <span class="pos">LONG</span> top 10% by predicted max peak (hold 2d, trail 5%)<br>
Breadth &lt; 35% or dropping fast → <span class="neg">SHORT</span> bottom 5 by predicted drawdown (hold 5d, trail 15%)<br>
Ambiguous → <span class="neu">CASH</span> (no trades)<br>
<br>
<b>Backtest:</b> +704% cumulative across 7 folds (2021-2026) | 6/7 positive
</div>

<h2>Event Log</h2>
<div class="card" style="max-height:300px;overflow-y:auto">
{log_html if log_html else '<p style="color:#8b949e">No events yet</p>'}
</div>

<p style="color:#30363d;margin-top:20px;font-size:11px">Paper trading — no real capital at risk. Models: CatBoost YetiRank (fwd_min + fwd_max). Data: Binance Futures.</p>
</body></html>'''

@app.route('/')
def index():
    return Response(build_html(), content_type='text/html')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()

    if args.export:
        RESULTS_DIR.mkdir(exist_ok=True)
        html = build_html()
        with open(RESULTS_DIR / 'dashboard.html', 'w') as f:
            f.write(html)
        print(f'Exported to {RESULTS_DIR / "dashboard.html"}')
    else:
        print(f'Starting dashboard on port {args.port}...')
        app.run(host='0.0.0.0', port=args.port, debug=False)
