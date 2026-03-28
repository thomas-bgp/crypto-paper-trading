"""
Recalculate paper trading equity history with corrected math:
1. Fixed weight = 1/N_original (not 1/len(remaining))
2. Estimate funding costs retroactively (use avg funding rate from position data)

Reads trades.json + equity.json, recalculates, and overwrites state/equity/trades.
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path(__file__).parent.parent
PAPER_DIR = PROJECT_DIR / 'paper_trading'

TOP_N = 5
COST_PER_SIDE = 0.002
INITIAL_CAPITAL = 10_000
STABLE_APY = 0.05
STABLE_YIELD_PER_4H = STABLE_APY / 365 / 6


def main():
    trades_path = PAPER_DIR / 'trades.json'
    equity_path = PAPER_DIR / 'equity.json'
    state_path = PAPER_DIR / 'state.json'

    trades = json.loads(trades_path.read_text()) if trades_path.exists() else []
    equity_hist = json.loads(equity_path.read_text()) if equity_path.exists() else []
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    if not trades:
        print("No trades to recalculate")
        return

    print(f"\n{'='*60}")
    print("  RECALCULATING PAPER TRADING HISTORY")
    print(f"{'='*60}")

    # Sort closed trades by exit_time
    closed_trades = [t for t in trades if t.get('exit_time')]
    closed_trades.sort(key=lambda t: t['exit_time'])

    # --- Replay equity with correct weights ---
    equity = INITIAL_CAPITAL
    # Entry cost: 0.2% on initial capital
    equity *= (1 - COST_PER_SIDE)
    n_at_entry = TOP_N  # fixed

    print(f"\n  Initial capital:     ${INITIAL_CAPITAL:,.2f}")
    print(f"  After entry cost:    ${equity:,.2f}")
    print(f"  Fixed weight:        1/{n_at_entry} = {1/n_at_entry:.1%}")
    print()

    # Track yield accrued between events
    cumulative_yield = 0.0
    cumulative_funding = 0.0

    for t in closed_trades:
        ret = t['return']  # already includes exit cost
        weight = 1.0 / n_at_entry

        # Estimate funding cost for this position's holding period
        funding_rate = t.get('funding_rate', 0)
        entry_time = t.get('entry_time', '')
        exit_time = t.get('exit_time', '')
        hours_held = 0
        if entry_time and exit_time:
            try:
                from dateutil.parser import parse
                dt_entry = parse(entry_time)
                dt_exit = parse(exit_time)
                hours_held = (dt_exit - dt_entry).total_seconds() / 3600
            except Exception:
                hours_held = 24  # fallback

        # Funding periods held (1 per 8h)
        funding_periods = hours_held / 8.0
        # Funding cost for this position (short pays when rate > 0)
        position_funding_cost = funding_rate * funding_periods * weight
        cumulative_funding += abs(position_funding_cost) * equity

        old_equity = equity
        # Apply trade PnL with fixed weight
        equity *= (1 + ret * weight)
        # Apply funding
        equity *= (1 - position_funding_cost)

        # Accrue yield on idle capital between stops
        # Count how many positions were idle after this close
        n_remaining = n_at_entry - (closed_trades.index(t) + 1)
        n_remaining = max(n_remaining, 0)
        idle_fraction = (n_at_entry - n_remaining) / n_at_entry
        # Approximate: yield for the hours this position was open
        cycles_4h = hours_held / 4.0
        # But idle fraction changes — use average: before this close, idle was (closed_so_far)/TOP_N
        closed_so_far = closed_trades.index(t)
        avg_idle = closed_so_far / n_at_entry
        yield_amount = equity * avg_idle * STABLE_YIELD_PER_4H * cycles_4h
        equity += yield_amount
        cumulative_yield += yield_amount

        pnl_dollar = equity - old_equity
        print(f"  {t['symbol']:12s}  ret={ret*100:+7.2f}%  weight={weight:.0%}  "
              f"funding={position_funding_cost*100:+.4f}%  "
              f"equity: ${old_equity:,.2f} -> ${equity:,.2f} ({pnl_dollar:+,.2f})")

    # Account for remaining yield from idle capital after all closes
    # (positions that are still open don't contribute to idle)
    total_return = (equity / INITIAL_CAPITAL - 1) * 100

    print(f"\n  {'-'*50}")
    print(f"  Final Equity (realized): ${equity:,.2f}")
    print(f"  Total Return:            {total_return:+.2f}%")
    print(f"  Cumulative Yield:        ${cumulative_yield:,.2f}")
    print(f"  Cumulative Funding:      ${cumulative_funding:,.2f}")

    # --- Recalculate the equity history snapshots ---
    # We need to replay through the original equity_hist timeline
    # and replace the equity values with corrected ones

    new_equity_hist = []
    replay_equity = INITIAL_CAPITAL * (1 - COST_PER_SIDE)  # after entry cost
    closed_idx = 0  # next closed trade to process
    replay_yield = 0.0
    replay_funding = 0.0

    for snap in equity_hist:
        ts = snap['timestamp']

        # Check if any trades closed at or before this timestamp
        while closed_idx < len(closed_trades):
            t = closed_trades[closed_idx]
            exit_ts = t.get('exit_time', '')
            if exit_ts and exit_ts <= ts:
                ret = t['return']
                weight = 1.0 / n_at_entry
                funding_rate = t.get('funding_rate', 0)
                entry_time = t.get('entry_time', '')
                hours_held = 0
                if entry_time and exit_ts:
                    try:
                        from dateutil.parser import parse
                        dt_entry = parse(entry_time)
                        dt_exit = parse(exit_ts)
                        hours_held = (dt_exit - dt_entry).total_seconds() / 3600
                    except Exception:
                        hours_held = 24
                funding_periods = hours_held / 8.0
                position_funding_cost = funding_rate * funding_periods * weight

                replay_equity *= (1 + ret * weight)
                replay_equity *= (1 - position_funding_cost)
                replay_funding += abs(position_funding_cost) * replay_equity
                closed_idx += 1
            else:
                break

        # Yield accrual: idle fraction based on how many have closed so far
        idle_fraction = closed_idx / n_at_entry
        if idle_fraction > 0:
            yield_amount = replay_equity * idle_fraction * STABLE_YIELD_PER_4H
            replay_equity += yield_amount
            replay_yield += yield_amount

        # Funding for still-open positions (approximate with stored rate)
        n_open = n_at_entry - closed_idx
        if n_open > 0 and state.get('positions'):
            avg_rate = np.mean([p.get('funding_rate', 0) for p in state['positions']
                               if p.get('funding_rate', 0) != 0] or [0])
            if avg_rate != 0:
                funding_cost = avg_rate * 0.5 * (n_open / n_at_entry)
                replay_equity *= (1 - funding_cost)

        # Recalculate mark equity from unrealized PnL
        unrealized_pct = snap.get('unrealized_pnl_pct', 0) / 100
        mark_equity = replay_equity * (1 + unrealized_pct)

        new_snap = dict(snap)
        new_snap['equity'] = replay_equity
        new_snap['mark_equity'] = mark_equity
        new_snap['idle_yield_accrued'] = replay_yield
        new_equity_hist.append(new_snap)

    # --- Update state ---
    state['equity'] = replay_equity
    state['cumulative_yield'] = replay_yield
    state['cumulative_funding'] = replay_funding
    state['n_positions_at_entry'] = n_at_entry
    # Recalculate unrealized for open positions
    for pos in state.get('positions', []):
        if pos.get('current_price') and pos.get('entry_price'):
            pos['unrealized_pnl'] = -(pos['current_price'] / pos['entry_price'] - 1)

    # --- Save corrected files ---
    # Backup originals
    for f in [equity_path, state_path]:
        backup = f.with_suffix('.json.bak')
        if f.exists() and not backup.exists():
            backup.write_text(f.read_text())
            print(f"\n  Backed up {f.name} -> {backup.name}")

    with open(equity_path, 'w') as f:
        json.dump(new_equity_hist, f, indent=2, default=str)
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    print(f"\n  Saved corrected equity.json ({len(new_equity_hist)} snapshots)")
    print(f"  Saved corrected state.json")

    # --- Compare old vs new ---
    if equity_hist:
        old_final = equity_hist[-1].get('equity', INITIAL_CAPITAL)
        new_final = new_equity_hist[-1]['equity']
        old_ret = (old_final / INITIAL_CAPITAL - 1) * 100
        new_ret = (new_final / INITIAL_CAPITAL - 1) * 100
        print(f"\n  OLD equity: ${old_final:,.2f} ({old_ret:+.2f}%)")
        print(f"  NEW equity: ${new_final:,.2f} ({new_ret:+.2f}%)")
        print(f"  Difference: ${new_final - old_final:,.2f} ({new_ret - old_ret:+.2f}pp)")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
