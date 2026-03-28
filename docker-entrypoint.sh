#!/bin/bash
set -e

echo "============================================"
echo "  CatBoost Short-Only — Paper Trading"
echo "  HOLDING: 5 days | Fixed weight: 1/N"
echo "============================================"

cd /app

# Seed persistent volume if empty (first deploy or fresh volume)
if [ ! -f paper_trading/state.json ]; then
    echo "[INIT] Volume empty — seeding with corrected historical data..."
    cp paper_trading_seed/*.json paper_trading/
    echo "[INIT] Seeded: state.json, trades.json, equity.json"
else
    echo "[INIT] Existing data found in volume, preserving."
fi

cd /app/src

# Train model if not present (but don't wipe state)
if [ ! -f /app/paper_trading/model_ensemble.cbm.0 ] && [ ! -f /app/paper_trading/model_meta.json ]; then
    echo "[INIT] No model found — training..."
    python -u paper_trading.py --retrain
    echo "[INIT] Model trained."
else
    echo "[INIT] Model already exists, skipping retrain."
fi

# Run one cycle
echo "[INIT] Running trading cycle..."
python -u paper_trading.py

# Export dashboard
echo "[INIT] Exporting dashboard..."
python -u paper_dashboard.py --export

# Start dashboard in background
echo "[START] Starting dashboard on port 5001..."
python -u paper_dashboard.py --port 5001 &
DASH_PID=$!

# Start paper trading daemon
echo "[START] Starting paper trading daemon (checks every 4h)..."
python -u paper_trading.py --daemon &
TRADE_PID=$!

echo "============================================"
echo "  Dashboard: http://localhost:5001"
echo "  Trading daemon PID: $TRADE_PID"
echo "  Dashboard PID: $DASH_PID"
echo "============================================"

# Wait for either process to exit
wait -n $DASH_PID $TRADE_PID
echo "[ERROR] A process exited unexpectedly. Shutting down."
kill $DASH_PID $TRADE_PID 2>/dev/null
exit 1
