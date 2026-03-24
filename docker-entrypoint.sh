#!/bin/bash
set -e

echo "============================================"
echo "  CatBoost Short-Only — Paper Trading"
echo "  HOLDING: 5 days (changed from 14)"
echo "============================================"

cd /app/src

# Force retrain with new 5-day holding period (preserves state/trades/equity)
echo "[INIT] Retraining model with 5-day holding period..."
python -u paper_trading.py --retrain
echo "[INIT] Model trained for 5-day hold."

# Run first cycle — will close positions >5d and open new ones
echo "[INIT] Running trading cycle (will close expired >5d positions)..."
python -u paper_trading.py

# Export initial dashboard
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
