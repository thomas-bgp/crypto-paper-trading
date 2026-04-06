#!/bin/bash
set -e

echo "============================================"
echo "  Adaptive Strategy — Paper Trading"
echo "  Regime R3: breadth + breadth momentum"
echo "  BULL  -> LONG top 10%, hold 2d, trail 5%"
echo "  BEAR  -> SHORT bottom 5, hold 5d, trail 15%"
echo "  CASH  -> no trades (ambiguous regime)"
echo "============================================"

cd /app

# FORCE RESET: always reseed from scratch on this deploy
echo "[INIT] Resetting state — fresh start for adaptive strategy..."
cp paper_trading_seed/*.json paper_trading/
# Remove old models to force retrain
rm -f paper_trading/model_*.cbm paper_trading/model_meta*.json paper_trading/feature_importance*.json paper_trading/log.jsonl
echo "[INIT] State reset. Old models cleared."

cd /app/src

# Train both models (fresh start)
echo "[INIT] Training short + long models..."
python -u paper_trading.py --retrain
echo "[INIT] Models trained."

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
