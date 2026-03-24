"""
ops_monitor.py — Monitoring and operations stack for crypto perpetual futures trading.

Components:
  - HealthCheck:       Scheduled checks every 30 min (API, positions, margin, concentration)
  - KillSwitch:        Emergency flatten — closes all positions + cancels all orders
  - TradeLogger:       SQLite-backed audit log (trades, rebalances, alerts)
  - DrawdownBreaker:   Circuit breaker at -15% (halt) and -20% (flatten)

Usage:
  python ops_monitor.py health   — run health check once
  python ops_monitor.py kill     — emergency flatten
  python ops_monitor.py status   — print drawdown / equity status

Environment variables required:
  BINANCE_API_KEY
  BINANCE_API_SECRET
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID

State file (optional, for position reconciliation):
  C:/Projects/crypto-investment/data/expected_positions.json
  Format: {"BTCUSDT": {"side": "LONG", "qty": 0.01}, ...}
"""

import hashlib
import hmac
import json
import logging
import os
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("C:/Projects/crypto-investment")
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "trading.db"
LOG_PATH = DATA_DIR / "ops_monitor.log"
EXPECTED_POSITIONS_PATH = DATA_DIR / "expected_positions.json"
HWM_PATH = DATA_DIR / "hwm.json"  # high-water mark persistence

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging — every function logs; silent failures are ruin
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_PATH), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ops_monitor")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

FAPI_BASE = "https://fapi.binance.com"

# Risk thresholds
FREE_MARGIN_FLOOR = 0.30       # 30 % minimum free margin
MAX_POSITION_EQUITY_RATIO = 0.25  # no single position > 25 % of equity
DRAWDOWN_HALT_THRESHOLD = -0.15   # -15 % → stop new positions
DRAWDOWN_FLATTEN_THRESHOLD = -0.20  # -20 % → full flatten

# ---------------------------------------------------------------------------
# Binance signed-request helpers
# ---------------------------------------------------------------------------


def _sign(params: dict, secret: str) -> str:
    """Return HMAC-SHA256 hex signature for the given params dict."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def _binance_get(path: str, params: Optional[dict] = None, signed: bool = False) -> Any:
    """
    Issue a GET request to the Binance FAPI.

    Raises requests.HTTPError on non-2xx responses.
    Returns the parsed JSON body.
    """
    if params is None:
        params = {}
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}

    if signed:
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = _sign(params, BINANCE_API_SECRET)

    url = FAPI_BASE + path
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _binance_post(path: str, params: dict) -> Any:
    """
    Issue a signed POST request to the Binance FAPI.

    Raises requests.HTTPError on non-2xx responses.
    Returns the parsed JSON body.
    """
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    params["timestamp"] = int(time.time() * 1000)
    params["signature"] = _sign(params, BINANCE_API_SECRET)

    url = FAPI_BASE + path
    resp = requests.post(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _binance_delete(path: str, params: dict) -> Any:
    """
    Issue a signed DELETE request to the Binance FAPI.

    Raises requests.HTTPError on non-2xx responses.
    Returns the parsed JSON body.
    """
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    params["timestamp"] = int(time.time() * 1000)
    params["signature"] = _sign(params, BINANCE_API_SECRET)

    url = FAPI_BASE + path
    resp = requests.delete(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Telegram alert helper
# ---------------------------------------------------------------------------


def send_telegram(message: str, parse_mode: str = "HTML") -> bool:
    """
    Send a Telegram message via the Bot API.

    Returns True on success, False on any failure.
    Never raises — alerts must not crash the monitor.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials not configured — skipping alert")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": parse_mode,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        log.info("Telegram alert sent OK")
        return True
    except Exception as exc:
        log.error("Failed to send Telegram alert: %s", exc)
        return False


def _ts() -> str:
    """ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ===========================================================================
# 1.  TradeLogger
# ===========================================================================


class TradeLogger:
    """
    SQLite-backed audit log for trades, rebalances, and alerts.

    The database is created on first use.  All writes are wrapped in
    try/except so a logging failure never kills the trading process.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        try:
            with self._conn() as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp    TEXT    NOT NULL,
                        symbol       TEXT    NOT NULL,
                        side         TEXT    NOT NULL,
                        qty          REAL    NOT NULL,
                        price        REAL    NOT NULL,
                        fee          REAL    NOT NULL DEFAULT 0,
                        model_score  REAL,
                        action_type  TEXT
                    );

                    CREATE TABLE IF NOT EXISTS rebalances (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp     TEXT  NOT NULL,
                        n_positions   INTEGER,
                        equity        REAL,
                        rank_ic       REAL,
                        regime        TEXT,
                        model_version TEXT
                    );

                    CREATE TABLE IF NOT EXISTS alerts (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp   TEXT NOT NULL,
                        alert_type  TEXT NOT NULL,
                        message     TEXT NOT NULL,
                        severity    TEXT NOT NULL DEFAULT 'INFO'
                    );

                    CREATE INDEX IF NOT EXISTS idx_trades_ts     ON trades     (timestamp);
                    CREATE INDEX IF NOT EXISTS idx_rebalances_ts ON rebalances (timestamp);
                    CREATE INDEX IF NOT EXISTS idx_alerts_ts     ON alerts     (timestamp);
                """)
            log.debug("TradeLogger DB initialised at %s", self.db_path)
        except Exception as exc:
            log.error("TradeLogger._init_db failed: %s", exc)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def log_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        fee: float = 0.0,
        model_score: Optional[float] = None,
        action_type: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        ts = timestamp or _ts()
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO trades
                       (timestamp, symbol, side, qty, price, fee, model_score, action_type)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (ts, symbol, side, qty, price, fee, model_score, action_type),
                )
            log.info("trade logged: %s %s %s @ %s", symbol, side, qty, price)
        except Exception as exc:
            log.error("log_trade failed: %s", exc)

    def log_rebalance(
        self,
        n_positions: int,
        equity: float,
        rank_ic: Optional[float] = None,
        regime: Optional[str] = None,
        model_version: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        ts = timestamp or _ts()
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO rebalances
                       (timestamp, n_positions, equity, rank_ic, regime, model_version)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (ts, n_positions, equity, rank_ic, regime, model_version),
                )
            log.info("rebalance logged: equity=%.2f n=%d regime=%s", equity, n_positions, regime)
        except Exception as exc:
            log.error("log_rebalance failed: %s", exc)

    def log_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "INFO",
        timestamp: Optional[str] = None,
    ) -> None:
        ts = timestamp or _ts()
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO alerts (timestamp, alert_type, message, severity)
                       VALUES (?, ?, ?, ?)""",
                    (ts, alert_type, message, severity),
                )
            log.info("alert logged [%s/%s]: %s", alert_type, severity, message)
        except Exception as exc:
            log.error("log_alert failed: %s", exc)

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def get_recent_trades(self, limit: int = 50, symbol: Optional[str] = None) -> list[dict]:
        """Return the most recent trades as a list of dicts."""
        try:
            with self._conn() as conn:
                if symbol:
                    rows = conn.execute(
                        "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                        (symbol, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            log.error("get_recent_trades failed: %s", exc)
            return []

    def get_recent_alerts(self, limit: int = 20) -> list[dict]:
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            log.error("get_recent_alerts failed: %s", exc)
            return []

    def get_last_rebalance(self) -> Optional[dict]:
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM rebalances ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
            return dict(row) if row else None
        except Exception as exc:
            log.error("get_last_rebalance failed: %s", exc)
            return None


# ===========================================================================
# 2.  KillSwitch
# ===========================================================================


class KillSwitch:
    """
    Emergency flatten: market-close every open perpetual position and cancel
    every open order on the account.

    All actions are logged to file and to the TradeLogger database.
    A Telegram alert is sent on completion (or on failure).
    """

    def __init__(self, logger: Optional[TradeLogger] = None):
        self.logger = logger or TradeLogger()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, reason: str = "manual") -> bool:
        """
        Flatten everything.

        Returns True if all closes + cancels succeeded without exception,
        False if any step encountered an error.
        """
        log.critical("KILL SWITCH ACTIVATED — reason: %s", reason)
        self.logger.log_alert(
            alert_type="KILL_SWITCH",
            message=f"Kill switch activated: {reason}",
            severity="CRITICAL",
        )
        send_telegram(
            f"🚨 <b>KILL SWITCH ACTIVATED</b>\nReason: {reason}\nTime: {_ts()}"
        )

        success = True

        # ── Step 1: cancel all open orders ────────────────────────────
        try:
            result = _binance_delete("/fapi/v1/allOpenOrders", params={})
            log.critical("All open orders cancelled: %s", result)
            self.logger.log_alert(
                "KILL_SWITCH", f"All orders cancelled: {result}", "CRITICAL"
            )
        except Exception as exc:
            log.error("Failed to cancel all orders: %s\n%s", exc, traceback.format_exc())
            self.logger.log_alert(
                "KILL_SWITCH", f"ERROR cancelling orders: {exc}", "CRITICAL"
            )
            success = False

        # ── Step 2: fetch open positions ───────────────────────────────
        positions = []
        try:
            account = _binance_get("/fapi/v2/account", signed=True)
            positions = [
                p for p in account.get("positions", [])
                if float(p.get("positionAmt", 0)) != 0
            ]
            log.critical("Open positions to close: %d", len(positions))
        except Exception as exc:
            log.error("Failed to fetch positions: %s\n%s", exc, traceback.format_exc())
            self.logger.log_alert(
                "KILL_SWITCH", f"ERROR fetching positions: {exc}", "CRITICAL"
            )
            success = False

        # ── Step 3: market-close each position ────────────────────────
        for pos in positions:
            symbol = pos["symbol"]
            amt = float(pos["positionAmt"])
            if amt == 0:
                continue

            close_side = "SELL" if amt > 0 else "BUY"
            close_qty = abs(amt)

            try:
                order = _binance_post(
                    "/fapi/v1/order",
                    params={
                        "symbol": symbol,
                        "side": close_side,
                        "type": "MARKET",
                        "quantity": close_qty,
                        "reduceOnly": "true",
                    },
                )
                log.critical(
                    "Closed %s %s qty=%s  orderId=%s",
                    symbol, close_side, close_qty, order.get("orderId"),
                )
                self.logger.log_trade(
                    symbol=symbol,
                    side=close_side,
                    qty=close_qty,
                    price=float(order.get("avgPrice", 0)),
                    fee=0.0,
                    action_type="KILL_SWITCH",
                )
            except Exception as exc:
                log.error(
                    "Failed to close %s: %s\n%s", symbol, exc, traceback.format_exc()
                )
                self.logger.log_alert(
                    "KILL_SWITCH", f"ERROR closing {symbol}: {exc}", "CRITICAL"
                )
                success = False

        # ── Step 4: final status report ────────────────────────────────
        status = "COMPLETED" if success else "COMPLETED WITH ERRORS"
        summary = (
            f"🏁 <b>Kill Switch {status}</b>\n"
            f"Reason: {reason}\n"
            f"Positions closed: {len(positions)}\n"
            f"Time: {_ts()}"
        )
        send_telegram(summary)
        log.critical("Kill switch %s — %d positions processed", status, len(positions))
        return success


# ===========================================================================
# 3.  HealthCheck
# ===========================================================================


class HealthCheck:
    """
    Scheduled health monitor.  Call run() every 30 minutes.

    Checks:
      1. Binance API connectivity  (GET /fapi/v1/ping)
      2. Open positions vs expected state  (JSON file)
      3. Free margin ratio  (> 30 %)
      4. Single-position concentration  (no position > 25 % of equity)

    Sends a Telegram alert on any finding.  All results are stored in the
    alerts table.
    """

    def __init__(self, logger: Optional[TradeLogger] = None):
        self.logger = logger or TradeLogger()

    # ------------------------------------------------------------------
    # Sub-checks
    # ------------------------------------------------------------------

    def _check_connectivity(self) -> bool:
        """Returns True if Binance FAPI is reachable."""
        try:
            _binance_get("/fapi/v1/ping")
            log.info("connectivity check: OK")
            return True
        except Exception as exc:
            msg = f"Binance API connectivity FAILED: {exc}"
            log.error(msg)
            self._alert("CONNECTIVITY", msg, "CRITICAL")
            return False

    def _get_account(self) -> Optional[dict]:
        """Fetch full FAPI v2 account data.  Returns None on error."""
        try:
            return _binance_get("/fapi/v2/account", signed=True)
        except Exception as exc:
            msg = f"Failed to fetch account data: {exc}"
            log.error(msg)
            self._alert("ACCOUNT_FETCH", msg, "CRITICAL")
            return None

    def _check_positions(self, account: dict) -> bool:
        """
        Compare live positions against expected state file.

        Returns True if state matches (or if no expected state file exists).
        """
        if not EXPECTED_POSITIONS_PATH.exists():
            log.info("position check: no expected state file — skipping reconciliation")
            return True

        try:
            with open(EXPECTED_POSITIONS_PATH, "r", encoding="utf-8") as fh:
                expected: dict = json.load(fh)
        except Exception as exc:
            msg = f"Cannot read expected positions file: {exc}"
            log.error(msg)
            self._alert("POSITION_CHECK", msg, "WARNING")
            return False

        live: dict[str, dict] = {}
        for pos in account.get("positions", []):
            amt = float(pos.get("positionAmt", 0))
            if amt != 0:
                live[pos["symbol"]] = {
                    "side": "LONG" if amt > 0 else "SHORT",
                    "qty": abs(amt),
                }

        mismatches = []

        # symbols in expected but wrong or missing in live
        for sym, exp in expected.items():
            lv = live.get(sym)
            if lv is None:
                mismatches.append(f"{sym}: expected {exp} but MISSING in live")
            elif lv["side"] != exp.get("side"):
                mismatches.append(f"{sym}: expected side={exp['side']} got {lv['side']}")
            else:
                qty_diff = abs(lv["qty"] - exp.get("qty", 0))
                if qty_diff / max(exp.get("qty", 1), 1e-9) > 0.05:  # > 5 % tolerance
                    mismatches.append(
                        f"{sym}: expected qty={exp.get('qty')} got {lv['qty']:.4f}"
                    )

        # symbols in live but not in expected (ghost positions)
        for sym in live:
            if sym not in expected:
                mismatches.append(f"{sym}: UNEXPECTED live position {live[sym]}")

        if mismatches:
            msg = "Position mismatch:\n" + "\n".join(mismatches)
            log.warning(msg)
            self._alert("POSITION_MISMATCH", msg, "WARNING")
            return False

        log.info("position check: OK (%d positions match expected)", len(expected))
        return True

    def _check_margin(self, account: dict) -> bool:
        """
        Verifies free margin ratio >= FREE_MARGIN_FLOOR.

        Binance FAPI account balance fields used:
          totalMarginBalance  — total margin (includes unrealised PnL)
          totalMaintMargin    — maintenance margin currently used
          availableBalance    — cash available
        """
        try:
            total_margin = float(account.get("totalMarginBalance", 0))
            avail = float(account.get("availableBalance", 0))

            if total_margin <= 0:
                msg = "totalMarginBalance is zero — cannot assess margin"
                log.warning(msg)
                self._alert("MARGIN", msg, "WARNING")
                return False

            free_ratio = avail / total_margin
            log.info(
                "margin check: availableBalance=%.2f totalMarginBalance=%.2f ratio=%.1f%%",
                avail, total_margin, free_ratio * 100,
            )

            if free_ratio < FREE_MARGIN_FLOOR:
                msg = (
                    f"Free margin BELOW threshold: {free_ratio:.1%} "
                    f"(threshold {FREE_MARGIN_FLOOR:.1%}). "
                    f"Available: {avail:.2f} USDT, Total: {total_margin:.2f} USDT"
                )
                log.warning(msg)
                self._alert("LOW_MARGIN", msg, "CRITICAL")
                return False

            return True

        except Exception as exc:
            msg = f"Margin check error: {exc}"
            log.error(msg)
            self._alert("MARGIN", msg, "WARNING")
            return False

    def _check_concentration(self, account: dict) -> bool:
        """
        Verifies no single position exceeds MAX_POSITION_EQUITY_RATIO of equity.
        """
        try:
            equity = float(account.get("totalMarginBalance", 0))
            if equity <= 0:
                return True  # cannot compute; skip

            violations = []
            for pos in account.get("positions", []):
                amt = float(pos.get("positionAmt", 0))
                if amt == 0:
                    continue
                notional = abs(float(pos.get("notional", 0)))
                ratio = notional / equity
                if ratio > MAX_POSITION_EQUITY_RATIO:
                    violations.append(
                        f"{pos['symbol']}: notional={notional:.2f} "
                        f"= {ratio:.1%} of equity ({equity:.2f})"
                    )

            if violations:
                msg = "Position concentration breach:\n" + "\n".join(violations)
                log.warning(msg)
                self._alert("CONCENTRATION", msg, "WARNING")
                return False

            log.info("concentration check: OK (equity=%.2f)", equity)
            return True

        except Exception as exc:
            msg = f"Concentration check error: {exc}"
            log.error(msg)
            self._alert("CONCENTRATION", msg, "WARNING")
            return False

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute all health checks.

        Returns a summary dict with keys: connectivity, positions, margin,
        concentration, overall (all True = healthy).
        """
        log.info("=== HealthCheck START %s ===", _ts())

        results: dict[str, bool] = {
            "connectivity": False,
            "positions": False,
            "margin": False,
            "concentration": False,
        }

        # Connectivity must pass first; everything else needs API
        if not self._check_connectivity():
            summary = {**results, "overall": False}
            self._send_summary(summary)
            return summary

        results["connectivity"] = True

        account = self._get_account()
        if account is None:
            summary = {**results, "overall": False}
            self._send_summary(summary)
            return summary

        results["positions"] = self._check_positions(account)
        results["margin"] = self._check_margin(account)
        results["concentration"] = self._check_concentration(account)

        results["overall"] = all(results.values())
        self._send_summary(results)

        log.info("=== HealthCheck END: overall=%s ===", results["overall"])
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _alert(self, alert_type: str, message: str, severity: str) -> None:
        self.logger.log_alert(alert_type, message, severity)
        icon = "🚨" if severity == "CRITICAL" else "⚠️"
        send_telegram(f"{icon} <b>[{severity}] {alert_type}</b>\n{message}\n{_ts()}")

    def _send_summary(self, results: dict) -> None:
        if results.get("overall", False):
            # All green — no alert needed (avoid noise)
            log.info("HealthCheck summary: ALL CLEAR")
            return
        lines = [f"{'✅' if v else '❌'} {k}" for k, v in results.items() if k != "overall"]
        msg = "🏥 <b>HealthCheck FAILED</b>\n" + "\n".join(lines) + f"\n{_ts()}"
        send_telegram(msg)


# ===========================================================================
# 4.  DrawdownBreaker
# ===========================================================================


class DrawdownBreaker:
    """
    Circuit breaker based on equity drawdown from the high-water mark.

      -15 %:  HALT — sets halt flag, logs alert, sends Telegram
      -20 %:  FLATTEN — calls KillSwitch.execute()

    The high-water mark is persisted to disk so a process restart does not
    reset the protection.

    Call check() on every rebalance or at regular intervals (e.g., hourly).
    Returns a status string: "OK", "HALTED", or "FLATTENED".
    """

    HALT_STATE = "HALTED"
    FLATTEN_STATE = "FLATTENED"
    OK_STATE = "OK"

    def __init__(
        self,
        logger: Optional[TradeLogger] = None,
        kill_switch: Optional[KillSwitch] = None,
    ):
        self.logger = logger or TradeLogger()
        self.kill_switch = kill_switch or KillSwitch(self.logger)
        self._hwm_data = self._load_hwm()

    # ------------------------------------------------------------------
    # High-water mark persistence
    # ------------------------------------------------------------------

    def _load_hwm(self) -> dict:
        if HWM_PATH.exists():
            try:
                with open(HWM_PATH, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                log.info(
                    "Loaded HWM: peak=%.2f state=%s", data.get("peak", 0), data.get("state", "OK")
                )
                return data
            except Exception as exc:
                log.error("Failed to load HWM file, starting fresh: %s", exc)
        return {"peak": 0.0, "state": self.OK_STATE, "updated_at": _ts()}

    def _save_hwm(self) -> None:
        try:
            with open(HWM_PATH, "w", encoding="utf-8") as fh:
                json.dump(self._hwm_data, fh, indent=2)
        except Exception as exc:
            log.error("Failed to save HWM: %s", exc)

    @property
    def peak_equity(self) -> float:
        return float(self._hwm_data.get("peak", 0))

    @property
    def state(self) -> str:
        return self._hwm_data.get("state", self.OK_STATE)

    @state.setter
    def state(self, value: str) -> None:
        self._hwm_data["state"] = value
        self._hwm_data["updated_at"] = _ts()
        self._save_hwm()

    def is_halted(self) -> bool:
        return self.state in (self.HALT_STATE, self.FLATTEN_STATE)

    def reset(self) -> None:
        """Manually reset the circuit breaker (e.g., after a drawdown recovery)."""
        log.warning("DrawdownBreaker RESET by operator")
        self._hwm_data = {"peak": 0.0, "state": self.OK_STATE, "updated_at": _ts()}
        self._save_hwm()
        self.logger.log_alert("DRAWDOWN_BREAKER", "Circuit breaker reset by operator", "INFO")

    # ------------------------------------------------------------------
    # Equity fetch
    # ------------------------------------------------------------------

    def get_equity(self) -> Optional[float]:
        """Fetch current account equity (totalMarginBalance) from Binance."""
        try:
            account = _binance_get("/fapi/v2/account", signed=True)
            equity = float(account.get("totalMarginBalance", 0))
            log.info("Current equity: %.4f USDT", equity)
            return equity
        except Exception as exc:
            msg = f"DrawdownBreaker: failed to fetch equity: {exc}"
            log.error(msg)
            self.logger.log_alert("DRAWDOWN_BREAKER", msg, "WARNING")
            return None

    # ------------------------------------------------------------------
    # Main check
    # ------------------------------------------------------------------

    def check(self, current_equity: Optional[float] = None) -> str:
        """
        Evaluate current equity against the high-water mark.

        Parameters
        ----------
        current_equity : float, optional
            If provided, use this value instead of fetching from Binance.
            Useful when the caller already has account data.

        Returns
        -------
        str : "OK", "HALTED", or "FLATTENED"
        """
        if current_equity is None:
            current_equity = self.get_equity()

        if current_equity is None:
            log.error("DrawdownBreaker.check: cannot determine equity — no action taken")
            return self.state  # maintain existing state; do not flip to OK

        # Update high-water mark
        if current_equity > self.peak_equity:
            log.info(
                "New equity high-water mark: %.4f (prev %.4f)",
                current_equity, self.peak_equity,
            )
            self._hwm_data["peak"] = current_equity
            # Reset halt state when equity recovers to a new peak
            if self.state == self.HALT_STATE:
                log.warning(
                    "Equity recovered to new peak %.4f — lifting HALT", current_equity
                )
                self._hwm_data["state"] = self.OK_STATE
                self.logger.log_alert(
                    "DRAWDOWN_BREAKER",
                    f"HALT lifted — equity recovered to {current_equity:.2f}",
                    "INFO",
                )
            self._save_hwm()

        peak = self.peak_equity
        if peak <= 0:
            # No meaningful peak yet; just record and return
            self._hwm_data["peak"] = current_equity
            self._save_hwm()
            return self.OK_STATE

        drawdown = (current_equity - peak) / peak
        log.info(
            "Drawdown check: equity=%.4f peak=%.4f drawdown=%.2f%%",
            current_equity, peak, drawdown * 100,
        )

        # ── -20 %: full flatten ────────────────────────────────────────
        if drawdown <= DRAWDOWN_FLATTEN_THRESHOLD and self.state != self.FLATTEN_STATE:
            msg = (
                f"DRAWDOWN BREAKER TRIGGERED — FLATTEN\n"
                f"Equity: {current_equity:.2f} USDT\n"
                f"Peak:   {peak:.2f} USDT\n"
                f"Drawdown: {drawdown:.2%}"
            )
            log.critical(msg)
            self.logger.log_alert("DRAWDOWN_BREAKER", msg, "CRITICAL")
            send_telegram(f"🚨🚨 <b>DRAWDOWN BREAKER — FLATTEN</b>\n{msg}\n{_ts()}")

            self.state = self.FLATTEN_STATE
            self.kill_switch.execute(reason=f"DrawdownBreaker at {drawdown:.2%}")
            return self.FLATTEN_STATE

        # ── -15 %: halt new positions ──────────────────────────────────
        if drawdown <= DRAWDOWN_HALT_THRESHOLD and self.state == self.OK_STATE:
            msg = (
                f"DRAWDOWN BREAKER TRIGGERED — HALT\n"
                f"Equity: {current_equity:.2f} USDT\n"
                f"Peak:   {peak:.2f} USDT\n"
                f"Drawdown: {drawdown:.2%}"
            )
            log.warning(msg)
            self.logger.log_alert("DRAWDOWN_BREAKER", msg, "WARNING")
            send_telegram(f"⚠️ <b>DRAWDOWN BREAKER — HALT</b>\n{msg}\n{_ts()}")

            self.state = self.HALT_STATE
            return self.HALT_STATE

        return self.state


# ===========================================================================
# CLI entry point
# ===========================================================================


def _cmd_health() -> None:
    """Run health check and print summary."""
    hc = HealthCheck()
    results = hc.run()
    print("\n--- Health Check Results ---")
    for key, val in results.items():
        status = "OK" if val else "FAIL"
        print(f"  {key:<20} {status}")
    sys.exit(0 if results.get("overall") else 1)


def _cmd_kill() -> None:
    """Emergency flatten — market-close all positions and cancel all orders."""
    print("=" * 60)
    print("WARNING: This will market-close ALL open positions.")
    print("Type 'CONFIRM' to proceed, or anything else to abort.")
    print("=" * 60)
    try:
        answer = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("Aborted.")
        sys.exit(1)

    if answer != "CONFIRM":
        print("Aborted.")
        sys.exit(1)

    ks = KillSwitch()
    success = ks.execute(reason="cli-command")
    sys.exit(0 if success else 1)


def _cmd_status() -> None:
    """Print drawdown / equity status."""
    db = TradeLogger()
    breaker = DrawdownBreaker(logger=db)

    equity = breaker.get_equity()
    peak = breaker.peak_equity
    state = breaker.state

    print("\n--- DrawdownBreaker Status ---")
    if equity is not None and peak > 0:
        dd = (equity - peak) / peak
        print(f"  Current equity : {equity:.4f} USDT")
        print(f"  Peak equity    : {peak:.4f} USDT")
        print(f"  Drawdown       : {dd:.2%}")
    elif equity is not None:
        print(f"  Current equity : {equity:.4f} USDT")
        print(f"  Peak equity    : (not yet established)")
    else:
        print("  Equity         : (could not fetch — check API keys)")

    print(f"  Breaker state  : {state}")
    print(f"  Halt threshold : {DRAWDOWN_HALT_THRESHOLD:.0%}")
    print(f"  Flatten threshold: {DRAWDOWN_FLATTEN_THRESHOLD:.0%}")

    print("\n--- Recent Alerts ---")
    alerts = db.get_recent_alerts(limit=10)
    if alerts:
        for a in alerts:
            print(f"  [{a['severity']:8}] {a['timestamp']}  {a['alert_type']}: {a['message'][:80]}")
    else:
        print("  (none)")

    print("\n--- Recent Trades ---")
    trades = db.get_recent_trades(limit=10)
    if trades:
        for t in trades:
            print(f"  {t['timestamp']}  {t['symbol']:12} {t['side']:5} {t['qty']} @ {t['price']}")
    else:
        print("  (none)")

    sys.exit(0)


USAGE = """
Usage:
  python ops_monitor.py health   Run health check (connectivity, positions, margin, concentration)
  python ops_monitor.py kill     Emergency flatten (market-close all positions)
  python ops_monitor.py status   Show equity, drawdown status, and recent activity
""".strip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "health":
        _cmd_health()
    elif cmd == "kill":
        _cmd_kill()
    elif cmd == "status":
        _cmd_status()
    else:
        print(f"Unknown command: {cmd}\n")
        print(USAGE)
        sys.exit(1)
