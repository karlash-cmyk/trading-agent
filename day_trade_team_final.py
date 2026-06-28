import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import json
import os
import re
import atexit
import threading
import urllib.request
import urllib.parse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone
import pytz
import anthropic
import yfinance as yf
from binance.client import Client
from dotenv import load_dotenv

try:
    from pybit.unified_trading import HTTP as BybitHTTP
except Exception:
    BybitHTTP = None

# ib_insync needs an asyncio event loop to exist at import time; on Python 3.12+
# none exists in the main thread by default, so create one before importing.
try:
    import asyncio as _asyncio
    try:
        _asyncio.get_event_loop()
    except RuntimeError:
        _asyncio.set_event_loop(_asyncio.new_event_loop())
    from ib_insync import IB, Crypto, MarketOrder, LimitOrder, StopOrder
except Exception:
    IB = None

# ===========================================================================
# DAY TRADE TEAM - FINAL (regime-aware, validated universe)
# ===========================================================================
# BACKTEST VALIDATION SUMMARY:
#   Crypto 5 pairs : 43.5% win rate, +0.18 R, 6 months   -> VALIDATED
#                    (AVAX/XRP/DOT/ATOM/APT, SELL-only, no EMA, 45/55 RSI,
#                     4-candle cooldown, net of 0.1% fees)
#   CL=F (WTI)     : PROVISIONAL - 15m shows edge (52.3% / +0.444 R, ~60-70d)
#                    but the 6-month 1h cross-check was INCONCLUSIVE (37.9% /
#                    +0.012 R, sitting on the 37.5% breakeven). Needs paid
#                    intraday data to confirm. Gated behind TRADE_OIL.
#   BZ=F (Brent)   : REMOVED - negative expectancy in the 6-month sample
#                    (35.5% / -0.06 R) and only marginal at 15m.
#   Stocks/forex/gold: EXCLUDED -- negative expectancy after fees
#                    (stocks 23.1%/-0.33R, forex 25.0%/-0.28R, gold 33.3%/-0.03R)
#
# SIMULATION FINDINGS (500-trade replay, 96-candle/24h outcome window):
#   490 trades simulated | 41.2% win rate (vs 43.5% baseline -- holds up)
#   BEAR regime 44.8% vs BULL 39.1% (short side is the strongest component)
#   APT removed -- 35.0% win over 103 trades was a significant portfolio drag
#   Time filter applied -- 21:00-22:00 UTC and 05:00-06:00 UTC suppressed
#   Longs kept (BUY 40.4% ~ SELL 41.5%) but gated behind ALLOW_LONGS
#
# REGIME DETECTOR (new in final):
#   At the start of each rotation we read BTC's 4h trend via the 50-period 4h
#   EMA.  BTC above EMA50 = BULL regime -> BUY and SELL both enabled.
#                BTC below EMA50 = BEAR regime -> SELL only.
#   Rationale: the short-only edge was validated across a full 6 months. Long
#   signals had NO standalone edge (33.7% win), so we only permit them when the
#   macro tide (BTC) is rising, where mean-reversion longs have a tailwind.
#
#   *** CAVEAT: regime-gated BUY is an UNTESTED hypothesis. The backtest never
#   validated longs conditioned on BTC regime -- it only showed longs are
#   unprofitable unconditionally. Treat every BUY trade as experimental and
#   watch its win rate separately on paper before trusting it. ***
# ===========================================================================

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

BINANCE_KEY = os.getenv("BINANCE_KEY")
BINANCE_SECRET = os.getenv("BINANCE_SECRET")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")
BYBIT_KEY = os.getenv("BYBIT_KEY")
BYBIT_SECRET = os.getenv("BYBIT_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not all([BINANCE_KEY, BINANCE_SECRET, ANTHROPIC_KEY]):
    raise EnvironmentError("Missing required environment variables. Check your .env file.")

# --- Telegram notifications (NOTIFY-ONLY: never affects trading) ------------
# Messages are built from EXISTING memory data via string templates -- NO LLM
# calls, zero token cost. If TELEGRAM_TOKEN or TELEGRAM_CHAT_ID is missing,
# notifications are silently disabled (no crash). Every send is wrapped in
# try/except routed through log_runtime_error, so a failed notification can
# NEVER affect trading or crash the loop.
NOTIFY_TRADES = True       # alert on each paper trade opened / closed
NOTIFY_DRAWDOWN = True     # alert on risk-mode TRANSITIONS only (not per rotation)
NOTIFY_HEALTH = True       # alert on health-check failures (deduplicated)
NOTIFY_DIGEST = True       # weekly template digest
TELEGRAM_ENABLED = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
DIGEST_INTERVAL_SECONDS = 7 * 24 * 3600
WIN_RATE_BASELINE = 43.5   # validated 6-month crypto backtest baseline (digest)

# ===========================================================================
# KILL-SWITCH / SAFETY CIRCUIT BREAKER  (configurable; default ON)
# ===========================================================================
# This layer ONLY stops opening NEW trades and raises alerts. It NEVER changes
# signal rules, position sizing, or stop/target levels -- it is a gate in front
# of new entries, not a change to how entries are computed.
#
# It is deliberately SEPARATE from the drawdown PROTECTION above. Ordinary
# losing is NOT an emergency: a ~42% win-rate system has long losing runs as a
# matter of course, and the drawdown logic (REDUCED at 6%, PAUSED at 12%, with
# auto-resume) already handles "we are losing". The kill switch fires only when
# something is GENUINELY BROKEN and must NOT auto-resume:
#   1. a capital emergency far beyond the pause level (hard equity floor),
#   2. the health check failing repeatedly (a persistent broken state), or
#   3. an impossible/anomalous state (inverted stops, cap breach, a loss bigger
#      than the risk model permits).
# None of these can be produced by a normal losing streak -- see notes below.
# Once tripped the system stays killed until clear_kill_switch() is called
# DELIBERATELY; it never auto-resumes.
KILL_SWITCH_ENABLED = True
# 1. HARD EQUITY FLOOR -- kill if validated drawdown exceeds this. Set well
#    past DD_PAUSE_PCT (12%) so a normal losing run can never reach it.
KILL_HARD_DRAWDOWN_PCT = 20.0
# 2. REPEATED HEALTH FAILURES -- kill if run_health_check fails this many
#    rotations IN A ROW (a broken feed/state, not a losing streak).
KILL_HEALTH_CONSECUTIVE = 3
# 3. ANOMALY -- a closed loss larger than this multiple of its OWN recorded
#    risk means the stop logic failed. Paper exits are normally exact (loss ==
#    risk), so this multiple has wide headroom and cannot trip on variance.
KILL_LOSS_RISK_MULTIPLE = 1.5

ACCOUNT_SIZE = 1000
MAX_RISK_PERCENT = 1
# --- V3 drawdown protection thresholds (risk sizing / pause only -- these never
#     change which signals fire, entry prices, or stop/target logic) ---
DD_REDUCE_PCT = 6.0    # > this drawdown from peak -> halve risk (0.5% not 1%)
DD_PAUSE_PCT = 12.0    # > this -> pause opening NEW trades (existing still tracked)
DD_RESUME_PCT = 8.0    # resume opening trades once drawdown recovers below this
REDUCED_RISK_PERCENT = 0.5  # risk-% used while in REDUCED mode
# Absolute path so the bot always uses THIS folder's memory regardless of the
# working directory it is launched from (a relative path silently split memory
# across two files depending on CWD).
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_memory.json")
OUTCOME_DELAY_SECONDS = 7200  # 2 hours (120 min) = 8 x 15min candles forward,
                              # matching the backtest's forward-looking window
REGIME_EMA_PERIOD = 50        # 50-period 4h EMA on BTC for regime detection
REGIME_SLOPE_LOOKBACK = 10    # slope of the EMA50 over the last 10 4h periods
# --- Regime-strength SELL sizing (from validation_results.json) -------------
# Validated SELL win-rate / expectancy by 4-state regime:
#   STRONG_BEAR 47.1% / +0.287R | STRONG_BULL 41.6% / +0.122R
#   WEAK_BEAR (21 trades, noise) | WEAK_BULL 35.0% / -0.075R (NEGATIVE edge)
# Value = risk % for a SELL in that regime; None = SKIP the trade entirely.
# (Only the SKIP differs today; sizing is kept at 1.0% but routed through this
#  map so per-regime sizing is a one-line change later. BUY is NOT affected --
#  we have no BUY validation yet. Drawdown REDUCED mode still halves on top.)
REGIME_SELL_RISK_PCT = {
    "STRONG_BEAR": 1.0,   # strongest edge -> full risk
    "STRONG_BULL": 1.0,   # positive edge -> normal risk
    "WEAK_BEAR":   1.0,   # small sample -> normal risk
    "WEAK_BULL":   None,  # negative edge -> SKIP shorting into weak bull
}
BULL_STATES = ("STRONG_BULL", "WEAK_BULL")  # map 4-state -> 2-state for BUY/SELL gating

# --- Self-diagnostic / health-check state (in-process, PURE PYTHON, no API) ---
_freshness = {}        # symbol -> {"candle_ts": ms, "price": float, "seen_at": ts} | {"error":..}
_freshness_prev = {}   # previous rotation's freshness snapshot (advancement check)
_validated_opens_this_rotation = 0  # validated paper positions opened this rotation
DATA_STALE_MIN = 35    # latest candle older than this (min) = stale feed
DATA_DUP_MIN = 20      # unchanged candle+price older than this (min) = duplicate feed

# --- Asset-class toggles -------------------------------------------------
# Code for each class is kept fully intact; flip these to enable/disable.
TRADE_STOCKS = False  # disabled - 60 day backtest showed negative expectancy
                      # with current signal logic, needs more validation
TRADE_FOREX = False   # disabled - same reason, re-enable when 6 month data
                      # available
TRADE_OIL = True      # enabled but PROVISIONAL - CL=F (WTI) shows edge on 15m
                      # but a 6-month 1h cross-check was inconclusive. Switch to
                      # False if paper trading disappoints.

# Regime-gated longs. The 500-trade 96-candle simulation showed BUY (40.4%) is
# within ~1pt of SELL (41.5%), so longs are NOT clearly broken -- left enabled.
# (The earlier 30.2% BUY figure was a 2h-window artefact.) Note BULL-regime
# trades still underperform BEAR overall (39.1% vs 44.8%). Set False to revert
# to a pure short-only desk if the live BUY win rate disappoints.
ALLOW_LONGS = True

# --- Execution -----------------------------------------------------------
# When True, APPROVED signals place REAL orders on a TESTNET (fake money, no
# real funds). Set False to run signal-only / paper-log mode instantly.
EXECUTE_TRADES = False  # signal-only: no broker. The internal paper-trading
                        # simulator (below) tracks positions with REAL live prices.
EXEC_VENUE = "ibkr"    # "ibkr" (active) | "bybit" | "binance" (only used if EXECUTE_TRADES)
EXEC_LEVERAGE = 5      # Binance futures testnet leverage (used only for binance)
BYBIT_LEVERAGE = 2     # Bybit linear perpetual leverage
MAX_CONCURRENT_POSITIONS = 3  # cap on simultaneous open positions across all symbols

# SAFETY: known venue limitations for THIS strategy (altcoin shorts). Before you
# ever set EXECUTE_TRADES=True, double-check EXEC_VENUE is one that can actually
# run the universe. The startup readiness check (run_trading_team) warns loudly
# if the selected venue is flagged here.
KNOWN_INCOMPATIBLE = {
    "ibkr": "IBKR crypto is spot (no shorting) and does not list AVAX/XRP/DOT/ATOM",
    "binance": "Binance derivatives restricted for UK accounts; spot cannot short",
}

# IBKR paper trading via TWS/IB Gateway. Start TWS in PAPER mode, enable
# "ActiveX and Socket Clients" in API settings, and confirm the socket port.
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497       # 7497 = TWS paper (7496 live; 4002/4001 = Gateway)
IBKR_CLIENT_ID = 17
#
# NOTE: IBKR crypto trades SPOT via Paxos (exchange PAXOS) and CANNOT be shorted,
# so SELL signals on IBKR will be rejected (no inventory to sell). Limited coins
# are supported. The error is caught and logged; the loop never crashes.
#
# Both venues are USDT-margined linear perpetuals (support long AND short, which
# the short-heavy strategy needs -- spot cannot short). Only crypto USDT pairs
# are executable; oil (CL=F) is logged and skipped.
#
# BYBIT: connects to Bybit TESTNET via pybit using BYBIT_KEY/BYBIT_SECRET from
# .env. Order calls are wrapped so a bad key / margin / filter error is logged
# and never crashes the loop.
#
# TESTNET LIQUIDITY NOTE: AVAXUSDT and ATOMUSDT have no active ticker/liquidity
# on Bybit TESTNET, so their signals are SIGNAL-ONLY here -- still generated,
# tracked and logged, but execution is recorded as EXECUTION_SKIPPED rather than
# erroring. On Bybit MAINNET all four pairs (AVAX, XRP, DOT, ATOM) are liquid and
# WILL execute normally once you switch to real trading. XRP and DOT execute on
# testnet today.

bybit = None
if EXEC_VENUE == "bybit" and BybitHTTP is not None and BYBIT_KEY and BYBIT_SECRET:
    try:
        bybit = BybitHTTP(testnet=True, api_key=BYBIT_KEY, api_secret=BYBIT_SECRET)
    except Exception as _e:
        print(f"Bybit session init failed: {_e}")
        bybit = None

binance = Client(BINANCE_KEY, BINANCE_SECRET, testnet=True)  # testnet (execution only)

# Public MAINNET client for REAL live market data + paper-sim prices.
# WHY NO KEYS: every call we make on this client -- get_symbol_ticker,
# get_klines, get_ticker -- is a PUBLIC market-data endpoint that needs no auth.
# Constructing it without keys guarantees real mainnet prices and avoids
# accidentally hitting testnet. DO NOT call private/account methods on this
# client (e.g. get_account, create_order); they will fail without credentials.
binance_data = Client()
try:
    binance_data.ping()  # surface a network/region problem now, not mid-rotation
except Exception as _e:
    print(f"WARNING: mainnet Binance market-data ping failed ({_e}). "
          f"Prices/signals may be unavailable until connectivity is restored.")
claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# CRYPTO: focused universe (6-month validated, SELL-edge).
#   AVAXUSDT 45.0%/+0.225R  XRPUSDT 44.8%/+0.219R  DOTUSDT 44.2%/+0.201R
#   ATOMUSDT 42.5%/+0.150R
# APTUSDT REMOVED: the 500-trade 96-candle simulation flagged APT as a
# statistically significant drag -- 35.0% win over 103 trades (its 67 losses
# were the most of any pair, ~9pts below the 43.5% baseline). Dropping it lifts
# the blended portfolio win rate.
CRYPTO_PAIRS = [
    "AVAXUSDT", "XRPUSDT", "DOTUSDT", "ATOMUSDT", "XLMUSDT"
]

# WATCH-ONLY candidates flagged by research (SYNUSDT 43.7%, XLMUSDT 43.8% win)
# -- both just UNDER the 200-signal promotion threshold, so NOT yet validated.
# Their signals are generated and paper-tracked SEPARATELY (tagged "watch") to
# build a live track record, but they are kept out of the validated stats,
# equity/drawdown, and the position cap. Promote a pair by moving it into
# CRYPTO_PAIRS once its live record confirms the edge.
WATCH_PAIRS = [
    "SYNUSDT"
]

# STOCKS: backtest showed NO EDGE -- 23.1% win / -0.327 R over 65 signals vs
# 34.0% breakeven. Shorting equities fought a persistent uptrend. Code is kept
# intact but gated behind TRADE_STOCKS (default False).
STOCKS = ["TSLA", "NVDA", "AAPL", "SPY", "QQQ", "RKLB"]

# FOREX: NO EDGE -- 25.0% win / -0.283 R over 60 signals vs 34.4% breakeven, all
# four pairs negative. yfinance also returns zero volume for FX so the VWAP
# condition is degenerate on that data. Gated behind TRADE_FOREX (default False).
FOREX = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GBPEUR=X"]

# COMMODITIES: WTI crude (CL=F) only.
#   CL=F (WTI): PROVISIONAL - 15m shows edge (52.3% / +0.444 R, ~60-70d) but the
#     6-month 1h cross-check was inconclusive (37.9% / +0.012 R, at breakeven).
#     Needs paid intraday data to confirm. Gated behind TRADE_OIL.
#   BZ=F (Brent): REMOVED - negative expectancy in the 6-month sample
#     (35.5% / -0.06 R) and only marginal at 15m. Dropped.
#   GC=F (Gold): excluded earlier (33.3% / -0.026 R, below breakeven).
COMMODITIES = ["CL=F"]

# --- Agent model selection -------------------------------------------------
# Analyst runs on Haiku: it is the highest-volume agent and now makes a SINGLE
# call (market data is passed inline in its prompt, so the get_market_data
# tool round-trip is gone). The Risk Manager -- the approval gatekeeper --
# stays on Sonnet. News and Reflection (low volume) also stay on Sonnet.
ANALYST_MODEL = "claude-haiku-4-5"      # was claude-sonnet-4-6 + tool round-trip
RISK_MODEL = "claude-sonnet-4-6"
NEWS_MODEL = "claude-sonnet-4-6"
REFLECTION_MODEL = "claude-sonnet-4-6"
# NOTE: the get_market_data tool schema was removed -- the Analyst no longer
# fetches data via a tool; it receives the same data block inline in its prompt.

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {
        "trades": [],
        "pending_outcomes": [],
        "paper_positions": [],
        "paper_closed": [],
        "lessons": [],
        "news_cache": {},
        "stats": {"approved": 0, "rejected": 0, "wait": 0, "skipped": 0, "wins": 0, "losses": 0,
                  # Savings-tracking counters (no effect on trading logic):
                  "prefilter_skips": 0, "analyst_haiku_calls": 0, "risk_sonnet_calls": 0,
                  "news_calls": 0},
        "paper_stats": {"wins": 0, "losses": 0, "total_pnl": 0.0},
        # WATCH track record (kept separate from validated paper_stats):
        "watch_stats": {"wins": 0, "losses": 0, "total_pnl": 0.0},
        # V3 additions (measurement/safety only -- no effect on signal logic):
        "paper_breakdown": {"by_direction": {}, "by_regime": {}, "by_pair": {}},
        "risk_state": {"peak_equity": ACCOUNT_SIZE, "equity": ACCOUNT_SIZE,
                       "drawdown_pct": 0.0, "paused": False, "reduce_risk": False,
                       "mode": "NORMAL"},
    }

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

# ---------------------------------------------------------------------------
# Runtime-error capture (change C): a single API error / network blip logs
# clearly AND is recorded in the health section, then the loop continues to the
# next pair/rotation. Errors are NEVER swallowed silently.
# ---------------------------------------------------------------------------

_runtime_errors = []  # cleared each rotation; surfaced in memory["health"]

def log_runtime_error(context, exc):
    """Print the error clearly to the console and queue it for the health
    section so an unattended run keeps a record of every transient failure."""
    msg = f"{context}: {type(exc).__name__}: {exc}"
    print(f"  !! ERROR [{context}] -> {type(exc).__name__}: {exc}")
    _runtime_errors.append(msg)

def record_runtime_errors_to_health(rotation_count):
    """Flush queued runtime errors into memory['health'] when the rotation body
    aborts before run_health_check() runs (so they are never lost)."""
    if not _runtime_errors:
        return
    try:
        mem = load_memory()
        h = mem.setdefault("health", {})
        fails = list(h.get("failures", []))
        fails.extend(f"RUNTIME: {e}" for e in _runtime_errors)
        h.update({
            "status": "WARNING",
            "last_check": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "rotation": rotation_count,
            "failures": fails,
        })
        mem["health"] = h
        save_memory(mem)
    except Exception as e:
        print(f"  !! ERROR [health-persist] -> {e}")

def bump_stat(key, n=1):
    """Increment a counter in memory['stats'] (savings tracking only -- never
    affects signals, sizing or outcomes)."""
    try:
        mem = load_memory()
        mem.setdefault("stats", {})
        mem["stats"][key] = mem["stats"].get(key, 0) + n
        save_memory(mem)
    except Exception as e:
        print(f"  !! ERROR [bump_stat {key}] -> {e}")

# ===========================================================================
# TELEGRAM NOTIFICATIONS  (NOTIFY-ONLY LAYER)
# ===========================================================================
# This whole section is observability only. It reads EXISTING memory data and
# formats it with plain string templates -- there are NO LLM calls here, so it
# costs zero tokens. It NEVER changes signal rules, sizing, stops/targets, the
# drawdown logic, or the health check. Every network send is wrapped so a
# Telegram outage just logs an error and trading continues untouched.
# ---------------------------------------------------------------------------

def telegram_send(text):
    """Fire-and-forget Telegram message via a plain HTTPS POST (stdlib urllib,
    no heavy deps). Silently disabled when token/chat id are absent. Any error
    is logged through log_runtime_error and swallowed -- a failed notification
    can never crash the loop or touch trading state. Returns True on success."""
    if not TELEGRAM_ENABLED:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True
    except Exception as e:
        log_runtime_error("telegram", e)
        return False

def send_test_telegram():
    """One-off connectivity test -- call this manually to confirm TELEGRAM_TOKEN
    and TELEGRAM_CHAT_ID work end to end. Does not touch trading state."""
    if not TELEGRAM_ENABLED:
        print("Telegram: DISABLED -- set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env.")
        return False
    ok = telegram_send("✅ Day-trade bot: Telegram test message. "
                       "Token + chat ID are working.")
    print("Telegram: test message sent OK." if ok
          else "Telegram: test FAILED (see error above).")
    return ok

def notify_trade_opened(pos):
    """Alert 1: a paper position was opened. Built from the position dict."""
    if not (NOTIFY_TRADES and TELEGRAM_ENABLED):
        return
    wtag = " [WATCH]" if pos.get("watch") else ""
    side = "\U0001f7e2 LONG" if pos.get("direction") == "BUY" else "\U0001f534 SHORT"
    telegram_send(
        f"\U0001f4c8 TRADE OPENED{wtag}\n"
        f"{side} {pos.get('pair')}\n"
        f"Regime : {pos.get('regime')}\n"
        f"Entry  : {pos.get('entry')}\n"
        f"Stop   : {pos.get('stop')} | Target: {pos.get('target')}\n"
        f"Risk   : ${pos.get('risk_usdt', 0):.2f} ({pos.get('risk_percent')}%)"
    )

def notify_trade_closed(closed, win_rate, closed_count):
    """Alert 2: a paper position closed. win_rate/closed_count are the running
    figures for the relevant book (validated or watch) passed in by the caller."""
    if not (NOTIFY_TRADES and TELEGRAM_ENABLED):
        return
    wtag = " [WATCH]" if closed.get("watch") else ""
    result = "✅ WIN" if closed.get("outcome") == "WIN" else "❌ LOSS"
    telegram_send(
        f"\U0001f4c9 TRADE CLOSED{wtag}\n"
        f"{closed.get('direction')} {closed.get('pair')} -> {result}\n"
        f"PnL      : ${closed.get('pnl_usdt', 0):+.2f}\n"
        f"Win rate : {win_rate}% ({closed_count} closed)"
    )

def notify_drawdown_transition(old_mode, new_mode, rs):
    """Alert 3: risk-mode TRANSITION only (caller guarantees old != new)."""
    if not (NOTIFY_DRAWDOWN and TELEGRAM_ENABLED):
        return
    telegram_send(
        f"⚠️ RISK MODE: {old_mode} -> {new_mode}\n"
        f"Equity   : ${rs.get('equity', 0):.2f} | Peak: ${rs.get('peak_equity', 0):.2f}\n"
        f"Drawdown : {rs.get('drawdown_pct', 0)}%"
    )

def notify_health(fails):
    """Alert 4: health-check failures (dedup is handled by the caller)."""
    if not (NOTIFY_HEALTH and TELEGRAM_ENABLED):
        return
    body = "\n".join(f"- {f}" for f in fails[:10])
    extra = f"\n(+{len(fails) - 10} more)" if len(fails) > 10 else ""
    telegram_send(f"\U0001f6a8 HEALTH WARNING ({len(fails)})\n{body}{extra}")

def _digest_ts(s):
    """Parse a 'YYYY-mm-dd HH:MM:SS' local timestamp to unix seconds (0 on fail)."""
    try:
        return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return 0.0

def build_weekly_digest(mem, since):
    """Template-based weekly summary string. Reads ONLY existing memory fields;
    no LLM, no token cost. `since` is the unix ts of the previous digest."""
    closed_all = mem.get("paper_closed", [])
    # This week's VALIDATED closes (watch excluded), by close_time.
    week = [t for t in closed_all
            if not t.get("watch") and _digest_ts(t.get("close_time", "")) >= since]
    ww = sum(1 for t in week if t.get("outcome") == "WIN")
    wl = sum(1 for t in week if t.get("outcome") == "LOSS")
    wpnl = round(sum(t.get("pnl_usdt", 0.0) for t in week), 2)

    ps = mem.get("paper_stats", {})
    cw, cl = ps.get("wins", 0), ps.get("losses", 0)
    closed_total = cw + cl
    live_wr = round(cw / closed_total * 100, 1) if closed_total else 0.0
    delta = round(live_wr - WIN_RATE_BASELINE, 1)

    by_pair = mem.get("paper_breakdown", {}).get("by_pair", {})
    ranked = sorted(by_pair.items(), key=lambda kv: kv[1].get("pnl", 0.0))
    worst = ranked[0] if ranked else None
    best = ranked[-1] if ranked else None

    rs = mem.get("risk_state", {})
    xlm = by_pair.get("XLMUSDT")
    wps = mem.get("watch_stats", {})
    w_closed = wps.get("wins", 0) + wps.get("losses", 0)

    lines = [
        "\U0001f4ca WEEKLY DIGEST (template, no LLM)",
        f"This week: {len(week)} closed | {ww}W / {wl}L | PnL ${wpnl:+.2f}",
        f"Live win rate: {live_wr}% ({closed_total} closed) "
        f"vs {WIN_RATE_BASELINE}% baseline ({delta:+.1f} pts)",
    ]
    if best:
        lines.append(f"Best pair : {best[0]} ${best[1].get('pnl', 0.0):+.2f} "
                     f"({best[1].get('wins', 0)}W/{best[1].get('losses', 0)}L)")
    if worst and worst is not best:
        lines.append(f"Worst pair: {worst[0]} ${worst[1].get('pnl', 0.0):+.2f} "
                     f"({worst[1].get('wins', 0)}W/{worst[1].get('losses', 0)}L)")
    lines.append(f"Equity: ${rs.get('equity', 0):.2f} | "
                 f"Drawdown: {rs.get('drawdown_pct', 0)}% | mode {rs.get('mode', 'NORMAL')}")
    # Research status: promotions/demotions are MANUAL config changes -- report
    # the current universe so any change is visible week to week.
    lines.append(f"Universe: validated {CRYPTO_PAIRS} | watch {WATCH_PAIRS} "
                 f"(promotions are manual)")
    if w_closed:
        w_wr = round(wps.get("wins", 0) / w_closed * 100, 1)
        lines.append(f"Watch record: {w_wr}% ({w_closed} closed) "
                     f"PnL ${wps.get('total_pnl', 0.0):+.2f}")
    if xlm:
        x_closed = xlm.get("wins", 0) + xlm.get("losses", 0)
        x_wr = round(xlm.get("wins", 0) / x_closed * 100, 1) if x_closed else 0.0
        lines.append(f"XLM (paper-trial): {x_wr}% ({xlm.get('wins', 0)}W/"
                     f"{xlm.get('losses', 0)}L) PnL ${xlm.get('pnl', 0.0):+.2f}")
    else:
        lines.append("XLM (paper-trial): no closed trades yet")
    return "\n".join(lines)

def maybe_send_weekly_digest():
    """Send the weekly digest if >= 7 days since the last one. Tracks
    notify.digest_last_sent in memory. On first run it just arms the timer
    (first digest goes out ~7 days later). Notify-only."""
    if not (NOTIFY_DIGEST and TELEGRAM_ENABLED):
        return
    try:
        mem = load_memory()
        notify = mem.setdefault("notify", {})
        now = time.time()
        last = notify.get("digest_last_sent")
        if last is None:
            notify["digest_last_sent"] = now      # arm; no immediate send
            save_memory(mem)
            return
        if now - last < DIGEST_INTERVAL_SECONDS:
            return
        if telegram_send(build_weekly_digest(mem, since=last)):
            notify["digest_last_sent"] = now      # advance only on success
            save_memory(mem)
    except Exception as e:
        log_runtime_error("weekly-digest", e)

# ===========================================================================
# KILL-SWITCH / SAFETY CIRCUIT BREAKER  (implementation)
# ===========================================================================
# Read the trigger constants at the top of the file (KILL_SWITCH_ENABLED,
# KILL_HARD_DRAWDOWN_PCT, KILL_HEALTH_CONSECUTIVE, KILL_LOSS_RISK_MULTIPLE) to
# see EXACTLY what can stop the system and why. Nothing in this section reads
# or writes signal rules, sizing, or stop/target levels -- it only flips a
# KILLED flag in memory and alerts. evaluate_kill_switch() is the single entry
# point called once per rotation; clear_kill_switch() is the manual reset.
# ---------------------------------------------------------------------------

def is_killed():
    """True if the kill switch is currently latched (reads memory only)."""
    return bool(load_memory().get("kill_switch", {}).get("killed"))

def _kill_switch_anomalies(mem):
    """Trigger 3: return a list of 'impossible' states. Each of these should be
    unreachable in normal operation, so any hit means something is broken --
    NOT that we are merely losing."""
    out = []
    positions = mem.get("paper_positions", [])

    # (a) Inverted stop/target on an OPEN position. A SELL must have stop above
    #     and target below entry; a BUY the reverse. Anything else is corrupt.
    for p in positions:
        d, e, s, t = p.get("direction"), p.get("entry"), p.get("stop"), p.get("target")
        if None in (e, s, t):
            continue
        if d == "SELL" and not (s > e and t < e):
            out.append(f"inverted SELL stop/target on {p.get('pair')} "
                       f"(entry {e}, stop {s}, target {t})")
        if d == "BUY" and not (s < e and t > e):
            out.append(f"inverted BUY stop/target on {p.get('pair')} "
                       f"(entry {e}, stop {s}, target {t})")

    # (b) Validated open positions exceeding the concurrent cap. The guard in
    #     can_open_position() should make this impossible.
    validated_open = sum(1 for p in positions if not p.get("watch"))
    if validated_open > MAX_CONCURRENT_POSITIONS:
        out.append(f"{validated_open} validated open positions exceed cap "
                   f"{MAX_CONCURRENT_POSITIONS}")

    # (c) A closed trade whose realised loss exceeds what its OWN risk model
    #     allowed. The simulator exits exactly at the stop, so a real loss is
    #     bounded by risk_usdt; a loss well beyond it means the stop logic
    #     failed. The multiple gives wide headroom so variance never trips it.
    for tr in mem.get("paper_closed", []):
        pnl = tr.get("pnl_usdt", 0.0)
        risk = tr.get("risk_usdt")
        if pnl < 0 and risk and risk > 0 and abs(pnl) > risk * KILL_LOSS_RISK_MULTIPLE:
            out.append(f"{tr.get('pair')} closed loss ${pnl:.2f} exceeds "
                       f"{KILL_LOSS_RISK_MULTIPLE}x its risk model (${risk:.2f})")
            break  # one is enough to trip

    return out

def trip_kill_switch(mem, reasons):
    """Latch the KILLED flag, print a loud banner, and alert. Never auto-clears."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    mem["kill_switch"] = {"killed": True, "reasons": reasons, "time": ts}
    save_memory(mem)
    bar = "!" * 64
    print("\n" + bar)
    print("!!! KILL SWITCH TRIPPED -- NEW TRADES HALTED !!!")
    print(f"!!! Time: {ts}")
    for r in reasons:
        print(f"!!!  - {r}")
    print("!!! Existing open positions are still tracked to completion.")
    print("!!! The system will NOT auto-resume. To restart deliberately, run:")
    print('!!!   python -c "import day_trade_team_final as t; t.clear_kill_switch()"')
    print(bar + "\n")
    if TELEGRAM_ENABLED:
        telegram_send(
            "\U0001f6d1 KILL SWITCH TRIPPED -- new trades halted.\n"
            + "\n".join(f"- {r}" for r in reasons)
            + "\nExisting positions still tracked. Manual restart required."
        )

def evaluate_kill_switch():
    """Single per-rotation entry point. Returns True if NEW trades must be
    blocked this rotation. If already killed, stays killed (no re-evaluation,
    no auto-resume). Otherwise checks the three triggers and latches on the
    first hit. Notify-only / gate-only: never touches signal or sizing logic."""
    if not KILL_SWITCH_ENABLED:
        return False
    mem = load_memory()
    if mem.get("kill_switch", {}).get("killed"):
        return True  # latched -- requires a manual clear_kill_switch()

    reasons = []

    # 1. HARD EQUITY FLOOR (a capital emergency, NOT ordinary losing variance).
    dd = mem.get("risk_state", {}).get("drawdown_pct", 0.0)
    if dd > KILL_HARD_DRAWDOWN_PCT:
        reasons.append(f"HARD EQUITY FLOOR: validated drawdown {dd}% "
                       f"> {KILL_HARD_DRAWDOWN_PCT}%")

    # 2. REPEATED HEALTH FAILURES (a persistent broken state across rotations).
    consec = mem.get("health", {}).get("consecutive_failures", 0)
    if consec >= KILL_HEALTH_CONSECUTIVE:
        reasons.append(f"REPEATED HEALTH FAILURES: {consec} consecutive rotations "
                       f">= {KILL_HEALTH_CONSECUTIVE}")

    # 3. ANOMALY: an impossible/corrupt state.
    for a in _kill_switch_anomalies(mem):
        reasons.append(f"ANOMALY: {a}")

    if reasons:
        trip_kill_switch(mem, reasons)
        return True
    return False

def clear_kill_switch():
    """MANUAL reset. The system never auto-resumes -- you must call this
    deliberately (e.g. python -c "import day_trade_team_final as t;
    t.clear_kill_switch()") after investigating what tripped it."""
    mem = load_memory()
    mem["kill_switch"] = {
        "killed": False, "reasons": [],
        "cleared_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    save_memory(mem)
    print("Kill switch CLEARED. New trades may resume on the next rotation.")
    if TELEGRAM_ENABLED:
        telegram_send("✅ Kill switch cleared manually. New trades re-enabled.")

def save_trade(pair, signal, sentiment, decision, price, stop=None, target=None,
               direction=None, market_type=None, regime=None, watch=False):
    memory = load_memory()
    trade = {
        "pair": pair,
        "market_type": market_type,
        "regime": regime,
        "watch": watch,
        "signal": signal[:100],
        "sentiment": sentiment[:100] if sentiment != "N/A" else "N/A",
        "decision": decision,
        "price": price,
        "stop": stop,
        "target": target,
        "direction": direction,
        "outcome": "OPEN",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "utc_hour": datetime.now(timezone.utc).hour,  # for health dead-zone check
    }
    trade_index = len(memory["trades"])
    memory["trades"].append(trade)

    if "APPROVED" in decision:
        memory["stats"]["approved"] += 1
        # Outcome tracking is handled by the paper-trading simulator
        # (open_paper_position / update_paper_positions), not pending_outcomes.
    elif "WAIT" in decision:
        memory["stats"]["wait"] += 1
    elif "SKIPPED" in decision:
        # Regime-strength filter (e.g. WEAK_BULL) -- NOT an approval or rejection.
        memory["stats"]["skipped"] = memory["stats"].get("skipped", 0) + 1
    else:
        memory["stats"]["rejected"] += 1

    save_memory(memory)

def get_lessons():
    memory = load_memory()
    if not memory["lessons"]:
        return "No lessons learned yet."
    return "\n".join([l["lesson"][:300] for l in memory["lessons"][-3:]])

def get_stats():
    memory = load_memory()
    s = memory["stats"]
    total = s["approved"] + s["rejected"] + s["wait"]
    if total == 0:
        return "No trades yet."
    rate = round((s["approved"] / total) * 100, 1)
    wins = s.get("wins", 0)
    losses = s.get("losses", 0)
    closed = wins + losses
    win_rate = round((wins / closed) * 100, 1) if closed > 0 else 0
    return (
        f"Total: {total} | Approved: {s['approved']} ({rate}%) | "
        f"Rejected: {s['rejected']} | Wait: {s['wait']} | "
        f"Skipped(regime): {s.get('skipped', 0)} | "
        f"Win rate: {wins}W/{losses}L ({win_rate}%)"
    )

# ---------------------------------------------------------------------------
# Outcome tracking
# ---------------------------------------------------------------------------

def get_window_candles(pair, market_type, start_ts, end_ts):
    """Return [(high, low), ...] for the 15m candles between start_ts and end_ts
    (unix seconds), in chronological order. This lets the outcome checker see
    intracandle highs/lows -- a trade can touch stop or target mid-candle even
    if the close does not reflect it. Returns None on fetch failure."""
    try:
        if market_type == "crypto":
            klines = binance.get_klines(
                symbol=pair, interval="15m",
                startTime=int(start_ts * 1000), endTime=int(end_ts * 1000),
                limit=20)
            return [(float(k[2]), float(k[3])) for k in klines]
        else:
            start = datetime.fromtimestamp(start_ts, timezone.utc)
            end = datetime.fromtimestamp(end_ts, timezone.utc)
            hist = yf.Ticker(pair).history(start=start, end=end, interval="15m")
            if hist.empty:
                return []
            return [(float(h), float(l)) for h, l in zip(hist["High"], hist["Low"])]
    except Exception:
        return None

def check_pending_outcomes():
    memory = load_memory()
    pending = memory.get("pending_outcomes", [])
    still_pending = []
    now = time.time()

    for p in pending:
        if now < p["check_at"]:
            still_pending.append(p)
            continue

        # The 2-hour window has elapsed -- pull every 15m candle from entry to
        # the check time and look for an intracandle touch of stop or target.
        entry_time = p.get("entry_time", p["check_at"] - OUTCOME_DELAY_SECONDS)
        candles = get_window_candles(p["pair"], p["market_type"],
                                     entry_time, p["check_at"])
        if candles is None:
            # Fetch failed (e.g. network); retry next rotation.
            still_pending.append(p)
            continue

        direction = p["direction"]
        entry = p["entry"]
        stop = p["stop"]
        target = p["target"]
        idx = p["trade_index"]

        # Walk candles in order; conservatively check the STOP before the TARGET
        # within the same candle (matches the backtest's simulate() logic).
        outcome = "OPEN"
        touch_price = None
        for high, low in candles:
            if direction == "BUY":
                if low <= stop:
                    outcome, touch_price = "LOSS", stop
                    break
                if high >= target:
                    outcome, touch_price = "WIN", target
                    break
            else:  # SELL
                if high >= stop:
                    outcome, touch_price = "LOSS", stop
                    break
                if low <= target:
                    outcome, touch_price = "WIN", target
                    break

        # The window is fully elapsed, so the outcome is terminal regardless of
        # result (OPEN = neither level touched in 2 hours).
        if idx < len(memory["trades"]):
            memory["trades"][idx]["outcome"] = outcome
            if touch_price is not None:
                memory["trades"][idx]["exit_price"] = touch_price
        if outcome == "WIN":
            memory["stats"]["wins"] = memory["stats"].get("wins", 0) + 1
        elif outcome == "LOSS":
            memory["stats"]["losses"] = memory["stats"].get("losses", 0) + 1
        # OPEN is terminal but not counted as a win or loss (matches backtest).

        detail = f"touched {touch_price}" if touch_price is not None else "no touch in 2h"
        print(f"  Outcome: {p['pair']} {direction} -> {outcome} "
              f"(entry {entry}, {detail}, {len(candles)} candles)")

    memory["pending_outcomes"] = still_pending
    save_memory(memory)

# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)

def calculate_vwap(highs, lows, closes, volumes):
    total_pv, total_vol = 0, 0
    for h, l, c, v in zip(highs, lows, closes, volumes):
        typical = (h + l + c) / 3
        total_pv += typical * v
        total_vol += v
    return round(total_pv / total_vol, 6) if total_vol > 0 else closes[-1]

def calculate_ema(closes, period=50):
    # Re-added in final ONLY for the BTC regime detector (not used as a per-pair
    # signal filter -- that was shown to add no edge in backtesting).
    if len(closes) < period:
        return sum(closes) / len(closes)
    k = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
    return round(ema, 6)

def calculate_ema_series(closes, period=50):
    """Full EMA series (so we can measure the EMA's slope over a lookback)."""
    if len(closes) < period:
        avg = sum(closes) / len(closes)
        return [avg] * len(closes)
    k = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    series = [None] * (period - 1) + [ema]
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
        series.append(ema)
    return series

# ---------------------------------------------------------------------------
# Market regime detector (BTC 4h vs 50-period 4h EMA)
# ---------------------------------------------------------------------------

def get_btc_regime():
    """4-state regime from BTC price vs its 50-period 4h EMA, plus the EMA's
    slope over the last REGIME_SLOPE_LOOKBACK periods:
        price > EMA & slope rising        -> STRONG_BULL
        price > EMA & slope flat/falling  -> WEAK_BULL
        price < EMA & slope falling       -> STRONG_BEAR
        price < EMA & slope flat/rising   -> WEAK_BEAR
    Returns (regime4, price, ema, slope). On error defaults to WEAK_BEAR (a
    SELL-allowed, non-skip BEAR state) so a data glitch never enables longs."""
    try:
        price = float(binance_data.get_symbol_ticker(symbol="BTCUSDT")["price"])
        k4h = binance_data.get_klines(symbol="BTCUSDT", interval="4h",
                                      limit=REGIME_EMA_PERIOD + 70)
        closes_4h = [float(k[4]) for k in k4h]
        series = calculate_ema_series(closes_4h, period=REGIME_EMA_PERIOD)
        ema_now = series[-1]
        prev = series[-1 - REGIME_SLOPE_LOOKBACK]
        slope = round(ema_now - prev, 6) if prev is not None else 0.0
        if price > ema_now:
            regime = "STRONG_BULL" if slope > 0 else "WEAK_BULL"
        else:
            regime = "STRONG_BEAR" if slope < 0 else "WEAK_BEAR"
        return regime, price, round(ema_now, 6), slope
    except Exception as e:
        print(f"Regime check error: {e} -- defaulting to WEAK_BEAR (SELL-only).")
        return "WEAK_BEAR", None, None, None

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def get_crypto_data(symbol):
    try:
        price = float(binance_data.get_symbol_ticker(symbol=symbol)["price"])
        k15 = binance_data.get_klines(symbol=symbol, interval="15m", limit=20)
        k1h = binance_data.get_klines(symbol=symbol, interval="1h", limit=10)

        # Health: record latest 15m candle open time + price for the freshness check.
        _freshness[symbol] = {"candle_ts": int(k15[-1][0]), "price": price,
                              "seen_at": time.time()}

        closes_15 = [float(k[4]) for k in k15]
        volumes_15 = [float(k[5]) for k in k15]
        highs_15 = [float(k[2]) for k in k15]
        lows_15 = [float(k[3]) for k in k15]
        closes_1h = [float(k[4]) for k in k1h]

        rsi = calculate_rsi(closes_15)
        vwap = calculate_vwap(highs_15, lows_15, closes_15, volumes_15)

        avg_vol = sum(volumes_15[:-1]) / len(volumes_15[:-1])
        vol = "HIGH" if volumes_15[-1] > avg_vol * 1.5 else "LOW" if volumes_15[-1] < avg_vol * 0.7 else "NORMAL"
        mom_15 = round(closes_15[-1] - closes_15[-3], 6)
        mom_1h = round(closes_1h[-1] - closes_1h[-3], 6)

        stats = binance_data.get_ticker(symbol=symbol)
        high_24 = float(stats["highPrice"])
        low_24 = float(stats["lowPrice"])
        change_24 = stats["priceChangePercent"]
        pos = round(((price - low_24) / (high_24 - low_24)) * 100, 1) if high_24 != low_24 else 50

        return f"""
=== {symbol} (CRYPTO) ===
Price: {price} | 24h Change: {change_24}%
24h Range position: {pos}%
RSI(14): {rsi} | VWAP: {vwap} | {"ABOVE" if price > vwap else "BELOW"} VWAP
Volume: {vol}
15min momentum: {mom_15} | 1hr momentum: {mom_1h}
15min closes: {closes_15[-5:]}
1hr closes: {closes_1h[-5:]}
"""
    except Exception as e:
        _freshness[symbol] = {"error": str(e), "seen_at": time.time()}
        return f"Error: {e}"

def get_yfinance_data(symbol, market_type):
    try:
        ticker = yf.Ticker(symbol)
        hist_15 = ticker.history(period="2d", interval="15m")
        hist_1h = ticker.history(period="5d", interval="1h")
        if hist_15.empty:
            _freshness[symbol] = {"error": "empty history", "seen_at": time.time()}
            return f"No data for {symbol}"

        # Health: record latest 15m candle time + price for the freshness check.
        _freshness[symbol] = {"candle_ts": int(hist_15.index[-1].timestamp() * 1000),
                              "price": float(hist_15["Close"].iloc[-1]),
                              "seen_at": time.time()}

        closes_15 = list(hist_15["Close"])[-20:]
        highs_15 = list(hist_15["High"])[-20:]
        lows_15 = list(hist_15["Low"])[-20:]
        volumes_15 = list(hist_15["Volume"])[-20:]
        closes_1h = list(hist_1h["Close"])[-10:]

        price = closes_15[-1]
        rsi = calculate_rsi(closes_15)
        vwap = calculate_vwap(highs_15, lows_15, closes_15, volumes_15)

        avg_vol = sum(volumes_15[:-1]) / max(len(volumes_15[:-1]), 1)
        vol = "HIGH" if volumes_15[-1] > avg_vol * 1.5 else "LOW" if volumes_15[-1] < avg_vol * 0.7 else "NORMAL"
        mom_15 = round(closes_15[-1] - closes_15[-3], 6) if len(closes_15) >= 3 else 0
        mom_1h = round(closes_1h[-1] - closes_1h[-3], 6) if len(closes_1h) >= 3 else 0

        day_high = max(highs_15)
        day_low = min(lows_15)
        pos = round(((price - day_low) / (day_high - day_low)) * 100, 1) if day_high != day_low else 50
        display = symbol.replace("=F", "").replace("=X", "")

        return f"""
=== {display} ({market_type.upper()}) ===
Price: {round(price, 5)}
Day Range position: {pos}%
RSI(14): {rsi} | VWAP: {round(vwap, 5)} | {"ABOVE" if price > vwap else "BELOW"} VWAP
Volume: {vol}
15min momentum: {round(mom_15, 6)} | 1hr momentum: {round(mom_1h, 6)}
15min closes: {[round(c, 4) for c in closes_15[-5:]]}
1hr closes: {[round(c, 4) for c in closes_1h[-5:]]}
"""
    except Exception as e:
        _freshness[symbol] = {"error": str(e), "seen_at": time.time()}
        return f"Error: {e}"

def get_market_data(symbol, market_type):
    if market_type == "crypto":
        return get_crypto_data(symbol)
    else:
        return get_yfinance_data(symbol, market_type)

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def parse_signal(text):
    """Read the analyst's intended signal from the FIRST word of the response,
    after stripping any leading markdown, asterisks, emojis, numbering or
    punctuation (e.g. '**SELL', '\U0001f534 SELL', '1. BUY', '- WAIT').

    FIRST-WORD-ONLY BY DESIGN: if that first word is not exactly BUY/SELL/WAIT
    we return WAIT. We deliberately do NOT scan the rest of the text for a
    keyword, so a genuine WAIT that merely MENTIONS 'buy'/'sell' in its reasoning
    can never be turned into a trade. This only ever rescues a signal the analyst
    put first but that leading formatting mangled -- it never invents one."""
    if not text:
        return "WAIT"
    cleaned = re.sub(r"^[^A-Za-z]+", "", str(text).strip())  # drop leading *,#,digits,emoji,etc
    m = re.match(r"[A-Za-z]+", cleaned)
    word = m.group(0).upper() if m else ""
    return word if word in ("BUY", "SELL", "WAIT") else "WAIT"

def analyst_agent(symbol, market_type, regime, market_data):
    """SINGLE Haiku call. The market_data block (RSI / VWAP / momentum / price /
    volume / ranges) is passed inline -- it is the EXACT SAME text the
    get_market_data tool used to return, so the analyst's decision basis is
    unchanged. This removes the tool-use round trip (2 calls -> 1). The rules,
    regime gating and required-output format are all identical to before."""
    lessons = get_lessons()

    if regime == "BULL" and ALLOW_LONGS:
        desk = ("The market regime is BULL (BTC above its 50-period 4h EMA), so "
                "BOTH long and short setups are permitted.")
        rules = (
            "- BUY if: RSI below 45, price above VWAP, 15min AND 1hr momentum both positive\n"
            "- SELL if: RSI above 55, price below VWAP, 15min AND 1hr momentum both negative\n"
            "- WAIT in every other case (RSI 45-55 neutral, or conflicting momentum)")
        start = ("Your response MUST begin with exactly one word in plain uppercase -- "
                 "BUY, SELL, or WAIT -- with NO markdown, asterisks, emojis or "
                 "punctuation before it. Then two sentences of reasoning.")
    else:
        desk = ("The market regime is BEAR (BTC below its 50-period 4h EMA), so this "
                "desk is SHORT-ONLY -- you never issue BUY signals, only SELL or WAIT.")
        rules = (
            "- SELL if: RSI above 55, price below VWAP, 15min AND 1hr momentum both negative\n"
            "- WAIT in every other case (including any bullish-looking setup -- no longs in BEAR)")
        start = ("Your response MUST begin with exactly one word in plain uppercase -- "
                 "SELL or WAIT -- with NO markdown, asterisks, emojis or punctuation "
                 "before it. Then two sentences of reasoning.")

    messages = [{
        "role": "user",
        "content": f"""You are a professional day trader specialising in {market_type} markets.
{desk}

Learned lessons:
{lessons}

Analyse {symbol} on 15 minute candles with 1 hour trend context.

Current market data:
{market_data}

Rules:
{rules}
- Volume LOW is a warning not a veto

{start}"""
    }]

    # Change C: a transient API/network error logs and returns WAIT (no trade)
    # rather than crashing the rotation.
    try:
        response = claude.messages.create(
            model=ANALYST_MODEL,
            max_tokens=200,
            messages=messages
        )
        bump_stat("analyst_haiku_calls")
    except Exception as e:
        log_runtime_error(f"analyst {symbol}", e)
        return "WAIT"

    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return "WAIT"

def news_agent(symbol, market_type):
    # Informational only - not used in approval decisions
    clean = symbol.replace("USDT", "").replace("=X", "").replace("=F", "")
    messages = [{
        "role": "user",
        "content": f"""You are a financial news analyst covering {market_type} markets.
For {clean} what are the key themes or catalysts a day trader should know?
Consider macro events, sector news, and key technical levels.
Two sentences max. End with BULLISH, BEARISH, or NEUTRAL."""
    }]
    response = claude.messages.create(
        model=NEWS_MODEL,
        max_tokens=100,
        messages=messages
    )
    return response.content[0].text

# News cache: avoid re-calling the news API for the same pair within 45 minutes.
# Persisted to agent_memory.json ("news_cache") so it SURVIVES RESTARTS -- a
# fresh start no longer re-calls the news API for every pair immediately.
NEWS_CACHE_SECONDS = 45 * 60
NEWS_CACHE_MAX = 50  # cap entries so the persisted cache can't grow unbounded
_news_cache = {}     # symbol -> (timestamp, sentiment)

def load_news_cache():
    """Restore the in-memory news cache from agent_memory.json on startup."""
    mem = load_memory()
    for sym, e in mem.get("news_cache", {}).items():
        _news_cache[sym] = (e.get("ts", 0), e.get("text", ""))

def _persist_news_cache():
    """Write the most-recent NEWS_CACHE_MAX entries back to memory."""
    mem = load_memory()
    items = sorted(_news_cache.items(), key=lambda kv: kv[1][0], reverse=True)[:NEWS_CACHE_MAX]
    mem["news_cache"] = {s: {"ts": ts, "text": txt} for s, (ts, txt) in items}
    save_memory(mem)

def get_cached_news(symbol, market_type):
    """Return (sentiment, was_cached). Reuses cached news if < 45 min old.
    News is INFO-ONLY (never used in approval), so on a transient error we log
    it, return a neutral placeholder, and do NOT cache it -- the next rotation
    will retry. This can never change a trade decision."""
    now = time.time()
    hit = _news_cache.get(symbol)
    if hit and (now - hit[0]) < NEWS_CACHE_SECONDS:
        return hit[1], True
    try:
        sentiment = news_agent(symbol, market_type)
        bump_stat("news_calls")
    except Exception as e:
        log_runtime_error(f"news {symbol}", e)
        return "NEUTRAL [news unavailable]", False
    _news_cache[symbol] = (now, sentiment)
    _persist_news_cache()
    return sentiment, False

def risk_agent(symbol, signal, current_price, market_type, regime):
    # Robust, markdown/emoji-safe parse -- same rule, just a more reliable read.
    parsed = parse_signal(signal)
    # Longs are only permitted in BULL regime AND when ALLOW_LONGS is on.
    if ALLOW_LONGS and regime == "BULL" and parsed == "BUY":
        direction = "BUY"
    else:
        direction = "SELL"

    if market_type == "forex":
        stop_pct = 0.003
        target_pct = 0.006
    elif market_type == "commodity":
        stop_pct = 0.008
        target_pct = 0.016
    elif market_type == "stock":
        stop_pct = 0.01
        target_pct = 0.02
    else:
        stop_pct = 0.008
        target_pct = 0.016

    if direction == "BUY":
        stop = round(current_price * (1 - stop_pct), 6)
        target = round(current_price * (1 + target_pct), 6)
        rsi_rule = "RSI below 45 (oversold enough to buy)"
        vwap_rule = "Price above VWAP"
        mom_rule = "Momentum positive on both timeframes (15min and 1hr)"
    else:
        stop = round(current_price * (1 + stop_pct), 6)
        target = round(current_price * (1 - target_pct), 6)
        rsi_rule = "RSI above 55 (overbought enough to short)"
        vwap_rule = "Price below VWAP"
        mom_rule = "Momentum negative on both timeframes (15min and 1hr)"

    max_loss = ACCOUNT_SIZE * (MAX_RISK_PERCENT / 100)
    stop_distance = abs(current_price - stop)
    position_size = round(max_loss / stop_distance, 4) if stop_distance > 0 else 0
    lessons = get_lessons()

    messages = [{
        "role": "user",
        "content": f"""You are a strict but fair risk manager for a day trading desk.
Current market regime: {regime} (BULL = longs+shorts allowed, BEAR = shorts only).

Learned lessons:
{lessons}

Signal: {signal[:150]}
Instrument: {symbol} ({market_type})
Direction: {direction}
Entry: {current_price}
Stop: {stop} | Target: {target} | R:R 2:1
Position: {position_size} units | Max loss: {max_loss} USDT

Approve if ALL 3 technical conditions are met (no EMA filter -- it added no edge
in backtesting):
1. {rsi_rule}
2. {vwap_rule}
3. {mom_rule}

Note: News sentiment is provided separately for context and must NOT be used as a reason to reject.

Your response MUST begin with exactly one word in plain uppercase -- APPROVED or REJECTED -- with NO markdown, asterisks, emojis or punctuation before it. Then one sentence why."""
    }]

    # Change C: on a transient API/network error, fail SAFE -- return a
    # REJECTED decision so NO trade opens (stop/target/direction were already
    # computed deterministically above). The error is logged to health.
    try:
        response = claude.messages.create(
            model=RISK_MODEL,
            max_tokens=120,
            messages=messages
        )
        bump_stat("risk_sonnet_calls")
        return response.content[0].text, stop, target, direction
    except Exception as e:
        log_runtime_error(f"risk {symbol}", e)
        return "REJECTED - risk agent error (logged)", stop, target, direction

def reflection_agent():
    memory = load_memory()
    if len(memory["trades"]) < 5:
        return

    closed_trades = [t for t in memory["trades"] if t.get("outcome") in ("WIN", "LOSS")]
    outcome_summary = f"Closed trades: {len(closed_trades)} | " + get_stats()

    messages = [{
        "role": "user",
        "content": f"""You are a head trader reviewing performance of a regime-aware desk.

{outcome_summary}
Last 15 decisions: {json.dumps(memory["trades"][-15:], indent=2)}

1. Is the SELL win rate holding up versus the 43.5% backtest baseline?
2. How are the (experimental) BULL-regime BUY trades performing vs shorts?
3. Are losses concentrated in any particular pair or regime?
4. What one change improves performance most?

Write 2 lessons under 100 words each."""
    }]

    # Change C: a reflection error logs and returns -- it never blocks trading.
    try:
        response = claude.messages.create(
            model=REFLECTION_MODEL,
            max_tokens=250,
            messages=messages
        )
    except Exception as e:
        log_runtime_error("reflection", e)
        return

    lesson = response.content[0].text
    memory = load_memory()
    memory["lessons"].append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lesson": lesson
    })
    save_memory(memory)
    print(f"\n*** REFLECTION ***\n{lesson}\n")

# ---------------------------------------------------------------------------
# Market hours
# ---------------------------------------------------------------------------

def is_us_market_open():
    uk = pytz.timezone("Europe/London")
    now = datetime.now(uk)
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=14, minute=30, second=0)
    close_time = now.replace(hour=21, minute=0, second=0)
    return open_time <= now <= close_time

def is_forex_open():
    uk = pytz.timezone("Europe/London")
    now = datetime.now(uk)
    return now.weekday() < 5

def is_commodity_open():
    uk = pytz.timezone("Europe/London")
    now = datetime.now(uk)
    return now.weekday() < 5

# Dead-zone UTC hours: the 500-trade simulation showed these windows produce
# poor win rates (21:00-22:00 UTC = 23.8%, 05:00-06:00 UTC = sub-30%), likely
# thin-liquidity periods. We suppress all new signals during them.
DEAD_ZONE_UTC_HOURS = {21, 5}

def in_dead_zone():
    return datetime.now(timezone.utc).hour in DEAD_ZONE_UTC_HOURS

# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Execution agent (Binance USD-M FUTURES testnet)
# ---------------------------------------------------------------------------

_exchange_filters = {}  # symbol -> {"step": float, "tick": float}

def _get_filters(symbol):
    """Fetch and cache LOT_SIZE step and PRICE_FILTER tick for a futures symbol.
    Falls back to coarse defaults if the info call fails."""
    if symbol in _exchange_filters:
        return _exchange_filters[symbol]
    step, tick = 0.001, 0.001
    try:
        info = binance.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        step = float(f["stepSize"])
                    elif f["filterType"] == "PRICE_FILTER":
                        tick = float(f["tickSize"])
                break
    except Exception:
        pass
    _exchange_filters[symbol] = {"step": step, "tick": tick}
    return _exchange_filters[symbol]

def _round_step(value, step):
    if step <= 0:
        return value
    # round DOWN to the nearest step to avoid exceeding risk / price filters
    return round((value // step) * step, 8)

def log_execution(record):
    memory = load_memory()
    memory.setdefault("executions", [])
    memory["executions"].append(record)
    save_memory(memory)

def execution_agent(symbol, direction, entry, stop, target):
    """Place a market entry plus separate stop-loss and take-profit orders on
    Binance FUTURES testnet. Quantity is sized so the stop = MAX_RISK (~$10).
    Every step is wrapped so a failure logs and returns without crashing."""
    record = {
        "symbol": symbol, "direction": direction,
        "entry": entry, "stop": stop, "target": target,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "PENDING", "orders": {},
    }

    # Only Binance crypto USDT pairs are executable here.
    if not symbol.endswith("USDT"):
        record["status"] = "SKIPPED"
        record["error"] = "non-crypto symbol not executable on Binance futures"
        print(f"Exec    : skipped {symbol} (not a Binance USDT pair).")
        log_execution(record)
        return record

    try:
        filt = _get_filters(symbol)
        # Position size: risk $ / stop distance, rounded down to lot step.
        rp = current_risk_percent()  # 1% normal, 0.5% in reduced-risk mode
        max_loss = ACCOUNT_SIZE * (rp / 100)
        stop_distance = abs(entry - stop)
        raw_qty = max_loss / stop_distance if stop_distance > 0 else 0
        qty = _round_step(raw_qty, filt["step"])
        record["quantity"] = qty
        record["risk_percent"] = rp
        record["risk_usdt"] = round(stop_distance * qty, 2)  # REAL $ risk after rounding
        if qty <= 0:
            record["status"] = "ERROR"
            record["error"] = "computed quantity rounded to zero"
            print(f"Exec    : {symbol} quantity rounded to zero, no order placed.")
            log_execution(record)
            return record

        side = "BUY" if direction == "BUY" else "SELL"
        close_side = "SELL" if side == "BUY" else "BUY"
        stop_px = _round_step(stop, filt["tick"])
        tp_px = _round_step(target, filt["tick"])

        # Best-effort leverage set (non-fatal if it fails).
        try:
            binance.futures_change_leverage(symbol=symbol, leverage=EXEC_LEVERAGE)
        except Exception as e:
            record["leverage_warning"] = str(e)

        # 1) Market entry
        entry_order = binance.futures_create_order(
            symbol=symbol, side=side, type="MARKET", quantity=qty)
        record["orders"]["entry"] = entry_order.get("orderId")

        # 2) Stop loss (closes the position when hit)
        try:
            sl = binance.futures_create_order(
                symbol=symbol, side=close_side, type="STOP_MARKET",
                stopPrice=stop_px, closePosition=True)
            record["orders"]["stop"] = sl.get("orderId")
        except Exception as e:
            record["orders"]["stop"] = None
            record["stop_error"] = str(e)
            print(f"Exec    : {symbol} stop-loss order failed: {e}")

        # 3) Take profit (closes the position when hit)
        try:
            tp = binance.futures_create_order(
                symbol=symbol, side=close_side, type="TAKE_PROFIT_MARKET",
                stopPrice=tp_px, closePosition=True)
            record["orders"]["take_profit"] = tp.get("orderId")
        except Exception as e:
            record["orders"]["take_profit"] = None
            record["tp_error"] = str(e)
            print(f"Exec    : {symbol} take-profit order failed: {e}")

        record["status"] = "PLACED"
        print(f"Exec    : {symbol} {direction} qty {qty} | "
              f"entry#{record['orders'].get('entry')} "
              f"stop#{record['orders'].get('stop')} "
              f"tp#{record['orders'].get('take_profit')}")
    except Exception as e:
        # Catch-all: bad keys, insufficient margin, filter errors, network, etc.
        record["status"] = "ERROR"
        record["error"] = str(e)
        print(f"Exec    : ORDER FAILED for {symbol} -> {e}")

    log_execution(record)
    return record

# ---------------------------------------------------------------------------
# Execution agent (Bybit linear perpetual testnet)
# ---------------------------------------------------------------------------

_bybit_filters = {}  # symbol -> {"step": float, "tick": float}

def bybit_check_balance():
    """Read-only connection test. Returns USDT balance string or None."""
    if bybit is None:
        print("Bybit    : session not initialised (missing pybit/keys).")
        return None
    try:
        resp = bybit.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        lst = resp["result"]["list"]
        if lst:
            coins = lst[0].get("coin", [])
            usdt = next((c for c in coins if c["coin"] == "USDT"), None)
            bal = usdt["walletBalance"] if usdt else lst[0].get("totalEquity", "n/a")
            print(f"Bybit    : testnet connected. USDT balance: {bal}")
            return bal
        print("Bybit    : connected but no balance rows returned.")
        return None
    except Exception as e:
        print(f"Bybit    : balance check FAILED -> {e}")
        return None

def _bybit_get_filters(symbol):
    if symbol in _bybit_filters:
        return _bybit_filters[symbol]
    step, tick = 0.001, 0.001
    try:
        info = bybit.get_instruments_info(category="linear", symbol=symbol)
        item = info["result"]["list"][0]
        step = float(item["lotSizeFilter"]["qtyStep"])
        tick = float(item["priceFilter"]["tickSize"])
    except Exception:
        pass
    _bybit_filters[symbol] = {"step": step, "tick": tick}
    return _bybit_filters[symbol]

def bybit_has_liquidity(symbol):
    """True if the symbol has a live ticker (a real, tradeable market) on Bybit.
    Some symbols (e.g. AVAXUSDT, ATOMUSDT) exist on testnet but have no ticker /
    liquidity, so we treat those as signal-only rather than erroring on them."""
    if bybit is None:
        return False
    try:
        t = bybit.get_tickers(category="linear", symbol=symbol)
        lst = t["result"]["list"]
        return bool(lst) and float(lst[0].get("lastPrice") or 0) > 0
    except Exception:
        return False

def bybit_execution_agent(symbol, direction, entry, stop, target):
    """Place a market entry plus separate reduce-only stop-loss and take-profit
    conditional orders on Bybit linear perpetual testnet. Sized to ~$10 risk."""
    record = {
        "venue": "bybit", "symbol": symbol, "direction": direction,
        "entry": entry, "stop": stop, "target": target,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "PENDING", "orders": {},
    }

    if not symbol.endswith("USDT"):
        record["status"] = "SKIPPED"
        record["error"] = "non-crypto symbol not executable on Bybit linear"
        print(f"Exec    : skipped {symbol} (not a USDT linear pair).")
        log_execution(record)
        return record

    if bybit is None:
        record["status"] = "ERROR"
        record["error"] = "bybit session not initialised"
        print(f"Exec    : Bybit session unavailable, no order placed.")
        log_execution(record)
        return record

    # Signal still valid & tracked, but skip execution where testnet has no
    # liquidity (e.g. AVAXUSDT, ATOMUSDT) instead of erroring on the order.
    if not bybit_has_liquidity(symbol):
        record["status"] = "EXECUTION_SKIPPED"
        record["reason"] = ("no Bybit testnet liquidity for this symbol -- "
                            "signal tracked but not executed (will trade on mainnet)")
        print(f"Exec    : {symbol} has no Bybit testnet liquidity -> "
              f"EXECUTION_SKIPPED (signal-only).")
        log_execution(record)
        return record

    try:
        filt = _bybit_get_filters(symbol)
        rp = current_risk_percent()  # 1% normal, 0.5% in reduced-risk mode
        max_loss = ACCOUNT_SIZE * (rp / 100)
        stop_distance = abs(entry - stop)
        raw_qty = max_loss / stop_distance if stop_distance > 0 else 0
        qty = _round_step(raw_qty, filt["step"])
        record["quantity"] = qty
        record["risk_percent"] = rp
        record["risk_usdt"] = round(stop_distance * qty, 2)  # REAL $ risk after rounding
        if qty <= 0:
            record["status"] = "ERROR"
            record["error"] = "computed quantity rounded to zero"
            print(f"Exec    : {symbol} quantity rounded to zero, no order placed.")
            log_execution(record)
            return record

        side = "Buy" if direction == "BUY" else "Sell"
        close_side = "Sell" if side == "Buy" else "Buy"
        stop_px = _round_step(stop, filt["tick"])
        tp_px = _round_step(target, filt["tick"])
        # triggerDirection: 1 = price rising to trigger, 2 = price falling.
        if direction == "SELL":   # short: SL above (rise=1), TP below (fall=2)
            sl_dir, tp_dir = 1, 2
        else:                     # long: SL below (fall=2), TP above (rise=1)
            sl_dir, tp_dir = 2, 1

        # 1) Leverage (non-fatal; "leverage not modified" is fine)
        try:
            bybit.set_leverage(category="linear", symbol=symbol,
                               buyLeverage=str(BYBIT_LEVERAGE),
                               sellLeverage=str(BYBIT_LEVERAGE))
        except Exception as e:
            record["leverage_warning"] = str(e)

        # 2) Market entry
        entry_resp = bybit.place_order(
            category="linear", symbol=symbol, side=side,
            orderType="Market", qty=str(qty))
        record["orders"]["entry"] = entry_resp["result"].get("orderId")

        # 3) Stop loss (reduce-only conditional market)
        try:
            sl_resp = bybit.place_order(
                category="linear", symbol=symbol, side=close_side,
                orderType="Market", qty=str(qty), reduceOnly=True,
                triggerPrice=str(stop_px), triggerDirection=sl_dir,
                triggerBy="LastPrice")
            record["orders"]["stop"] = sl_resp["result"].get("orderId")
        except Exception as e:
            record["orders"]["stop"] = None
            record["stop_error"] = str(e)
            print(f"Exec    : {symbol} stop-loss order failed: {e}")

        # 4) Take profit (reduce-only conditional market)
        try:
            tp_resp = bybit.place_order(
                category="linear", symbol=symbol, side=close_side,
                orderType="Market", qty=str(qty), reduceOnly=True,
                triggerPrice=str(tp_px), triggerDirection=tp_dir,
                triggerBy="LastPrice")
            record["orders"]["take_profit"] = tp_resp["result"].get("orderId")
        except Exception as e:
            record["orders"]["take_profit"] = None
            record["tp_error"] = str(e)
            print(f"Exec    : {symbol} take-profit order failed: {e}")

        record["status"] = "PLACED"
        print(f"Exec    : [bybit] {symbol} {direction} qty {qty} | "
              f"entry#{record['orders'].get('entry')} "
              f"stop#{record['orders'].get('stop')} "
              f"tp#{record['orders'].get('take_profit')}")
    except Exception as e:
        record["status"] = "ERROR"
        record["error"] = str(e)
        print(f"Exec    : ORDER FAILED for {symbol} -> {e}")

    log_execution(record)
    return record

# ---------------------------------------------------------------------------
# Execution agent (Interactive Brokers paper trading via TWS / ib_insync)
# ---------------------------------------------------------------------------

_ibkr = None

def ibkr_connect():
    """Connect to TWS/IB Gateway (cached). Returns an IB instance or None."""
    global _ibkr
    if IB is None:
        return None
    if _ibkr is not None and _ibkr.isConnected():
        return _ibkr
    try:
        ib = IB()
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=8)
        _ibkr = ib
        return ib
    except Exception as e:
        print(f"IBKR    : connection failed -> {e}")
        _ibkr = None
        return None

def ibkr_check_connection():
    """Startup connectivity test. Returns True if connected."""
    ib = ibkr_connect()
    if ib is None:
        print("IBKR    : NOT connected -- start TWS/Gateway in PAPER mode on "
              f"port {IBKR_PORT} with API socket clients enabled.")
        return False
    try:
        summ = ib.accountSummary()
        nl = next((s.value for s in summ if s.tag == "NetLiquidation"), "n/a")
        print(f"IBKR    : connected (paper). NetLiquidation: {nl}")
    except Exception as e:
        print(f"IBKR    : connected; account query failed -> {e}")
    return True

def ibkr_contract(symbol):
    # AVAXUSDT -> Crypto('AVAX', 'PAXOS', 'USD')
    base = symbol.replace("USDT", "")
    return Crypto(base, "PAXOS", "USD")

def ibkr_open_position_symbols():
    """Symbols with a non-zero IBKR position, mapped back to *USDT names."""
    if IB is None:
        return set()
    ib = ibkr_connect()
    if ib is None:
        return set()
    try:
        return {p.contract.symbol + "USDT" for p in ib.positions()
                if float(p.position or 0) != 0}
    except Exception as e:
        print(f"IBKR    : positions query failed -> {e}")
        return set()

def ibkr_execution_agent(symbol, direction, entry, stop, target):
    """Place a market entry plus bracket stop-loss and take-profit on IBKR paper.
    Sized to ~$10 risk. Note: IBKR crypto is spot -- SELL/short will be rejected."""
    record = {
        "venue": "ibkr", "symbol": symbol, "direction": direction,
        "entry": entry, "stop": stop, "target": target,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "PENDING", "orders": {},
    }

    if not symbol.endswith("USDT"):
        record["status"] = "SKIPPED"
        record["error"] = "non-crypto symbol not handled by IBKR crypto agent"
        print(f"Exec    : skipped {symbol} (not a crypto pair).")
        log_execution(record)
        return record

    ib = ibkr_connect()
    if ib is None:
        record["status"] = "ERROR"
        record["error"] = "IBKR not connected (TWS/Gateway unavailable)"
        print(f"Exec    : IBKR not connected, no order placed.")
        log_execution(record)
        return record

    try:
        contract = ibkr_contract(symbol)
        try:
            ib.qualifyContracts(contract)
        except Exception as e:
            record["qualify_warning"] = str(e)

        rp = current_risk_percent()  # 1% normal, 0.5% in reduced-risk mode
        max_loss = ACCOUNT_SIZE * (rp / 100)
        stop_distance = abs(entry - stop)
        qty = round(max_loss / stop_distance, 6) if stop_distance > 0 else 0
        record["quantity"] = qty
        record["risk_percent"] = rp
        record["risk_usdt"] = round(stop_distance * qty, 2)  # REAL $ risk after rounding
        if qty <= 0:
            record["status"] = "ERROR"
            record["error"] = "computed quantity rounded to zero"
            print(f"Exec    : {symbol} quantity zero, no order placed.")
            log_execution(record)
            return record

        action = "BUY" if direction == "BUY" else "SELL"
        opp = "SELL" if action == "BUY" else "BUY"

        # Bracket: parent market (untransmitted) + TP limit + SL stop (transmits).
        parent = MarketOrder(action, qty)
        parent.transmit = False
        pt = ib.placeOrder(contract, parent)
        record["orders"]["entry"] = parent.orderId

        tp = LimitOrder(opp, qty, round(target, 6))
        tp.parentId = parent.orderId
        tp.transmit = False
        tt = ib.placeOrder(contract, tp)
        record["orders"]["take_profit"] = tp.orderId

        sl = StopOrder(opp, qty, round(stop, 6))
        sl.parentId = parent.orderId
        sl.transmit = True  # transmits the whole bracket
        st = ib.placeOrder(contract, sl)
        record["orders"]["stop"] = sl.orderId

        # Verify the orders actually went live -- placeOrder assigns IDs even when
        # TWS later rejects (read-only API, unknown contract, etc.).
        ib.sleep(1.5)
        statuses = {"entry": pt.orderStatus.status,
                    "take_profit": tt.orderStatus.status,
                    "stop": st.orderStatus.status}
        record["order_status"] = statuses
        errs = []
        for tr in (pt, tt, st):
            for le in getattr(tr, "log", []):
                code = getattr(le, "errorCode", 0)
                if code:
                    errs.append(f"{code}: {getattr(le, 'message', '')}")
        rejected = any(s in ("Cancelled", "Inactive", "ApiCancelled")
                       for s in statuses.values())
        if errs or rejected:
            record["status"] = "REJECTED"
            record["order_errors"] = errs[:5]
            print(f"Exec    : [ibkr] {symbol} REJECTED -> {statuses} | {errs[:2]}")
        else:
            record["status"] = "PLACED"
            print(f"Exec    : [ibkr] {symbol} {direction} qty {qty} | "
                  f"entry#{parent.orderId} stop#{sl.orderId} tp#{tp.orderId} "
                  f"({statuses['entry']})")
    except Exception as e:
        record["status"] = "ERROR"
        record["error"] = str(e)
        print(f"Exec    : ORDER FAILED for {symbol} -> {e}")

    log_execution(record)
    return record

def route_execution(symbol, direction, entry, stop, target):
    """Dispatch an approved trade to the configured execution venue."""
    if EXEC_VENUE == "ibkr":
        return ibkr_execution_agent(symbol, direction, entry, stop, target)
    if EXEC_VENUE == "bybit":
        return bybit_execution_agent(symbol, direction, entry, stop, target)
    return execution_agent(symbol, direction, entry, stop, target)

# ---------------------------------------------------------------------------
# Position cap + existing-position guard
# ---------------------------------------------------------------------------

def bybit_open_position_symbols():
    """Symbols with a live (size > 0) position on the Bybit testnet account."""
    if bybit is None:
        return set()
    try:
        resp = bybit.get_positions(category="linear", settleCoin="USDT")
        return {p["symbol"] for p in resp["result"]["list"]
                if float(p.get("size", 0) or 0) > 0}
    except Exception as e:
        print(f"Bybit    : get_positions failed -> {e}")
        return set()

def memory_open_symbols():
    """Symbols with an open VALIDATED paper position (watch positions excluded
    so they never consume the validated concurrent-position cap)."""
    mem = load_memory()
    return {p["pair"] for p in mem.get("paper_positions", []) if not p.get("watch")}

def watch_open_symbols():
    """Symbols with an open WATCH paper position."""
    mem = load_memory()
    return {p["pair"] for p in mem.get("paper_positions", []) if p.get("watch")}

def open_positions_overview():
    """Union of open paper positions and (only when executing) real venue ones."""
    mem_syms = memory_open_symbols()
    if EXECUTE_TRADES and EXEC_VENUE == "bybit":
        venue_syms = bybit_open_position_symbols()
    elif EXECUTE_TRADES and EXEC_VENUE == "ibkr":
        venue_syms = ibkr_open_position_symbols()
    else:
        venue_syms = set()
    return (mem_syms | venue_syms), mem_syms, venue_syms

def can_open_position(symbol):
    """Return (allowed, reason). Blocks if the symbol already has an open
    position, or if the concurrent-position cap is already reached."""
    combined, _mem, _venue = open_positions_overview()
    if symbol in combined:
        return False, f"already an open position on {symbol}"
    if len(combined) >= MAX_CONCURRENT_POSITIONS:
        return (False, f"position cap reached "
                       f"({len(combined)}/{MAX_CONCURRENT_POSITIONS}): {sorted(combined)}")
    return True, ""

# ---------------------------------------------------------------------------
# Paper-trading simulator (real live prices, no broker)
# ---------------------------------------------------------------------------

def paper_price(pair, market_type):
    """Current REAL live price -- mainnet Binance for crypto, yfinance otherwise."""
    try:
        if market_type == "crypto":
            return float(binance_data.get_symbol_ticker(symbol=pair)["price"])
        hist = yf.Ticker(pair).history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None

def paper_candle(pair, market_type):
    """Return (high, low, close) covering roughly the last rotation so stop/target
    can be checked against intracandle HIGH/LOW, not just the close. Uses the last
    two 15m candles (real mainnet prices for crypto)."""
    try:
        if market_type == "crypto":
            k = binance_data.get_klines(symbol=pair, interval="15m", limit=2)
            highs = [float(c[2]) for c in k]
            lows = [float(c[3]) for c in k]
            return max(highs), min(lows), float(k[-1][4])
        hist = yf.Ticker(pair).history(period="1d", interval="15m")
        if not hist.empty:
            tail = hist.tail(2)
            return (float(tail["High"].max()), float(tail["Low"].min()),
                    float(tail["Close"].iloc[-1]))
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# V3: Drawdown protection + risk sizing (measurement/safety only -- never
# changes signal rules, entry prices, or stop/target logic)
# ---------------------------------------------------------------------------

def paper_equity(mem):
    """Mark-to-market VALIDATED paper equity = start capital + realised +
    unrealised PnL. Watch positions are excluded so the watch experiment can
    never trip drawdown protection on the validated book."""
    realized = mem.get("paper_stats", {}).get("total_pnl", 0.0)
    unreal = sum(p.get("unrealized_pnl", 0.0)
                 for p in mem.get("paper_positions", []) if not p.get("watch"))
    return round(ACCOUNT_SIZE + realized + unreal, 2)

def update_risk_state():
    """Recompute peak equity, drawdown %, and risk MODE (NORMAL/REDUCED/PAUSED)
    with hysteresis. Returns the risk_state dict. Affects only position SIZE and
    whether NEW entries open -- never the signal rules or stop/target levels."""
    mem = load_memory()
    rs = mem.setdefault("risk_state", {"peak_equity": ACCOUNT_SIZE, "equity": ACCOUNT_SIZE,
                                       "drawdown_pct": 0.0, "paused": False,
                                       "reduce_risk": False, "mode": "NORMAL"})
    old_mode = rs.get("mode", "NORMAL")  # for transition-only notification
    equity = paper_equity(mem)
    peak = max(rs.get("peak_equity", ACCOUNT_SIZE), equity)
    dd = round((peak - equity) / peak * 100, 2) if peak > 0 else 0.0

    paused = rs.get("paused", False)
    if dd > DD_PAUSE_PCT:
        paused = True
    elif paused and dd < DD_RESUME_PCT:
        paused = False  # recovered enough to resume opening trades

    reduce_risk = (not paused) and dd > DD_REDUCE_PCT
    mode = "PAUSED" if paused else ("REDUCED" if reduce_risk else "NORMAL")
    rs.update({"peak_equity": round(peak, 2), "equity": equity, "drawdown_pct": dd,
               "paused": paused, "reduce_risk": reduce_risk, "mode": mode})
    mem["risk_state"] = rs
    save_memory(mem)
    # Notify-only: fire ONLY on an actual mode transition (never per rotation).
    if mode != old_mode:
        notify_drawdown_transition(old_mode, mode, rs)
    return rs

def current_risk_percent():
    """Effective risk % per NEW trade given the current drawdown mode."""
    mem = load_memory()
    return REDUCED_RISK_PERCENT if mem.get("risk_state", {}).get("reduce_risk") else MAX_RISK_PERCENT

def _bump_breakdown(mem, group, key, outcome, pnl):
    """Increment per-direction / per-regime / per-pair stats on a closed trade."""
    g = mem.setdefault("paper_breakdown", {}).setdefault(group, {}).setdefault(
        key or "UNKNOWN", {"wins": 0, "losses": 0, "pnl": 0.0})
    if outcome == "WIN":
        g["wins"] += 1
    else:
        g["losses"] += 1
    g["pnl"] = round(g.get("pnl", 0.0) + pnl, 2)

def today_signal_counts():
    """(approved, wait, rejected) for today's date, from the trades log."""
    today = time.strftime("%Y-%m-%d")
    a = w = r = 0
    for t in load_memory().get("trades", []):
        if not str(t.get("time", "")).startswith(today):
            continue
        d = str(t.get("decision", "")).upper()
        if "APPROVED" in d:
            a += 1
        elif "WAIT" in d:
            w += 1
        else:
            r += 1
    return a, w, r

# ---------------------------------------------------------------------------
# Paper position open (now risk-mode aware; sizing only -- prices untouched)
# ---------------------------------------------------------------------------

def open_paper_position(pair, market_type, direction, entry, stop, target, regime,
                        risk_percent=None, watch=False):
    """Record an APPROVED signal as an OPEN paper position. Position SIZE follows
    the current risk mode (1% normal, 0.5% reduced); entry/stop/target are passed
    through unchanged. Logs the REAL dollar risk after sizing (not an assumed $10).
    WATCH positions are tagged so they are tracked but kept out of validated stats."""
    mem = load_memory()
    mem.setdefault("paper_positions", [])
    rp = MAX_RISK_PERCENT if risk_percent is None else risk_percent
    max_loss = ACCOUNT_SIZE * (rp / 100)
    qty = round(max_loss / abs(entry - stop), 6) if entry != stop else 0
    actual_risk = round(abs(entry - stop) * qty, 2)  # real $ risk after rounding
    pos = {
        "id": int(time.time() * 1000),
        "pair": pair, "market_type": market_type, "direction": direction,
        "entry": entry, "stop": stop, "target": target, "qty": qty,
        "regime": regime, "entry_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "risk_percent": rp, "risk_usdt": actual_risk, "watch": watch,
        "current_price": entry, "unrealized_pnl": 0.0,
    }
    mem["paper_positions"].append(pos)
    save_memory(mem)
    if not watch:
        global _validated_opens_this_rotation
        _validated_opens_this_rotation += 1   # for the PAUSED-mode health check
    label = "OPEN[WATCH]" if watch else "OPEN"
    print(f"Paper   : {label} {direction} {pair} qty {qty} @ {entry} "
          f"(stop {stop} / target {target}) | risk ${actual_risk:.2f} ({rp}%)")
    notify_trade_opened(pos)  # notify-only
    return pos

def update_paper_positions():
    """Fetch live prices for open paper positions; close any that hit stop/target
    and record WIN/LOSS with actual USDT PnL. Refresh unrealized PnL otherwise."""
    mem = load_memory()
    positions = mem.get("paper_positions", [])
    if not positions:
        return
    ps = mem.setdefault("paper_stats", {"wins": 0, "losses": 0, "total_pnl": 0.0})
    still_open = []
    for p in positions:
        candle = paper_candle(p["pair"], p["market_type"])
        if candle is None:
            still_open.append(p)
            continue
        high, low, price = candle
        direction, entry = p["direction"], p["entry"]
        stop, target, qty = p["stop"], p["target"], p["qty"]
        # Check stop/target against the candle HIGH and LOW (intracandle touch),
        # conservatively checking the STOP before the TARGET within a candle.
        outcome, exitp = None, None
        if direction == "SELL":
            if high >= stop:
                outcome, exitp = "LOSS", stop
            elif low <= target:
                outcome, exitp = "WIN", target
        else:  # BUY
            if low <= stop:
                outcome, exitp = "LOSS", stop
            elif high >= target:
                outcome, exitp = "WIN", target

        if outcome:
            pnl = (entry - exitp) * qty if direction == "SELL" else (exitp - entry) * qty
            pnl = round(pnl, 2)
            is_watch = p.get("watch", False)
            closed = {k: v for k, v in p.items()
                      if k not in ("current_price", "unrealized_pnl")}
            closed.update({"exit_price": exitp, "outcome": outcome, "pnl_usdt": pnl,
                           "close_time": time.strftime("%Y-%m-%d %H:%M:%S")})
            mem.setdefault("paper_closed", []).append(closed)
            if is_watch:
                # WATCH: tracked in a SEPARATE record, never in validated stats.
                wps = mem.setdefault("watch_stats", {"wins": 0, "losses": 0, "total_pnl": 0.0})
                if outcome == "WIN":
                    wps["wins"] = wps.get("wins", 0) + 1
                else:
                    wps["losses"] = wps.get("losses", 0) + 1
                wps["total_pnl"] = round(wps.get("total_pnl", 0.0) + pnl, 2)
                print(f"Paper   : CLOSE[WATCH] {direction} {p['pair']} -> {outcome} "
                      f"@ {exitp} | PnL ${pnl:+.2f}")
                w_closed = wps.get("wins", 0) + wps.get("losses", 0)
                w_wr = round(wps.get("wins", 0) / w_closed * 100, 1) if w_closed else 0.0
                notify_trade_closed(closed, w_wr, w_closed)  # notify-only
            else:
                if outcome == "WIN":
                    ps["wins"] = ps.get("wins", 0) + 1
                else:
                    ps["losses"] = ps.get("losses", 0) + 1
                ps["total_pnl"] = round(ps.get("total_pnl", 0.0) + pnl, 2)
                # V3: per-direction / per-regime / per-pair breakdown stats.
                _bump_breakdown(mem, "by_direction", direction, outcome, pnl)
                _bump_breakdown(mem, "by_regime", p.get("regime"), outcome, pnl)
                _bump_breakdown(mem, "by_pair", p.get("pair"), outcome, pnl)
                print(f"Paper   : CLOSE {direction} {p['pair']} -> {outcome} "
                      f"@ {exitp} | PnL ${pnl:+.2f}")
                v_closed = ps.get("wins", 0) + ps.get("losses", 0)
                v_wr = round(ps.get("wins", 0) / v_closed * 100, 1) if v_closed else 0.0
                notify_trade_closed(closed, v_wr, v_closed)  # notify-only
        else:
            unreal = (entry - price) * qty if direction == "SELL" else (price - entry) * qty
            p["current_price"] = round(price, 6)
            p["unrealized_pnl"] = round(unreal, 2)
            still_open.append(p)

    mem["paper_positions"] = still_open
    save_memory(mem)

def print_paper_portfolio():
    """Print the paper portfolio: open positions w/ PnL, closed record, total PnL."""
    mem = load_memory()
    all_open = mem.get("paper_positions", [])
    opens = [p for p in all_open if not p.get("watch")]      # validated
    watch_open = [p for p in all_open if p.get("watch")]     # watch-only
    ps = mem.get("paper_stats", {"wins": 0, "losses": 0, "total_pnl": 0.0})
    wins, losses = ps.get("wins", 0), ps.get("losses", 0)
    realized = ps.get("total_pnl", 0.0)
    unreal_total = round(sum(p.get("unrealized_pnl", 0.0) for p in opens), 2)
    closed_total = wins + losses
    wr = round(wins / closed_total * 100, 1) if closed_total else 0.0
    print("----- PAPER PORTFOLIO (real live prices, VALIDATED) -----")
    print(f"Open: {len(opens)}/{MAX_CONCURRENT_POSITIONS}")
    for p in opens:
        print(f"  {p['direction']:<4} {p['pair']:<9} entry {p['entry']} "
              f"now {p.get('current_price', '?')} | uPnL ${p.get('unrealized_pnl', 0.0):+.2f}")
    print(f"Closed: {closed_total} ({wins}W/{losses}L, {wr}% win)")
    print(f"Realized: ${realized:+.2f} | Unrealized: ${unreal_total:+.2f} "
          f"| TOTAL PAPER PnL: ${round(realized + unreal_total, 2):+.2f}")
    # WATCH track record -- separate, NOT validated.
    wps = mem.get("watch_stats", {"wins": 0, "losses": 0, "total_pnl": 0.0})
    w_closed = wps.get("wins", 0) + wps.get("losses", 0)
    w_wr = round(wps.get("wins", 0) / w_closed * 100, 1) if w_closed else 0.0
    w_unreal = round(sum(p.get("unrealized_pnl", 0.0) for p in watch_open), 2)
    print(f"-- WATCH (not validated): open {len(watch_open)} {sorted(WATCH_PAIRS)} | "
          f"closed {w_closed} ({wps.get('wins',0)}W/{wps.get('losses',0)}L, {w_wr}%) | "
          f"realized ${wps.get('total_pnl',0.0):+.2f} | unreal ${w_unreal:+.2f}")
    print("----------------------------------------------")

def write_dashboard_state(regime):
    """Write a compact dashboard snapshot (regime, totals, timestamp) to memory
    each rotation so trading_dashboard.html can render live PnL. Open positions
    and their current prices are already persisted by update_paper_positions."""
    mem = load_memory()
    all_open = mem.get("paper_positions", [])
    opens = [p for p in all_open if not p.get("watch")]   # validated only
    watch_open = [p for p in all_open if p.get("watch")]
    ps = mem.get("paper_stats", {"wins": 0, "losses": 0, "total_pnl": 0.0})
    wins, losses = ps.get("wins", 0), ps.get("losses", 0)
    realized = round(ps.get("total_pnl", 0.0), 2)
    unreal = round(sum(p.get("unrealized_pnl", 0.0) for p in opens), 2)
    closed_total = wins + losses
    # WATCH summary (separate, not validated)
    wps = mem.get("watch_stats", {"wins": 0, "losses": 0, "total_pnl": 0.0})
    w_closed = wps.get("wins", 0) + wps.get("losses", 0)
    # LLM usage / savings (steps 2-4): Haiku analyst vs Sonnet risk, pre-filter
    # skips, and approx Analyst calls saved vs the old 2-Sonnet-call design.
    st = mem.get("stats", {})
    haiku = st.get("analyst_haiku_calls", 0)
    skips = st.get("prefilter_skips", 0)
    mem["dashboard"] = {
        "regime": regime,
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "open_count": len(opens),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / closed_total * 100, 1) if closed_total else 0.0,
        "realized_pnl": realized,
        "unrealized_pnl": unreal,
        "total_pnl": round(realized + unreal, 2),
        # V3 observability: risk/drawdown state + breakdown stats.
        "risk_state": mem.get("risk_state", {}),
        "breakdown": mem.get("paper_breakdown", {}),
        # WATCH track record (not validated) for the dashboard WATCH section.
        "watch": {
            "pairs": WATCH_PAIRS,
            "open_count": len(watch_open),
            "wins": wps.get("wins", 0),
            "losses": wps.get("losses", 0),
            "win_rate": round(wps.get("wins", 0) / w_closed * 100, 1) if w_closed else 0.0,
            "realized_pnl": round(wps.get("total_pnl", 0.0), 2),
            "unrealized_pnl": round(sum(p.get("unrealized_pnl", 0.0) for p in watch_open), 2),
        },
        # LLM usage / savings tracking (Haiku analyst, Sonnet risk, pre-filter).
        "llm_usage": {
            "analyst_haiku_calls": haiku,
            "risk_sonnet_calls": st.get("risk_sonnet_calls", 0),
            "news_calls": st.get("news_calls", 0),
            "prefilter_skips": skips,
            # Old design = 2 Sonnet Analyst calls per scanned pair, no pre-filter.
            "analyst_calls_saved_vs_old": haiku + 2 * skips,
        },
    }
    save_memory(mem)

# ---------------------------------------------------------------------------
# Self-diagnostic health check (PURE PYTHON, zero API/LLM calls)
# ---------------------------------------------------------------------------

def run_health_check(rotation_count):
    """Assert invariants against data already in memory + in-process state.
    Writes memory['health'] and prints a warning block on any failure. No API."""
    global _freshness_prev
    mem = load_memory()
    fails = []
    now_ms = time.time() * 1000
    positions = mem.get("paper_positions", [])
    closed = mem.get("paper_closed", [])
    ps = mem.get("paper_stats", {})
    wps = mem.get("watch_stats", {})
    rs = mem.get("risk_state", {})

    # 1. STOP/TARGET INTEGRITY
    for p in positions:
        d, e, s, t = p.get("direction"), p.get("entry"), p.get("stop"), p.get("target")
        if None in (e, s, t):
            continue
        if d == "SELL" and not (s > e and t < e):
            fails.append(f"STOP/TARGET: {p.get('pair')} SELL inverted (entry {e}, stop {s}, target {t})")
        if d == "BUY" and not (s < e and t > e):
            fails.append(f"STOP/TARGET: {p.get('pair')} BUY inverted (entry {e}, stop {s}, target {t})")

    # 2. PNL RECONCILIATION (live and watch reconciled SEPARATELY)
    live_sum = round(sum(t.get("pnl_usdt", 0) for t in closed if not t.get("watch")), 2)
    if abs(live_sum - round(ps.get("total_pnl", 0), 2)) > 0.01:
        fails.append(f"PNL RECON (live): closed-sum {live_sum} != paper_stats {ps.get('total_pnl')}")
    watch_sum = round(sum(t.get("pnl_usdt", 0) for t in closed if t.get("watch")), 2)
    if abs(watch_sum - round(wps.get("total_pnl", 0), 2)) > 0.01:
        fails.append(f"PNL RECON (watch): closed-sum {watch_sum} != watch_stats {wps.get('total_pnl')}")

    # 3. DRAWDOWN STATE CONSISTENCY
    if rs.get("mode") == "PAUSED" and _validated_opens_this_rotation > 0:
        fails.append(f"DRAWDOWN: mode PAUSED but {_validated_opens_this_rotation} validated "
                     f"position(s) opened this rotation")
    if rs.get("equity") is not None and rs.get("peak_equity") is not None:
        if rs["equity"] > rs["peak_equity"] + 0.01:
            fails.append(f"DRAWDOWN: equity {rs['equity']} exceeds peak {rs['peak_equity']} "
                         f"(peak not updated)")

    # 4. FILTER INTEGRITY (approved trades)
    for tr in mem.get("trades", []):
        if "APPROVED" not in str(tr.get("decision", "")):
            continue
        if tr.get("utc_hour") in (5, 21):
            fails.append(f"FILTER: approved {tr.get('pair')} during dead-zone UTC hour "
                         f"{tr.get('utc_hour')} ({tr.get('time')})")
        if tr.get("direction") == "SELL" and tr.get("regime") == "WEAK_BULL" and not tr.get("watch"):
            fails.append(f"FILTER: validated SELL approved in WEAK_BULL "
                         f"({tr.get('pair')} {tr.get('time')}) -- should be SKIPPED")

    # 5. DATA FRESHNESS (from in-process _freshness; NO API calls)
    for sym, info in _freshness.items():
        if info.get("error"):
            fails.append(f"DATA FRESHNESS: {sym} fetch error/empty: {info['error']}")
            continue
        age = (now_ms - info.get("candle_ts", now_ms)) / 60000.0
        if age > DATA_STALE_MIN:
            fails.append(f"DATA FRESHNESS: {sym} latest 15m candle {age:.0f} min old (stale feed)")
        prev = _freshness_prev.get(sym)
        if (prev and not prev.get("error")
                and prev.get("candle_ts") == info.get("candle_ts")
                and prev.get("price") == info.get("price") and age > DATA_DUP_MIN):
            fails.append(f"DATA FRESHNESS: {sym} candle+price unchanged since last rotation "
                         f"({age:.0f} min old -- possible duplicate/stuck feed)")

    # 6. POSITION INTEGRITY
    seen = {}
    for p in positions:
        key = (p.get("pair"), bool(p.get("watch")))
        seen[key] = seen.get(key, 0) + 1
    for (pair, w), c in seen.items():
        if c > 1:
            fails.append(f"POSITION: {c} duplicate open positions on {pair} "
                         f"({'watch' if w else 'validated'})")
    validated_open = sum(1 for p in positions if not p.get("watch"))
    if validated_open > MAX_CONCURRENT_POSITIONS:
        fails.append(f"POSITION: validated open count {validated_open} exceeds cap "
                     f"{MAX_CONCURRENT_POSITIONS}")

    # 7. RUNTIME ERRORS captured this rotation (change C) -- surfaced here so an
    # unattended run keeps a clear record of every transient API/network failure.
    for e in _runtime_errors:
        fails.append(f"RUNTIME: {e}")

    status = "OK" if not fails else "WARNING"

    # Consecutive-failure counter (read OLD value before overwriting health).
    # Used by the kill-switch's "repeated health failures" trigger. This is
    # measurement only -- it does not change any check result.
    prev_consecutive = mem.get("health", {}).get("consecutive_failures", 0)
    consecutive_failures = (prev_consecutive + 1) if fails else 0

    # Notify-only dedup: only alert when the failure set CHANGES, so the same
    # warning is not resent every rotation. Reset on OK so a recurrence re-alerts.
    sig = "|".join(sorted(fails)) if fails else ""
    nf = mem.setdefault("notify", {})
    should_notify_health = bool(fails) and sig != nf.get("last_health_sig", "")
    nf["last_health_sig"] = sig

    mem["health"] = {
        "status": status,
        "last_check": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "rotation": rotation_count,
        "checks_run": 7,
        "failures": fails,
        "consecutive_failures": consecutive_failures,
    }
    save_memory(mem)
    _freshness_prev = dict(_freshness)  # snapshot for next rotation's advancement check

    if fails:
        print("\n*** HEALTH WARNING ***")
        for f in fails:
            print(f"  - {f}")
        print("*** END HEALTH WARNING ***\n")
        if should_notify_health:
            notify_health(fails)  # notify-only (deduplicated)
    else:
        print(f"Health  : OK ({mem['health']['checks_run']} checks passed)")
    return mem["health"]

# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def _parse_indicators(data):
    """Pull (price, rsi, below_vwap, above_vwap) out of a get_market_data()
    block. Returns None if the block is an error / unparseable. Reads the SAME
    'ABOVE/BELOW VWAP' token the data formatter already computed (price vs vwap),
    so the pre-filter uses the EXACT same comparison the rest of the system uses."""
    try:
        price_line = next(l for l in data.split("\n") if "Price:" in l)
        price = float(price_line.split("Price:")[1].split("|")[0].strip().replace("$", ""))
        rsi_line = next(l for l in data.split("\n") if "RSI(14):" in l)
        rsi = float(rsi_line.split("RSI(14):")[1].split("|")[0].strip())
        below_vwap = "BELOW VWAP" in data
        above_vwap = "ABOVE VWAP" in data
        return price, rsi, below_vwap, above_vwap
    except Exception:
        return None

def scan_market(symbol, market_type, regime4, risk_mode="NORMAL", watch=False):
    label = "WATCH" if watch else market_type
    print(f"--- {symbol} ({label}) ---")

    # 2-state regime drives the EXISTING BUY/SELL gating + analyst/risk prompts
    # (unchanged). The 4-state regime drives SELL filtering, sizing and stats.
    regime2 = "BULL" if regime4 in BULL_STATES else "BEAR"

    # Time-of-day filter: skip dead-zone UTC hours flagged by the simulation.
    if in_dead_zone():
        print(f"Skip    : {datetime.now(timezone.utc).strftime('%H:%M')} UTC is in the "
              f"dead-zone {sorted(DEAD_ZONE_UTC_HOURS)} -- no signals this hour.\n")
        return

    # Fetch market data ONCE, up front. This single snapshot is used for (a) the
    # pre-filter, (b) the data passed inline to the Analyst (no more tool round
    # trip), and (c) the entry price. It also records data-freshness for the
    # health check. Entry price / stop / target math downstream is unchanged.
    data = get_market_data(symbol, market_type)
    ind = _parse_indicators(data)
    if ind is None:
        print("Could not parse market data (feed error?) -- skipping.\n")
        return
    current_price, rsi, below_vwap, above_vwap = ind

    # PRE-FILTER (provably behaviour-neutral): skip the Analyst call ONLY when a
    # valid signal is MATHEMATICALLY IMPOSSIBLE, using the EXACT live thresholds.
    #   SELL possible iff  RSI > 55 AND price < VWAP
    #   BUY  possible iff  RSI < 45 AND price > VWAP
    # In BEAR (or ALLOW_LONGS off) only SELL can ever fire, so only SELL counts.
    # A trade only opens if risk_agent APPROVES, and APPROVAL requires these same
    # RSI/VWAP conditions (plus momentum) -- so this skip set is a strict subset
    # of "no valid signal exists" and can NEVER drop a candle the Analyst could
    # have turned into a valid signal.
    sell_possible = rsi > 55 and below_vwap
    buy_possible = rsi < 45 and above_vwap
    if regime2 == "BULL" and ALLOW_LONGS:
        signal_possible = sell_possible or buy_possible
    else:
        signal_possible = sell_possible
    if not signal_possible:
        vwap_side = "below" if below_vwap else "above"
        print(f"PRE-FILTER-SKIP: {symbol} no signal possible "
              f"(RSI {rsi}, price {vwap_side} VWAP) -- Analyst not called.\n")
        # Counted as WAIT so stats stay IDENTICAL to before; tallied separately
        # (prefilter_skips) for savings tracking only.
        save_trade(symbol, "PRE-FILTER-SKIP: no signal possible", "N/A", "WAIT", 0,
                   market_type=market_type, regime=regime4, watch=watch)
        bump_stat("prefilter_skips")
        return

    # Analyst: SINGLE Haiku call, market data passed inline (same data as before).
    signal = analyst_agent(symbol, market_type, regime2, data)
    print(f"Analyst : {signal[:120]}")

    # Robust, formatting-safe parse (same rules, reliable read).
    parsed = parse_signal(signal)
    if regime2 == "BULL" and ALLOW_LONGS:
        actionable = parsed in ("BUY", "SELL")
    else:
        actionable = parsed == "SELL"

    if not actionable:
        print(f"Risk    : No actionable setup ({regime4}) -> WAIT.\n")
        save_trade(symbol, signal, "N/A", "WAIT", 0, market_type=market_type,
                   regime=regime4, watch=watch)
        return

    # REGIME-STRENGTH FILTER (validated SELLs only): skip regimes with validated
    # negative edge (WEAK_BULL). Logged as SKIPPED -- NOT an approval or rejection.
    # BUY is never filtered (no BUY validation yet). WATCH pairs are EXEMPT: they
    # still record WEAK_BULL shorts so we can measure whether the negative edge
    # also holds for them before promoting -- their results stay in watch_stats.
    if parsed == "SELL" and not watch and REGIME_SELL_RISK_PCT.get(regime4) is None:
        print(f"Risk    : SKIPPED - {regime4} negative edge (no SELL).\n")
        save_trade(symbol, signal, "N/A", f"SKIPPED - {regime4} negative edge", 0,
                   market_type=market_type, regime=regime4, watch=watch)
        return

    # News is informational only - printed but not passed to risk agent (cached 45m)
    sentiment, cached = get_cached_news(symbol, market_type)
    tag = "[CACHED] " if cached else ""
    print(f"News    : {tag}{sentiment[:100]} [INFO ONLY]")

    # Entry price comes from the single up-front snapshot parsed above; stop/
    # target math inside risk_agent is unchanged.
    decision, stop, target, direction = risk_agent(symbol, signal, current_price, market_type, regime2)
    print(f"Risk    : {decision[:120]}")

    if "APPROVED" in decision:
        save_trade(symbol, signal, sentiment, "APPROVED",
                   current_price, stop, target, direction, market_type, regime4, watch=watch)
        tag = "LONG" if direction == "BUY" else "SHORT"
        flag = "  [EXPERIMENTAL - regime-gated long]" if direction == "BUY" else ""
        wtag = " [WATCH - NOT VALIDATED]" if watch else ""
        print(f"\n*** TRADE ALERT ({tag}){flag}{wtag} ***")
        print(f"Instrument: {symbol} ({market_type})")
        print(f"Regime    : {regime4}")
        print(f"Direction : {direction}")
        print(f"Entry     : {current_price}")
        print(f"Stop      : {stop} | Target: {target} | R:R 2:1")

        if watch:
            # WATCH candidate: tracked separately, never executed, never counted
            # against the validated cap or drawdown. Just dedupe per symbol.
            if symbol in watch_open_symbols():
                print(f"Paper   : WATCH already tracking {symbol}, no duplicate opened.")
            else:
                open_paper_position(symbol, market_type, direction, current_price,
                                    stop, target, regime4, watch=True)
        else:
            # Validated path: position guard, drawdown risk mode, real execution.
            allowed, reason = can_open_position(symbol)
            if risk_mode == "PAUSED":
                # Drawdown pause: signal recorded, but NO new entry opens.
                print(f"Paper   : PAUSED (drawdown protection) -> signal logged, NO new entry.")
            elif allowed:
                # SELL size follows regime-strength map; BUY size unchanged (no
                # BUY validation). Drawdown REDUCED mode halves either on top.
                if direction == "SELL":
                    base_rp = REGIME_SELL_RISK_PCT.get(regime4) or MAX_RISK_PERCENT
                    rp = round(base_rp * 0.5, 3) if risk_mode == "REDUCED" else base_rp
                else:
                    rp = REDUCED_RISK_PERCENT if risk_mode == "REDUCED" else MAX_RISK_PERCENT
                open_paper_position(symbol, market_type, direction,
                                    current_price, stop, target, regime4, risk_percent=rp)
                # Real broker execution only when EXECUTE_TRADES is on (off here).
                if EXECUTE_TRADES and market_type == "crypto":
                    route_execution(symbol, direction, current_price, stop, target)
            else:
                print(f"Paper   : SKIP open -> {reason}")
        print()
    else:
        save_trade(symbol, signal, sentiment, "REJECTED",
                   current_price, stop, target, direction, market_type, regime4, watch=watch)
    print()

# ---------------------------------------------------------------------------
# Dashboard HTTP server (background thread)
# ---------------------------------------------------------------------------

DASHBOARD_PORT = 8000

class _QuietHandler(SimpleHTTPRequestHandler):
    # Suppress per-request logging so it doesn't spam the trading output.
    def log_message(self, *args):
        pass

def start_dashboard_server():
    """Serve THIS folder over HTTP in a daemon thread so the dashboard can fetch
    agent_memory.json. Daemon + atexit ensures it stops cleanly with the script."""
    folder = os.path.dirname(os.path.abspath(__file__))
    handler = partial(_QuietHandler, directory=folder)
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", DASHBOARD_PORT), handler)
    except OSError as e:
        print(f"Dashboard: port {DASHBOARD_PORT} unavailable ({e}); "
              f"server not started (it may already be running).")
        return None
    threading.Thread(target=httpd.serve_forever, daemon=True,
                     name="dashboard-http").start()
    atexit.register(httpd.shutdown)
    print(f"Dashboard: http://localhost:{DASHBOARD_PORT}/trading_dashboard.html")
    return httpd

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def llm_savings_line():
    """One-line LLM usage / savings summary for the console (step 4)."""
    s = load_memory().get("stats", {})
    skips = s.get("prefilter_skips", 0)
    haiku = s.get("analyst_haiku_calls", 0)
    sonnet_risk = s.get("risk_sonnet_calls", 0)
    # Old design: every scanned pair cost 2 Sonnet Analyst calls and there was
    # no pre-filter. New: skipped pairs cost 0; the rest cost 1 Haiku call each.
    saved = haiku + 2 * skips  # (haiku + skips)*2 old  -  haiku new
    return (f"LLM   : Analyst(Haiku) {haiku} | Risk(Sonnet) {sonnet_risk} | "
            f"pre-filter skips {skips} | ~{saved} Analyst calls saved vs old design (cum.)")

def _run_all_scans(regime4, risk_mode):
    """Run every per-pair scan_market for this rotation. Extracted so the
    kill-switch can gate ALL new-trade scanning with a single, readable guard.
    Each scan_market is individually wrapped so one bad pair never aborts the
    rest of the rotation (change C)."""
    print("=== CRYPTO (24/7) ===")
    for pair in CRYPTO_PAIRS:
        try:
            scan_market(pair, "crypto", regime4, risk_mode)
        except Exception as e:
            log_runtime_error(f"scan {pair}", e)
        time.sleep(8)

    # WATCH candidates: signals tracked in paper SEPARATELY (not validated,
    # not executed, not counted in validated stats/cap/drawdown).
    if WATCH_PAIRS:
        print("=== WATCH (candidates -- NOT validated) ===")
        for pair in WATCH_PAIRS:
            try:
                scan_market(pair, "crypto", regime4, risk_mode, watch=True)
            except Exception as e:
                log_runtime_error(f"scan {pair} (watch)", e)
            time.sleep(8)

    # STOCKS and FOREX are gated behind toggles (default off) -- no edge in
    # backtesting. Flip TRADE_STOCKS / TRADE_FOREX to True to re-enable.
    if TRADE_STOCKS:
        if is_us_market_open():
            print("=== STOCKS (Market Open) ===")
            for stock in STOCKS:
                try:
                    scan_market(stock, "stock", regime4, risk_mode)
                except Exception as e:
                    log_runtime_error(f"scan {stock}", e)
                time.sleep(8)
        else:
            print("=== STOCKS (Market Closed) ===\n")

    if TRADE_OIL and COMMODITIES:
        if is_commodity_open():
            print("=== COMMODITIES (WTI crude, PROVISIONAL) ===")
            for commodity in COMMODITIES:
                try:
                    scan_market(commodity, "commodity", regime4, risk_mode)
                except Exception as e:
                    log_runtime_error(f"scan {commodity}", e)
                time.sleep(8)
        else:
            print("=== COMMODITIES (Weekend Closed) ===\n")

    if TRADE_FOREX:
        if is_forex_open():
            print("=== FOREX ===")
            for pair in FOREX:
                try:
                    scan_market(pair, "forex", regime4, risk_mode)
                except Exception as e:
                    log_runtime_error(f"scan {pair}", e)
                time.sleep(8)
        else:
            print("=== FOREX (Weekend Closed) ===\n")

def run_one_rotation(rotation_count):
    """One full scan cycle. Wrapped by the caller in try/except (change C) so a
    transient API/network error in any single pair or rotation logs and
    continues, rather than crashing the whole loop. Returns the (possibly
    incremented) rotation_count."""
    global _validated_opens_this_rotation
    # Reset the per-rotation opened-positions counter (used by the health check).
    _validated_opens_this_rotation = 0

    # Update the paper portfolio against real live prices (close any hits).
    update_paper_positions()

    # --- V3: recompute drawdown / risk mode against fresh equity ---
    rs = update_risk_state()
    risk_mode = rs["mode"]

    # --- KILL SWITCH: evaluate the circuit breaker against fresh state. If it
    # trips (or was already tripped), NO new trades are scanned/opened this
    # rotation; existing open positions were already marked-to-market above and
    # continue to be tracked to completion. Requires a manual restart. ---
    killed = evaluate_kill_switch()

    # --- Detect 4-state market regime at the start of each rotation ---
    regime4, btc_price, btc_ema, btc_slope = get_btc_regime()
    regime2 = "BULL" if regime4 in BULL_STATES else "BEAR"
    sell_action = ("SELL SKIPPED (negative edge)"
                   if REGIME_SELL_RISK_PCT.get(regime4) is None else "SELL enabled")
    if btc_price is not None:
        sl = "rising" if (btc_slope or 0) > 0 else "falling" if (btc_slope or 0) < 0 else "flat"
        print(f"REGIME: {regime4}  (BTC {btc_price} vs 50-4h-EMA {btc_ema}, "
              f"slope {btc_slope} {sl})")
        print(f"        -> base {regime2}: "
              f"{'BUY + SELL' if regime2 == 'BULL' else 'SELL only'}; {sell_action}")
    else:
        print(f"REGIME: {regime4} (BTC data unavailable -- conservative default)")

    # --- V3: rich rotation status line + drawdown report ---
    open_n = len(load_memory().get("paper_positions", []))
    a, w, r = today_signal_counts()
    risk_label = {"NORMAL": "NORMAL (1%)", "REDUCED": "REDUCED (0.5%)",
                  "PAUSED": "PAUSED (no new entries)"}[risk_mode]
    print(f"STATUS | Regime: {regime4} | Risk: {risk_label} | "
          f"Equity: ${rs['equity']:.2f} | Drawdown: {rs['drawdown_pct']}% | "
          f"Open: {open_n} | Today: {a} approved / {w} wait / {r} rejected")
    print(f"Equity: ${rs['equity']:.2f} | Peak: ${rs['peak_equity']:.2f} | "
          f"Drawdown: {rs['drawdown_pct']}% (reduce>{DD_REDUCE_PCT}% / "
          f"pause>{DD_PAUSE_PCT}% / resume<{DD_RESUME_PCT}%)")

    print(f"Stats: {get_stats()}")
    print(llm_savings_line())
    print_paper_portfolio()
    write_dashboard_state(regime4)  # persist snapshot for the HTML dashboard
    now = datetime.now(pytz.timezone("Europe/London"))
    print(f"Time: {now.strftime('%H:%M')} UK\n")

    # Open NEW trades only when the kill switch has NOT tripped. Existing
    # positions are tracked regardless (update_paper_positions above).
    if killed:
        print("KILL SWITCH ACTIVE -- skipping all scans; no new trades will be "
              "opened. Existing positions still tracked. Manual restart required.\n")
    else:
        _run_all_scans(regime4, risk_mode)

    rotation_count += 1
    print(f"--- Rotation {rotation_count} complete ---")

    # Pure-Python self-diagnostic (no API calls) at the END of each rotation.
    # It also flushes this rotation's runtime errors into the health section.
    run_health_check(rotation_count)

    # Skip the (LLM) reflection agent while killed -- the desk is halted.
    if rotation_count % 10 == 0 and not killed:
        print("Running reflection agent...")
        reflection_agent()

    # Notify-only: weekly template digest (sends at most once / 7 days).
    maybe_send_weekly_digest()

    return rotation_count

def run_trading_team():
    start_dashboard_server()  # one command starts everything
    load_news_cache()         # restore the 45-min news cache across restarts

    print("=" * 50)
    print("PRODUCTION TRADING SYSTEM - V3 (REGIME-AWARE + SAFETY LAYER)")
    print("Adds: drawdown protection, real-risk logging, BUY/SELL & regime & pair "
          "stats, robust signal parsing -- signal rules UNCHANGED")
    print(f"Crypto top-4 | Watch {WATCH_PAIRS} (paper-only) | "
          f"Oil(WTI) {'ON' if TRADE_OIL else 'OFF'} | "
          f"Stocks {'ON' if TRADE_STOCKS else 'OFF'} | Forex {'ON' if TRADE_FOREX else 'OFF'}")
    print(f"EXECUTION: {('LIVE on ' + EXEC_VENUE + ' testnet') if EXECUTE_TRADES else 'OFF (signal-only)'}"
          f" | Longs {'ON' if ALLOW_LONGS else 'OFF'}")
    print("PAPER TRADER: ON -- internal simulator, real live prices, no broker")
    print(f"TELEGRAM: {'ON' if TELEGRAM_ENABLED else 'OFF (token/chat id not set)'}"
          f" | trades {NOTIFY_TRADES} drawdown {NOTIFY_DRAWDOWN} "
          f"health {NOTIFY_HEALTH} digest {NOTIFY_DIGEST}")
    print(f"KILL SWITCH: {'ON' if KILL_SWITCH_ENABLED else 'OFF'} "
          f"(floor {KILL_HARD_DRAWDOWN_PCT}% dd / {KILL_HEALTH_CONSECUTIVE} "
          f"health fails / {KILL_LOSS_RISK_MULTIPLE}x-risk anomaly)")
    if KILL_SWITCH_ENABLED and is_killed():
        print("!!! KILL SWITCH IS LATCHED from a previous run -- NO new trades "
              "will open until clear_kill_switch() is called. !!!")

    # --- Execution readiness gate: only relevant when EXECUTE_TRADES is on ---
    if EXECUTE_TRADES:
        print(f">>> EXECUTION IS LIVE on venue '{EXEC_VENUE}'. Verify this is correct. <<<")
        if EXEC_VENUE in KNOWN_INCOMPATIBLE:
            print(f"!!! WARNING: '{EXEC_VENUE}' is flagged incompatible with this "
                  f"strategy: {KNOWN_INCOMPATIBLE[EXEC_VENUE]}")
            print("    Orders will likely be rejected. Set EXEC_VENUE correctly or "
                  "EXECUTE_TRADES=False.")
        if EXEC_VENUE == "bybit":
            bybit_check_balance()
        elif EXEC_VENUE == "ibkr":
            ibkr_check_connection()
    print()
    print("BEAR regime: SELL-only | BULL regime: BUY+SELL (longs experimental)")
    print("Crypto 43.5% win/+0.18R (6mo VALIDATED) | Oil 47%/+0.39R (60d PROVISIONAL)")
    print("=" * 50 + "\n")

    rotation_count = 0

    while True:
        # Clear this rotation's runtime-error queue, then run one full cycle
        # wrapped in try/except (change C). A transient API/network error logs
        # clearly + records to the health section, and the loop keeps running
        # unattended instead of crashing. The 15-min sleep ALWAYS runs.
        _runtime_errors.clear()
        try:
            rotation_count = run_one_rotation(rotation_count)
        except Exception as e:
            log_runtime_error("rotation", e)
            record_runtime_errors_to_health(rotation_count)

        print("Waiting 15 minutes...\n")
        time.sleep(900)

if __name__ == "__main__":
    run_trading_team()
