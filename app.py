import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# V5.3 PRO — Conservative, no-lookahead, multi-provider engine
# ============================================================
# Research / paper trading only. No real orders.
# Design goals:
#   - closed-candle signals only
#   - no same-candle entry/exit look-ahead in backtests
#   - risk-based compounding rather than fixed notional
#   - 2% hard SL / 4% base TP
#   - trailing activates only after meaningful profit
#   - break-even protection
#   - early exit only after confirmation
#   - BTC/ETH stricter quality filter
#   - provider fallback: OKX -> KuCoin -> Kraken -> Coinbase
#   - 2H constructed from closed 1H candles
#   - SQLite active-trade persistence
# ============================================================

st.set_page_config(
    page_title="V5.3 Crypto Trading Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

COINS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
TIMEFRAMES = ["1h", "2h", "4h", "1d"]
PROVIDERS = ["okx", "kucoin", "kraken", "coinbase"]
DB_FILE = "v5_trades.db"
START_CAPITAL = 10_000.0  # INR account equity
RISK_PER_TRADE = 0.01      # 1% of current equity
MAX_OPEN_TRADES = 2
INITIAL_SL_PCT = 0.02
BASE_TP_PCT = 0.04
TRAIL_ACTIVATION_PCT = 0.02
TRAIL_ATR_MULT = 2.0
BREAK_EVEN_BUFFER = 0.001   # 0.10% beyond entry
ENTRY_THRESHOLD_DEFAULT = 80
MIN_RR = 2.0
FEE_RATE = 0.0005           # 0.05% per side, paper estimate
DEFAULT_USDT_INR = 88.0      # quote-currency conversion assumption; editable in sidebar

HISTORY_TARGET = {"1h": 2200, "2h": 1200, "4h": 700, "1d": 500}
MIN_WARMUP = 260

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def utc_now():
    return pd.Timestamp.now(tz="UTC")


def safe_float(x, default=0.0):
    try:
        v = float(x)
        return default if not np.isfinite(v) else v
    except Exception:
        return default


# ------------------------------------------------------------
# SQLite persistence
# ------------------------------------------------------------

def db():
    con = sqlite3.connect(DB_FILE, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS active_trades (
            symbol TEXT PRIMARY KEY,
            side TEXT NOT NULL,
            state TEXT NOT NULL,
            entry REAL NOT NULL,
            initial_sl REAL NOT NULL,
            base_tp REAL NOT NULL,
            trailing_stop REAL,
            quantity REAL NOT NULL,
            risk_inr REAL NOT NULL,
            opened_at TEXT NOT NULL,
            last_price REAL,
            pnl REAL DEFAULT 0,
            signal_strength REAL DEFAULT 0,
            signal_label TEXT,
            exit_reason TEXT,
            updated_at TEXT NOT NULL
        )
    """)
    con.commit()
    return con


def load_trades():
    con = db()
    df = pd.read_sql_query("SELECT * FROM active_trades", con)
    con.close()
    return df


def upsert_trade(t):
    con = db()
    con.execute("""
        INSERT INTO active_trades
        (symbol,side,state,entry,initial_sl,base_tp,trailing_stop,quantity,
         risk_inr,opened_at,last_price,pnl,signal_strength,signal_label,
         exit_reason,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(symbol) DO UPDATE SET
        side=excluded.side,state=excluded.state,entry=excluded.entry,
        initial_sl=excluded.initial_sl,base_tp=excluded.base_tp,
        trailing_stop=excluded.trailing_stop,quantity=excluded.quantity,
        risk_inr=excluded.risk_inr,opened_at=excluded.opened_at,
        last_price=excluded.last_price,pnl=excluded.pnl,
        signal_strength=excluded.signal_strength,
        signal_label=excluded.signal_label,exit_reason=excluded.exit_reason,
        updated_at=excluded.updated_at
    """, tuple(t[k] for k in [
        "symbol", "side", "state", "entry", "initial_sl", "base_tp",
        "trailing_stop", "quantity", "risk_inr", "opened_at", "last_price",
        "pnl", "signal_strength", "signal_label", "exit_reason", "updated_at"
    ]))
    con.commit()
    con.close()


def delete_trade(symbol):
    con = db()
    con.execute("DELETE FROM active_trades WHERE symbol=?", (symbol,))
    con.commit()
    con.close()


# ------------------------------------------------------------
# Market data: Binance-free provider chain
# ------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_exchange(name):
    cfg = {"enableRateLimit": True, "timeout": 15000}
    if name == "okx":
        return ccxt.okx(cfg)
    if name == "kucoin":
        return ccxt.kucoin(cfg)
    if name == "kraken":
        return ccxt.kraken(cfg)
    if name == "coinbase":
        return ccxt.coinbase(cfg)
    raise ValueError(name)


def native_symbol(exchange, symbol):
    markets = exchange.load_markets()
    if symbol in markets:
        return symbol
    base, _ = symbol.split("/")
    for alt in (f"{base}/USDT", f"{base}/USD", f"{base}/USDC"):
        if alt in markets:
            return alt
    return None


def fetch_paginated(exchange, symbol, timeframe, target):
    actual = native_symbol(exchange, symbol)
    if not actual:
        raise RuntimeError(f"{symbol} not listed")

    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - tf_ms * (target + 10)
    rows, seen = [], set()
    page_limit = 1000

    for _ in range(8):
        batch = exchange.fetch_ohlcv(
            actual, timeframe=timeframe, since=since,
            limit=min(page_limit, target)
        )
        if not batch:
            break
        for row in batch:
            if row[0] not in seen:
                seen.add(row[0])
                rows.append(row)
        if len(rows) >= target or len(batch) < page_limit:
            break
        since = batch[-1][0] + tf_ms

    if not rows:
        raise RuntimeError("empty OHLCV")
    return sorted(rows, key=lambda r: r[0])[-target:]


def remove_open_candle(df, source_tf):
    tf = pd.Timedelta(hours={"1h": 1, "4h": 4, "1d": 24}[source_tf])
    now = utc_now()
    return df[(df["timestamp"] + tf) <= now].copy()


def resample_2h(df):
    x = df.set_index("timestamp")
    out = x.resample("2h", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna().reset_index()
    return out


@st.cache_data(ttl=30, show_spinner=False)
def fetch_ohlcv(symbol, timeframe, limit=None):
    target = limit or HISTORY_TARGET[timeframe]
    source_tf = "1h" if timeframe == "2h" else timeframe
    source_target = target * 2 + 10 if timeframe == "2h" else target
    errors = []

    for provider in PROVIDERS:
        try:
            ex = get_exchange(provider)
            rows = fetch_paginated(ex, symbol, source_tf, source_target)
            df = pd.DataFrame(rows, columns=[
                "timestamp", "open", "high", "low", "close", "volume"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.drop_duplicates("timestamp").sort_values("timestamp")
            df = remove_open_candle(df, source_tf)
            if timeframe == "2h":
                df = resample_2h(df)
            df = df.tail(target).reset_index(drop=True)

            if len(df) < MIN_WARMUP:
                raise RuntimeError(f"only {len(df)} closed candles")

            df.attrs["provider"] = provider
            return df
        except Exception as exc:
            errors.append(f"{provider}: {type(exc).__name__}: {str(exc)[:120]}")

    raise RuntimeError("All providers failed | " + " | ".join(errors))


# ------------------------------------------------------------
# Indicators
# ------------------------------------------------------------

def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100)
    out = out.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return out.clip(0, 100)


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def enrich(df):
    x = df.copy()
    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200)
    x["rsi"] = rsi(x["close"], 14)
    x["rsi_sma"] = x["rsi"].rolling(14, min_periods=14).mean()
    x["atr"] = atr(x, 14)
    x["vol_sma"] = x["volume"].rolling(20, min_periods=20).mean()
    x["atr_pct"] = x["atr"] / x["close"] * 100
    x["body_pct"] = (x["close"] - x["open"]).abs() / x["close"] * 100
    return x


def candle_flags(row):
    rng = max(float(row.high - row.low), 1e-12)
    body = abs(float(row.close - row.open))
    pos = (float(row.close) - float(row.low)) / rng
    bull = row.close > row.open and body / rng >= 0.45 and pos >= 0.65
    bear = row.close < row.open and body / rng >= 0.45 and pos <= 0.35
    return bull, bear


def score_row(row, symbol):
    long_score = 0.0
    short_score = 0.0
    lr, sr = [], []

    # Trend regime: highest weight.
    if row.close > row.ema200:
        long_score += 25; lr.append("above EMA200")
    elif row.close < row.ema200:
        short_score += 25; sr.append("below EMA200")

    # Fast/medium alignment.
    if row.ema20 > row.ema50 > row.ema200:
        long_score += 20; lr.append("EMA20>50>200")
    elif row.ema20 < row.ema50 < row.ema200:
        short_score += 20; sr.append("EMA20<50<200")

    # Momentum must agree with RSI SMA.
    if 52 <= row.rsi <= 70 and row.rsi > row.rsi_sma:
        long_score += 20; lr.append("RSI momentum")
    elif 30 <= row.rsi <= 48 and row.rsi < row.rsi_sma:
        short_score += 20; sr.append("RSI momentum")

    # Volume confirmation.
    vol_ratio = row.volume / max(row.vol_sma, 1e-12)
    if vol_ratio >= 1.10:
        if row.close > row.open:
            long_score += 15; lr.append("volume expansion")
        elif row.close < row.open:
            short_score += 15; sr.append("volume expansion")

    bull, bear = candle_flags(row)
    if bull:
        long_score += 15; lr.append("bullish candle")
    if bear:
        short_score += 15; sr.append("bearish candle")

    # Volatility sanity filter: avoid dead markets and extreme chase.
    if row.atr_pct < 0.15:
        long_score -= 8; short_score -= 8
    if row.atr_pct > 8.0:
        long_score -= 8; short_score -= 8

    # BTC/ETH: require better quality and less RSI chasing.
    if symbol in ("BTC/USDT", "ETH/USDT"):
        if vol_ratio < 1.0:
            long_score -= 6; short_score -= 6
        if row.rsi > 72:
            long_score -= 8
        if row.rsi < 28:
            short_score -= 8

    long_score = float(np.clip(long_score, 0, 100))
    short_score = float(np.clip(short_score, 0, 100))

    # High-quality signal only when direction dominates.
    if long_score >= 88 and long_score >= short_score + 12:
        label, direction, score = "Strong Buy", "LONG", long_score
    elif long_score >= 80 and long_score >= short_score + 10:
        label, direction, score = "Buy", "LONG", long_score
    elif short_score >= 88 and short_score >= long_score + 12:
        label, direction, score = "Strong Sell", "SHORT", short_score
    elif short_score >= 80 and short_score >= long_score + 10:
        label, direction, score = "Sell", "SHORT", short_score
    else:
        label, direction, score = "Wait", None, max(long_score, short_score)

    return {
        "label": label,
        "direction": direction,
        "score": score,
        "price": float(row.close),
        "ema20": float(row.ema20),
        "ema50": float(row.ema50),
        "ema200": float(row.ema200),
        "rsi": float(row.rsi),
        "rsi_sma": float(row.rsi_sma),
        "atr": float(row.atr),
        "atr_pct": float(row.atr_pct),
        "vol_ratio": float(vol_ratio),
        "long_score": long_score,
        "short_score": short_score,
        "bull_candle": bool(bull),
        "bear_candle": bool(bear),
        "long_reasons": lr,
        "short_reasons": sr,
    }


def signal_for(df, symbol):
    x = enrich(df).dropna().reset_index(drop=True)
    if len(x) < 2:
        raise RuntimeError("Not enough EMA200 warm-up candles")
    return score_row(x.iloc[-1], symbol)


# ------------------------------------------------------------
# Paper-trade lifecycle
# ------------------------------------------------------------

def make_trade(symbol, sig, equity, usdt_inr):
    side = sig["direction"]
    entry = sig["price"]
    sl = entry * (1 - INITIAL_SL_PCT) if side == "LONG" else entry * (1 + INITIAL_SL_PCT)
    tp = entry * (1 + BASE_TP_PCT) if side == "LONG" else entry * (1 - BASE_TP_PCT)
    risk_inr = equity * RISK_PER_TRADE
    risk_per_coin = abs(entry - sl)
    qty = risk_inr / max(risk_per_coin * usdt_inr, 1e-12)
    return {
        "symbol": symbol,
        "side": side,
        "state": f"ENTER {side}",
        "entry": entry,
        "initial_sl": sl,
        "base_tp": tp,
        "trailing_stop": sl,
        "quantity": qty,
        "risk_inr": risk_inr,
        "opened_at": utc_now().isoformat(),
        "last_price": entry,
        "pnl": 0.0,
        "signal_strength": sig["score"],
        "signal_label": sig["label"],
        "exit_reason": "",
        "updated_at": utc_now().isoformat(),
    }


def manage_live_trade(trade, sig, usdt_inr):
    side = trade["side"]
    price = sig["price"]
    entry = trade["entry"]
    pnl = ((price - entry) * trade["quantity"] if side == "LONG" else (entry - price) * trade["quantity"]) * usdt_inr
    move = (price - entry) / entry if side == "LONG" else (entry - price) / entry

    # Trailing only activates after +2%; before that the hard SL protects the trade.
    trail = float(trade["trailing_stop"] or trade["initial_sl"])
    if move >= TRAIL_ACTIVATION_PCT:
        atr_dist = max(float(sig["atr"]) * TRAIL_ATR_MULT, entry * 0.01)
        if side == "LONG":
            trail = max(trail, entry * (1 + BREAK_EVEN_BUFFER), price - atr_dist)
        else:
            trail = min(trail, entry * (1 - BREAK_EVEN_BUFFER), price + atr_dist)

    reason = None
    if side == "LONG":
        if price <= trade["initial_sl"]:
            reason = "Initial 2% SL"
        elif price >= trade["base_tp"]:
            reason = "Base 4% TP"
        elif move >= TRAIL_ACTIVATION_PCT and price <= trail:
            reason = "Trailing stop"
        elif sig["score"] < 70 and price < sig["ema50"]:
            reason = "Confirmed trend/momentum reversal"
    else:
        if price >= trade["initial_sl"]:
            reason = "Initial 2% SL"
        elif price <= trade["base_tp"]:
            reason = "Base 4% TP"
        elif move >= TRAIL_ACTIVATION_PCT and price >= trail:
            reason = "Trailing stop"
        elif sig["score"] < 70 and price > sig["ema50"]:
            reason = "Confirmed trend/momentum reversal"

    trade.update({
        "state": "EXIT NOW" if reason else "HOLD",
        "last_price": price,
        "pnl": float(pnl),
        "trailing_stop": float(trail),
        "signal_strength": float(sig["score"]),
        "signal_label": sig["label"],
        "exit_reason": reason or "",
        "updated_at": utc_now().isoformat(),
    })
    return trade, reason


# ------------------------------------------------------------
# Backtest — no look-ahead
# ------------------------------------------------------------

def backtest(df, symbol, eval_start, eval_end, initial=START_CAPITAL, usdt_inr=DEFAULT_USDT_INR):
    # IMPORTANT: indicators are calculated on the COMPLETE warm-up dataset
    # before the evaluation window is sliced. This prevents EMA200 from being
    # re-initialized at the start of the 30-day test.
    x = enrich(df).dropna().reset_index(drop=True)
    x = x[(x.timestamp >= eval_start) & (x.timestamp <= eval_end)].reset_index(drop=True)
    if len(x) < 30:
        return pd.DataFrame(), initial, 0.0, pd.Series(dtype=float)

    equity = float(initial)
    peak = equity
    max_dd = 0.0
    equity_curve = []
    trades = []
    pos = None

    def close_position(position, exit_price, candle_time, reason):
        side = position["side"]
        gross = ((exit_price - position["entry"]) * position["qty"] if side == "LONG" else
                 (position["entry"] - exit_price) * position["qty"]) * usdt_inr
        fees = (position["entry"] * position["qty"] + exit_price * position["qty"]) * FEE_RATE * usdt_inr
        pnl = gross - fees
        return {
            "symbol": symbol,
            "side": side,
            "entry_time": position["entry_time"],
            "exit_time": candle_time,
            "entry": position["entry"],
            "exit": exit_price,
            "pnl": pnl,
            "return_pct": pnl / max(position["risk_equity"], 1e-12) * 100,
            "reason": reason,
            "signal_score": position["score"],
        }, pnl

    # Signal on candle i is based only on candle i's CLOSED values.
    # Entry happens at candle i+1 OPEN. No same-candle signal look-ahead.
    for i in range(1, len(x)):
        signal_row = x.iloc[i - 1]
        candle = x.iloc[i]
        sig = score_row(signal_row, symbol)

        # 1) Manage an existing position on this fully completed candle.
        if pos is not None:
            high, low = float(candle.high), float(candle.low)
            exit_price, reason = None, None

            if pos["side"] == "LONG":
                # Conservative same-candle ambiguity: SL before TP.
                if low <= pos["sl"]:
                    exit_price, reason = pos["sl"], "SL"
                elif high >= pos["tp"]:
                    exit_price, reason = pos["tp"], "TP"
                elif pos["trail_active"] and low <= pos["trail"]:
                    exit_price, reason = pos["trail"], "TRAIL"
                elif sig["score"] < 70 and signal_row.close < signal_row.ema50:
                    exit_price, reason = float(candle.open), "EARLY_EXIT"
            else:
                if high >= pos["sl"]:
                    exit_price, reason = pos["sl"], "SL"
                elif low <= pos["tp"]:
                    exit_price, reason = pos["tp"], "TP"
                elif pos["trail_active"] and high >= pos["trail"]:
                    exit_price, reason = pos["trail"], "TRAIL"
                elif sig["score"] < 70 and signal_row.close > signal_row.ema50:
                    exit_price, reason = float(candle.open), "EARLY_EXIT"

            if exit_price is not None:
                record, pnl = close_position(pos, exit_price, candle.timestamp, reason)
                equity += pnl
                trades.append(record)
                pos = None
            else:
                # Trailing is updated only after this candle closes, so it can
                # never retroactively stop the same candle.
                if pos["side"] == "LONG" and float(candle.close) >= pos["entry"] * (1 + TRAIL_ACTIVATION_PCT):
                    pos["trail_active"] = True
                    atr_dist = max(float(candle.atr) * TRAIL_ATR_MULT, pos["entry"] * 0.01)
                    pos["trail"] = max(pos["trail"], pos["entry"] * (1 + BREAK_EVEN_BUFFER), float(candle.close) - atr_dist)
                elif pos["side"] == "SHORT" and float(candle.close) <= pos["entry"] * (1 - TRAIL_ACTIVATION_PCT):
                    pos["trail_active"] = True
                    atr_dist = max(float(candle.atr) * TRAIL_ATR_MULT, pos["entry"] * 0.01)
                    pos["trail"] = min(pos["trail"], pos["entry"] * (1 - BREAK_EVEN_BUFFER), float(candle.close) + atr_dist)

        # 2) New entry at this candle's OPEN, using the previous closed candle's signal.
        if pos is None and sig["direction"] and sig["score"] >= ENTRY_THRESHOLD_DEFAULT:
            entry = float(candle.open)
            side = sig["direction"]
            sl = entry * (1 - INITIAL_SL_PCT) if side == "LONG" else entry * (1 + INITIAL_SL_PCT)
            tp = entry * (1 + BASE_TP_PCT) if side == "LONG" else entry * (1 - BASE_TP_PCT)
            risk_equity = equity * RISK_PER_TRADE
            qty = risk_equity / max(abs(entry - sl) * usdt_inr, 1e-12)
            pos = {
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "trail": sl,
                "trail_active": False,
                "qty": qty,
                "risk_equity": risk_equity,
                "entry_time": candle.timestamp,
                "score": sig["score"],
            }

            # If the entry candle itself hits SL/TP, count it. The ordering is
            # deliberately conservative when both are touched.
            if side == "LONG":
                if float(candle.low) <= sl:
                    record, pnl = close_position(pos, sl, candle.timestamp, "SL")
                    equity += pnl; trades.append(record); pos = None
                elif float(candle.high) >= tp:
                    record, pnl = close_position(pos, tp, candle.timestamp, "TP")
                    equity += pnl; trades.append(record); pos = None
            else:
                if float(candle.high) >= sl:
                    record, pnl = close_position(pos, sl, candle.timestamp, "SL")
                    equity += pnl; trades.append(record); pos = None
                elif float(candle.low) <= tp:
                    record, pnl = close_position(pos, tp, candle.timestamp, "TP")
                    equity += pnl; trades.append(record); pos = None

        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)
        equity_curve.append((candle.timestamp, equity))

    # Force-close at the last available close so the report never hides an open position.
    if pos is not None:
        last = x.iloc[-1]
        record, pnl = close_position(pos, float(last.close), last.timestamp, "END_OF_TEST")
        equity += pnl
        trades.append(record)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)
        equity_curve.append((last.timestamp, equity))

    return pd.DataFrame(trades), equity, max_dd, pd.Series(dict(equity_curve), dtype=float)

def summarize(trades, final, max_dd):
    if trades.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "pnl": final - START_CAPITAL, "final": final,
            "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
            "max_dd": max_dd, "tp": 0, "sl": 0, "trail": 0,
            "early": 0, "end": 0,
        }
    wins = trades[trades.pnl > 0]
    losses = trades[trades.pnl < 0]
    gross_win = float(wins.pnl.sum())
    gross_loss = abs(float(losses.pnl.sum()))
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100,
        "pnl": final - START_CAPITAL,
        "final": final,
        "avg_win": float(wins.pnl.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.pnl.mean()) if len(losses) else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss else np.inf,
        "max_dd": max_dd,
        "tp": int((trades.reason == "TP").sum()),
        "sl": int((trades.reason == "SL").sum()),
        "trail": int((trades.reason == "TRAIL").sum()),
        "early": int((trades.reason == "EARLY_EXIT").sum()),
        "end": int((trades.reason == "END_OF_TEST").sum()),
    }


def run_30_30(symbol, timeframe, usdt_inr):
    df = fetch_ohlcv(symbol, timeframe, HISTORY_TARGET[timeframe])
    end = df.timestamp.max()
    oos_start = end - pd.Timedelta(days=30)
    is_start = end - pd.Timedelta(days=60)

    # The full dataset contains warm-up history. Each evaluation gets its
    # own warm-up prefix so EMA200 is never initialized inside the test window.
    warm_is = df[df.timestamp < is_start].tail(MIN_WARMUP)
    is_df = df[(df.timestamp >= is_start) & (df.timestamp < oos_start)]
    warm_oos = df[df.timestamp < oos_start].tail(MIN_WARMUP)
    oos_df = df[df.timestamp >= oos_start]

    is_trades, is_final, is_dd, _ = backtest(
        pd.concat([warm_is, is_df], ignore_index=True), symbol, is_start, end, usdt_inr=usdt_inr
    )
    oos_trades, oos_final, oos_dd, eq = backtest(
        pd.concat([warm_oos, oos_df], ignore_index=True), symbol, oos_start, end, usdt_inr=usdt_inr
    )

    return summarize(is_trades, is_final, is_dd), summarize(oos_trades, oos_final, oos_dd), oos_trades, eq, df.attrs.get("provider", "unknown")


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

st.title("📈 V5.3 Crypto Trading Signal Dashboard")
st.caption("Conservative rule-based paper engine • closed candles • no-lookahead backtest • no profit guarantee")

with st.sidebar:
    st.header("Controls")
    timeframe = st.selectbox("Primary timeframe", TIMEFRAMES, index=0)
    auto_refresh = st.checkbox("Auto-refresh", True)
    refresh_sec = st.slider("Refresh seconds", 15, 300, 30)
    threshold = st.slider("Entry threshold", 75, 95, ENTRY_THRESHOLD_DEFAULT)
    usdt_inr = st.number_input("USDT → INR assumption", min_value=50.0, max_value=150.0, value=DEFAULT_USDT_INR, step=0.5)
    st.divider()
    st.write("**Risk model**")
    st.write(f"Risk/trade: {RISK_PER_TRADE*100:.1f}%")
    st.write(f"Initial SL: {INITIAL_SL_PCT*100:.1f}%")
    st.write(f"Base TP: {BASE_TP_PCT*100:.1f}%")
    st.write(f"Trail activates: +{TRAIL_ACTIVATION_PCT*100:.1f}%")
    st.write(f"Max open trades: {MAX_OPEN_TRADES}")
    st.info("Signals use closed candles only. Active trades persist in SQLite. Binance is not used.")

if auto_refresh:
    st.markdown(f"<meta http-equiv='refresh' content='{refresh_sec}'>", unsafe_allow_html=True)

active = load_trades()
cols = st.columns(5)

for col, symbol in zip(cols, COINS):
    with col:
        try:
            df = fetch_ohlcv(symbol, timeframe, 500)
            sig = signal_for(df, symbol)
            provider = df.attrs.get("provider", "?")

            st.subheader(symbol.split("/")[0])
            st.metric("Price", f"${sig['price']:,.4f}")
            label = sig["label"]
            if label in ("Strong Buy", "Buy"):
                st.success(f"{label} • {sig['score']:.0f}/100")
            elif label in ("Strong Sell", "Sell"):
                st.error(f"{label} • {sig['score']:.0f}/100")
            else:
                st.warning(f"WAIT • {sig['score']:.0f}/100")

            st.caption(f"Data: {provider.upper()}")
            st.write(f"RSI **{sig['rsi']:.1f}** / SMA **{sig['rsi_sma']:.1f}**")
            st.write(f"EMA200 **${sig['ema200']:,.4f}**")
            st.write(f"ATR **{sig['atr_pct']:.2f}%**")
            st.write(f"Volume **{sig['vol_ratio']:.2f}×** avg")

            row = active[active.symbol == symbol]
            if row.empty:
                open_count = len(active)
                if sig["direction"] and sig["score"] >= threshold and open_count < MAX_OPEN_TRADES:
                    t = make_trade(symbol, sig, START_CAPITAL, usdt_inr)
                    upsert_trade(t)
                    st.success(f"ENTER {t['side']}")
                    st.write(f"Entry ${t['entry']:,.4f}")
                    st.write(f"SL ${t['initial_sl']:,.4f}")
                    st.write(f"TP ${t['base_tp']:,.4f}")
                else:
                    st.info("WAIT")
            else:
                t = row.iloc[0].to_dict()
                t, reason = manage_live_trade(t, sig, usdt_inr)
                if reason:
                    st.error(f"EXIT NOW — {reason}")
                    st.write(f"Final P/L: ₹{t['pnl']:,.2f}")
                    delete_trade(symbol)
                else:
                    upsert_trade(t)
                    st.warning("HOLD")
                    st.write(f"Entry: ${t['entry']:,.4f}")
                    st.write(f"Current P/L: ₹{t['pnl']:,.2f}")
                    st.write(f"SL: ${t['initial_sl']:,.4f}")
                    st.write(f"TP: ${t['base_tp']:,.4f}")
                    st.write(f"Trail: ${t['trailing_stop']:,.4f}")
                    st.write(f"Risk: ₹{t['risk_inr']:,.2f}")
        except Exception as exc:
            st.warning("Market data temporarily unavailable")
            st.caption("Tried OKX → KuCoin → Kraken → Coinbase. Next refresh will retry.")
            with st.expander("Technical details"):
                st.code(str(exc))

st.divider()
st.header("📊 Backtest")
b1, b2, b3 = st.columns([1, 1, 2])
with b1:
    bt_symbol = st.selectbox("Coin", COINS, key="bt_coin")
with b2:
    bt_tf = st.selectbox("Timeframe", TIMEFRAMES, key="bt_tf")
with b3:
    run_bt = st.button("Run 30D IS + 30D OOS", type="primary")

if run_bt:
    with st.spinner("Fetching warm-up history and running no-lookahead backtest..."):
        try:
            ins, oos, trades, eq, provider = run_30_30(bt_symbol, bt_tf, usdt_inr)
            st.caption(f"Historical provider used: {provider.upper()}")

            st.subheader("Out-of-sample — recent 30 days")
            m = st.columns(8)
            m[0].metric("Trades", str(oos["trades"]))
            m[1].metric("Win rate", f"{oos['win_rate']:.1f}%")
            m[2].metric("P/L", f"₹{oos['pnl']:,.2f}")
            m[3].metric("Profit factor", f"{oos['profit_factor']:.2f}" if np.isfinite(oos['profit_factor']) else "∞")
            m[4].metric("Avg win", f"₹{oos['avg_win']:,.2f}")
            m[5].metric("Avg loss", f"₹{oos['avg_loss']:,.2f}")
            m[6].metric("Max DD", f"{oos['max_dd']:.2f}%")
            m[7].metric("Final", f"₹{oos['final']:,.2f}")

            st.write("**Exit counts:**", {
                "TP": oos["tp"], "SL": oos["sl"], "Trailing": oos["trail"],
                "Early": oos["early"], "End of test": oos["end"]
            })

            st.subheader("In-sample — prior 30 days")
            st.json(ins)

            if trades.empty:
                st.info("No closed trades in OOS. This is a valid result; do not interpret it as a forced signal.")
            else:
                st.dataframe(trades.sort_values("exit_time", ascending=False), use_container_width=True)
                st.download_button(
                    "Download OOS trade history CSV",
                    trades.to_csv(index=False).encode("utf-8"),
                    file_name=f"{bt_symbol.replace('/', '-')}_{bt_tf}_OOS_trades.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Backtest failed safely: {exc}")

st.divider()
st.header("Active Trade Ledger")
latest = load_trades()
if latest.empty:
    st.info("No active paper trades.")
else:
    st.dataframe(latest, use_container_width=True)

st.caption(
    f"P/L uses the editable USDT→INR assumption ({usdt_inr:.2f}) and estimated 0.05% per-side fees. No real orders. No guaranteed win rate or profit. "
    "Backtests are estimates and can differ materially from live execution."
)
