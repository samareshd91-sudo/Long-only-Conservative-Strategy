
import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import sqlite3
import time
from datetime import datetime, timezone, timedelta

st.set_page_config(
    page_title="V5 Crypto Trading Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COINS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
TIMEFRAMES = ["1h", "2h", "4h", "1d"]
START_CAPITAL = 10_000.0
DB_FILE = "v5_trades.db"

# -----------------------------
# Exchange / persistence
# -----------------------------
# IMPORTANT:
# Binance is intentionally NOT used. Streamlit Cloud deployments can receive
# Binance HTTP 451 errors because the cloud runtime may be in a restricted
# jurisdiction. We therefore use a provider chain and automatically fail over.
#
# Provider order:
#   OKX -> KuCoin -> Kraken -> Coinbase
#
# No API keys are required because only public OHLCV endpoints are used.

PROVIDERS = ["okx", "kucoin", "kraken", "coinbase"]

@st.cache_resource
def get_exchange(name):
    cfg = {
        "enableRateLimit": True,
        "timeout": 15000,
    }
    if name == "okx":
        return ccxt.okx(cfg)
    if name == "kucoin":
        return ccxt.kucoin(cfg)
    if name == "kraken":
        return ccxt.kraken(cfg)
    if name == "coinbase":
        return ccxt.coinbase(cfg)
    raise ValueError(f"Unsupported provider: {name}")

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
         opened_at,last_price,pnl,signal_strength,signal_label,exit_reason,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(symbol) DO UPDATE SET
        side=excluded.side,state=excluded.state,entry=excluded.entry,
        initial_sl=excluded.initial_sl,base_tp=excluded.base_tp,
        trailing_stop=excluded.trailing_stop,quantity=excluded.quantity,
        opened_at=excluded.opened_at,last_price=excluded.last_price,pnl=excluded.pnl,
        signal_strength=excluded.signal_strength,signal_label=excluded.signal_label,
        exit_reason=excluded.exit_reason,updated_at=excluded.updated_at
    """, tuple(t[k] for k in [
        "symbol","side","state","entry","initial_sl","base_tp","trailing_stop",
        "quantity","opened_at","last_price","pnl","signal_strength",
        "signal_label","exit_reason","updated_at"
    ]))
    con.commit()
    con.close()

def delete_trade(symbol):
    con = db()
    con.execute("DELETE FROM active_trades WHERE symbol=?", (symbol,))
    con.commit()
    con.close()

def _native_symbol(exchange, symbol):
    """Return a symbol only if the provider actually lists it."""
    try:
        markets = exchange.load_markets()
        if symbol in markets:
            return symbol
        # Some providers list XRP/USD rather than XRP/USDT.
        base, quote = symbol.split("/")
        alternatives = [
            f"{base}/USDT",
            f"{base}/USD",
            f"{base}/USDC",
        ]
        for alt in alternatives:
            if alt in markets:
                return alt
    except Exception:
        pass
    return None

def _fetch_paginated(exchange, symbol, timeframe, target=1800):
    """
    Fetch enough candles for both EMA200 warm-up and a full 60-day test.
    Repeated requests avoid the common exchange limit of ~500-1000 candles.
    """
    actual = _native_symbol(exchange, symbol)
    if not actual:
        raise RuntimeError(f"{symbol} is not listed by this provider")

    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    now_ms = exchange.milliseconds()
    since = now_ms - tf_ms * (target + 5)
    rows = []
    seen = set()

    for _ in range(5):
        batch = exchange.fetch_ohlcv(
            actual, timeframe=timeframe, since=since, limit=min(1000, target)
        )
        if not batch:
            break
        for r in batch:
            if r[0] not in seen:
                seen.add(r[0])
                rows.append(r)
        if len(batch) < min(1000, target):
            break
        since = batch[-1][0] + tf_ms
        if len(rows) >= target:
            break

    if not rows:
        raise RuntimeError(f"No OHLCV returned for {symbol}")

    rows = sorted(rows, key=lambda r: r[0])[-target:]
    return rows

def _resample_2h_from_1h(df):
    """Build 2H candles from closed 1H candles."""
    x = df.copy().set_index("timestamp")
    out = x.resample("2h", label="right", closed="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna().reset_index()
    return out

@st.cache_data(ttl=30, show_spinner=False)
def fetch_ohlcv(symbol, timeframe, limit=1800):
    errors = []

    # 2H is constructed from 1H so it does not depend on an exchange
    # supporting a native 2H market timeframe.
    source_tf = "1h" if timeframe == "2h" else timeframe
    source_limit = max(limit * 2 + 20, 1000) if timeframe == "2h" else max(limit, 1000)

    for provider in PROVIDERS:
        try:
            ex = get_exchange(provider)
            rows = _fetch_paginated(ex, symbol, source_tf, target=source_limit)
            df = pd.DataFrame(
                rows,
                columns=["timestamp","open","high","low","close","volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.drop_duplicates("timestamp").sort_values("timestamp")

            # Remove the currently forming candle.
            if len(df) > 2:
                tf_delta = pd.Timedelta(hours=1 if source_tf == "1h" else
                                        4 if source_tf == "4h" else
                                        24 if source_tf == "1d" else 1)
                cutoff = pd.Timestamp.now(tz="UTC") - tf_delta
                df = df[df["timestamp"] <= cutoff].copy()

            if timeframe == "2h":
                df = _resample_2h_from_1h(df)

            df = df.tail(limit).reset_index(drop=True)
            if len(df) < 250:
                raise RuntimeError(f"Only {len(df)} usable candles")

            return df

        except Exception as e:
            errors.append(f"{provider}: {type(e).__name__}: {str(e)[:100]}")
            continue

    raise RuntimeError("All market-data providers failed. " + " | ".join(errors))

# -----------------------------
# Signal engine
# -----------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=1).mean()

def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))

    # Handle a permanently rising/flat series safely.
    out = out.where(avg_loss.notna(), 50)
    out = out.fillna(50)
    return out.clip(0, 100)

def atr(df, n=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def enrich(df):
    x = df.copy()

    # EMA200 is calculated over the full warm-up history supplied by the
    # data layer, not just the 30-day evaluation window.
    x["ema200"] = ema(x["close"], 200)
    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)

    x["rsi"] = rsi(x["close"], 14)
    x["rsi_sma"] = x["rsi"].rolling(14, min_periods=14).mean()
    x["atr"] = atr(x, 14)

    x["vol_sma"] = x["volume"].rolling(20, min_periods=20).mean()
    x["atr_pct"] = (x["atr"] / x["close"].replace(0, np.nan)) * 100

    return x

def candle_confirmation(x):
    o, h, l, c = x[["open","high","low","close"]]
    body = abs(c-o)
    rng = max(h-l, 1e-12)
    close_pos = (c-l)/rng
    bull = c > o and close_pos >= 0.65 and body/rng >= 0.35
    bear = c < o and close_pos <= 0.35 and body/rng >= 0.35
    return bull, bear

def signal_for(df, symbol):
    enriched = enrich(df).dropna(subset=[
        "ema200", "rsi", "rsi_sma", "atr", "vol_sma", "ema20", "ema50"
    ])
    if len(enriched) < 2:
        raise RuntimeError("Not enough warmed-up candles for signal calculation")
    x = enriched.iloc[-1]
    bull_c, bear_c = candle_confirmation(x)

    long_score = 0.0
    short_score = 0.0
    reasons_long, reasons_short = [], []

    if x.close > x.ema200:
        long_score += 25; reasons_long.append("price > EMA200")
    elif x.close < x.ema200:
        short_score += 25; reasons_short.append("price < EMA200")

    if x.ema20 > x.ema50 > x.ema200:
        long_score += 15; reasons_long.append("EMA alignment")
    if x.ema20 < x.ema50 < x.ema200:
        short_score += 15; reasons_short.append("EMA alignment")

    if 52 <= x.rsi <= 72 and x.rsi > x.rsi_sma:
        long_score += 20; reasons_long.append("RSI momentum")
    elif 28 <= x.rsi <= 48 and x.rsi < x.rsi_sma:
        short_score += 20; reasons_short.append("RSI momentum")

    if x.volume > x.vol_sma * 1.05:
        if x.close > x.open:
            long_score += 15; reasons_long.append("volume confirmation")
        elif x.close < x.open:
            short_score += 15; reasons_short.append("volume confirmation")

    if bull_c:
        long_score += 15; reasons_long.append("bullish candle")
    if bear_c:
        short_score += 15; reasons_short.append("bearish candle")

    # BTC/ETH quality filter: avoid low-quality momentum and extreme RSI chasing.
    quality_penalty = 0
    if symbol in ("BTC/USDT", "ETH/USDT"):
        if x.atr_pct < 0.20:
            quality_penalty += 10
        if x.rsi > 76:
            long_score -= 10
        if x.rsi < 24:
            short_score -= 10
        if x.volume < x.vol_sma * 0.85:
            quality_penalty += 10
    long_score = max(0, min(100, long_score-quality_penalty))
    short_score = max(0, min(100, short_score-quality_penalty))

    if long_score >= 82 and long_score > short_score + 8:
        label = "Strong Buy"
        direction = "LONG"
        score = long_score
    elif long_score >= 70 and long_score > short_score + 5:
        label = "Buy"
        direction = "LONG"
        score = long_score
    elif short_score >= 82 and short_score > long_score + 8:
        label = "Strong Sell"
        direction = "SHORT"
        score = short_score
    elif short_score >= 70 and short_score > long_score + 5:
        label = "Sell"
        direction = "SHORT"
        score = short_score
    else:
        label = "Wait"
        direction = None
        score = max(long_score, short_score)

    momentum_weak = (
        (direction == "LONG" and (x.rsi < x.rsi_sma or x.rsi < 45)) or
        (direction == "SHORT" and (x.rsi > x.rsi_sma or x.rsi > 55))
    )
    trend_reverse = (
        (direction == "LONG" and x.close < x.ema50) or
        (direction == "SHORT" and x.close > x.ema50)
    )

    return {
        "label": label, "direction": direction, "score": float(score),
        "price": float(x.close), "ema200": float(x.ema200),
        "rsi": float(x.rsi), "rsi_sma": float(x.rsi_sma),
        "atr": float(x.atr), "volume": float(x.volume),
        "vol_sma": float(x.vol_sma), "atr_pct": float(x.atr_pct),
        "momentum_weak": bool(momentum_weak),
        "trend_reverse": bool(trend_reverse),
        "reasons": reasons_long if direction=="LONG" else reasons_short,
    }

def manage_trade(trade, sig):
    side = trade["side"]
    price = sig["price"]
    entry = trade["entry"]
    if side == "LONG":
        pnl = (price-entry)*trade["quantity"]
        move = (price-entry)/entry
        if price > entry:
            trail = max(float(trade["trailing_stop"] or trade["initial_sl"]),
                        price * (1 - max(0.012, sig["atr"]/price*2)))
        else:
            trail = trade["initial_sl"]
        exit_reason = None
        if price <= trade["initial_sl"]:
            exit_reason = "Initial 2% SL"
        elif price >= trade["base_tp"]:
            exit_reason = "Base 4% TP"
        elif price <= trail:
            exit_reason = "Trailing stop"
        elif sig["trend_reverse"]:
            exit_reason = "Trend reversal"
        elif sig["momentum_weak"]:
            exit_reason = "Momentum weakened"
    else:
        pnl = (entry-price)*trade["quantity"]
        move = (entry-price)/entry
        if price < entry:
            trail = min(float(trade["trailing_stop"] or trade["initial_sl"]),
                        price * (1 + max(0.012, sig["atr"]/price*2)))
        else:
            trail = trade["initial_sl"]
        exit_reason = None
        if price >= trade["initial_sl"]:
            exit_reason = "Initial 2% SL"
        elif price <= trade["base_tp"]:
            exit_reason = "Base 4% TP"
        elif price >= trail:
            exit_reason = "Trailing stop"
        elif sig["trend_reverse"]:
            exit_reason = "Trend reversal"
        elif sig["momentum_weak"]:
            exit_reason = "Momentum weakened"

    trade.update({
        "state": "EXIT NOW" if exit_reason else ("HOLD" if abs(move) > 0.005 else "ENTER "+side),
        "last_price": price, "pnl": float(pnl),
        "trailing_stop": float(trail),
        "signal_strength": sig["score"], "signal_label": sig["label"],
        "exit_reason": exit_reason,
        "updated_at": datetime.now(timezone.utc).isoformat()
    })
    return trade, exit_reason

def open_trade(symbol, sig):
    side = sig["direction"]
    p = sig["price"]
    if side == "LONG":
        sl, tp = p*0.98, p*1.04
    else:
        sl, tp = p*1.02, p*0.96
    # Fixed notional sizing for dashboard paper-trading; no leverage.
    notional = START_CAPITAL * 0.10
    qty = notional / p
    return {
        "symbol": symbol, "side": side, "state": "ENTER "+side,
        "entry": p, "initial_sl": sl, "base_tp": tp,
        "trailing_stop": sl, "quantity": qty,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "last_price": p, "pnl": 0.0,
        "signal_strength": sig["score"], "signal_label": sig["label"],
        "exit_reason": "", "updated_at": datetime.now(timezone.utc).isoformat()
    }

# -----------------------------
# Backtest
# -----------------------------
def backtest(df, symbol, initial=START_CAPITAL):
    # Uses only data after warm-up; EMA200 is therefore not initialized from
    # the 30-day test window alone.
    x = enrich(df).dropna().reset_index(drop=True)
    capital = initial
    equity = initial
    peak = initial
    max_dd = 0
    trades = []
    position = None

    for i in range(1, len(x)):
        window = x.iloc[:i+1]
        if len(window) < 220:
            continue
        sig = signal_for(window, symbol)
        price = float(x.iloc[i].close)

        if position is None and sig["direction"] and sig["score"] >= 70:
            side = sig["direction"]
            entry = price
            sl = entry*0.98 if side=="LONG" else entry*1.02
            tp = entry*1.04 if side=="LONG" else entry*0.96
            qty = (capital*0.10)/entry
            position = dict(side=side, entry=entry, sl=sl, tp=tp, qty=qty,
                            trail=sl, opened=x.iloc[i].timestamp, peak=entry)
            continue

        if position:
            side = position["side"]
            high, low = float(x.iloc[i].high), float(x.iloc[i].low)
            exit_price, reason = None, None

            if side == "LONG":
                if low <= position["sl"]:
                    exit_price, reason = position["sl"], "SL"
                elif high >= position["tp"]:
                    exit_price, reason = position["tp"], "TP"
                elif price > position["peak"]:
                    position["peak"] = price
                    position["trail"] = max(position["trail"],
                        price*(1-max(0.012, float(x.iloc[i].atr)/price*2)))
                if exit_price is None and low <= position["trail"]:
                    exit_price, reason = position["trail"], "TRAIL"
                if exit_price is None and (sig["trend_reverse"] or sig["momentum_weak"]):
                    exit_price, reason = price, "EARLY_EXIT"
                pnl = (exit_price-position["entry"])*position["qty"] if exit_price else 0
            else:
                if high >= position["sl"]:
                    exit_price, reason = position["sl"], "SL"
                elif low <= position["tp"]:
                    exit_price, reason = position["tp"], "TP"
                elif price < position["peak"]:
                    position["peak"] = price
                    position["trail"] = min(position["trail"],
                        price*(1+max(0.012, float(x.iloc[i].atr)/price*2)))
                if exit_price is None and high >= position["trail"]:
                    exit_price, reason = position["trail"], "TRAIL"
                if exit_price is None and (sig["trend_reverse"] or sig["momentum_weak"]):
                    exit_price, reason = price, "EARLY_EXIT"
                pnl = (position["entry"]-exit_price)*position["qty"] if exit_price else 0

            if exit_price is not None:
                capital += pnl
                trades.append({
                    "symbol": symbol, "side": side, "entry_time": position["opened"],
                    "exit_time": x.iloc[i].timestamp, "entry": position["entry"],
                    "exit": exit_price, "pnl": pnl, "reason": reason
                })
                position = None

        equity = capital
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak-equity)/peak*100)

    return pd.DataFrame(trades), capital, max_dd

def prepare_backtest_data(symbol, timeframe):
    # Fetch enough history for EMA200 warm-up PLUS the full 60-day evaluation.
    # For 1H/2H this is intentionally > 60 days; for 4H/1D it is also ample.
    return fetch_ohlcv(symbol, timeframe, limit=1800)

def run_periodic_backtest(symbol, timeframe):
    df = prepare_backtest_data(symbol, timeframe)
    # Recent 30 calendar days; split the available history into a 30-day
    # in-sample and recent 30-day out-of-sample segment where possible.
    end = df["timestamp"].max()
    recent_start = end - pd.Timedelta(days=30)
    is_start = end - pd.Timedelta(days=60)
    warm = df[df["timestamp"] < is_start].tail(450)
    ins = df[(df["timestamp"] >= is_start) & (df["timestamp"] < recent_start)]
    oos = df[df["timestamp"] >= recent_start]
    ins_full = pd.concat([warm, ins], ignore_index=True)
    oos_full = pd.concat([warm, ins, oos], ignore_index=True)

    tr1, cap1, dd1 = backtest(ins_full, symbol)
    tr2, cap2, dd2 = backtest(oos_full, symbol)

    # Keep only trades whose exits are inside the requested evaluation window.
    tr1 = tr1[tr1["exit_time"] >= is_start] if not tr1.empty else tr1
    tr2 = tr2[tr2["exit_time"] >= recent_start] if not tr2.empty else tr2

    def stats(tr, cap, dd):
        if tr.empty:
            return dict(final=cap, pnl=cap-START_CAPITAL, win_rate=0,
                        avg_win=0, avg_loss=0, pf=0, max_dd=dd,
                        tp=0, sl=0, trail=0, early=0)
        wins = tr[tr.pnl > 0]
        losses = tr[tr.pnl < 0]
        gross_win = wins.pnl.sum()
        gross_loss = abs(losses.pnl.sum())
        return dict(
            final=cap, pnl=cap-START_CAPITAL,
            win_rate=len(wins)/len(tr)*100,
            avg_win=wins.pnl.mean() if not wins.empty else 0,
            avg_loss=losses.pnl.mean() if not losses.empty else 0,
            pf=gross_win/gross_loss if gross_loss else np.inf,
            max_dd=dd,
            tp=int((tr.reason=="TP").sum()),
            sl=int((tr.reason=="SL").sum()),
            trail=int((tr.reason=="TRAIL").sum()),
            early=int((tr.reason=="EARLY_EXIT").sum()),
        )
    return stats(tr1, cap1, dd1), stats(tr2, cap2, dd2), tr2

# -----------------------------
# UI
# -----------------------------
st.title("📈 V5 Crypto Trading Signal Dashboard")
st.caption("Rule-based paper signal engine • no win-rate/profit guarantee • live market data")

with st.sidebar:
    st.header("Controls")
    timeframe = st.selectbox("Primary timeframe", TIMEFRAMES, index=0)
    refresh = st.checkbox("Auto-refresh", value=True)
    refresh_sec = st.slider("Refresh seconds", 10, 300, 30)
    threshold = st.slider("Entry threshold", 60, 90, 70)
    st.info("Signals use closed candles only. Active trades persist in SQLite.")

if refresh:
    st.markdown(f"<meta http-equiv='refresh' content='{refresh_sec}'>", unsafe_allow_html=True)

active = load_trades()
cols = st.columns(5)

for col, symbol in zip(cols, COINS):
    with col:
        try:
            df = fetch_ohlcv(symbol, timeframe, 450)
            sig = signal_for(df, symbol)
            st.subheader(symbol.split("/")[0])
            st.metric("Price", f"${sig['price']:,.4f}")
            st.write(f"**{sig['label']}** · {sig['score']:.0f}/100")
            st.write(f"RSI {sig['rsi']:.1f} | RSI SMA {sig['rsi_sma']:.1f}")
            st.write(f"EMA200 ${sig['ema200']:,.4f}")
            st.write(f"ATR {sig['atr']:.4f} ({sig['atr_pct']:.2f}%)")
            st.write(f"Volume / avg: {sig['volume']/max(sig['vol_sma'],1e-12):.2f}×")

            row = active[active.symbol == symbol]
            if row.empty and sig["direction"] and sig["score"] >= threshold:
                t = open_trade(symbol, sig)
                upsert_trade(t)
                st.success(f"ENTER {t['side']}")
            elif not row.empty:
                t = row.iloc[0].to_dict()
                t, reason = manage_trade(t, sig)
                if reason:
                    st.error(f"EXIT NOW — {reason}")
                    delete_trade(symbol)
                else:
                    upsert_trade(t)
                    st.warning(t["state"])
                    st.write(f"Entry: ${t['entry']:,.4f}")
                    st.write(f"Current P/L: ₹{t['pnl']:,.2f}")
                    st.write(f"SL: ${t['initial_sl']:,.4f}")
                    st.write(f"TP: ${t['base_tp']:,.4f}")
                    st.write(f"Trail: ${t['trailing_stop']:,.4f}")
            else:
                st.info("WAIT")
        except Exception as e:
            st.warning("Market data temporarily unavailable")
            st.caption(
                "The app tried OKX → KuCoin → Kraken → Coinbase. "
                "Retrying automatically on the next refresh."
            )
            with st.expander("Technical details"):
                st.code(str(e))

st.divider()
st.header("📊 Backtest")
bcol1, bcol2 = st.columns([1,3])
with bcol1:
    bt_symbol = st.selectbox("Coin", COINS, key="bt_coin")
    bt_tf = st.selectbox("Timeframe", TIMEFRAMES, key="bt_tf")
    run_bt = st.button("Run 30D + 30D backtest", type="primary")
if run_bt:
    with st.spinner("Fetching warm-up history and running backtest..."):
        try:
            ins, oos, trades = run_periodic_backtest(bt_symbol, bt_tf)
            with bcol2:
                m = st.columns(4)
                m[0].metric("OOS P/L", f"₹{oos['pnl']:,.2f}")
                m[1].metric("OOS Win rate", f"{oos['win_rate']:.1f}%")
                m[2].metric("Profit factor", f"{oos['pf']:.2f}" if np.isfinite(oos['pf']) else "∞")
                m[3].metric("Max drawdown", f"{oos['max_dd']:.2f}%")
                st.write("**In-sample (30 days)**", ins)
                st.write("**Out-of-sample / recent 30 days**", oos)
                if trades.empty:
                    st.info("No closed trades in the selected OOS period. That is a valid result.")
                else:
                    st.dataframe(trades.sort_values("exit_time", ascending=False),
                                 use_container_width=True)
        except Exception as e:
            st.error(f"Backtest failed safely: {e}")

st.divider()
st.header("Active Trade Ledger")
latest = load_trades()
if latest.empty:
    st.info("No active paper trades.")
else:
    st.dataframe(latest, use_container_width=True)

st.caption(
    "Risk note: this is a rule-based research/paper-trading dashboard, not financial advice. "
    "Past/backtest performance does not predict future results. No 70% win-rate or profit guarantee is made."
)
