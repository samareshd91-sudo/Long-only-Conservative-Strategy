
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# ============================================================
# PAGE
# ============================================================
st.set_page_config(
    page_title="Conservative Pro Crypto Scanner",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container{max-width:1500px;padding:1rem 1rem 2rem}
h1{letter-spacing:-.5px}
.signal-box{
    border:1px solid rgba(148,163,184,.18);
    border-radius:16px;padding:16px;
    background:linear-gradient(145deg,#111827,#0f172a);
    min-height:190px;margin-bottom:12px
}
.buy{border-left:6px solid #22c55e}
.sell{border-left:6px solid #ef4444}
.wait{border-left:6px solid #f59e0b}
.coin{font-size:1.15rem;font-weight:700}
.signal{font-size:1.45rem;font-weight:800;margin:7px 0}
.muted{color:#94a3b8;font-size:.82rem}
.price{font-size:1.25rem;font-weight:700}
.tag{display:inline-block;padding:3px 8px;margin:2px;border-radius:99px;
background:#1e293b;color:#cbd5e1;font-size:.72rem}
div[data-testid="stMetric"]{
background:#0f172a;border:1px solid rgba(148,163,184,.14);
padding:12px;border-radius:14px
}
@media(max-width:700px){
 .block-container{padding:.65rem .55rem}
 .signal-box{min-height:170px;padding:13px}
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SETTINGS
# ============================================================
COINS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "BNB": "BNB-USD",
    "Solana": "SOL-USD",
    "XRP": "XRP-USD",
}

st.title("₿ Conservative Pro Crypto Scanner")
st.caption("200 EMA • RSI(10) • RSI SMA(14) • Long + Short • 2% SL • 4% TP")

with st.sidebar:
    st.header("⚙️ Scanner Settings")
    interval_choice = st.selectbox(
        "Candle timeframe",
        ["1h", "2h", "4h", "1d"],
        index=2
    )
    refresh_seconds = st.slider("Auto refresh (seconds)", 15, 300, 30, 15)
    auto_refresh = st.toggle("🔄 Auto Refresh", True)
    st.divider()
    st.subheader("Strategy")
    ema_len = st.number_input("EMA", 10, 500, 200)
    rsi_len = st.number_input("RSI", 2, 100, 10)
    rsi_sma_len = st.number_input("RSI SMA", 2, 100, 14)
    st.divider()
    st.subheader("Risk Management")
    sl_pct = st.number_input("Stop Loss %", 0.1, 20.0, 2.0, 0.1)
    tp_pct = st.number_input("Take Profit %", 0.1, 50.0, 4.0, 0.1)
    st.caption("Risk rules are fixed at the values above for every coin.")

if auto_refresh:
    st_autorefresh(interval=refresh_seconds * 1000, key="crypto_auto_refresh")

# ============================================================
# DATA HELPERS
# ============================================================
def rsi_wilder(close, length):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    d = df.copy()
    d["EMA"] = d["Close"].ewm(span=ema_len, adjust=False).mean()
    d["RSI"] = rsi_wilder(d["Close"], rsi_len)
    d["RSI_SMA"] = d["RSI"].rolling(rsi_sma_len).mean()

    d["BullCross"] = (
        (d["RSI"] > d["RSI_SMA"]) &
        (d["RSI"].shift(1) <= d["RSI_SMA"].shift(1))
    )
    d["BearCross"] = (
        (d["RSI"] < d["RSI_SMA"]) &
        (d["RSI"].shift(1) >= d["RSI_SMA"].shift(1))
    )

    d["LONG_SIGNAL"] = d["BullCross"] & (d["Close"] > d["EMA"])
    d["SHORT_SIGNAL"] = d["BearCross"] & (d["Close"] < d["EMA"])

    return d.dropna()

def fetch_coin(ticker):
    # 2h is created by resampling 1h candles because Yahoo Finance
    # does not consistently expose a native 2h interval.
    native = "1h" if interval_choice in ["1h", "2h", "4h"] else "1d"
    period = "60d" if native == "1h" else "2y"

    raw = yf.download(
        ticker,
        period=period,
        interval=native,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    time_col = "Datetime" if "Datetime" in raw.columns else "Date"
    raw = raw.rename(columns={time_col: "Date"})

    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    raw = raw[[c for c in cols if c in raw.columns]].copy()
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", utc=True)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw = raw.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    raw = raw.sort_values("Date").set_index("Date")

    if interval_choice in ["2h", "4h"]:
        rule = interval_choice
        raw = raw.resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum" if "Volume" in raw.columns else "last",
        }).dropna(subset=["Open", "High", "Low", "Close"])

    return raw

def signal_info(d):
    r = d.iloc[-1]
    if bool(r["LONG_SIGNAL"]):
        signal = "BUY"
        cls = "buy"
        reason = "Bullish RSI crossover while price is above 200 EMA."
        entry = float(r["Close"])
        sl = entry * (1 - sl_pct/100)
        tp = entry * (1 + tp_pct/100)
    elif bool(r["SHORT_SIGNAL"]):
        signal = "SELL"
        cls = "sell"
        reason = "Bearish RSI crossover while price is below 200 EMA."
        entry = float(r["Close"])
        sl = entry * (1 + sl_pct/100)
        tp = entry * (1 - tp_pct/100)
    else:
        signal = "WAIT"
        cls = "wait"
        reason = "No fresh confirmed crossover on the latest candle."
        entry = sl = tp = None

    trend = "Bullish" if r["Close"] > r["EMA"] else "Bearish"
    momentum = "Bullish" if r["RSI"] > r["RSI_SMA"] else "Bearish"

    return {
        "signal": signal, "class": cls, "reason": reason,
        "entry": entry, "sl": sl, "tp": tp,
        "price": float(r["Close"]), "ema": float(r["EMA"]),
        "rsi": float(r["RSI"]), "rsi_sma": float(r["RSI_SMA"]),
        "trend": trend, "momentum": momentum, "time": r.name,
    }

# ============================================================
# LOAD ALL FIVE
# ============================================================
results = {}
errors = {}

with st.spinner("Loading BTC • ETH • BNB • SOL • XRP..."):
    for name, ticker in COINS.items():
        try:
            df = fetch_coin(ticker)
            if df is None or len(df) < max(ema_len, rsi_len + rsi_sma_len) + 5:
                errors[name] = "Not enough market data"
                continue
            ind = add_indicators(df)
            if len(ind) < 2:
                errors[name] = "Not enough indicator data"
                continue
            results[name] = {"ticker": ticker, "data": ind, "info": signal_info(ind)}
        except Exception as e:
            errors[name] = str(e)[:100]

# ============================================================
# HEADER STATUS
# ============================================================
now = pd.Timestamp.now(tz="UTC")
h1, h2, h3, h4 = st.columns(4)
h1.metric("Markets", f"{len(results)}/5")
h2.metric("BUY", sum(x["info"]["signal"] == "BUY" for x in results.values()))
h3.metric("SELL", sum(x["info"]["signal"] == "SELL" for x in results.values()))
h4.metric("Updated", now.strftime("%H:%M:%S UTC"))

if errors:
    st.warning("Some markets could not be loaded: " + " • ".join(f"{k}: {v}" for k,v in errors.items()))

# ============================================================
# MARKET SCANNER CARDS
# ============================================================
st.subheader("🎯 Live Market Scanner")

names = list(COINS.keys())
for start in range(0, len(names), 2):
    cols = st.columns(2)
    for j, name in enumerate(names[start:start+2]):
        with cols[j]:
            if name not in results:
                st.error(f"{name}: data unavailable")
                continue
            x = results[name]["info"]
            arrow = "🟢" if x["signal"] == "BUY" else "🔴" if x["signal"] == "SELL" else "🟡"
            entry_text = f"{x['entry']:,.6f}" if x["entry"] is not None else "—"
            sl_text = f"{x['sl']:,.6f}" if x["sl"] is not None else "—"
            tp_text = f"{x['tp']:,.6f}" if x["tp"] is not None else "—"
            st.markdown(f"""
            <div class="signal-box {x['class']}">
              <div class="coin">{arrow} {name} <span class="muted">({results[name]['ticker']})</span></div>
              <div class="signal">{x['signal']}</div>
              <div class="price">${x['price']:,.6f}</div>
              <div class="muted" style="margin:7px 0">{x['reason']}</div>
              <span class="tag">Trend: {x['trend']}</span>
              <span class="tag">RSI: {x['rsi']:.2f}</span>
              <span class="tag">RSI SMA: {x['rsi_sma']:.2f}</span>
              <span class="tag">EMA: {x['ema']:,.6f}</span>
              <div style="margin-top:10px">
                <b>Entry:</b> {entry_text} &nbsp; | &nbsp;
                <b>SL:</b> {sl_text} &nbsp; | &nbsp;
                <b>TP:</b> {tp_text}
              </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# DETAIL VIEW
# ============================================================
st.subheader("🔎 Detailed Chart")
selected = st.selectbox("Select coin", names)
if selected in results:
    d = results[selected]["data"]
    info = results[selected]["info"]

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Signal", info["signal"])
    c2.metric("Price", f"${info['price']:,.6f}")
    c3.metric("EMA 200", f"${info['ema']:,.6f}")
    c4.metric("RSI", f"{info['rsi']:.2f}")
    c5.metric("Momentum", info["momentum"])

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close,
        name="Price"
    ))
    fig.add_trace(go.Scatter(
        x=d.index, y=d.EMA, mode="lines", name=f"EMA {ema_len}"
    ))

    longs = d[d.LONG_SIGNAL]
    shorts = d[d.SHORT_SIGNAL]
    if not longs.empty:
        fig.add_trace(go.Scatter(
            x=longs.index, y=longs.Low*0.995, mode="markers",
            marker=dict(symbol="triangle-up", size=12),
            name="BUY"
        ))
    if not shorts.empty:
        fig.add_trace(go.Scatter(
            x=shorts.index, y=shorts.High*1.005, mode="markers",
            marker=dict(symbol="triangle-down", size=12),
            name="SELL"
        ))

    # Current SL/TP guide lines when there is a fresh signal
    if info["entry"] is not None:
        fig.add_hline(y=info["entry"], line_dash="dot", annotation_text="Entry")
        fig.add_hline(y=info["sl"], line_dash="dash", annotation_text="2% SL")
        fig.add_hline(y=info["tp"], line_dash="dash", annotation_text="4% TP")

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=5,r=5,t=20,b=5),
    )
    st.plotly_chart(fig, use_container_width=True)

    rf = go.Figure()
    rf.add_trace(go.Scatter(x=d.index, y=d.RSI, mode="lines", name=f"RSI {rsi_len}"))
    rf.add_trace(go.Scatter(x=d.index, y=d.RSI_SMA, mode="lines", name=f"RSI SMA {rsi_sma_len}"))
    rf.add_hline(y=70, line_dash="dash")
    rf.add_hline(y=30, line_dash="dash")
    rf.update_layout(height=330, hovermode="x unified")
    st.plotly_chart(rf, use_container_width=True)

# ============================================================
# BACKTEST PER COIN
# ============================================================
st.subheader("📊 Signal History")
b1,b2,b3,b4,b5 = st.columns(5)

for col, name in zip([b1,b2,b3,b4,b5], names):
    if name not in results:
        col.metric(name, "N/A")
        continue
    d = results[name]["data"]
    long_count = int(d["LONG_SIGNAL"].sum())
    short_count = int(d["SHORT_SIGNAL"].sum())
    col.metric(name, f"B {long_count} / S {short_count}")

st.caption(
    "Signal definition: BUY = fresh RSI(10) cross above RSI SMA(14) + price above EMA(200). "
    "SELL = fresh RSI(10) cross below RSI SMA(14) + price below EMA(200). "
    "WAIT means there is no fresh confirmed crossover on the latest candle."
)

st.divider()
st.caption(
    "Educational scanner only. Crypto markets are volatile. "
    "The 2% stop-loss and 4% take-profit are rule-based targets, not guaranteed execution prices."
)
