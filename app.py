import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Conservative Trading Strategy", page_icon="📈", layout="wide")
st.title("📈 Conservative Trading Strategy")
st.caption("200 EMA + RSI(10) + RSI SMA(14) Backtesting Dashboard")

def indicators(df, ema_len, rsi_len, sma_len):
    d = df.copy()
    d["EMA"] = d["Close"].ewm(span=ema_len, adjust=False).mean()
    delta = d["Close"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/rsi_len, min_periods=rsi_len, adjust=False).mean()
    al = loss.ewm(alpha=1/rsi_len, min_periods=rsi_len, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    d["RSI"] = 100 - 100/(1+rs)
    d["RSI_SMA"] = d["RSI"].rolling(sma_len).mean()
    bull = (d["RSI"] > d["RSI_SMA"]) & (d["RSI"].shift(1) <= d["RSI_SMA"].shift(1))
    bear = (d["RSI"] < d["RSI_SMA"]) & (d["RSI"].shift(1) >= d["RSI_SMA"].shift(1))
    trend = d["Close"] > d["EMA"]
    trend_break = (d["Close"] < d["EMA"]) & (d["Close"].shift(1) >= d["EMA"].shift(1))
    d["BUY_SIGNAL"] = bull & trend
    d["EXIT_SIGNAL"] = bear | trend_break
    return d.dropna()

def backtest(d, initial, pos_pct, sl_pct, tp_pct, commission):
    capital = float(initial)
    position = 0.0
    entry_price = 0.0
    entry_date = None
    entry_capital = 0.0
    fee = commission / 100
    trades, curve = [], []

    for i, (date, r) in enumerate(d.iterrows()):
        close, high, low = float(r.Close), float(r.High), float(r.Low)

        if position == 0 and r.BUY_SIGNAL:
            entry_price, entry_date = close, date
            entry_capital = capital * pos_pct / 100
            capital -= entry_capital * fee
            position = entry_capital / entry_price

        elif position > 0:
            exit_price, reason = None, None

            if sl_pct > 0 and low <= entry_price * (1-sl_pct/100):
                exit_price, reason = entry_price * (1-sl_pct/100), "Stop Loss"
            elif tp_pct > 0 and high >= entry_price * (1+tp_pct/100):
                exit_price, reason = entry_price * (1+tp_pct/100), "Take Profit"
            elif r.EXIT_SIGNAL:
                exit_price, reason = close, "Strategy Exit"

            if reason:
                gross = position * exit_price
                net = gross - gross * fee
                pnl = net - entry_capital
                trades.append({
                    "Entry Date": entry_date, "Exit Date": date,
                    "Entry Price": entry_price, "Exit Price": exit_price,
                    "P&L": pnl, "P&L %": pnl/entry_capital*100,
                    "Reason": reason
                })
                capital += net
                position = 0.0

        curve.append(capital + position * close if position else capital)

    if position:
        final_price = float(d.Close.iloc[-1])
        gross = position * final_price
        net = gross - gross * fee
        pnl = net - entry_capital
        trades.append({
            "Entry Date": entry_date, "Exit Date": d.index[-1],
            "Entry Price": entry_price, "Exit Price": final_price,
            "P&L": pnl, "P&L %": pnl/entry_capital*100,
            "Reason": "End of Backtest"
        })
        capital += net
        curve[-1] = capital

    return pd.DataFrame(trades), pd.DataFrame({"Equity": curve}, index=d.index)

def stats(trades, equity, initial):
    if trades.empty:
        return 0, 0, 0, 0, 0
    wins = trades[trades["P&L"] > 0]
    losses = trades[trades["P&L"] < 0]
    pf = wins["P&L"].sum()/abs(losses["P&L"].sum()) if not losses.empty else np.inf
    dd = ((equity["Equity"] / equity["Equity"].cummax()) - 1).min() * 100
    return len(trades), len(wins)/len(trades)*100, equity.Equity.iloc[-1]-initial, pf, abs(dd)

st.sidebar.header("⚙️ Settings")
ema = st.sidebar.number_input("EMA Length", 10, 500, 200)
rsi = st.sidebar.number_input("RSI Length", 2, 100, 10)
rsi_sma = st.sidebar.number_input("RSI SMA Length", 2, 100, 14)
initial = st.sidebar.number_input("Initial Capital", 100.0, 10000000.0, 10000.0, 100.0)
pos = st.sidebar.slider("Position Size %", 1, 100, 100)
commission = st.sidebar.number_input("Commission %", 0.0, 5.0, 0.10, 0.01)
sl_on = st.sidebar.checkbox("Use Stop Loss")
sl = st.sidebar.number_input("Stop Loss %", 0.1, 50.0, 2.0, 0.1) if sl_on else 0.0
tp_on = st.sidebar.checkbox("Use Take Profit")
tp = st.sidebar.number_input("Take Profit %", 0.1, 100.0, 4.0, 0.1) if tp_on else 0.0

file = st.file_uploader("📂 Upload OHLC CSV", type=["csv"])
if not file:
    st.info("CSV must contain: Date, Open, High, Low, Close")
    st.stop()

try:
    df = pd.read_csv(file)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ["date","open","high","low","close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()
    df = df.rename(columns={"date":"Date","open":"Open","high":"High","low":"Low","close":"Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open","High","Low","Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("Date").set_index("Date")
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if len(df) < ema:
    st.warning(f"At least {ema} candles are required.")
    st.stop()

d = indicators(df, ema, rsi, rsi_sma)
trades, equity = backtest(d, initial, pos, sl, tp, commission)
n, win, profit, pf, dd = stats(trades, equity, initial)

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Net Profit", f"${profit:,.2f}")
c2.metric("Win Rate", f"{win:.2f}%")
c3.metric("Trades", n)
c4.metric("Profit Factor", "∞" if np.isinf(pf) else f"{pf:.2f}")
c5.metric("Max Drawdown", f"{dd:.2f}%")

st.subheader("📈 Equity Curve")
fig = go.Figure(go.Scatter(x=equity.index, y=equity.Equity, mode="lines", name="Equity"))
fig.update_layout(height=420, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🕯️ Price + Signals")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="Price"))
fig.add_trace(go.Scatter(x=d.index, y=d.EMA, mode="lines", name=f"EMA {ema}"))
b = d[d.BUY_SIGNAL]
e = d[d.EXIT_SIGNAL]
fig.add_trace(go.Scatter(x=b.index, y=b.Low*0.995, mode="markers", marker_symbol="triangle-up", marker_size=10, name="BUY"))
fig.add_trace(go.Scatter(x=e.index, y=e.High*1.005, mode="markers", marker_symbol="triangle-down", marker_size=9, name="EXIT"))
fig.update_layout(height=620, xaxis_rangeslider_visible=False, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📉 RSI")
rfig = go.Figure()
rfig.add_trace(go.Scatter(x=d.index, y=d.RSI, mode="lines", name=f"RSI {rsi}"))
rfig.add_trace(go.Scatter(x=d.index, y=d.RSI_SMA, mode="lines", name=f"RSI SMA {rsi_sma}"))
rfig.add_hline(y=70, line_dash="dash")
rfig.add_hline(y=30, line_dash="dash")
rfig.update_layout(height=380)
st.plotly_chart(rfig, use_container_width=True)

st.subheader("📋 Trade History")
if not trades.empty:
    st.dataframe(trades.round(2), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download Trades CSV", trades.to_csv(index=False), "trade_history.csv", "text/csv")
else:
    st.warning("No trades with current settings.")

st.subheader("ℹ️ Strategy")
st.markdown(f"""
**BUY:** Close > EMA {ema} AND RSI({rsi}) crosses above RSI SMA({rsi_sma}).

**EXIT:** RSI crosses below its SMA OR price crosses below EMA.
Optional Stop Loss and Take Profit are configurable from the sidebar.

*For educational/backtesting purposes only. Past performance does not guarantee future results.*
""")
