# V5.3 Pro Crypto Trading Signal Dashboard

A conservative Streamlit + Python **paper/research** dashboard for BTC, ETH, BNB, SOL and XRP.

## What was fixed from V5.2

The previous versions had two important problems:

1. Binance HTTP 451 / restricted-location errors on Streamlit Cloud.
2. The backtest could make unrealistic decisions around the same candle used for a signal.

V5.3 removes Binance completely and uses:

**OKX → KuCoin → Kraken → Coinbase**

The backtest now follows a strict sequence:

**closed candle signal → next candle OPEN entry → future candle management**

It does not enter at a candle close and then use that same candle's high/low to manufacture a result.

## Risk / reward design

The objective is not to force a 70% win rate. The engine is designed around controlled risk:

- Starting capital: ₹10,000
- Risk per trade: 1% of current equity
- Initial hard stop: 2%
- Base take profit: 4%
- Minimum intended reward/risk: 2:1
- Trailing activates only after approximately +2%
- Break-even protection is added when trailing activates
- ATR-based trailing distance
- Early exit only after a confirmed momentum/trend deterioration
- 0.05% estimated fee per side in backtest
- Compounding: position risk is based on current equity

This does **not** guarantee profit. A strategy can still lose money.

## Signal model

Signals use closed candles and combine:

- EMA20
- EMA50
- EMA200
- RSI(14)
- RSI SMA(14)
- ATR(14)
- Volume vs 20-period average
- Candle-body / close-position confirmation
- Volatility sanity filter
- Extra quality restrictions for BTC and ETH

Labels:

- Strong Buy
- Buy
- Wait
- Sell
- Strong Sell

## Trade lifecycle

`WAIT → ENTER LONG/SHORT → HOLD → EXIT NOW`

The persistent ledger stores:

- Entry
- Current price
- Initial SL
- Base TP
- Trailing stop
- Risk amount
- Current P/L
- Signal strength
- Signal label
- Exit reason

SQLite keeps active paper trades across Streamlit reruns.

## Backtest

The Backtest button runs:

- Previous 30 days: In-Sample
- Most recent 30 days: Out-of-Sample
- EMA200 warm-up before both windows
- Compounding
- Win rate
- Total P/L
- Average win
- Average loss
- Profit factor
- Maximum drawdown
- TP exits
- SL exits
- Trailing exits
- Early exits
- End-of-test exits
- Full OOS trade history CSV download

### Look-ahead protection

The signal is calculated only from a completed candle. The trade enters at the next candle's open. If the next candle touches both SL and TP, the backtest assumes **SL first** (conservative ambiguity handling).

Trailing stops are updated only after a candle closes, so a newly created trailing stop cannot retroactively exit the same candle.

## Historical data

The engine fetches substantially more than 200 candles before the evaluation windows. EMA200 is calculated on that warm-up history and only then is the 30-day test window evaluated.

2H is constructed from closed 1H candles, so a provider does not need a native 2H market timeframe.

## INR conversion

Exchange OHLCV is quoted in USDT/USD. The sidebar contains an editable **USDT → INR assumption** (default 88.0) so the ₹10,000 account and P/L calculations have consistent units.

Change it if you want to use a different conversion assumption. It is not a live FX feed.

## Streamlit Cloud

No exchange API keys are required because the dashboard uses public OHLCV endpoints.

Upload these files to GitHub:

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `README.md`
- `LICENSE`

Then deploy `app.py` on Streamlit Cloud.

## Local run

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Data-provider failure behavior

If OKX fails, the app tries KuCoin. If KuCoin fails, it tries Kraken, then Coinbase. If all fail, only that card/backtest reports the failure; the entire app does not crash.

## Important limitations

- This is a rule-based research/paper system.
- No real orders are placed.
- No win-rate or profit is guaranteed.
- A 30-day OOS sample can be statistically small.
- Public exchange data can differ slightly between providers.
- SQLite on Streamlit Cloud is not guaranteed to survive container replacement. Use a persistent database for production-grade state.
