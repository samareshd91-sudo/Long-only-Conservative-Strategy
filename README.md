# V5.2 Crypto Trading Signal Dashboard

Streamlit + Python crypto research / paper-signal dashboard for BTC, ETH, BNB, SOL and XRP.

## The Streamlit Cloud / Binance 451 problem is handled

This version **does not use Binance**.

Some Streamlit Cloud deployments can receive Binance HTTP `451 Service unavailable from a restricted location` responses. To prevent that single-provider failure from taking down the dashboard, V5 uses a public-data fallback chain:

1. OKX
2. KuCoin
3. Kraken
4. Coinbase

If one provider fails, the next provider is tried automatically. If all providers fail, only the affected card/backtest is reported as unavailable; the entire app does not crash.

No API keys are required.

## Timeframes

- 1H
- 2H
- 4H
- 1D

2H is constructed from closed 1H candles, so the dashboard does not depend on an exchange offering a native 2H timeframe.

## Signal engine

- EMA200
- RSI(14)
- RSI SMA(14)
- Volume vs 20-period average
- ATR(14)
- Candle confirmation
- Strong Buy / Buy / Wait / Sell / Strong Sell
- Extra quality filters for BTC and ETH
- Closed-candle analysis

## Trade lifecycle

`WAIT → ENTER LONG/SHORT → HOLD → EXIT NOW`

Each active trade tracks:

- Entry
- Current price
- Initial 2% SL
- Base 4% TP
- ATR-based trailing stop
- Current P/L
- Signal strength
- Exit reason

Early exit can happen when momentum weakens or the trend reverses.

Active paper trades are persisted in SQLite so Streamlit reruns/refreshes do not erase them.

## Backtest

- ₹10,000 starting capital
- Compounding
- 30-day in-sample
- Recent 30-day out-of-sample
- Win rate
- Total P/L
- Average win/loss
- Profit factor
- Max drawdown
- TP / SL / trailing / early exits
- Trade history

The backtest fetches substantially more than 200 candles before the evaluation window. EMA200 is therefore **not** initialized from only the 30-day test period.

## No fake performance

This project does not promise:

- 70% win rate
- ₹7,000 profit
- guaranteed returns

Results are actual results from the retrieved historical data and the coded rules.

## Run locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

Upload:

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `README.md`

Then deploy `app.py`.

No exchange credentials are needed for public OHLCV.

### Persistence note

SQLite is suitable for local use. Streamlit Cloud containers can be recycled, so production-grade persistent trade state should use hosted PostgreSQL or another persistent database.

## Paper trading only

The dashboard does not place real orders. ENTER/HOLD/EXIT is a paper-trading state machine.

Past performance does not guarantee future results.


## V5.2 bug fix

V5.1's provider failover was correct, but a packaging regression omitted the
indicator helper functions used by the signal/backtest engine. V5.2 restores
the EMA/RSI/ATR/enrichment layer and adds warm-up validation before signals
are evaluated.
