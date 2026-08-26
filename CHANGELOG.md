# Changelog

## V5.3 Pro

- Removed Binance dependency entirely.
- Added OKX → KuCoin → Kraken → Coinbase fallback.
- Added 2H construction from closed 1H candles.
- Added strict closed-candle signal processing.
- Reworked backtest to avoid same-candle signal/entry look-ahead.
- Entry occurs at next candle OPEN.
- Conservative SL-first handling when SL and TP both touch a candle.
- Trailing activation delayed until meaningful profit.
- Added break-even protection.
- Added risk-based position sizing and compounding.
- Corrected INR P/L unit conversion using editable USDT/INR assumption.
- Added estimated fees to backtest.
- Added OOS trade-history CSV download.
- Added provider name to live cards.
- Preserved SQLite active-trade persistence.
