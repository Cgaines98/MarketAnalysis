# Financial ML

A financial data pipeline and ML project for collecting market data, visualizing price history with technical indicators, and predicting next-day price direction using an LSTM classifier.

## Setup

Requires Python 3.10+.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys (see [Environment variables](#environment-variables)).

## Project structure

| File | Purpose |
|------|---------|
| `cryptoDataCollector.py` | Fetches OHLCV history from Binance and writes CSVs to `./data/crypto/` |
| `stockDataCollector.py` | Fetches OHLCV history from yfinance and writes CSVs to `./data/yfinance/` |
| `dataVisualization.py` | Builds interactive Plotly charts from collected CSVs and saves them to `./charts/` |
| `prediction.py` | Trains an LSTM direction classifier and saves the model to `./models/` |
| `reportingBot.py` | Discord bot that responds to `c^` commands with live stock data |

## Data collection

```bash
python stockDataCollector.py     # fetches MSFT, AAPL, TSLA (1 year, daily)
python cryptoDataCollector.py    # fetches BTCUSDT, ETHUSDT, DOGEUSDT (30 days, 1h)
```

CSVs are written to `./data/yfinance/` and `./data/crypto/` respectively, named `<SYMBOL>-<DATE>.csv`.

## Visualization

```bash
python dataVisualization.py
```

Reads all CSVs from both data directories and writes one interactive HTML chart per symbol to `./charts/`. Each chart has three panels:

- **Price** — candlestick with SMA 20/50, EMA 20, and Bollinger Bands
- **Volume** — bars colored by candle direction
- **RSI (14)** — with 70/30 reference lines

Open any `./charts/<SYMBOL>.html` in a browser.

## Prediction

```bash
python prediction.py                            # default: AAPL, 50 epochs, 30-day lookback
python prediction.py --symbol TSLA --epochs 100
python prediction.py --symbol MSFT --lookback 20
```

Trains a multivariate LSTM to classify whether the next trading day's close will be **higher (1) or lower (0)** than the current day.

**Features:** open, high, low, close, volume, SMA 20, SMA 50, EMA 20, RSI 14, Bollinger Band width, daily return

**Pipeline:**
1. Loads the most recent CSV for the symbol, or fetches 2 years from yfinance if no CSV is present
2. 70 / 15 / 15 time-aware train/val/test split (no shuffling)
3. `MinMaxScaler` fit on train only
4. LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16) → sigmoid

Saves the best model to `./models/<SYMBOL>_lstm.keras` and writes two charts to `./charts/`:
- `<SYMBOL>_training.html` — train/val loss and accuracy curves
- `<SYMBOL>_predictions.html` — predicted directions on the test set with confidence bars

## Discord bot

The bot listens for messages prefixed with `c^`.

| Command | Description |
|---------|-------------|
| `c^ hist` | Returns the most recent stock data for TSLA (1d, 1h interval) |

See [botCommands.md](botCommands.md) for the full command reference.

```bash
python reportingBot.py
```

## Environment variables

| Variable | Used by |
|----------|---------|
| `PYTHON_BINANCE_API_KEY` | `cryptoDataCollector.py` |
| `PYTHON_BINANCE_SECRET` | `cryptoDataCollector.py` |
| `BOT_TOKEN` | `reportingBot.py` |
| `DISCORD_GUILD` | `reportingBot.py` |
