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
| `dataCollector.py` | Fetches OHLCV history for stocks and crypto via yfinance and writes CSVs to `./data/` |
| `dataVisualization.py` | Builds interactive Plotly charts from collected CSVs and saves them to `./charts/` |
| `prediction.py` | Trains an LSTM direction classifier and saves the model to `./models/` |
| `api.py` | FastAPI backend — serves Plotly chart JSON for any symbol |
| `reportingBot.py` | Discord bot that responds to `c^` commands with live stock data |

## Data collection

```bash
python dataCollector.py                                     # default: AAPL MSFT TSLA BTC-USD ETH-USD DOGE-USD
python dataCollector.py --symbols AAPL BTC-USD              # custom symbols
python dataCollector.py --symbols TSLA --period 2y          # longer history
python dataCollector.py --symbols BTC-USD --interval 1h     # hourly bars
```

CSVs are written to `./data/`, named `<SYMBOL>-<DATE>.csv`. Uses yfinance for both stocks and crypto — no API key required.

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

## API

```bash
pip install fastapi "uvicorn[standard]"
uvicorn api:app --reload --port 8000
```

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /symbols` | Default symbol list |
| `GET /chart/{symbol}` | Plotly figure JSON for the symbol |

**Query params for `/chart/{symbol}`:**

| Param | Default | Options |
|-------|---------|---------|
| `period` | `1y` | `1d 5d 1mo 3mo 6mo 1y 2y 5y max` |
| `interval` | `1d` | `1m 5m 15m 30m 1h 1d 1wk 1mo` |

Examples:
```
GET /chart/AAPL
GET /chart/BTC-USD?period=3mo&interval=1h
GET /chart/TSLA?period=2y
```

Interactive API docs are available at `http://localhost:8000/docs` when the server is running.

**Rendering a chart with plotly.js:**
```html
<div id="chart"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
  fetch('/chart/AAPL')
    .then(r => r.json())
    .then(fig => Plotly.newPlot('chart', fig.data, fig.layout));
</script>
```

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
| `BOT_TOKEN` | `reportingBot.py` |
| `DISCORD_GUILD` | `reportingBot.py` |
