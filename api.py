"""
Financial ML API — serves Plotly chart JSON for stock and crypto symbols.

Run:
    uvicorn api:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""
import json
import logging

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from dataCollector import DEFAULT_SYMBOLS, clean_data, get_history
from dataVisualization import build_chart

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial ML API",
    description="Fetch interactive Plotly charts for stocks and crypto.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"}
VALID_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"}


def _fetch_and_normalize(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = get_history(symbol, period=period, interval=interval)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data returned for '{symbol}'. Check the symbol and try again.")
    df = clean_data(df)
    df.columns = [c.lower() for c in df.columns]
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/symbols", tags=["Meta"])
def symbols() -> dict:
    """Returns the default symbol list."""
    return {"symbols": DEFAULT_SYMBOLS}


@app.get("/chart/{symbol}", tags=["Charts"])
def chart(
    symbol: str,
    period: str = Query(default="1y", description="History length (1d 5d 1mo 3mo 6mo 1y 2y 5y max)"),
    interval: str = Query(default="1d", description="Bar interval (1m 5m 15m 30m 1h 1d 1wk 1mo)"),
) -> dict:
    """
    Returns a Plotly figure as JSON for the given symbol.

    Render on the frontend with:
    ```js
    const fig = await fetch('/chart/AAPL').then(r => r.json());
    Plotly.newPlot('div-id', fig.data, fig.layout);
    ```
    """
    symbol = symbol.upper()

    if period not in VALID_PERIODS:
        raise HTTPException(status_code=422, detail=f"Invalid period '{period}'. Choose from: {sorted(VALID_PERIODS)}")
    if interval not in VALID_INTERVALS:
        raise HTTPException(status_code=422, detail=f"Invalid interval '{interval}'. Choose from: {sorted(VALID_INTERVALS)}")

    logger.info("Chart request: %s  period=%s  interval=%s", symbol, period, interval)

    try:
        df = _fetch_and_normalize(symbol, period, interval)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch data for %s", symbol)
        raise HTTPException(status_code=503, detail=f"Failed to fetch data: {exc}") from exc

    fig = build_chart(df, symbol)
    return json.loads(fig.to_json())
