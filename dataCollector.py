"""
Fetches OHLCV history for stocks and crypto via yfinance and writes CSVs to ./data/.

Usage:
    python dataCollector.py                          # default symbol list
    python dataCollector.py --symbols AAPL BTC-USD   # custom symbols
    python dataCollector.py --symbols TSLA --period 2y --interval 1d
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "TSLA", "BTC-USD", "ETH-USD", "DOGE-USD"]


def get_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    return yf.Ticker(symbol).history(period=period, interval=interval)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")


def write_to_csv(df: pd.DataFrame, symbol: str, data_dir: str = "./data") -> None:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "-")
    path = f"{data_dir}/{safe_symbol}-{datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(path, encoding="utf-8")
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fetch OHLCV data for stocks and crypto")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, metavar="SYM",
                        help="Ticker symbols (default: AAPL MSFT TSLA BTC-USD ETH-USD DOGE-USD)")
    parser.add_argument("--period", default="1y", help="History length, e.g. 1y 6mo 30d (default: 1y)")
    parser.add_argument("--interval", default="1d", help="Bar interval, e.g. 1d 1h 15m (default: 1d)")
    args = parser.parse_args()

    for symbol in args.symbols:
        logger.info("Fetching %s  period=%s  interval=%s", symbol, args.period, args.interval)
        df = get_history(symbol, period=args.period, interval=args.interval)
        df = clean_data(df)
        write_to_csv(df, symbol)
