import logging
from datetime import datetime

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_stock_history(symbol: str = "AAPL", period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    return yf.Ticker(symbol).history(period=period, interval=interval)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["Dividends", "Stock Splits"])


def write_to_csv(df: pd.DataFrame, symbol: str) -> None:
    path = f"./data/yfinance/{symbol}-{datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(path, sep=",", encoding="utf-8")
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    stock_list = ["MSFT", "AAPL", "TSLA"]
    for symbol in stock_list:
        logger.info("Processing %s", symbol)
        df = get_stock_history(symbol=symbol, period="1y")
        df = clean_data(df)
        write_to_csv(df, symbol)
