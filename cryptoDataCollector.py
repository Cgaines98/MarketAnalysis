import logging
import os
import time
from datetime import datetime

import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_KLINE_COLUMNS = [
    "date", "open", "high", "low", "close", "volume",
    "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore",
]
_DROP_COLUMNS = ["close_time", "qav", "taker_base_vol", "taker_quote_vol", "ignore"]
_FLOAT_COLUMNS = ["open", "high", "low", "close", "volume"]


def clean_data(data: list) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=_KLINE_COLUMNS)
    df = df.drop(columns=_DROP_COLUMNS)
    df[_FLOAT_COLUMNS] = df[_FLOAT_COLUMNS].astype(float)
    return df


def log_summary(df: pd.DataFrame, symbol: str) -> None:
    highest_high = df["high"].max()
    lowest_low = df["low"].min()
    logger.info("%s — 30d high: %.4f  low: %.4f", symbol, highest_high, lowest_low)


def write_to_csv(df: pd.DataFrame, symbol: str) -> None:
    df = df.copy()
    df["date"] = df["date"].apply(
        lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t / 1000))
    )
    path = f"./data/crypto/{symbol}-{datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(path, index=False)
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    client = Client(
        api_key=os.getenv("PYTHON_BINANCE_API_KEY"),
        api_secret=os.getenv("PYTHON_BINANCE_SECRET"),
    )

    symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]
    for symbol in symbols:
        logger.info("Processing %s", symbol)
        data = client.get_historical_klines(symbol, interval="1h", start_str="30d")
        df = clean_data(data)
        log_summary(df, symbol)
        write_to_csv(df, symbol)
