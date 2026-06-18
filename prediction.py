"""
LSTM direction classifier — predicts whether the next trading day closes
higher (1) or lower (0) than the current day.

Usage:
    python prediction.py                         # default: AAPL, 50 epochs
    python prediction.py --symbol TSLA --epochs 100
    python prediction.py --symbol MSFT --lookback 20
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)

FEATURE_COLS = ["open", "high", "low", "close", "volume",
                "sma_20", "sma_50", "ema_20", "rsi", "bb_width", "pct_return"]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(symbol: str, data_dir: str = "./data/yfinance") -> pd.DataFrame:
    files = sorted(Path(data_dir).glob(f"{symbol}-*.csv"))
    if files:
        path = files[-1]
        logger.info("Loading %s", path)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
    else:
        logger.info("No CSV found for %s — fetching 2y from yfinance", symbol)
        df = yf.Ticker(symbol).history(period="2y", interval="1d")
        df = df.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        df.columns = [c.lower() for c in df.columns]

    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["bb_width"] = (4 * std) / mid   # (upper - lower) / mid

    df["pct_return"] = close.pct_change()

    return df


def create_target(df: pd.DataFrame) -> pd.Series:
    return (df["close"].shift(-1) > df["close"]).astype(int)


# ── Sequence building ─────────────────────────────────────────────────────────

def make_sequences(
    X: np.ndarray, y: np.ndarray, start: int, end: int, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(max(start, lookback), end):
        xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(lookback: int, n_features: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_training_history(history: keras.callbacks.History, symbol: str, output_dir: str) -> None:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))

    for col, metric, val_metric in (
        (1, "loss", "val_loss"),
        (2, "accuracy", "val_accuracy"),
    ):
        fig.add_trace(
            go.Scatter(y=history.history[metric], name="Train", line=dict(color="#3b82f6")),
            row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(y=history.history[val_metric], name="Val", line=dict(color="#f59e0b")),
            row=1, col=col,
        )

    fig.update_layout(title=f"{symbol} — Training history", template="plotly_dark", height=400, showlegend=True)
    out = os.path.join(output_dir, f"{symbol}_training.html")
    fig.write_html(out)
    logger.info("Saved training history → %s", out)


def plot_predictions(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    symbol: str,
    output_dir: str,
) -> None:
    correct = y_true == y_pred

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
        subplot_titles=("Predicted direction (▲ Up / ▼ Down)", "Prediction confidence"),
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_pred.astype(float),
            mode="markers",
            marker=dict(
                symbol=["triangle-up" if p == 1 else "triangle-down" for p in y_pred],
                color=["#26a69a" if c else "#ef5350" for c in correct],
                size=9,
            ),
            name="Predicted (green=correct)",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=y_true.astype(float),
            mode="lines", line=dict(color="rgba(200,200,200,0.4)", width=1),
            name="Actual",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=dates, y=y_prob,
            marker_color=["#26a69a" if p >= 0.5 else "#ef5350" for p in y_prob],
            name="Confidence",
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0.5, line=dict(color="gray", dash="dash", width=1), row=2, col=1)

    fig.update_layout(
        title=f"{symbol} — Test set predictions",
        template="plotly_dark",
        height=600,
        yaxis=dict(tickvals=[0, 1], ticktext=["Down", "Up"]),
        yaxis2=dict(title="P(Up)", range=[0, 1]),
    )
    out = os.path.join(output_dir, f"{symbol}_predictions.html")
    fig.write_html(out)
    logger.info("Saved prediction chart → %s", out)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(symbol: str = "AAPL", epochs: int = 50, lookback: int = 30) -> None:
    df = load_data(symbol)
    df = add_features(df)
    target = create_target(df)

    # Drop NaN rows (from rolling windows), then drop last row (no future target)
    df = df.dropna().iloc[:-1]
    target = target.loc[df.index]

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y = target.values
    n = len(X)

    if n < lookback + 40:
        raise ValueError(f"Not enough data ({n} rows) for lookback={lookback}. Collect more history.")

    # ── Time-aware split: 70 / 15 / 15 ──
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    logger.info("Samples — train: %d  val: %d  test: %d", train_end, val_end - train_end, n - val_end)

    # ── Scale (fit on train only, transform all) ──
    scaler = MinMaxScaler()
    X_scaled = np.empty_like(X, dtype=float)
    X_scaled[:train_end] = scaler.fit_transform(X[:train_end])
    X_scaled[train_end:] = scaler.transform(X[train_end:])

    # ── Sequences (val/test windows may reach back into prior splits) ──
    X_train, y_train = make_sequences(X_scaled, y, 0, train_end, lookback)
    X_val, y_val = make_sequences(X_scaled, y, train_end, val_end, lookback)
    X_test, y_test = make_sequences(X_scaled, y, val_end, n, lookback)
    test_dates = df.index[val_end:]

    logger.info("Sequences — train: %d  val: %d  test: %d", len(X_train), len(X_val), len(X_test))

    # ── Train ──
    os.makedirs("./models", exist_ok=True)
    model = build_model(lookback, n_features=len(feature_cols))
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                f"./models/{symbol}_lstm.keras", monitor="val_loss", save_best_only=True
            ),
        ],
        verbose=1,
    )

    # ── Evaluate ──
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    logger.info(
        "\nTest set results:\n%s",
        classification_report(y_test, y_pred, target_names=["Down", "Up"]),
    )
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

    os.makedirs("./charts", exist_ok=True)
    plot_training_history(history, symbol, "./charts")
    plot_predictions(test_dates, y_test, y_pred, y_prob, symbol, "./charts")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="LSTM direction classifier for stocks")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs (default: 50)")
    parser.add_argument("--lookback", type=int, default=30, help="Sequence lookback in trading days (default: 30)")
    args = parser.parse_args()

    run(symbol=args.symbol, epochs=args.epochs, lookback=args.lookback)
