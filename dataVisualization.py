import glob
import logging
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


# ── Indicator calculations ──────────────────────────────────────────────────

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    return mid - num_std * std, mid, mid + num_std * std


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ── Data loading ────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    # strip timezone so plotly doesn't mangle the x-axis labels
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# ── Chart builder ───────────────────────────────────────────────────────────

def build_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    close = df["close"]

    sma_20 = sma(close, 20)
    sma_50 = sma(close, 50)
    ema_20 = ema(close, 20)
    bb_lower, bb_mid, bb_upper = bollinger_bands(close)
    rsi_vals = rsi(close)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20],
        vertical_spacing=0.02,
        subplot_titles=(symbol, "Volume", "RSI (14)"),
    )

    # ── Candlestick ──
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # ── Moving averages ──
    fig.add_trace(
        go.Scatter(x=df.index, y=sma_20, name="SMA 20", line=dict(color="#f59e0b", width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=sma_50, name="SMA 50", line=dict(color="#3b82f6", width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=ema_20, name="EMA 20", line=dict(color="#a855f7", width=1, dash="dot")),
        row=1, col=1,
    )

    # ── Bollinger Bands (upper first, then lower fills back to upper) ──
    fig.add_trace(
        go.Scatter(
            x=df.index, y=bb_upper,
            name="BB Upper",
            line=dict(color="rgba(150,150,150,0.4)", width=1, dash="dash"),
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=bb_lower,
            name="BB Lower",
            line=dict(color="rgba(150,150,150,0.4)", width=1, dash="dash"),
            fill="tonexty",
            fillcolor="rgba(150,150,150,0.08)",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=bb_mid, name="BB Mid", line=dict(color="rgba(150,150,150,0.6)", width=1)),
        row=1, col=1,
    )

    # ── Volume ──
    bar_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=bar_colors, showlegend=False),
        row=2, col=1,
    )

    # ── RSI ──
    fig.add_trace(
        go.Scatter(x=df.index, y=rsi_vals, name="RSI", line=dict(color="#f59e0b", width=1), showlegend=False),
        row=3, col=1,
    )
    for level, color in ((70, "rgba(239,83,80,0.5)"), (30, "rgba(38,166,154,0.5)")):
        fig.add_hline(y=level, line=dict(color=color, dash="dash", width=1), row=3, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=850,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

    return fig


# ── Entry point ─────────────────────────────────────────────────────────────

def plot_all(data_dirs: list[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for data_dir in data_dirs:
        for path in glob.glob(os.path.join(data_dir, "*.csv")):
            symbol = os.path.basename(path).split("-")[0]
            logger.info("Plotting %s from %s", symbol, path)
            try:
                df = load_csv(path)
                fig = build_chart(df, symbol)
                out_path = os.path.join(output_dir, f"{symbol}.html")
                fig.write_html(out_path)
                logger.info("Saved → %s", out_path)
            except Exception:
                logger.exception("Failed to plot %s", symbol)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    plot_all(
        data_dirs=["./data"],
        output_dir="./charts",
    )
