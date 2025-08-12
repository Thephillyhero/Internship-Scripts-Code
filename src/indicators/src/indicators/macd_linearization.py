#!/usr/bin/env python3
"""
macd_linearization.py

MACD + Least-Squares Moving Average (LSMA) "linearization".
- Computes MACD (12, 26, 9) from yfinance OHLCV
- Computes LSMA (linear regression over a rolling window) on the MACD line
- Trade rule: long when MACD crosses ABOVE LSMA, flat when crosses BELOW
- Saves plot + trades CSV + equity CSV under results/macd_linearization/

Examples:
  Daily (1y):
    python src/indicators/macd_linearization.py --ticker AAPL --period 1y --interval 1d

  Intraday (5d of 5m):
    python src/indicators/macd_linearization.py --ticker AAPL --period 5d --interval 5m --window 25
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def compute_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["Close"].ewm(span=fast, adjust=False).mean()
    out["ema_slow"] = out["Close"].ewm(span=slow, adjust=False).mean()
    out["macd"] = out["ema_fast"] - out["ema_slow"]
    out["signal"] = out["macd"].ewm(span=signal, adjust=False).mean()
    out["hist"] = out["macd"] - out["signal"]
    return out


def lsma(series: pd.Series, window: int) -> pd.Series:
    """
    Least-Squares Moving Average:
    For each rolling window, fit y = a*x + b where x = [0..window-1],
    then return the fitted value at the LAST point in the window.

    This is effectively a linear regression smoothing of the series.
    """
    x = np.arange(window)

    def fit_last(y_window: np.ndarray) -> float:
        # Fit line to window: returns value at last x (window-1)
        # polyfit gives slope a and intercept b for y = a*x + b
        a, b = np.polyfit(x, y_window, 1)
        return a * (window - 1) + b

    return series.rolling(window=window, min_periods=window).apply(
        lambda y: fit_last(np.asarray(y)), raw=False
    )


def backtest_macd_lsma(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trading logic:
      - Enter long when MACD crosses above LSMA
      - Exit (flat) when MACD crosses below LSMA
    No shorting, no fees/slippage.
    """
    out = df.copy()
    macd = out["macd"]
    lin  = out["lsma"]

    cross_up = (macd > lin) & (macd.shift(1) <= lin.shift(1))
    cross_dn = (macd < lin) & (macd.shift(1) >= lin.shift(1))

    position = 0
    buys, sells = [], []

    out["long"] = 0
    for ts in out.index:
        if cross_up.loc[ts] and position == 0:
            position = 1
            buys.append(ts)
        elif cross_dn.loc[ts] and position == 1:
            position = 0
            sells.append(ts)
        out.at[ts, "long"] = position

    # Strategy equity on price (not on MACD)
    out["ret"] = out["Close"].pct_change().fillna(0)
    out["strat_ret"] = out["long"].shift(1).fillna(0) * out["ret"]
    out["equity"] = (1 + out["strat_ret"]).cumprod()

    # Pair trades
    trades = []
    open_buy = None
    for ts in out.index:
        if open_buy is None and ts in buys:
            open_buy = ts
        elif open_buy is not None and ts in sells:
            buy_ts = open_buy
            sell_ts = ts
            buy_px = float(out.at[buy_ts, "Close"])
            sell_px = float(out.at[sell_ts, "Close"])
            profit = sell_px - buy_px
            trades.append({
                "buy_time": str(buy_ts),
                "buy_px": buy_px,
                "sell_time": str(sell_ts),
                "sell_px": sell_px,
                "profit": profit
            })
            open_buy = None

    return out, pd.DataFrame(trades)


def plot_macd_lsma(df: pd.DataFrame, ticker: str, out_png: Path, title_extra: str = ""):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Price
    ax[0].plot(df.index, df["Close"], label="Close")
    ax[0].set_title(f"{ticker} — Price {title_extra}".strip())
    ax[0].grid(True)
    ax[0].legend()

    # MACD + LSMA
    ax[1].plot(df.index, df["macd"], label="MACD")
    ax[1].plot(df.index, df["lsma"], label="LSMA(MACD)")
    ax[1].plot(df.index, df["signal"], label="Signal", alpha=0.6)
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].set_title("MACD, LSMA(MACD), Signal")
    ax[1].grid(True)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="MACD linearization with LSMA and crossover strategy.")
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--period", default="1y", help="e.g., 1y, 6mo, 5d")
    ap.add_argument("--interval", default="1d", help="e.g., 1d, 1h, 5m")
    ap.add_argument("--window", type=int, default=20, help="LSMA window (bars)")
    ap.add_argument("--out-dir", default="results/macd_linearization", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = yf.download(args.ticker, period=args.period, interval=args.interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        print(f"[WARN] No data for {args.ticker}.")
        return
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    dfi = compute_macd(df)
    dfi["lsma"] = lsma(dfi["macd"], window=args.window)

    # Drop rows before LSMA is available
    dfi = dfi.dropna(subset=["lsma"]).copy()

    bt, trades = backtest_macd_lsma(dfi)

    # Save outputs
    csv_trades = out_dir / f"macd_lsma_trades_{args.ticker}_{args.period}_{args.interval}_w{args.window}.csv"
    trades.to_csv(csv_trades, index=False)
    print(f"[CSV] Trades -> {csv_trades}")

    csv_equity = out_dir / f"macd_lsma_equity_{args.ticker}_{args.period}_{args.interval}_w{args.window}.csv"
    bt[["equity", "long", "Close", "macd", "lsma", "signal"]].to_csv(csv_equity)
    print(f"[CSV] Equity & signals -> {csv_equity}")

    png = out_dir / f"macd_lsma_{args.ticker}_{args.period}_{args.interval}_w{args.window}.png"
    title_extra = f"(window={args.window})"
    plot_macd_lsma(bt, args.ticker, png, title_extra=title_extra)
    print(f"[PLOT] Chart -> {png}")


if __name__ == "__main__":
    main()
