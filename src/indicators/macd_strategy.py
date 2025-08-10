#!/usr/bin/env python3
"""
macd_strategy.py

Simple MACD backtest (daily or intraday) using yfinance.
- Computes MACD (12, 26, 9)
- Buy on MACD line crossing above signal; sell on crossing below (flat when out)
- Saves equity curve plot + trade CSV to results/indicators/macd/

Examples:
  Daily bars (1y):
    python src/indicators/macd_strategy.py --ticker AAPL --period 1y --interval 1d

  Intraday bars (5d of 5m):
    python src/indicators/macd_strategy.py --ticker AAPL --period 5d --interval 5m
"""

import argparse
from pathlib import Path
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


def backtest_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Naive rules:
      - Enter long when MACD crosses above signal.
      - Exit (flat) when MACD crosses below signal.
    No shorting, no slippage/fees.
    """
    out = df.copy()
    out["long"] = 0

    macd = out["macd"]
    sig = out["signal"]
    cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    cross_dn = (macd < sig) & (macd.shift(1) >= sig.shift(1))

    position = 0
    buys = []
    sells = []

    for idx in out.index:
        if cross_up.loc[idx] and position == 0:
            position = 1
            buys.append(idx)
        elif cross_dn.loc[idx] and position == 1:
            position = 0
            sells.append(idx)
        out.at[idx, "long"] = position

    out["ret"] = out["Close"].pct_change().fillna(0)
    out["strat_ret"] = out["long"].shift(1).fillna(0) * out["ret"]
    out["equity"] = (1 + out["strat_ret"]).cumprod()

    trades = []
    buy_iter = iter(buys)
    sell_iter = iter(sells)
    open_buy = None

    # Pair up buy/sell chronologically
    for ts in out.index:
        if open_buy is None and ts in buys:
            open_buy = ts
        elif open_buy is not None and ts in sells:
            # close trade
            buy_ts = open_buy
            sell_ts = ts
            buy_px = float(out.at[buy_ts, "Close"])
            sell_px = float(out.at[sell_ts, "Close"])
            profit = sell_px - buy_px
            trades.append({"buy_time": str(buy_ts), "buy_px": buy_px,
                           "sell_time": str(sell_ts), "sell_px": sell_px,
                           "profit": profit})
            open_buy = None

    return out, pd.DataFrame(trades)


def plot_equity(df: pd.DataFrame, ticker: str, out_png: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(df["equity"], label="MACD Strategy Equity")
    plt.title(f"{ticker} — MACD Strategy Equity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="MACD strategy backtest (yfinance).")
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--period", default="1y", help="e.g., 1y, 6mo, 5d")
    ap.add_argument("--interval", default="1d", help="e.g., 1d, 1h, 5m")
    ap.add_argument("--out-dir", default="results/indicators/macd", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = yf.download(args.ticker, period=args.period, interval=args.interval, auto_adjust=True, progress=False)
    if df.empty:
        print(f"[WARN] No data for {args.ticker}.")
        return

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    dfi = compute_macd(df)
    bt, trades = backtest_macd(dfi)

    # Save outputs
    csv_path = out_dir / f"macd_trades_{args.ticker}_{args.period}_{args.interval}.csv"
    trades.to_csv(csv_path, index=False)
    print(f"[CSV] Saved trades -> {csv_path}")

    png_path = out_dir / f"macd_equity_{args.ticker}_{args.period}_{args.interval}.png"
    plot_equity(bt, args.ticker, png_path)
    print(f"[PLOT] Saved equity curve -> {png_path}")


if __name__ == "__main__":
    main()
