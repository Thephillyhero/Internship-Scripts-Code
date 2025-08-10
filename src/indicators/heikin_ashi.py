#!/usr/bin/env python3
"""
heikin_ashi.py

Heikin-Ashi transform + simple trend detection.
- Converts OHLC to Heikin-Ashi candles
- Flags bullish/bearish stretches
- Saves HA chart + summary CSV to results/indicators/heikin_ashi/

Examples:
  Daily bars (6mo):
    python src/indicators/heikin_ashi.py --ticker TSLA --period 6mo --interval 1d

  Intraday (5d of 5m):
    python src/indicators/heikin_ashi.py --ticker TSLA --period 5d --interval 5m
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


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with Heikin-Ashi OHLC (HA_Open, HA_High, HA_Low, HA_Close)."""
    o = df["Open"].copy()
    h = df["High"].copy()
    l = df["Low"].copy()
    c = df["Close"].copy()

    ha = pd.DataFrame(index=df.index)
    ha["HA_Close"] = (o + h + l + c) / 4.0
    ha["HA_Open"] = 0.0
    ha["HA_High"] = 0.0
    ha["HA_Low"] = 0.0

    # Initialize HA_Open
    ha.iloc[0, ha.columns.get_loc("HA_Open")] = (o.iloc[0] + c.iloc[0]) / 2.0

    for i in range(1, len(df)):
        ha.iloc[i, ha.columns.get_loc("HA_Open")] = (
            ha.iloc[i-1, ha.columns.get_loc("HA_Open")] + ha.iloc[i-1, ha.columns.get_loc("HA_Close")]
        ) / 2.0

    ha["HA_High"] = pd.concat([ha["HA_Open"], ha["HA_Close"], h], axis=1).max(axis=1)
    ha["HA_Low"] = pd.concat([ha["HA_Open"], ha["HA_Close"], l], axis=1).min(axis=1)
    return ha


def label_trend(ha: pd.DataFrame) -> pd.DataFrame:
    """
    Simple trend labeling:
      bullish if HA_Close > HA_Open
      bearish if HA_Close < HA_Open
    """
    out = ha.copy()
    out["bullish"] = (out["HA_Close"] > out["HA_Open"]).astype(int)
    out["bearish"] = (out["HA_Close"] < out["HA_Open"]).astype(int)
    return out


def plot_ha(ha: pd.DataFrame, ticker: str, out_png: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(ha.index, ha["HA_Close"], label="HA Close")
    plt.fill_between(ha.index, ha["HA_Open"], ha["HA_Close"],
                     where=(ha["HA_Close"] >= ha["HA_Open"]),
                     alpha=0.2, label="Bullish", step=None)
    plt.fill_between(ha.index, ha["HA_Open"], ha["HA_Close"],
                     where=(ha["HA_Close"] < ha["HA_Open"]),
                     alpha=0.2, label="Bearish", step=None)
    plt.title(f"{ticker} — Heikin-Ashi Close")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Heikin-Ashi transform and trend flags.")
    ap.add_argument("--ticker", default="TSLA")
    ap.add_argument("--period", default="6mo", help="e.g., 6mo, 1y, 5d")
    ap.add_argument("--interval", default="1d", help="e.g., 1d, 1h, 5m")
    ap.add_argument("--out-dir", default="results/indicators/heikin_ashi", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = yf.download(args.ticker, period=args.period, interval=args.interval, auto_adjust=True, progress=False)
    if df.empty:
        print(f"[WARN] No data for {args.ticker}.")
        return

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if not {'Open','High','Low','Close'}.issubset(df.columns):
        print("[ERROR] Missing OHLC columns.")
        return

    ha = heikin_ashi(df)
    lab = label_trend(ha)

    # Save CSV summary
    csv_path = out_dir / f"heikin_ashi_{args.ticker}_{args.period}_{args.interval}.csv"
    lab.to_csv(csv_path, index=True)
    print(f"[CSV] Saved HA summary -> {csv_path}")

    # Save plot
    png_path = out_dir / f"heikin_ashi_{args.ticker}_{args.period}_{args.interval}.png"
    plot_ha(ha, args.ticker, png_path)
    print(f"[PLOT] Saved HA chart -> {png_path}")


if __name__ == "__main__":
    main()
