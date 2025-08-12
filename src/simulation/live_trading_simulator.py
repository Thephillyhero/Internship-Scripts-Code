#!/usr/bin/env python3
"""
Live/Backtest Trading Simulator (Yahoo + Stooq)
----------------------------------------------
- Primary source: Yahoo Finance (intraday + daily).
- Fallback/alternative: Stooq (daily only, very reliable behind strict networks).
- Works in both modes: --mode {live, backtest}
- Produces plots and logs in user-provided output folders.
- Optional RandomForest model gating (features: [Close, Volume, return]).

CLI (examples):
  Backtest intraday 5m for a prior weekday via Yahoo:
    python src/simulation/live_trading_simulator.py \
      --mode backtest --tickers AAPL MSFT --interval 5m \
      --lookback 50 --min-profit 0.5 --date 2024-07-02 \
      --out-plots results/plots --out-logs results/logs --source yahoo

  Backtest daily via Stooq (bypasses Yahoo issues):
    python src/simulation/live_trading_simulator.py \
      --mode backtest --tickers SPY --interval 1d \
      --lookback 50 --min-profit 0.5 --date 2025-07-31 \
      --out-plots results/plots --out-logs results/logs --source stooq

  Live (during market hours) with automatic source (Yahoo preferred, fallback to Stooq daily):
    python src/simulation/live_trading_simulator.py \
      --mode live --tickers AAPL MSFT --interval 5m --lookback 50 \
      --min-profit 0.5 --out-plots results/plots --out-logs results/logs --source auto

Notes:
- Stooq provides DAILY bars only. If you request an intraday interval with --source stooq,
  we will downgrade to 1d and warn.
- For model gating, pass --model-path to a joblib RandomForest trained on columns
  [Close, Volume, return]. If not provided or load fails, the simulator runs without gating.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from datetime import datetime, time as dtime
import warnings

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------
# Helpers
# ----------------------------------------

MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_intraday_interval(interval: str) -> bool:
    interval = interval.lower()
    return any(s in interval for s in ["m", "h"]) and interval != "1d"


def between_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return df
    # If tz-aware, convert to local naive for .between_time
    if df.index.tz is not None:
        df = df.tz_convert(None)
    try:
        return df.between_time(MARKET_OPEN.strftime('%H:%M'), MARKET_CLOSE.strftime('%H:%M'))
    except Exception:
        return df


# ---------- Network/session hardening for Yahoo ----------

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    })
    s.trust_env = True  # respect corporate proxy env vars if present
    return s

YF_SESSION = make_session()


def fetch_intraday_period_then_filter(ticker: str, interval: str, date_str: str, period: str = "7d") -> pd.DataFrame:
    """Yahoo's intraday works more reliably with period-based queries.
    We fetch a few days then filter to the specific yyyy-mm-dd locally.
    """
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        session=YF_SESSION,
    )
    if df.empty:
        return df
    # Drop to naive for date filtering
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        idx = df.index.tz_convert(None)
    else:
        idx = df.index
    mask = idx.strftime('%Y-%m-%d') == date_str
    return df.loc[mask]


def fetch_yahoo(ticker: str, interval: str, lookback: int, date: str | None, mode: str) -> pd.DataFrame:
    if mode == "backtest":
        if is_intraday_interval(interval):
            if not date:
                raise ValueError("--date is required for intraday backtest")
            df = fetch_intraday_period_then_filter(ticker, interval, date)
        else:
            df = yf.download(ticker, period="1y", interval=interval, auto_adjust=False, progress=False, session=YF_SESSION)
            if df.empty:
                return df
            if date:
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    idx = df.index.tz_convert(None)
                else:
                    idx = df.index
                mask = idx <= pd.to_datetime(date)
                df = df.loc[mask]
    else:  # live
        period = "5d" if is_intraday_interval(interval) else "1y"
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, session=YF_SESSION)
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df


# ---------- Stooq (daily only) ----------

def stooq_symbol(ticker: str) -> str:
    t = ticker.strip().lower()
    # For US symbols Stooq commonly uses ".us" suffix
    if "." not in t:
        t = f"{t}.us"
    return t


def fetch_stooq_daily(ticker: str, date: str | None, lookback: int) -> pd.DataFrame:
    sym = stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        df = pd.read_csv(url)
    except Exception:
        return pd.DataFrame()
    if df.empty or 'Date' not in df.columns:
        return pd.DataFrame()
    df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).set_index('Date')
    # Keep up to date if provided, otherwise tail lookback
    if date:
        df = df.loc[df.index <= pd.to_datetime(date)]
    # mimic lookback window
    df = df.tail(max(lookback, 2))
    return df


def fetch_data(ticker: str, mode: str, interval: str, lookback: int, date: str | None, source: str) -> pd.DataFrame:
    """Fetches data from the requested source with fallbacks.
    - source=yahoo: use Yahoo only
    - source=stooq: use Stooq (forces daily)
    - source=auto: try Yahoo; if empty or network errors, fall back to Stooq daily
    """
    # If stooq and intraday requested, warn & force daily
    if source == "stooq" and is_intraday_interval(interval):
        warnings.warn("Stooq supports daily data only; downgrading interval to 1d.")
        interval = "1d"

    if source in ("yahoo", "auto"):
        try:
            df = fetch_yahoo(ticker, interval, lookback, date, mode)
            if not df.empty:
                return df
            else:
                print(f"[WARN] Yahoo returned empty for {ticker} ({interval}).")
        except Exception as e:
            print(f"[WARN] Yahoo error for {ticker}: {e}")
            df = pd.DataFrame()
        if source == "yahoo":
            return df
        # else fall through to stooq

    # Stooq path (daily only)
    df = fetch_stooq_daily(ticker, date, lookback)
    if df.empty:
        print(f"[WARN] Stooq returned empty for {ticker}.")
    return df


def apply_model_gate(model, df: pd.DataFrame) -> bool:
    """Returns True if model predicts to allow a trade, False to skip.
    Expects RandomForestClassifier trained on [Close, Volume, return].
    Uses the latest bar as a simple feature vector.
    If anything fails, returns True (do not gate).
    """
    if model is None:
        return True
    try:
        if df.empty or any(col not in df.columns for col in ["Close", "Volume"]):
            return True
        ret = df["Close"].pct_change().fillna(0)
        features = pd.DataFrame({
            "Close": [float(df["Close"].iloc[-1])],
            "Volume": [float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0.0],
            "return": [float(ret.iloc[-1])],
        })
        pred = int(model.predict(features)[0])
        return pred == 1
    except Exception as e:
        warnings.warn(f"[ML gate error] {e}")
        return True


def find_trade(df: pd.DataFrame, lookback: int, min_profit: float) -> tuple[bool, dict | None]:
    """Simple min->max swing inside the last `lookback` bars within market hours.
    Returns (found, trade_dict).
    """
    if df.empty:
        return False, None

    # Focus on recent bars
    tail = df.tail(lookback).copy()
    tail = between_market_hours(tail)
    if tail.empty or "Close" not in tail.columns:
        return False, None

    # Find min then subsequent max
    min_idx = tail["Close"].idxmin()
    min_price = float(tail.at[min_idx, "Close"])
    after = tail.loc[min_idx:]
    max_idx = after["Close"].idxmax()
    max_price = float(after.at[max_idx, "Close"]) 

    profit = max_price - min_price
    if max_idx > min_idx and profit >= min_profit:
        minutes_held = int((max_idx - min_idx).total_seconds() // 60)
        return True, {
            "buy_time": str(min_idx),
            "buy_price": min_price,
            "sell_time": str(max_idx),
            "sell_price": max_price,
            "profit": profit,
            "minutes_held": minutes_held,
        }
    return False, None


def plot_trade(df: pd.DataFrame, buy_ts: pd.Timestamp, sell_ts: pd.Timestamp, ticker: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["Close"], label="Close")
    ax.axvline(buy_ts, color="green", linestyle="--", label="Buy")
    ax.axvline(sell_ts, color="red", linestyle="--", label="Sell")
    ax.set_title(f"{ticker} — Trade window")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------------------
# Main
# ----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Live/backtest trading simulator with Yahoo + Stooq fallback")
    ap.add_argument("--mode", required=True, choices=["live", "backtest"], help="Run mode")
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT"], help="List of tickers")
    ap.add_argument("--interval", default="5m", help="Yahoo interval, e.g., 1d, 1h, 5m")
    ap.add_argument("--lookback", type=int, default=50, help="Number of recent bars to examine")
    ap.add_argument("--min-profit", type=float, default=0.5, help="Min $ profit to log a trade")
    ap.add_argument("--model-path", default=None, help="Optional path to joblib RF model")
    ap.add_argument("--date", default=None, help="Backtest date YYYY-MM-DD (required for intraday backtest)")
    ap.add_argument("--out-plots", default="results/plots", help="Folder to save plots")
    ap.add_argument("--out-logs", default="results/logs", help="Folder to save CSV logs")
    ap.add_argument("--source", default="auto", choices=["yahoo", "stooq", "auto"], help="Data source")
    args = ap.parse_args()

    out_plots = Path(args.out_plots); ensure_dir(out_plots)
    out_logs = Path(args.out_logs); ensure_dir(out_logs)

    # Load model if provided
    model = None
    if args.model_path:
        try:
            model = joblib.load(args.model_path)
            print(f"✅ Loaded model: {args.model_path}")
        except Exception as e:
            warnings.warn(f"[WARN] Failed to load model: {e}. Continuing without ML gating.")

    trades_all = []

    for ticker in args.tickers:
        print(f"\nFetching {ticker}...")
        try:
            df = fetch_data(ticker, args.mode, args.interval, args.lookback, args.date, args.source)
            if df.empty:
                print(f"[WARN] No data for {ticker}.")
                continue

            if not apply_model_gate(model, df):
                print("[ML] Model gate: skip trade for now.")
                continue

            found, trade = find_trade(df, args.lookback, args.min_profit)
            if not found:
                print(f"No qualifying trade for {ticker} in the last {args.lookback} bars.")
                continue

            # Log & plot
            buy_ts = pd.to_datetime(trade["buy_time"]) 
            sell_ts = pd.to_datetime(trade["sell_time"]) 
            print(
                f"✅ Trade {ticker}: Buy ${trade['buy_price']:.2f} at {buy_ts.time()} → "
                f"Sell ${trade['sell_price']:.2f} at {sell_ts.time()} | Profit ${trade['profit']:.2f} "
                f"(held {trade['minutes_held']}m)"
            )

            plot_file = out_plots / f"{ticker}_{args.mode}_{args.interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_trade(df.tail(args.lookback), buy_ts, sell_ts, ticker, plot_file)

            trades_all.append({"ticker": ticker, **trade})
        except Exception as e:
            warnings.warn(f"[ERROR] {ticker}: {e}")
            continue

    # Save trades CSV
    if trades_all:
        df_trades = pd.DataFrame(trades_all)
        csv_file = out_logs / "trades.csv"
        # Append if exists
        if csv_file.exists():
            old = pd.read_csv(csv_file)
            df_trades = pd.concat([old, df_trades], ignore_index=True)
        df_trades.to_csv(csv_file, index=False)
        print(f"Saved trades -> {csv_file}")
    else:
        print("No trades found this run.")


if __name__ == "__main__":
    # Quiet some yfinance noisy warnings
    warnings.simplefilter("ignore", category=UserWarning)
    main()
