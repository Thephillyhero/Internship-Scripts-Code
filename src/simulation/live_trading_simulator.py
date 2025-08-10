#!/usr/bin/env python3
"""
live_trading_simulator.py

Portable intraday simulator that can run in two modes:
- backtest: simulate a single market day (works anytime)
- live:     run during market hours ET

Features:
- Uses yfinance only (no broker keys required)
- Optional Random Forest gating (if model file exists)
- Headless plotting (saves PNGs; no GUI needed)
- Clean CLI flags and sensible defaults

Usage examples:
  Backtest yesterday (Linux/macOS):
    python src/simulation/live_trading_simulator.py --mode backtest --date 2025-08-07

  Live mode (run during market hours ET):
    python src/simulation/live_trading_simulator.py --mode live

  With explicit options:
    python src/simulation/live_trading_simulator.py \
      --mode backtest --date 2025-08-07 \
      --tickers AAPL MSFT TSLA \
      --interval 5m --lookback 100 --min-profit 1.0 \
      --model-path src/models/random_forest/rf_trade_model.pkl \
      --out-plots results/plots --out-logs results/logs/trades.csv
"""

import argparse
from datetime import time as dtime
from pathlib import Path

import pandas as pd
import yfinance as yf
import joblib

# Headless plotting so this runs fine on servers / GitHub Codespaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytz

# ---------------------------------------------------------------------
# Optional import from your shared features module. If it's not present,
# we define a local fallback so this file works standalone.
# ---------------------------------------------------------------------
def _add_basic_features_local(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ['Close','Volume'] exist and add 'return' column."""
    need = {'Close', 'Volume'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df.copy()
    out['return'] = out['Close'].pct_change().fillna(0)
    return out

def _latest_feature_row_local(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 1-row DataFrame with ['Close','Volume','return'] for the latest bar."""
    df_feat = _add_basic_features_local(df)
    last = df_feat.iloc[-1]
    return pd.DataFrame(
        {'Close': [last['Close']], 'Volume': [last['Volume']], 'return': [last['return']]}
    )

try:
    # If your repo has src/features/engineering.py, we use that for consistency
    from src.features.engineering import latest_feature_row as _latest_feature_row
except Exception:
    # Fallback to local helpers if the module isn't available
    _latest_feature_row = _latest_feature_row_local


DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA", "PLTR", "UBER", "FUBO", "RIOT"]


def is_market_open(now_utc: pd.Timestamp) -> bool:
    """Return True if current time in US/Eastern is within 09:30–16:00 on a weekday."""
    eastern = pytz.timezone("US/Eastern")
    now_est = now_utc.astimezone(eastern)
    if now_est.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    return dtime(9, 30) <= now_est.time() <= dtime(16, 0)


def fetch_recent_intraday(ticker: str, interval: str, lookback_minutes: int) -> pd.DataFrame:
    """
    Live mode: use period='1d' to avoid future time windows, then filter the last lookback minutes.
    """
    df = yf.download(ticker, period="1d", interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # yfinance timestamps are usually tz-aware UTC; align if needed
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(minutes=lookback_minutes)
    df = df[df.index >= cutoff]
    return df


def fetch_day_intraday(ticker: str, interval: str, day_ymd: str) -> pd.DataFrame:
    """
    Backtest mode: fetch bars for a specific calendar date (market hours in US/Eastern).
    """
    eastern = pytz.timezone("US/Eastern")
    day = pd.Timestamp(day_ymd).tz_localize(eastern)
    start_est = day.replace(hour=9, minute=30, second=0)
    end_est = day.replace(hour=16, minute=0, second=0)
    start_utc = start_est.tz_convert("UTC")
    end_utc = end_est.tz_convert("UTC")
    df = yf.download(ticker, start=start_utc, end=end_utc, interval=interval, progress=False, auto_adjust=True)
    if not df.empty and not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # ensure tz-aware
    if not df.empty and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Live/Backtest intraday simulator with optional RF gating.")
    ap.add_argument("--mode", choices=["live", "backtest"], default="backtest", help="Run in live or backtest mode.")
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS, help="Tickers to scan.")
    ap.add_argument("--interval", default="5m", help="Bar interval (e.g., 5m, 1m, 15m).")
    ap.add_argument("--lookback", type=int, default=100, help="Minutes to include (live mode).")
    ap.add_argument("--min-profit", type=float, default=1.0, help="Minimum $ gain (max_after_min - min) to log a trade.")
    ap.add_argument("--model-path", default="src/models/random_forest/rf_trade_model.pkl",
                    help="Path to joblib RF model (optional; script runs fine without it).")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD for backtest mode.")
    ap.add_argument("--out-plots", default="results/plots", help="Directory to save charts.")
    ap.add_argument("--out-logs", default="results/logs/trades.csv", help="CSV path to append trade logs.")
    args = ap.parse_args()

    plots_dir = Path(args.out_plots)
    ensure_dir(plots_dir)
    logs_path = Path(args.out_logs)
    ensure_dir(logs_path.parent)

    # Load model if present (optional)
    model = None
    model_path = Path(args.model_path)
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            print(f"✅ Loaded model from {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load model at {model_path}: {e}. Proceeding without ML gating.")
    else:
        print("⚠️ No model found — running WITHOUT ML gating. (This is fine for Quick Verify.)")

    trades = []

    for ticker in args.tickers:
        print(f"\nFetching {ticker}...")
        # Pull data based on mode
        if args.mode == "live":
            now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
            if not is_market_open(now_utc):
                print("Market closed; skipping live fetch for this ticker.")
                continue
            df = fetch_recent_intraday(ticker, args.interval, args.lookback)
        else:
            if not args.date:
                print("Backtest mode requires --date YYYY-MM-DD. Skipping this ticker.")
                continue
            df = fetch_day_intraday(ticker, args.interval, args.date)

        if df.empty:
            print(f"[WARN] No data for {ticker}.")
            continue

        # Slice to market hours (defensive, even in backtest)
        df = df.between_time("09:30", "16:00")
        if df.empty or "Close" not in df or "Volume" not in df:
            print(f"[WARN] No usable bars for {ticker} after filtering.")
            continue

        # Find min close, then max close AFTER that min (simple swing logic)
        min_idx = df["Close"].idxmin()
        df_after_min = df.loc[min_idx:]
        if df_after_min.empty:
            print(f"[WARN] No bars after min for {ticker}.")
            continue

        max_idx = df_after_min["Close"].idxmax()

        # Scalar prices
        min_price = float(df.at[min_idx, "Close"])
        max_price = float(df_after_min.at[max_idx, "Close"])
        profit = max_price - min_price
        print(f"[DEBUG] {ticker} Min {min_price:.2f} at {min_idx}, Max {max_price:.2f} at {max_idx}, Gain ${profit:.2f}")

        # ML gating (optional)
        if model is not None:
            try:
                feats = _latest_feature_row(df)  # columns ['Close','Volume','return'] in that order
                pred = int(model.predict(feats)[0])
                if pred != 1:
                    print("[ML] Model says skip.")
                    # If you want to ENFORCE gating, uncomment the next line:
                    # continue
            except Exception as e:
                print(f"[ML ERROR] {e}  (continuing without gating)")

        # Log trade if it meets threshold and occurs in correct order
        if max_idx > min_idx and profit >= args.min_profit:
            minutes_held = int((max_idx - min_idx).total_seconds() // 60)
            trade = {
                "ticker": ticker,
                "buy_time": str(min_idx),
                "buy_price": min_price,
                "sell_time": str(max_idx),
                "sell_price": max_price,
                "profit": profit,
                "minutes_held": minutes_held,
            }
            trades.append(trade)
            print(f"✅ Trade: {trade}")

            # Plot and save
            plt.figure(figsize=(10, 5))
            plt.plot(df["Close"], label="Close")
            plt.axvline(min_idx, linestyle="--", label="Buy")
            plt.axvline(max_idx, linestyle="--", label="Sell")
            plt.title(f"{ticker} {args.mode} {args.interval}")
            plt.legend()
            plt.grid(True)
            out_png = plots_dir / f"{ticker}_{args.mode}_{args.interval}_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(out_png, dpi=120)
            plt.close()
            print(f"[PLOT] Saved {out_png}")

    # Append/Write CSV log
    if trades:
        df_tr = pd.DataFrame(trades)
        if logs_path.exists():
            try:
                old = pd.read_csv(logs_path)
                df_tr = pd.concat([old, df_tr], axis=0, ignore_index=True)
            except Exception as e:
                print(f"[WARN] Could not read existing log: {e}. Overwriting.")
        df_tr.to_csv(logs_path, index=False)
        print(f"[LOG] Saved {len(trades)} trades to {logs_path}")
    else:
        print("No qualifying trades this run.")


if __name__ == "__main__":
    main()
