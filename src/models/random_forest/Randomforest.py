# src/models/random_forest/Randomforest.py
"""
Random Forest trainer for intraday/daily trade classification.
- Fetches data from Yahoo Finance with a Stooq daily fallback.
- Trains on features: ['Close', 'Volume', 'return']  (matches live_trading_simulator.py)
- Saves model with joblib and writes a small training summary CSV.

Examples
--------
# Auto-source (Yahoo first, then Stooq daily fallback), 5m, last 30 days
python src/models/random_forest/Randomforest.py ^
  --tickers AAPL MSFT TSLA PLTR UBER FUBO RIOT ^
  --interval 5m ^
  --days 30 ^
  --future-shift 5 ^
  --profit-threshold 1.0 ^
  --model-out data\rf_trade_model.pkl ^
  --out-logs results\logs ^
  --source auto

# Stooq (daily-only), 1d, last 365 days (stable everywhere)
python src/models/random_forest/Randomforest.py ^
  --tickers AAPL MSFT TSLA PLTR UBER FUBO RIOT ^
  --interval 1d ^
  --days 365 ^
  --future-shift 3 ^
  --profit-threshold 1.0 ^
  --model-out data\rf_trade_model.pkl ^
  --out-logs results\logs ^
  --source stooq
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------- IO helpers ----------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 Chrome/124 Safari/537.36"})
    s.trust_env = True
    return s

YF_SESSION = make_session()

def is_intraday(interval: str) -> bool:
    i = interval.lower()
    return any(k in i for k in ["m", "h"]) and i != "1d"

def stooq_symbol(t: str) -> str:
    t = t.strip().lower()
    if "." not in t:
        t = f"{t}.us"
    return t

# ---------- Data fetchers ----------
def fetch_yahoo(ticker: str, interval: str, days: int) -> pd.DataFrame:
    # For intraday Yahoo, period must be <= 60d; choose based on interval
    period = f"{min(max(days, 1), 60)}d" if is_intraday(interval) else f"{max(days, 1)}d"
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        session=YF_SESSION,
    )
    return df

def fetch_stooq_daily(ticker: str, days: int) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol(ticker)}&i=d"
    try:
        df = pd.read_csv(url)
    except Exception:
        return pd.DataFrame()
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df.rename(
        columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"},
        inplace=True,
    )
    # Keep last N days
    if days > 0:
        df = df.tail(days)
    return df

def fetch_ohlcv(ticker: str, interval: str, days: int, source: str) -> tuple[pd.DataFrame, str]:
    """
    Returns (df, effective_source). If Yahoo fails or returns empty, uses Stooq daily.
    """
    if source in ("yahoo", "auto"):
        try:
            df = fetch_yahoo(ticker, interval, days)
            if df is not None and not df.empty:
                return df, "yahoo"
        except Exception as e:
            print(f"[WARN] Yahoo error for {ticker}: {e}")
        if source == "yahoo":
            return pd.DataFrame(), "yahoo"

    # Fallback to Stooq daily (no intraday)
    df = fetch_stooq_daily(ticker, days)
    return df, "stooq"

# ---------- Feature engineering ----------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten MultiIndex if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Basic sanity check
    needed = {"Close", "Volume"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    out = df[["Close", "Volume"]].copy()
    # Ensure numeric
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0)
    out.dropna(subset=["Close"], inplace=True)

    # Simple momentum/return
    out["return"] = out["Close"].pct_change().fillna(0)
    return out

def build_labels(df: pd.DataFrame, future_shift: int, profit_threshold: float) -> pd.Series:
    future_price = df["Close"].shift(-future_shift)
    label = ((future_price - df["Close"]) > profit_threshold).astype(int)
    return label.dropna()

# ---------- Trainer ----------
def train_model(
    tickers: list[str],
    interval: str,
    days: int,
    future_shift: int,
    profit_threshold: float,
    source: str,
) -> tuple[RandomForestClassifier, pd.DataFrame]:
    frames = []
    for t in tickers:
        print(f"\nDownloading {t} (interval={interval}, days={days}, src={source})")
        df, eff = fetch_ohlcv(t, interval, days, source=source)

        # If asking intraday but stooq fallback happened, warn
        if eff == "stooq" and is_intraday(interval):
            print(f"[INFO] {t}: Yahoo unavailable → Using Stooq DAILY data. (Effective interval=1d)")

        if df is None or df.empty:
            print(f"[WARN] No data for {t}. Skipping.")
            continue

        # Make tz-naive index
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)

        feats = prepare_features(df)
        if feats.empty:
            print(f"[WARN] Missing OHLC/Volume columns for {t}. Skipping.")
            continue

        # Labels must align with features (drop last n rows due to shift)
        y = build_labels(feats, future_shift=future_shift, profit_threshold=profit_threshold)
        X = feats.loc[y.index]
        X["ticker"] = t  # keep info for analysis
        frames.append(pd.concat([X, y.rename("label")], axis=1))

    if not frames:
        raise RuntimeError("No usable data across all tickers — training aborted.")

    data = pd.concat(frames, axis=0)
    features = data[["Close", "Volume", "return"]]
    labels = data["label"]

    # Basic train/holdout split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, shuffle=True, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n✅ Model trained. Holdout accuracy: {acc:.2%}")
    print("Class balance (holdout):")
    print(pd.Series(y_test).value_counts(normalize=True).rename("share").map(lambda v: f"{v:.1%}"))

    print("\nClassification report (holdout):")
    print(classification_report(y_test, preds, digits=3))

    return model, data

# ---------- CLI ----------
def main():
    warnings.simplefilter("ignore", category=UserWarning)

    ap = argparse.ArgumentParser(description="Random Forest trade model trainer")
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "TSLA", "PLTR", "UBER", "FUBO", "RIOT"])
    ap.add_argument("--interval", default="5m", help="e.g., 1d, 1h, 5m")
    ap.add_argument("--days", type=int, default=30, help="lookback window in days")
    ap.add_argument("--future-shift", type=int, default=5, help="bars ahead to label the target")
    ap.add_argument("--profit-threshold", type=float, default=1.0, help="USD move considered positive")
    ap.add_argument("--model-out", default="data/rf_trade_model.pkl")
    ap.add_argument("--out-logs", default="results/logs")
    ap.add_argument("--source", choices=["yahoo", "stooq", "auto"], default="auto",
                    help="yahoo (intraday+daily), stooq (daily-only), auto (yahoo→stooq fallback)")
    args = ap.parse_args()

    # Ensure folders
    model_out = Path(args.model_out); ensure_dir(model_out.parent)
    out_logs = Path(args.out_logs); ensure_dir(out_logs)

    # Train
    model, df_all = train_model(
        tickers=args.tickers,
        interval=args.interval,
        days=args.days,
        future_shift=args.future_shift,
        profit_threshold=args.profit_threshold,
        source=args.source,
    )

    # Persist
    joblib.dump(model, model_out)
    print(f"\n📦 Saved model → {model_out}")

    # Save lightweight training summary
    summary = (
        df_all.assign(label=df_all["label"].astype(int))
        .groupby("ticker")["label"]
        .agg(samples="count", positives="sum")
    )
    summary["positive_share"] = summary["positives"] / summary["samples"]
    summary_path = out_logs / "rf_training_summary.csv"
    summary.to_csv(summary_path)
    print(f"📝 Wrote training summary → {summary_path}")

if __name__ == "__main__":
    main()
