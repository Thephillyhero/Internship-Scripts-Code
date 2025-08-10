#!/usr/bin/env python3
# src/models/random_forest/Randomforest.py
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from src.features.engineering import add_basic_features

DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA", "PLTR", "UBER", "FUBO", "RIOT"]

def download_ticker(ticker: str, start, end, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
    # Flatten multiindex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def main():
    ap = argparse.ArgumentParser(description="Train RandomForest trade classifier.")
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS, help="List of tickers")
    ap.add_argument("--interval", default="5m", help="Bar interval (e.g., 5m, 1m, 15m)")
    ap.add_argument("--days", type=int, default=30, help="Days lookback")
    ap.add_argument("--future-shift", type=int, default=5, help="Bars ahead for label")
    ap.add_argument("--profit-threshold", type=float, default=1.0, help="Min $ gain to label=1")
    ap.add_argument("--model-path", default="src/models/random_forest/rf_trade_model.pkl", help="Output model path")
    args = ap.parse_args()

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    all_rows = []
    for t in args.tickers:
        df = download_ticker(t, start=start_date, end=end_date, interval=args.interval)
        if df.empty or len(df) <= args.future_shift:
            continue
        if not {'Close','Volume'}.issubset(df.columns):
            continue

        df = df[['Open','High','Low','Close','Volume']].copy()
        # Label: future price vs current price
        df['future_price'] = df['Close'].shift(-args.future_shift)
        df.dropna(inplace=True)
        if df.empty:
            continue

        df['label'] = ((df['future_price'] - df['Close']) > args.profit_threshold).astype(int)
        df = add_basic_features(df)  # adds 'return'
        use = df[['Close','Volume','return','label']].dropna()
        use['ticker'] = t
        all_rows.append(use)

    if not all_rows:
        raise SystemExit("No usable data. Try different tickers, more --days, or larger --future-shift.")

    data = pd.concat(all_rows, axis=0)
    X = data[['Close','Volume','return']]
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Trained RF on {len(data)} rows across {len(set(data['ticker']))} tickers. Accuracy: {acc:.2%}")

    out_path = Path(args.model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"📦 Saved model to {out_path}")

if __name__ == "__main__":
    main()

