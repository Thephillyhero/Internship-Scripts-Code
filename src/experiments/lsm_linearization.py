# src/experiments/lsm_linearization.py
from __future__ import annotations
import argparse, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Try TensorFlow; fail friendly if absent
try:
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except Exception as e:
    raise SystemExit(
        "TensorFlow + scikit-learn are required for this script.\n"
        "Install with: pip install tensorflow scikit-learn"
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent":"Mozilla/5.0 Chrome/124 Safari/537.36"})
    s.trust_env = True
    return s

YF_SESSION = make_session()

def stooq_symbol(t: str) -> str:
    t = t.strip().lower()
    if "." not in t: t = f"{t}.us"
    return t

def fetch_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    return yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False, session=YF_SESSION)

def fetch_stooq_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol(ticker)}&i=d"
    try:
        df = pd.read_csv(url)
    except Exception:
        return pd.DataFrame()
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}, inplace=True)
    if start: df = df.loc[df.index >= pd.to_datetime(start)]
    if end:   df = df.loc[df.index <= pd.to_datetime(end)]
    return df

def fetch_close_series(ticker: str, start: str, end: str, source: str) -> pd.DataFrame:
    if source in ("yahoo","auto"):
        try:
            df = fetch_yahoo(ticker, start, end)
            if not df.empty:
                return df[["Close"]].dropna()
        except Exception as e:
            print(f"[WARN] Yahoo error for {ticker}: {e}")
        if source == "yahoo":
            return pd.DataFrame()
    df = fetch_stooq_daily(ticker, start, end)
    return df[["Close"]].dropna() if not df.empty else pd.DataFrame()

def least_squares_smooth(close: pd.Series, window: int = 14) -> pd.Series:
    vals = close.values.astype(float)
    smoothed = np.full(len(vals), np.nan, dtype=float)
    for i in range(window, len(vals)):
        x = np.arange(window, dtype=float)
        y = vals[i-window:i]
        coeffs = np.polyfit(x, y, deg=1)
        trend = np.poly1d(coeffs)
        smoothed[i] = trend(window - 1)
    return pd.Series(smoothed, index=close.index)

def make_lstm_sets(arr_2d: np.ndarray, window: int = 60):
    X, y = [], []
    for i in range(window, len(arr_2d)):
        X.append(arr_2d[i-window:i, 0])
        y.append(arr_2d[i, 0])
    X = np.array(X); y = np.array(y)
    if X.size == 0: return None, None
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.1),
        LSTM(64),
        Dropout(0.1),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def dir_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    a = actual.flatten(); p = pred.flatten()
    if len(a) < 2 or len(p) < 2: return float("nan")
    ad = np.sign(a[1:] - a[:-1]); pdiff = np.sign(p[1:] - p[:-1])
    return float((ad == pdiff).mean() * 100.0)

def main():
    warnings.simplefilter("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(description="LSM linearization + LSTM forecast (daily data; Yahoo with Stooq fallback)")
    ap.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT TSLA")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--lsm-window", type=int, default=14)
    ap.add_argument("--lstm-window", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--source", choices=["yahoo","stooq","auto"], default="auto")
    ap.add_argument("--out-dir", default="results/lsm")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    preds_dir = out_dir / "predictions"
    graphs_dir = out_dir / "graphs"
    ensure_dir(preds_dir); ensure_dir(graphs_dir)

    for t in args.tickers:
        print(f"\n[LSM] Processing {t} ({args.source}) …")
        df = fetch_close_series(t, args.start, args.end, args.source)
        if df.empty:
            print(f"[WARN] No data for {t}. Skipping.")
            continue

        ls = least_squares_smooth(df["Close"], window=args.lsm_window).dropna()
        if ls.empty:
            print(f"[WARN] LSM produced no data for {t}. Skipping.")
            continue

        scaler = StandardScaler()
        scaled = scaler.fit_transform(ls.values.reshape(-1,1))
        n = len(scaled)
        tr = int(0.7*n); va = int(0.9*n)

        Xtr, ytr = make_lstm_sets(scaled[:tr], window=args.lstm_window)
        Xva, yva = make_lstm_sets(scaled[tr:va], window=args.lstm_window)
        Xte, yte = make_lstm_sets(scaled[va:], window=args.lstm_window)
        if any(v is None for v in (Xtr, ytr, Xva, yva, Xte, yte)):
            print(f"[WARN] Not enough data windows for {t}. Skipping.")
            continue

        model = build_model((Xtr.shape[1], 1))
        model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs, batch_size=args.batch, verbose=0)

        ptr = model.predict(Xtr); pva = model.predict(Xva); pte = model.predict(Xte)
        ptr = scaler.inverse_transform(ptr); pva = scaler.inverse_transform(pva); pte = scaler.inverse_transform(pte)
        atr = scaler.inverse_transform(ytr.reshape(-1,1)); ava = scaler.inverse_transform(yva.reshape(-1,1)); ate = scaler.inverse_transform(yte.reshape(-1,1))

        acc_tr = dir_accuracy(atr, ptr); acc_va = dir_accuracy(ava, pva); acc_te = dir_accuracy(ate, pte)
        print(f"   Accuracies — Train: {acc_tr:.2f}% | Val: {acc_va:.2f}% | Test: {acc_te:.2f}%")

        # Save CSVs
        pd.DataFrame({"Actual": atr.flatten(), "Predicted": ptr.flatten()}).to_csv(preds_dir / f"{t}_train_LSM_predictions.csv", index=False)
        pd.DataFrame({"Actual": ava.flatten(), "Predicted": pva.flatten()}).to_csv(preds_dir / f"{t}_val_LSM_predictions.csv", index=False)
        pd.DataFrame({"Actual": ate.flatten(), "Predicted": pte.flatten()}).to_csv(preds_dir / f"{t}_test_LSM_predictions.csv", index=False)

        # Save plots
        for name, a, p in [("train", atr, ptr), ("val", ava, pva), ("test", ate, pte)]:
            plt.figure(figsize=(10,5))
            plt.plot(a, label="Actual"); plt.plot(p, label="Predicted")
            plt.title(f"{t} — LSM+LSTM ({name})"); plt.xlabel("Time"); plt.ylabel("LSM")
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(graphs_dir / f"{t}_{name}_LSM_graph.png"); plt.close()

if __name__ == "__main__":
    main()
