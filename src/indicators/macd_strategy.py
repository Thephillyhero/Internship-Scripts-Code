# src/indicators/macd_strategy.py
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf

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

def is_intraday(interval: str) -> bool:
    i = interval.lower()
    return any(k in i for k in ["m","h"]) and i != "1d"

def stooq_symbol(t: str) -> str:
    t = t.strip().lower()
    if "." not in t: t = f"{t}.us"
    return t

def fetch_yahoo(ticker: str, interval: str, lookback: int) -> pd.DataFrame:
    period = "1y" if interval == "1d" else "30d"
    return yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                       progress=False, session=YF_SESSION)

def fetch_stooq_daily(ticker: str, lookback: int) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol(ticker)}&i=d"
    try:
        df = pd.read_csv(url)
    except Exception:
        return pd.DataFrame()
    if df.empty or "Date" not in df: return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}, inplace=True)
    return df.tail(max(lookback, 50))

def fetch_data(ticker: str, interval: str, lookback: int, source: str) -> pd.DataFrame:
    if source == "stooq" and is_intraday(interval):
        warnings.warn("Stooq is daily-only; downgrading interval to 1d.")
        interval = "1d"
    if source in ("yahoo","auto"):
        try:
            df = fetch_yahoo(ticker, interval, lookback)
            if not df.empty: return df.tail(max(lookback, 50))
        except Exception as e:
            print(f"[WARN] Yahoo error for {t}: {e}")
        if source == "yahoo":
            return pd.DataFrame()
    return fetch_stooq_daily(ticker, lookback)

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def macd_calc(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    out = df.copy()
    out["EMA_fast"] = ema(out["Close"], fast)
    out["EMA_slow"] = ema(out["Close"], slow)
    out["MACD"] = out["EMA_fast"] - out["EMA_slow"]
    out["Signal"] = ema(out["MACD"], signal)
    out["Hist"] = out["MACD"] - out["Signal"]
    out["CrossUp"]   = (out["MACD"].shift(1) < out["Signal"].shift(1)) & (out["MACD"] >= out["Signal"])
    out["CrossDown"] = (out["MACD"].shift(1) > out["Signal"].shift(1)) & (out["MACD"] <= out["Signal"])
    return out

def backtest(df: pd.DataFrame, fee_bp: float = 0.0):
    bt = macd_calc(df)
    cash, pos = 1_000.0, 0.0
    last_price = None
    trades = []

    for ts, row in bt.iterrows():
        price = row["Close"]
        if row["CrossUp"] and cash > 0:
            qty = cash / price
            fee = cash * (fee_bp/10000.0)
            cash = 0.0 - fee
            pos = qty
            trades.append({"time": ts, "side":"BUY", "price": float(price), "qty": float(qty), "cash": float(cash)})
        elif row["CrossDown"] and pos > 0:
            proceeds = pos * price
            fee = proceeds * (fee_bp/10000.0)
            cash = proceeds - fee
            trades.append({"time": ts, "side":"SELL", "price": float(price), "qty": float(pos), "cash": float(cash)})
            pos = 0.0
        last_price = price

    equity = cash + (pos * last_price if last_price and pos > 0 else 0.0)
    return bt, trades, equity

def plot_macd(bt: pd.DataFrame, ticker: str, out_png: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    ax1.plot(bt.index, bt["Close"], label="Close")
    ax1.set_title(f"{ticker} — Price"); ax1.grid(True); ax1.legend()

    ax2.plot(bt.index, bt["MACD"], label="MACD")
    ax2.plot(bt.index, bt["Signal"], label="Signal")
    ax2.bar(bt.index, bt["Hist"], alpha=0.3, label="Hist")
    for ts in bt.loc[bt["CrossUp"]].index:
        ax1.axvline(ts, color="green", linestyle="--", alpha=0.3)
    for ts in bt.loc[bt["CrossDown"]].index:
        ax1.axvline(ts, color="red", linestyle="--", alpha=0.3)
    ax2.legend(); ax2.grid(True)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)

def main():
    warnings.simplefilter("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(description="MACD crossover backtest (Yahoo + Stooq fallback)")
    ap.add_argument("--tickers", nargs="+", default=["AAPL","MSFT"])
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--lookback", type=int, default=200)
    ap.add_argument("--source", choices=["yahoo","stooq","auto"], default="auto")
    ap.add_argument("--fees-bp", type=float, default=0.0)
    ap.add_argument("--out-plots", default="results/plots")
    ap.add_argument("--out-logs", default="results/logs")
    args = ap.parse_args()

    out_plots = Path(args.out_plots); ensure_dir(out_plots)
    out_logs  = Path(args.out_logs);  ensure_dir(out_logs)

    rows = []
    for t in args.tickers:
        print(f"\n[MACD] Backtesting {t}…")
        df = fetch_data(t, args.interval, args.lookback, args.source)
        if df.empty or "Close" not in df.columns:
            print(f"[WARN] No data for {t}")
            continue

        bt, trades, equity = backtest(df)
        png = out_plots / f"macd_{t}_{args.interval}.png"
        plot_macd(bt, t, png)
        pd.DataFrame(trades).to_csv(out_logs / f"macd_trades_{t}_{args.interval}.csv", index=False)

        start, last = float(df["Close"].iloc[0]), float(df["Close"].iloc[-1])
        bh = (last/start - 1.0) * 100.0
        rows.append({"ticker": t, "bars": len(df), "final_equity": equity, "buy_hold_%": bh})
        print(f"   -> equity: ${equity:,.2f} | buy&hold: {bh:.2f}%")
        print(f"   -> plot: {png}")

    if rows:
        pd.DataFrame(rows).to_csv(out_logs / "macd_summary.csv", index=False)
        print(f"\nSaved summary -> {out_logs / 'macd_summary.csv'}")
    else:
        print("\nNo summaries created.")

if __name__ == "__main__":
    main()
