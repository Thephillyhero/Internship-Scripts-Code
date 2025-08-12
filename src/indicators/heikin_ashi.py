# src/indicators/heikin_ashi.py
from __future__ import annotations
import argparse, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- utils ----------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def make_session() -> requests.Session:
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

def fetch_yahoo(ticker: str, interval: str, lookback: int, date: str|None) -> pd.DataFrame:
    if is_intraday(interval):
        if not date:
            raise ValueError("--date is required for intraday runs (e.g. 2025-07-31)")
        df = yf.download(ticker, period="7d", interval=interval, auto_adjust=False,
                         progress=False, session=YF_SESSION)
        if df.empty: return df
        idx = df.index.tz_convert(None) if getattr(df.index,"tz",None) is not None else df.index
        df = df.loc[idx.strftime("%Y-%m-%d") == date]
    else:
        df = yf.download(ticker, period="1y", interval=interval, auto_adjust=False,
                         progress=False, session=YF_SESSION)
        if df.empty: return df
        if date:
            idx = df.index.tz_convert(None) if getattr(df.index,"tz",None) is not None else df.index
            df = df.loc[idx <= pd.to_datetime(date)]
    return df.tail(max(lookback, 2))

def fetch_stooq_daily(ticker: str, date: str|None, lookback: int) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol(ticker)}&i=d"
    try:
        df = pd.read_csv(url)
    except Exception:
        return pd.DataFrame()
    if df.empty or "Date" not in df: return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}, inplace=True)
    if date:
        df = df.loc[df.index <= pd.to_datetime(date)]
    return df.tail(max(lookback, 2))

def fetch_data(ticker: str, interval: str, lookback: int, date: str|None, source: str) -> pd.DataFrame:
    if source == "stooq" and is_intraday(interval):
        warnings.warn("Stooq is daily-only; downgrading interval to 1d.")
        interval = "1d"
    if source in ("yahoo","auto"):
        try:
            df = fetch_yahoo(ticker, interval, lookback, date)
            if not df.empty: return df
        except Exception as e:
            print(f"[WARN] Yahoo error for {ticker}: {e}")
        if source == "yahoo":
            return pd.DataFrame()
    return fetch_stooq_daily(ticker, date, lookback)

# ---------- Heikin-Ashi core ----------
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    o,h,l,c = df["Open"], df["High"], df["Low"], df["Close"]
    ha_close = (o+h+l+c)/4
    ha_open = np.zeros(len(df))
    ha_open[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
    ha_high = pd.concat([h, pd.Series(ha_open, index=df.index), ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([l, pd.Series(ha_open, index=df.index), ha_close], axis=1).min(axis=1)

    out = df.copy()
    out["HA_Open"]  = ha_open
    out["HA_High"]  = ha_high
    out["HA_Low"]   = ha_low
    out["HA_Close"] = ha_close
    out["HA_Bull"]  = (out["HA_Close"] > out["HA_Open"]).astype(int)
    out["HA_Shift"] = out["HA_Bull"].diff().fillna(0)
    return out

def plot_ha(df: pd.DataFrame, ticker: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(10,4.5))
    ax.plot(df.index, df["Close"], label="Close", alpha=0.6)
    ax.plot(df.index, df["HA_Close"], label="HA Close", linewidth=2)
    ax.fill_between(df.index, df["HA_Close"], df["HA_Open"],
                    where=(df["HA_Close"]>df["HA_Open"]), alpha=0.2, label="HA Bull")
    ax.set_title(f"Heikin-Ashi — {ticker}")
    ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)

def main():
    warnings.simplefilter("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(description="Heikin-Ashi scanner (Yahoo + Stooq fallback)")
    ap.add_argument("--tickers", nargs="+", default=["AAPL","MSFT"])
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (required for intraday)")
    ap.add_argument("--source", choices=["yahoo","stooq","auto"], default="auto")
    ap.add_argument("--out-plots", default="results/plots")
    ap.add_argument("--out-logs", default="results/logs")
    args = ap.parse_args()

    out_plots = Path(args.out_plots); ensure_dir(out_plots)
    out_logs  = Path(args.out_logs);  ensure_dir(out_logs)

    summary_rows = []
    for t in args.tickers:
        print(f"\n[HA] Scanning {t}…")
        df = fetch_data(t, args.interval, args.lookback, args.date, args.source)
        if df.empty or not {"Open","High","Low","Close"}.issubset(df.columns):
            print(f"[WARN] No OHLC data for {t}")
            continue
        ha = heikin_ashi(df)
        png = out_plots / f"ha_{t}_{args.interval}.png"
        csv = out_logs  / f"ha_{t}_{args.interval}.csv"
        plot_ha(ha, t, png)
        ha.to_csv(csv)

        last = ha.iloc[-1]
        state = "BULL" if last["HA_Bull"]==1 else "BEAR"
        flips = int((ha["HA_Shift"]!=0).sum())
        summary_rows.append({"ticker": t, "bars": len(ha), "state": state,
                             "last_close": float(last["Close"]), "flips": flips})
        print(f"   -> {state}, flips={flips} | plot: {png} | csv: {csv}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_logs / "ha_summary.csv", index=False)
        print(f"\nSaved summary -> {out_logs / 'ha_summary.csv'}")
    else:
        print("\nNo summaries created.")

if __name__ == "__main__":
    main()
