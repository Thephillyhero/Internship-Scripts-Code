# src/data/fetch_data.py
from __future__ import annotations
import argparse, warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests
import yfinance as yf

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 Chrome/124 Safari/537.36"})
    s.trust_env = True
    return s

YF_SESSION = make_session()

def stooq_symbol(t: str) -> str:
    t = t.strip().lower()
    if "." not in t: t = f"{t}.us"
    return t

def fetch_yahoo(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    return yf.download(
        ticker, start=start, end=end, interval=interval,
        auto_adjust=False, progress=False, session=YF_SESSION
    )

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
    # filter date range if provided
    if start: df = df.loc[df.index >= pd.to_datetime(start)]
    if end:   df = df.loc[df.index <= pd.to_datetime(end)]
    return df

def fetch_ohlcv(ticker: str, start: str, end: str, interval: str, source: str) -> tuple[pd.DataFrame, str]:
    if source in ("yahoo", "auto"):
        try:
            df = fetch_yahoo(ticker, start, end, interval)
            if not df.empty:
                return df, "yahoo"
        except Exception as e:
            print(f"[WARN] Yahoo error for {ticker}: {e}")
        if source == "yahoo":
            return pd.DataFrame(), "yahoo"
    # fallback to Stooq (daily only)
    df = fetch_stooq_daily(ticker, start, end)
    return df, "stooq"

def parse_list(s: str) -> list[str]:
    return [t.strip().upper() for t in s.split(",") if t.strip()]

def main():
    warnings.simplefilter("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(description="Download OHLCV CSVs by cap group with Yahoo→Stooq fallback")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--interval", default="1d", help="Yahoo interval (Stooq is daily-only)")
    ap.add_argument("--source", choices=["yahoo","stooq","auto"], default="auto")
    ap.add_argument("--tickers-high", default="AAPL,MSFT")
    ap.add_argument("--tickers-mid", default="ZS,OKTA")
    ap.add_argument("--tickers-low", default="DUOL,IONQ")
    ap.add_argument("--out-dir", default="data/raw")
    args = ap.parse_args()

    end = args.end or datetime.today().strftime("%Y-%m-%d")
    out_root = Path(args.out_dir)
    (out_root / "high").mkdir(parents=True, exist_ok=True)
    (out_root / "mid").mkdir(parents=True, exist_ok=True)
    (out_root / "low").mkdir(parents=True, exist_ok=True)

    groups = {
        "high": parse_list(args.tickers_high),
        "mid" : parse_list(args.tickers_mid),
        "low" : parse_list(args.tickers_low),
    }

    for cap, tickers in groups.items():
        for t in tickers:
            print(f"\n[{cap.upper()}] Fetching {t} ({args.source}) …")
            df, eff = fetch_ohlcv(t, args.start, end, args.interval, args.source)
            if df.empty:
                print(f"[WARN] No data for {t}. Skipped.")
                continue
            # ensure columns are as expected
            if "Close" not in df.columns:
                print(f"[WARN] Missing Close for {t}. Skipped.")
                continue
            out_file = out_root / cap / f"{cap}_{t}.csv"
            df.to_csv(out_file)
            print(f"Saved → {out_file}   (rows={len(df)}, source={eff})")

if __name__ == "__main__":
    main()
