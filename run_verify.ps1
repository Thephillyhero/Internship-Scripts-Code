# run_verify.ps1 — robust date picker for intraday
if (!(Test-Path ".venv")) { python -m venv .venv }
. .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

New-Item -ItemType Directory -Force -Path results\plots | Out-Null
New-Item -ItemType Directory -Force -Path results\logs  | Out-Null

function Has-IntradayData($dateStr) {
  try {
    $out = python - <<PY
import yfinance as yf, pandas as pd, sys
t="AAPL"; interval="5m"; date="$dateStr"
df = yf.download(t, period="7d", interval=interval, auto_adjust=False, progress=False)
# filter to the date
if not df.empty:
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        idx = df.index.tz_convert(None)
    else:
        idx = df.index
    df = df.loc[idx.strftime("%Y-%m-%d") == date]
print(1 if not df.empty else 0)
PY
    return [int]$out.Trim()
  } catch { return 0 }
}

# try last 10 weekdays for intraday
$chosen = $null
for ($i=1; $i -le 10; $i++) {
  $cand = (Get-Date).AddDays(-$i)
  if ($cand.DayOfWeek -eq "Saturday" -or $cand.DayOfWeek -eq "Sunday") { continue }
  $ds = $cand.ToString("yyyy-MM-dd")
  if (Has-IntradayData $ds) { $chosen = $ds; break }
}
if ($null -ne $chosen) {
  Write-Host "Using intraday backtest date: $chosen"
  $cmd = @(
    "python", "src\simulation\live_trading_simulator.py",
    "--mode","backtest",
    "--tickers","AAPL","MSFT",
    "--interval","5m",
    "--lookback","50",
    "--min-profit","0.5",
    "--date",$chosen,
    "--out-plots","results\plots",
    "--out-logs","results\logs"
  )
  & $cmd
} else {
  # fallback: daily bars
  $ds = (Get-Date).AddDays(-5).ToString("yyyy-MM-dd")
  Write-Host "No safe intraday day found; falling back to daily on $ds"
  python src\simulation\live_trading_simulator.py `
    --mode backtest `
    --tickers AAPL MSFT `
    --interval 1d `
    --lookback 50 `
    --min-profit 0.5 `
    --date $ds `
    --out-plots results\plots `
    --out-logs results\logs
}

Write-Host "✅ Verify complete. Check results\plots and results\logs."

Write-Host "✅ Verify complete. Check the results\plots and results\logs folders."



