# run_verify.ps1 — robust verify for PowerShell

# 1) Ensure venv + deps
if (!(Test-Path ".venv")) { python -m venv .venv }
. .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 2) Ensure output folders
New-Item -ItemType Directory -Force -Path results\plots | Out-Null
New-Item -ItemType Directory -Force -Path results\logs  | Out-Null

# 3) Helper: check if a date has intraday data via a small Python script
function Has-IntradayData {
    param([string]$DateStr)

    $pyCode = @'
import sys, yfinance as yf, pandas as pd
date = sys.argv[1]
t = "AAPL"; interval = "5m"
df = yf.download(t, period="7d", interval=interval, auto_adjust=False, progress=False)
if df.empty:
    print("0"); sys.exit(0)
idx = df.index.tz_convert(None) if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None else df.index
df = df.loc[idx.strftime("%Y-%m-%d") == date]
print("1" if not df.empty else "0")
'@

    $tmp = Join-Path $env:TEMP ("yfcheck_{0}.py" -f ([guid]::NewGuid().ToString()))
    Set-Content -Path $tmp -Value $pyCode -Encoding UTF8
    try {
        $out = python $tmp $DateStr 2>$null
        return [int]($out.Trim())
    } catch {
        return 0
    } finally {
        Remove-Item $tmp -ErrorAction SilentlyContinue
    }
}

# 4) Pick a recent weekday that actually has 5m data
$chosen = $null
for ($i = 1; $i -le 10; $i++) {
    $cand = (Get-Date).AddDays(-$i)
    if ($cand.DayOfWeek -eq "Saturday" -or $cand.DayOfWeek -eq "Sunday") { continue }
    $ds = $cand.ToString("yyyy-MM-dd")
    if (Has-IntradayData $ds) { $chosen = $ds; break }
}

if ($null -ne $chosen) {
    Write-Host "Using intraday backtest date: $chosen"
    python src\simulation\live_trading_simulator.py `
        --mode backtest `
        --tickers AAPL MSFT `
        --interval 5m `
        --lookback 50 `
        --min-profit 0.5 `
        --date $chosen `
        --out-plots results\plots `
        --out-logs results\logs
} else {
    # Fallback to daily bars if no intraday day found
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



