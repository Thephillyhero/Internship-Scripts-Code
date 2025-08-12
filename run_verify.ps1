# Quick verify script for Windows PowerShell (current CLI flags)

# 1) Create/activate venv
if (!(Test-Path ".venv")) {
  python -m venv .venv
}
. .\.venv\Scripts\Activate.ps1

# 2) Deps
pip install --upgrade pip
pip install -r requirements.txt

# 3) Make output folders
New-Item -ItemType Directory -Force -Path results | Out-Null
New-Item -ItemType Directory -Force -Path results\plots | Out-Null
New-Item -ItemType Directory -Force -Path results\logs  | Out-Null

# 4) Pick a recent weekday (avoid weekends)
function Get-LastWeekday {
  $d = Get-Date
  for ($i = 1; $i -le 7; $i++) {
    $cand = $d.AddDays(-$i)
    if ($cand.DayOfWeek -ne "Saturday" -and $cand.DayOfWeek -ne "Sunday") {
      return $cand.ToString("yyyy-MM-dd")
    }
  }
}
$DATE = Get-LastWeekday
Write-Host "Using backtest date: $DATE"

# 5) Backtest with CURRENT flags
python src\simulation\live_trading_simulator.py `
  --mode backtest `
  --tickers AAPL MSFT `
  --interval 5m `
  --lookback 50 `
  --min-profit 0.5 `
  --date $DATE `
  --out-plots results\plots `
  --out-logs results\logs

Write-Host "✅ Verify complete. Check results\plots and results\logs."


Write-Host "✅ Verify complete. Check the results\plots and results\logs folders."



