# Quick verify script for Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python src/simulation/live_trading_simulator.py --mode backtest --start 2024-07-01 --end 2024-07-05 --tickers AAPL,MSFT --save



