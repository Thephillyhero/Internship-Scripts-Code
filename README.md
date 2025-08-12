A reproducible toolkit for market signal exploration and a lightweight ML-gated trading simulator. It’s designed to be data-source agnostic (Yahoo Finance for intraday when available, Stooq daily fallback for reliability), with simple, explainable features and CLI-first scripts so anyone can run it on a clean machine.

Note: Educational project. Nothing here is financial advice.

Goals
I’m testing whether a small, explainable ML gate can select trades that are more able, accurate, and trustworthy than luck or simple baselines. Indicators (Heikin-Ashi, MACD) generate candidates; a Random Forest trained on Close, Volume, and simple return filters when to act. Everything logs to CSVs and plots so results are inspectable.

Internship-Scripts-Code/
├─ README.md
├─ requirements.txt
├─ (optional) requirements-tf.txt          # only if you’ll run the LSM experiment
├─ data/
│  └─ rf_trade_model.pkl                   # created after training (not pre-committed)
├─ results/                                # auto-created (plots & logs)
│  ├─ plots/
│  └─ logs/
├─ src/
│  ├─ data/
│  │  └─ fetch_data.py                     # download OHLCV to CSVs (Yahoo→Stooq)
│  ├─ experiments/
│  │  └─ lsm_linearization.py              # optional LSM + LSTM experiment
│  ├─ features/
│  │  └─ engineering.py                    # small helpers (imported by others)
│  ├─ indicators/
│  │  ├─ heikin_ashi.py                    # build HA candles; trend flips
│  │  └─ macd_strategy.py                  # MACD crossover backtest
│  ├─ models/
│  │  └─ random_forest/
│  │     └─ Randomforest.py                # train RF on Close/Volume/return
│  └─ simulation/
│     └─ live_trading_simulator.py         # backtest/live with ML gate + plots
└─ scripts/
   ├─ run_verify.ps1                       # optional convenience script (Windows)
   └─ run_verify.sh                        # optional (macOS/Linux)

   Prereqs & Setup
Python 3.10+ (Windows/macOS/Linux)

Internet access (for data fetching)

Create a virtual environment and install deps.

Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-tf.txt

python src\data\fetch_data.py `
  --start 2018-01-01 `
  --end 2024-12-31 `
  --interval 1d `
  --source auto `
  --tickers-high "AAPL,MSFT" `
  --tickers-mid "ZS,OKTA" `
  --tickers-low "DUOL,IONQ" `
  --out-dir data\raw

  python src\indicators\heikin_ashi.py `
  --tickers AAPL MSFT `
  --interval 1d `
  --lookback 120 `
  --source stooq `
  --out-plots results\plots `
  --out-logs results\logs

  python src\indicators\macd_strategy.py `
  --tickers AAPL MSFT TSLA `
  --interval 1d `
  --lookback 300 `
  --source stooq `
  --out-plots results\plots `
  --out-logs results\logs

  python src\models\random_forest\Randomforest.py `
  --tickers AAPL MSFT TSLA PLTR UBER FUBO RIOT `
  --interval 1d `
  --days 365 `
  --future-shift 3 `
  --profit-threshold 1.0 `
  --model-out data\rf_trade_model.pkl `
  --out-logs results\logs `
  --source stooq

  python src\simulation\live_trading_simulator.py `
  --mode backtest `
  --tickers AAPL MSFT `
  --interval 1d `
  --lookback 200 `
  --min-profit 1.0 `
  --model-path data\rf_trade_model.pkl `
  --source stooq `
  --out-plots results\plots `
  --out-logs results\logs

  python src\experiments\lsm_linearization.py `
  --tickers AAPL MSFT `
  --start 2020-01-01 `
  --end 2024-12-31 `
  --lsm-window 14 `
  --lstm-window 60 `
  --epochs 20 `
  --batch 32 `
  --source auto `
  --out-dir results\lsm

  python src\simulation\live_trading_simulator.py `
  --mode backtest `
  --tickers AAPL `
  --interval 5m `
  --lookback 120 `
  --date 2025-07-31 `
  --min-profit 1.0 `
  --model-path data\rf_trade_model.pkl `
  --source auto `
  --out-plots results\plots `
  --out-logs results\logs








