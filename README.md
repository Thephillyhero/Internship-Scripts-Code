# Internship-Scripts-Code
All the Code and Scripts that I used to complete this project
.
src/
├── simulation/
│ ├── live_trading_simulator.py # Main live trading & backtest script
│ ├── rf_trainer.py # Trains Random Forest model
│ └── utils/ # Utility functions for data handling & plots
│
├── indicators/
│ ├── RSI.py # Relative Strength Index calculation
│ ├── LSM.py # Least Squares Method (trendline) calculation
│ └── RV.py # Rolling Volatility calculation
│
├── archive/ # Old or unused scripts kept for reference
│ ├── historical_backtester.py
│ ├── intraday_profit_tracker2.py
│ └── alpaca_trade.py
│
├── data/ # Saved datasets and models
│ ├── rf_trade_model.pkl # Random Forest model (optional)
│ └── live_trade_results.csv # Trade logs
│
├── plots/ # Saved trade plot images
├── README.md # Project documentation
└── requirements.txt # Python dependencies

1️⃣ Install Dependencies
How to Run the Code - Make sure you have Python 3.10+ installed. Then, in the project root:

  pip install -r requirements.txt

2️⃣ Running Live Trading Mode (Market Hours Only)

python src/simulation/live_trading_simulator.py --mode live

3️⃣ Running Backtest Mode (Any Date)

python src/simulation/live_trading_simulator.py --mode backtest --date YYYY-MM-DD

4️⃣ Training the Random Forest Model (Optional)

python src/simulation/rf_trainer.py

⚙️ Configuration

INTERVAL = "1m"            # Data interval
LOOKBACK_MINUTES = 15      # Window for finding trades
MIN_PROFIT_DOLLARS = 0.25  # Minimum profit to trigger a trade
MODEL_PATH = "data/rf_trade_model.pkl"

📊 Output Files

  Trade Logs: data/live_trade_results.csv

Trade Plots: Saved to /plots folder with buy/sell markers

🗂 Archived Scripts
The /archive folder contains older or unused scripts. These are not required for running the main tools but are kept for reference.













