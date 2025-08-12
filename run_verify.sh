#!/usr/bin/env bash
# Quick verify script for macOS/Linux
set -euo pipefail

# Create and activate venv
python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Optional: train the RF model (safe to skip if you just want a fast run)
python src/models/random_forest/Randomforest.py --days 30 --interval 5m --future-shift 5 --profit-threshold 1.0 || true

# Backtest a known recent weekday; change date if you want
python src/simulation/live_trading_simulator.py --mode backtest --date 2025-08-07 --interval 5m --lookback 100 --min-profit 1.0

echo "✅ Verify complete. Check the results folder for plots/logs."
