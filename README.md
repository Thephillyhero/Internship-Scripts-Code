# Internship Scripts & Code

This repository contains all the code and scripts developed during my internship, organized for clarity and ease of use. It includes feature engineering tools, model training scripts, simulation environments, and supporting utilities.

---

## 📂 Project Structure

```
.
├── README.md                     # This file
├── requirements.txt               # Python dependencies
├── run_verify.ps1                 # PowerShell verification script
├── run_verify.sh                  # Bash verification script (for macOS/Linux)
│
├── src/
│   ├── features/
│   │   ├── engineering.py         # Feature engineering utilities
│   │   ├── split_data.py          # Splits stock data into training, validation, and test sets
│   │   └── __init__.py
│   │
│   ├── models/
│   │   ├── random_forest/
│   │   │   ├── Randomforest.py    # Random Forest model training script
│   │   ├── LSM_linearization_script.py  # Linearizes stock data using Least Squares Method
│   │   └── __init__.py
│   │
│   └── simulation/
│       ├── live_trading_sim.py    # Live trading simulator script
│       └── __init__.py
│
└── archive/                       # Unused or older scripts stored here for reference
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

   git clone https://github.com/Thephillyhero/Internship-Scripts-Code.git
   cd Internship-Scripts-Code
   ```

2. **Install dependencies**

   pip install -r requirements.txt
   ```

3. **Verify environment** (optional, ensures all dependencies are installed and paths are correct)

   * On Windows:

     ```powershell
     .\run_verify.ps1
     ```
   * On macOS/Linux:

     ```bash
     bash run_verify.sh
     ```

---

## 🚀 How to Run the Code

### Feature Engineering

#### split\_data.py

Splits historical stock market data into **training**, **validation**, and **test** datasets for modeling.

```bash
python src/features/split_data.py
```

#### engineering.py

Contains helper functions for generating additional features from raw stock data.

```bash
python src/features/engineering.py
```

---

### Modeling

#### Randomforest.py

Trains a Random Forest model for predicting stock market trends based on engineered features.

```bash
python src/models/random_forest/Randomforest.py
```

#### LSM\_linearization\_script.py

Uses the **Least Squares Method (LSM)** to linearize historical stock price trends for analysis or as model input.

```bash
python src/models/LSM_linearization_script.py
```

---

### Simulation

#### live\_trading\_sim.py

Runs a live trading simulation using your trained model and real-time market data.

```bash
python src/simulation/live_trading_sim.py

## 📌 Notes

* The **archive/** folder contains older or unused scripts for reference.
* Ensure you have a stable internet connection for scripts that fetch live or historical market data.
* All scripts are designed to be **machine-independent** — you can run them on any system with Python and the required dependencies installed.






