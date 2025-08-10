# src/features/engineering.py
import pandas as pd

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and add 'return' column."""
    need = {'Close', 'Volume'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df.copy()
    out['return'] = out['Close'].pct_change().fillna(0)
    return out

def latest_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 1-row DataFrame with ['Close','Volume','return'] from the latest bar."""
    df_feat = add_basic_features(df)
    last = df_feat.iloc[-1]
    return pd.DataFrame(
        {'Close': [last['Close']], 'Volume': [last['Volume']], 'return': [last['return']]}
    )
