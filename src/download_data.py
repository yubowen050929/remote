"""Download all market data needed for the drawdown-factor analysis.

Saves one CSV per ticker under data/, with Date index and OHLCV columns.
"""
import os

import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TICKERS = {
    "^GSPC": "spx",        # S&P 500 index
    "^NDX": "ndx",         # Nasdaq 100 index
    "^VIX": "vix",         # 30d implied vol
    "^VIX3M": "vix3m",     # 93d implied vol (term structure)
    "^VVIX": "vvix",       # vol of vol
    "^MOVE": "move",       # treasury implied vol
    "HYG": "hyg",          # high yield credit ETF
    "LQD": "lqd",          # investment grade credit ETF
    "TLT": "tlt",          # 20y+ treasuries
    "^TNX": "tnx",         # 10y yield
    "RSP": "rsp",          # equal-weight S&P (breadth proxy)
    "^SKEW": "skew",       # CBOE SKEW index (tail-risk pricing)
    "^VXN": "vxn",         # Nasdaq-100 implied vol
    "^VIX9D": "vix9d",     # 9d implied vol
    "^SOX": "sox",         # semiconductor index
    "JPY=X": "usdjpy",     # yen
    "GLD": "gld",          # gold ETF
    "QQQE": "qqqe",        # equal-weight Nasdaq 100
}

START = "2010-06-01"  # extra runway so 2011 features have lookback


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for ticker, name in TICKERS.items():
        df = yf.download(ticker, start=START, progress=False, auto_adjust=True)
        if df.empty:
            print(f"WARN: no data for {ticker}")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        path = os.path.join(DATA_DIR, f"{name}.csv")
        df.to_csv(path)
        print(f"{ticker:8s} -> {name}.csv  rows={len(df)}  {df.index.min().date()} .. {df.index.max().date()}")


if __name__ == "__main__":
    main()
