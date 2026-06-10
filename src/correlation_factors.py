"""Verification of two externally-claimed correlation signals (June 2026):

1. SPY-TLT correlation "at 100th percentile" -> fact-check the level and
   test whether stock-bond correlation predicts next-day large drops.
2. Average pairwise correlation as dispersion/herding gauge -> computed on
   the 11 SPDR sector ETFs (20d window) and tested the same way.

Run after download_data.py. Downloads sector ETFs itself.
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")

SECTORS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]


def quintile_table(x: pd.Series, target: pd.Series, label: str):
    ok = x.notna() & target.notna() & (x.index >= "2011-06-09")
    q = pd.qcut(x[ok], 5, labels=False, duplicates="drop")
    tab = target[ok].astype(float).groupby(q).mean()
    print(f"P(next-day SPX drop >1.3%) by {label} quintile (low->high):")
    print("  " + "  ".join(f"Q{int(k)+1}={v:.1%}" for k, v in tab.items()))


def main():
    spx = pd.read_csv(os.path.join(DATA, "spx.csv"), index_col=0, parse_dates=True)["Close"]
    tlt = pd.read_csv(os.path.join(DATA, "tlt.csv"), index_col=0, parse_dates=True)["Close"]
    r_spx, r_tlt = spx.pct_change(), tlt.pct_change()
    target = r_spx.shift(-1) < -0.013

    for win in [20, 60]:
        corr = r_spx.rolling(win).corr(r_tlt).dropna()
        print(f"SPX-TLT {win}d corr now={corr.iloc[-1]:+.2f}, "
              f"percentile since 2010={(corr < corr.iloc[-1]).mean():.1%}")
    quintile_table(r_spx.rolling(20).corr(r_tlt), target, "SPX-TLT 20d correlation")

    px = yf.download(SECTORS, start="2010-06-01", progress=False, auto_adjust=True)["Close"]
    rets = px.pct_change()
    out = {}
    for i in range(20, len(rets)):
        sub = rets.iloc[i - 20:i].dropna(axis=1)
        if sub.shape[1] < 8:
            continue
        c = sub.corr().values
        out[rets.index[i]] = c[np.triu_indices_from(c, 1)].mean()
    apc = pd.Series(out)
    apc.to_csv(os.path.join(DATA, "sector_pairwise_corr.csv"))
    print(f"\navg pairwise sector corr (20d) now={apc.iloc[-1]:.2f}, "
          f"percentile={(apc < apc.iloc[-1]).mean():.1%}, median={apc.median():.2f}")
    quintile_table(apc, target.reindex(apc.index), "avg pairwise sector correlation")


if __name__ == "__main__":
    main()
