"""Monthly OpEx (3rd Friday) calendar test: is the week after option
expiration more dangerous, as market folklore claims?

Result (2011-2026): no. P(big-drop day) in the post-OpEx week is
indistinguishable from any other day. Mild (noise-level) softness
appears in the two days BEFORE OpEx, not after.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")


def main():
    df = pd.read_csv(os.path.join(DATA, "spx.csv"), index_col=0, parse_dates=True)
    c = df["Close"]
    r = c.pct_change()
    ev = r < -0.013
    idx = df.index
    mask = (idx >= "2011-06-09") & r.notna()

    opex = []
    for y in range(2011, 2027):
        for m in range(1, 13):
            d = pd.Timestamp(y, m, 1)
            opex.append(pd.date_range(d, d + pd.offsets.MonthEnd(0), freq="W-FRI")[2])

    rel = pd.Series(np.nan, index=idx)
    for e in pd.DatetimeIndex(opex):
        loc = idx.searchsorted(e)
        for k in range(-4, 6):
            j = loc + k
            if 0 <= j < len(idx):
                rel.iloc[j] = k

    print("day relative to monthly OpEx (0 = 3rd Friday):")
    for k in range(-4, 6):
        m = mask & (rel == k)
        print(f"  {k:+d}: P(big drop)={ev[m].mean():6.1%}  mean ret={r[m].mean()*1e4:+6.1f}bp")
    wk = mask & rel.between(1, 5)
    print(f"\nweek after OpEx: P={ev[wk].mean():.1%} | other days: P={ev[mask & ~rel.between(1, 5)].mean():.1%}")


if __name__ == "__main__":
    main()
