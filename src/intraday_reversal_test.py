"""Test of the 'intraday reversal = the worst has printed' narrative template:

On days where the intraday low is more than 1.5% below the previous close,
does recovering more than half of the fall by the close (a 'V-reversal',
the 'panic exhausted / put wall held' day) predict stabilization, compared
with closing near the lows?
"""
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")


def main():
    df = pd.read_csv(os.path.join(DATA, "spx.csv"), index_col=0, parse_dates=True)
    l, c = df["Low"], df["Close"]
    pc = c.shift(1)
    r = c.pct_change()
    t1 = r.shift(-1) < -0.013
    fwd5 = r.rolling(5).sum().shift(-5)
    mask = (df.index >= "2011-06-09") & t1.notna()

    deep = (l / pc - 1) < -0.015
    recovery = (c - l) / (pc - l).where(pc > l)
    rev = mask & deep & (recovery > 0.5)
    norev = mask & deep & (recovery < 0.2)

    for name, m in [("V-reversal (recovered >50% of an intraday -1.5% fall)", rev),
                    ("closed near lows (recovered <20%)", norev),
                    ("unconditional", mask)]:
        print(f"{name:55s} n={int(m.sum()):4d}  P(next-day event)={t1[m].mean():6.1%}  "
              f"fwd5_med={fwd5[m].median():+.2%}")


if __name__ == "__main__":
    main()
