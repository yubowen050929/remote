"""Empirical test of the components inside Saty Mahajan's two TradingView
indicators (Pivot Ribbon: 8/21/34 EMA stack + 13/48 conviction cross;
ATR Levels: prev_close +/- fib multiples of ATR14) as one-day-ahead
drawdown predictors on SPX, 2011-2026.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")


def main():
    df = pd.read_csv(os.path.join(DATA, "spx.csv"), index_col=0, parse_dates=True)
    c, h, l = df["Close"], df["High"], df["Low"]
    target = (c.pct_change().shift(-1) < -0.013)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    e8, e21, e34, e13, e48 = [c.ewm(span=s, adjust=False).mean() for s in (8, 21, 34, 13, 48)]

    bull_stack = (c >= e8) & (e8 >= e21) & (e21 >= e34)
    bear_stack = (c <= e8) & (e8 <= e21) & (e21 <= e34)
    bear_conv = e13 < e48
    trig_dn = c < (c.shift(1) - 0.236 * atr.shift(1))
    below_1atr = c < (c.shift(1) - 1.0 * atr.shift(1))

    mask = (c.index >= "2011-06-09") & target.notna()
    base = target[mask].mean()
    print(f"base rate {base:.1%}")
    states = [
        ("bullish 8/21/34 stack", bull_stack),
        ("mixed/transition", ~bull_stack & ~bear_stack),
        ("bearish 8/21/34 stack", bear_stack),
        ("13<48 bear conviction", bear_conv),
        ("closed below -0.236 ATR trigger", trig_dn),
        ("closed below -1 ATR", below_1atr),
    ]
    for name, s in states:
        m = mask & s
        p = target[m].mean()
        print(f"{name:32s} days={s[mask].mean():5.1%}  P(next-day event)={p:5.1%}  lift={p / base:.2f}x")

    vix = pd.read_csv(os.path.join(DATA, "vix.csv"), index_col=0, parse_dates=True)["Close"].reindex(c.index)
    print("\njoint with VIX:")
    for vn, vm in [("VIX<=20", vix <= 20), ("VIX>20", vix > 20)]:
        for sn, sm in [("bull stack", bull_stack), ("bear stack", bear_stack)]:
            m = mask & vm & sm
            if m.sum() > 30:
                print(f"  {vn} & {sn:10s}: P(event)={target[m].mean():5.1%}  (days={int(m.sum())})")

    atr_units = 0.013 / (atr.shift(1) / c.shift(1))
    print(f"\nfixed -1.3% expressed in ATR units: 2017 median={atr_units['2017'].median():.1f}, "
          f"2022 median={atr_units['2022'].median():.1f}, sample median={atr_units[mask].median():.1f}")


if __name__ == "__main__":
    main()
