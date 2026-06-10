"""Candlestick pattern test: do long shadows / dojis (single or repeated)
predict next-day large drawdowns on SPX?

Patterns are ATR-normalized and tested with and without location context
(near 52w high / after a decline), plus a VIX control to check whether
wick 'signals' carry information beyond the volatility regime itself.
"""
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")


def main():
    df = pd.read_csv(os.path.join(DATA, "spx.csv"), index_col=0, parse_dates=True)
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    vix = pd.read_csv(os.path.join(DATA, "vix.csv"), index_col=0, parse_dates=True)["Close"].reindex(df.index)
    r = c.pct_change()
    t1 = r.shift(-1) < -0.013
    fwd5 = r.rolling(5).sum().shift(-5)
    mask = (df.index >= "2011-06-09") & t1.notna()

    rng = h - l
    body = (c - o).abs()
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

    doji = (body < 0.1 * rng) & (rng > 0)
    long_up = (upper > 2 * body) & (upper > 0.4 * atr)
    long_dn = (lower > 2 * body) & (lower > 0.4 * atr)
    near_high = c / c.rolling(252).max() > 0.98
    after_drop = r.rolling(5).sum() < -0.02

    base = t1[mask].mean()
    print(f"base {base:.1%}")
    pats = [
        ("doji", doji),
        ("2+ dojis in 3 days", doji.rolling(3).sum() >= 2),
        ("long upper wick", long_up),
        ("long lower wick", long_dn),
        ("shooting star (upper wick near 52w high)", long_up & near_high),
        ("hammer (lower wick after -2% 5d)", long_dn & after_drop),
        ("doji near 52w high", doji & near_high),
        ("plain wide-range day (range>1.5 ATR)", rng > 1.5 * atr),
    ]
    for name, s in pats:
        m = mask & s
        if m.sum() < 20:
            continue
        print(f"{name:44s} days={int(m.sum()):4d}  P(event)={t1[m].mean():6.1%}  "
              f"lift={t1[m].mean() / base:.2f}x  fwd5_med={fwd5[m].median():+.2%}")

    print("\nVIX control (is the wick anything beyond the vol regime?):")
    for vn, vm in [("VIX<=20", vix <= 20), ("VIX>20", vix > 20)]:
        for pn, pm in [("long upper wick", long_up), ("no long upper wick", ~long_up)]:
            m = mask & vm & pm
            print(f"  {vn} & {pn:20s}: P(event)={t1[m].mean():6.1%} (days={int(m.sum())})")


if __name__ == "__main__":
    main()
