"""Does the dealer-gamma regime mechanically amplify daily ranges?

Tests next-session SPX high-low range conditional on the GEX 1y rolling
percentile (SqueezeMetrics series), with a VIX control to confirm the
range information is not just volatility-regime overlap.
"""
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")


def main():
    df = pd.read_csv(os.path.join(DATA, "spx.csv"), index_col=0, parse_dates=True)
    h, l, c = df["High"], df["Low"], df["Close"]
    rng = ((h - l) / c.shift(1)).shift(-1)  # next session's range
    dg = pd.read_csv(os.path.join(DATA, "dix_gex.csv"), parse_dates=["date"]).set_index("date")
    gexp = (dg["gex"] / dg["price"] ** 2).rolling(252).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).mean()).reindex(df.index)
    vix = pd.read_csv(os.path.join(DATA, "vix.csv"), index_col=0, parse_dates=True)["Close"].reindex(df.index)

    mask = (df.index >= "2011-06-09") & gexp.notna() & rng.notna()
    q = pd.qcut(gexp[mask], 5, labels=False)
    med = rng[mask].groupby(q).median() * 100
    p90 = rng[mask].groupby(q).quantile(0.9) * 100
    print("next-session range by GEX percentile quintile (Q1=most negative gamma):")
    print("  median: " + "  ".join(f"Q{int(k)+1}={v:.2f}%" for k, v in med.items()))
    print("  p90:    " + "  ".join(f"Q{int(k)+1}={v:.2f}%" for k, v in p90.items()))

    lo = gexp < 0.2
    print("\nVIX control:")
    for vn, vm in [("VIX<=20", vix <= 20), ("VIX>20", vix > 20)]:
        for gn, gm in [("GEX pctl<0.2", lo), ("GEX pctl>=0.2", ~lo)]:
            m = mask & vm & gm
            if m.sum() > 50:
                print(f"  {vn} & {gn:14s}: median next-range={rng[m].median() * 100:.2f}% (n={int(m.sum())})")


if __name__ == "__main__":
    main()
