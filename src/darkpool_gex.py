"""Dark pool (DIX) and dealer gamma (GEX) as drawdown predictors.

Data: SqueezeMetrics free historical file (date, SPX price, DIX, GEX),
https://squeezemetrics.com/monitor/static/DIX.csv, 2011-05 onward.
  - DIX: dark-pool short-volume ratio (built from FINRA daily RegSHO short
    volume files); high = passive buy-side absorption, documented as a
    medium-horizon bullish signal.
  - GEX: dollar gamma exposure estimated from SPX option open interest.

GEX in dollars grows with index level, so we test both a price-squared
normalization and a 1y rolling percentile.
"""
import os
import urllib.request

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
URL = "https://squeezemetrics.com/monitor/static/DIX.csv"
PATH = os.path.join(DATA, "dix_gex.csv")


def qtest(x, target, label, n=5):
    ok = x.notna() & target.notna() & (x.index >= "2011-06-09")
    q = pd.qcut(x[ok], n, labels=False, duplicates="drop")
    tab = target[ok].astype(float).groupby(q).mean()
    print(f"{label}:")
    print("  " + "  ".join(f"Q{int(k)+1}={v:.1%}" for k, v in tab.items()))


def main():
    if not os.path.exists(PATH):
        urllib.request.urlretrieve(URL, PATH)
    dg = pd.read_csv(PATH, parse_dates=["date"]).set_index("date")
    spx = pd.read_csv(os.path.join(DATA, "spx.csv"), index_col=0, parse_dates=True)["Close"]
    ret = spx.pct_change()
    target = (ret.shift(-1) < -0.013).reindex(dg.index)

    dg["gex_norm"] = dg["gex"] / dg["price"] ** 2
    dg["gex_pctl"] = dg["gex_norm"].rolling(252).apply(lambda x: (x.iloc[:-1] < x.iloc[-1]).mean())

    qtest(dg["gex_norm"], target, "P(next-day SPX drop >1.3%) by normalized GEX quintile (low->high)")
    qtest(dg["gex_pctl"], target, "P(next-day drop) by GEX 1y-rolling percentile quintile")
    qtest(dg["dix"], target, "P(next-day drop) by DIX quintile (low->high)")

    neg = dg["gex_norm"] <= dg["gex_norm"].quantile(0.10)
    ok = target.notna() & (dg.index >= "2011-06-09")
    print(f"\nGEX bottom decile: P(event)={target[ok & neg].astype(float).mean():.1%} "
          f"vs rest={target[ok & ~neg].astype(float).mean():.1%}")

    fwd21 = spx.pct_change(21).shift(-21).reindex(dg.index)
    ok2 = dg["dix"].notna() & fwd21.notna()
    q = pd.qcut(dg.loc[ok2, "dix"], 5, labels=False)
    print("\nmedian forward 21d SPX return by DIX quintile (DIX is a horizon-weeks signal):")
    print("  " + "  ".join(f"Q{int(k)+1}={v:+.2%}" for k, v in fwd21[ok2].groupby(q).median().items()))
    print(f"\nlatest: {dg.index.max().date()}  DIX={dg['dix'].iloc[-1]:.3f}  "
          f"GEX 1y-percentile={dg['gex_pctl'].iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
