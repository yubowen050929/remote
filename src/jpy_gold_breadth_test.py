"""Test three user-proposed signal families as drawdown predictors:

1. USDJPY (yen carry unwind risk)
2. Gold, absolute and relative to SPX (flight-to-safety)
3. Breadth concentration via equal-weight vs cap-weight ratios
   (RSP/SPX, QQQE/NDX) and "narrow rally" day flags.

Horizons: next-day event (< -1.3%) and, for breadth, a -3% five-day
stretch within the next month.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")


def load(n):
    return pd.read_csv(os.path.join(DATA, f"{n}.csv"), index_col=0, parse_dates=True)["Close"]


def qtest(x, target, mask, label, n=5):
    ok = x.notna() & target.notna() & mask
    q = pd.qcut(x[ok], n, labels=False, duplicates="drop")
    tab = target[ok].astype(float).groupby(q).mean()
    print(f"{label}:")
    print("  " + "  ".join(f"Q{int(k)+1}={v:.1%}" for k, v in tab.items()))


def main():
    spx, ndx, rsp, vix = load("spx"), load("ndx"), load("rsp"), load("vix")
    jpy, gld, qqqe = load("usdjpy"), load("gld"), load("qqqe")
    idx = spx.index
    r_spx, r_ndx, r_rsp = spx.pct_change(), ndx.pct_change(), rsp.pct_change()
    t_spx = r_spx.shift(-1) < -0.013
    t_ndx = r_ndx.shift(-1) < -0.013
    mask = pd.Series(idx >= "2011-06-09", index=idx)
    ok = mask & t_spx.notna()
    print(f"SPX base rate {t_spx[ok].mean():.1%}\n")

    jpy_d = jpy.reindex(idx).ffill()
    qtest(jpy_d.pct_change(10), t_spx, mask, "USDJPY 10d change (Q1=yen surging)")
    yen_surge = jpy_d.pct_change(10) < jpy_d.pct_change(10).rolling(252).quantile(0.10)
    print(f"yen surge (bottom decile 1y): P={t_spx[ok & yen_surge].mean():.1%}; "
          f"AND VIX>20: P={t_spx[ok & yen_surge & (vix.reindex(idx) > 20)].mean():.1%}\n")

    gld_d = gld.reindex(idx).ffill()
    qtest(gld_d.pct_change(20), t_spx, mask, "gold 20d return")
    qtest(gld_d.pct_change(20) - r_spx.rolling(20).sum(), t_spx, mask,
          "gold minus SPX 20d (flight-to-safety spread)")
    print()

    qtest((rsp / spx).pct_change(20), t_spx, mask, "RSP/SPX 20d change (Q1=narrowing)")
    narrow = (r_spx > 0) & (r_rsp < 0)
    print(f"narrow-rally day (SPX up, RSP down): P(next-day event)={t_spx[ok & narrow].mean():.1%}")
    r_qqqe = qqqe.reindex(idx).pct_change()
    ok_n = mask & t_ndx.notna()
    narrow_n = (r_ndx > 0) & (r_qqqe < 0)
    print(f"NDX narrow-rally day: P(next-day NDX event)={t_ndx[ok_n & narrow_n].mean():.1%} "
          f"vs base={t_ndx[ok_n].mean():.1%}\n")

    t_week = (spx.pct_change(5).shift(-5).rolling(21).min().shift(-16) < -0.03)
    qtest((rsp / spx).pct_change(60), t_week, mask,
          "P(-3% 5d stretch within next month) by 60d RSP/SPX change (Q1=narrowest)")


if __name__ == "__main__":
    main()
