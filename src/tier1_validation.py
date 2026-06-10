"""Same-axis validation of newly acquired Tier-1 data (run after
download_tier1.py / download_cot.py).

Verdicts recorded in SIGNALS.md:
  - COR1M (implied correlation): PASS - monotonic 3.3%->13.4%, corr 0.71
    with realized sector pairwise (same axis, forward-looking version)
  - VX curve backwardation: PASS - ->16.5% top quintile, corr -0.85 with
    VIX/VIX3M (same axis; run in parallel one quarter before any switch)
  - VXN replacing VIX in the NDX model: FAIL gate 3 (AUC 0.670->0.662);
    VIX retained as the global risk anchor even for NDX
  - COT VIX leveraged-fund net percentile: monthly-horizon candidate,
    47% vs 32% - note direction: HIGH net (funds long vol) = risk
  - NDX top-10 earnings window (unconditional): NULL (11.9% vs 10.6%);
    calendar retained only as a conditional-flag candidate
"""
import os
import warnings

import pandas as pd

from precursor_analysis import load

warnings.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def qt(x, target, label, n=5):
    ok = x.notna() & target.notna() & (x.index >= "2011-06-09")
    q = pd.qcut(x[ok], n, labels=False, duplicates="drop")
    tab = target[ok].astype(float).groupby(q).mean()
    print(f"{label}: " + "  ".join(f"Q{int(k)+1}={v:.1%}" for k, v in tab.items()))


def main():
    spx, ndx = load("spx"), load("ndx")
    t_spx = spx.pct_change().shift(-1) < -0.013
    t_ndx = ndx.pct_change().shift(-1) < -0.013

    cor = pd.read_csv(os.path.join(DATA, "cor1m.csv"))
    cor["DATE"] = pd.to_datetime(cor["DATE"])
    cor = cor.set_index("DATE")["CLOSE"].reindex(spx.index).ffill()
    qt(cor, t_spx, "COR1M implied correlation")
    apc = pd.read_csv(os.path.join(DATA, "sector_pairwise_corr.csv"),
                      index_col=0, parse_dates=True).iloc[:, 0].reindex(spx.index)
    print(f"  corr with realized pairwise: {cor.corr(apc):.2f}")

    vx = pd.read_csv(os.path.join(DATA, "vx_curve.csv"), parse_dates=["date"]).set_index("date")
    f2f1 = vx["f2_f1"].reindex(spx.index).ffill()
    qt(-f2f1, t_spx, "VX backwardation (-f2/f1)")
    vix, vix3m = load("vix"), load("vix3m")
    print(f"  corr(f2/f1, VIX/VIX3M): {f2f1.corr(vix / vix3m):.2f}")

    cot = pd.read_csv(os.path.join(DATA, "cot_positioning.csv"), parse_dates=["date"]).set_index("date")
    vixlev = cot["vix_lev_net"].reindex(spx.index, method="ffill").shift(3)  # release lag
    vl_pct = vixlev.rolling(252).apply(lambda x: (x.iloc[:-1] < x.iloc[-1]).mean())
    t21 = spx.pct_change(5).shift(-5).rolling(21).min().shift(-16) < -0.03
    qt(vl_pct, t21, "COT VIX lev-net pctl vs monthly -3% stretch")

    earn = pd.read_csv(os.path.join(DATA, "ndx_top10_earnings.csv"), parse_dates=["date"])
    idx = ndx.index
    after = pd.Series(False, index=idx)
    for e in pd.DatetimeIndex(earn["date"].unique()):
        loc = idx.searchsorted(e)
        for j in (loc, loc + 1):
            if 0 <= j < len(idx):
                after.iloc[j] = True
    mask = (idx >= "2012-01-01") & t_ndx.notna()
    print(f"top-10 earnings window: P(NDX event)={t_ndx[mask & after].mean():.1%} "
          f"vs other={t_ndx[mask & ~after].mean():.1%}")


if __name__ == "__main__":
    main()
