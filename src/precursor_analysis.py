"""Precursor analysis: what does the market look like on day T-1, before a
large drawdown on day T, compared with an ordinary day?

For each feature measured at T-1 we report the median on pre-event days vs
all other days, plus the event probability by feature quintile (computed
on expanding history to avoid lookahead in the quintile cut would be
overkill here - this section is descriptive, the predictive test lives in
predict_model.py).
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")


def load(name, col="Close"):
    df = pd.read_csv(os.path.join(DATA, f"{name}.csv"), index_col=0, parse_dates=True)
    return df[col].astype(float)


def build_features() -> pd.DataFrame:
    spx = load("spx")
    ndx = load("ndx")
    vix = load("vix")
    vix3m = load("vix3m")
    vvix = load("vvix")
    move = load("move")
    hyg = load("hyg")
    skew = load("skew")
    rsp = load("rsp")

    f = pd.DataFrame(index=spx.index)
    f["spx_ret"] = spx.pct_change()
    f["ndx_ret"] = ndx.pct_change()
    f["vix"] = vix
    f["vix_chg5"] = vix.diff(5)
    f["ts_ratio"] = vix / vix3m                      # >1 = inverted term structure
    f["vvix"] = vvix
    f["move"] = move.reindex(spx.index).ffill()
    f["skew"] = skew.reindex(spx.index).ffill()
    f["rv21"] = f["spx_ret"].rolling(21).std() * np.sqrt(252) * 100
    f["vrp"] = f["vix"] - f["rv21"]                  # variance risk premium proxy
    f["dd_52w"] = spx / spx.rolling(252).max() - 1   # distance from 52w high
    f["ret_5d"] = spx.pct_change(5)
    f["hyg_ret20"] = hyg.pct_change(20)
    f["breadth_20"] = rsp.pct_change(20) - spx.pct_change(20)  # equal-wt minus cap-wt
    f["ndx_rv21"] = f["ndx_ret"].rolling(21).std() * np.sqrt(252) * 100
    return f


def main():
    f = build_features()
    f = f[f.index >= "2011-06-09"]

    rows = []
    for idx_name, retcol in [("SPX", "spx_ret"), ("NDX", "ndx_ret")]:
        target = (f[retcol].shift(-1) < -0.013)  # event happens TOMORROW
        target = target[f.index]
        feats = ["vix", "vix_chg5", "ts_ratio", "vvix", "move", "skew",
                 "rv21", "vrp", "dd_52w", "ret_5d", "hyg_ret20", "breadth_20"]
        for col in feats:
            x = f[col]
            ok = x.notna() & target.notna()
            pre = x[ok & target]
            normal = x[ok & ~target]
            rows.append({
                "index": idx_name, "feature": col,
                "median_pre_event_T-1": round(float(pre.median()), 4),
                "median_normal": round(float(normal.median()), 4),
                "p75_pre_event": round(float(pre.quantile(0.75)), 4),
            })
        # quintile event-rates for the headline features
        for col in ["vix", "ts_ratio", "rv21", "dd_52w"]:
            x = f[col]
            ok = x.notna() & target.notna()
            q = pd.qcut(x[ok], 5, labels=False, duplicates="drop")
            tab = target[ok].groupby(q).mean()
            print(f"\n{idx_name}  P(next-day event) by {col} quintile (low->high):")
            print("  " + "  ".join(f"Q{int(k)+1}={v:.1%}" for k, v in tab.items()))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "precursor_features.csv"), index=False)
    print("\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
