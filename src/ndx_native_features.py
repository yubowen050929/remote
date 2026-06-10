"""NDX feature-localization test.

Diagnosis: the walk-forward NDX model (AUC 0.643) was built on SPX-centric
features (SPX dd_52w / ret_5d / rv21, SPX-option-derived VIX family, SPX
GEX). This script swaps in NDX-native equivalents plus a semiconductor
relative-strength factor and re-runs the walk-forward comparison.

Result: AUC 0.643 -> 0.670 from feature localization alone. Remaining gap
vs SPX (0.726) is attributed to instrument mismatch (VIX vs VXN, SPX-chain
GEX vs QQQ-chain GEX) and NDX idiosyncratic drivers (mega-cap earnings).
"""
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from precursor_analysis import build_features, load
from predict_model import walk_forward

warnings.filterwarnings("ignore")

SPX_BORROWED = ["vix", "vix_chg5", "ts_ratio", "vvix", "move", "skew",
                "rv21", "vrp", "dd_52w", "ret_5d", "hyg_ret20", "breadth_20"]
NDX_NATIVE = ["vix", "vix_chg5", "ts_ratio", "vvix", "move", "skew",
              "ndx_rv21", "vrp", "ndx_dd_52w", "ndx_ret_5d", "hyg_ret20", "sox_rel_20"]


def main():
    f = build_features()
    ndx, sox = load("ndx"), load("sox")
    f["ndx_dd_52w"] = ndx / ndx.rolling(252).max() - 1
    f["ndx_ret_5d"] = ndx.pct_change(5)
    f["sox_rel_20"] = sox.reindex(f.index).ffill().pct_change(20) - ndx.pct_change(20)
    f = f[f.index >= "2011-06-09"]
    target = (f["ndx_ret"].shift(-1) < -0.013).astype(int)

    for name, feats in [("SPX-borrowed (baseline)", SPX_BORROWED),
                        ("NDX-native", NDX_NATIVE)]:
        data = f[feats]
        ok = data.notna().all(axis=1) & target.notna()
        ok.iloc[-1] = False
        X, y = data[ok], target[ok]
        p = walk_forward(X, y, lambda: LogisticRegression(max_iter=2000, C=0.5))
        yy = y.loc[p.index]
        top10 = p >= p.quantile(0.90)
        print(f"{name:28s} AUC={roc_auc_score(yy, p):.3f}  "
              f"top10%-precision={yy[top10].mean():.1%}  base={yy.mean():.1%}")


if __name__ == "__main__":
    main()
