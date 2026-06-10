"""Model gate for validated new signals.

Adds GEX percentile, COR1M, VX term structure, gold-SPX spread and realized
sector pairwise correlation to the walk-forward SPX/NDX models. All variants
within an experiment are evaluated on the SAME sample (intersection of
feature availability) so AUCs are comparable.

Experiment A (sample from GEX/COR availability, ~2012):
  baseline 12 features vs +gex_pctl vs +cor1m vs +gold vs +apc vs combined
Experiment B (sample from VX curve availability, 2013-05):
  best of A vs best of A + vx f2/f1
"""
import os
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from precursor_analysis import build_features, load
from predict_model import walk_forward, FEATS

warnings.filterwarnings("ignore")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def extra_features(f):
    dg = pd.read_csv(os.path.join(DATA, "dix_gex.csv"), parse_dates=["date"]).set_index("date")
    gexn = dg["gex"] / dg["price"] ** 2
    f["gex_pctl"] = gexn.rolling(252).apply(
        lambda x: (x.iloc[:-1] < x.iloc[-1]).mean()).reindex(f.index)
    cor = pd.read_csv(os.path.join(DATA, "cor1m.csv"))
    cor["DATE"] = pd.to_datetime(cor["DATE"])
    f["cor1m"] = cor.set_index("DATE")["CLOSE"].reindex(f.index).ffill()
    vx = pd.read_csv(os.path.join(DATA, "vx_curve.csv"), parse_dates=["date"]).set_index("date")
    f["vx_f2f1"] = vx["f2_f1"].reindex(f.index).ffill()
    gld = load("gld")
    spx = load("spx")
    f["gold_spx_20"] = gld.reindex(f.index).ffill().pct_change(20) - spx.pct_change(20)
    apc = pd.read_csv(os.path.join(DATA, "sector_pairwise_corr.csv"),
                      index_col=0, parse_dates=True).iloc[:, 0]
    f["apc"] = apc.reindex(f.index)
    return f


def run(f, target, feats, sample_cols, label):
    """Evaluate feats on the sample where ALL sample_cols are available."""
    ok = f[sample_cols].notna().all(axis=1) & target.notna()
    ok.iloc[-1] = False
    X, y = f.loc[ok, feats], target[ok]
    p = walk_forward(X, y, lambda: LogisticRegression(max_iter=2000, C=0.5))
    yy = y.loc[p.index]
    top10 = p >= p.quantile(0.90)
    top5 = p >= p.quantile(0.95)
    print(f"{label:38s} AUC={roc_auc_score(yy, p):.3f}  "
          f"top10%={yy[top10].mean():.1%}  top5%={yy[top5].mean():.1%}  "
          f"(n={len(yy)}, base={yy.mean():.1%})")
    return roc_auc_score(yy, p)


def main():
    f = build_features()
    f = extra_features(f)
    ndx, sox = load("ndx"), load("sox")
    f["ndx_dd_52w"] = ndx / ndx.rolling(252).max() - 1
    f["ndx_ret_5d"] = ndx.pct_change(5)
    f["sox_rel_20"] = sox.reindex(f.index).ffill().pct_change(20) - ndx.pct_change(20)
    f = f[f.index >= "2011-06-09"]

    A_COLS = FEATS + ["gex_pctl", "cor1m", "gold_spx_20", "apc"]
    B_COLS = A_COLS + ["vx_f2f1"]

    for idx_name, retcol, base_feats in [
            ("SPX", "spx_ret", FEATS),
            ("NDX", "ndx_ret", ["vix", "vix_chg5", "ts_ratio", "vvix", "move", "skew",
                                "ndx_rv21", "vrp", "ndx_dd_52w", "ndx_ret_5d",
                                "hyg_ret20", "sox_rel_20"])]:
        target = (f[retcol].shift(-1) < -0.013).astype(int)
        cols = list(dict.fromkeys(base_feats + ["gex_pctl", "cor1m", "gold_spx_20", "apc"]))
        print(f"\n=== {idx_name} | Experiment A (common sample: GEX+COR era) ===")
        run(f, target, base_feats, cols, "baseline")
        run(f, target, base_feats + ["gex_pctl"], cols, "+ gex_pctl")
        run(f, target, base_feats + ["cor1m"], cols, "+ cor1m")
        run(f, target, base_feats + ["gold_spx_20"], cols, "+ gold_spx_20")
        run(f, target, base_feats + ["apc"], cols, "+ apc (realized pairwise)")
        run(f, target, base_feats + ["gex_pctl", "cor1m"], cols, "+ gex_pctl + cor1m")
        run(f, target, base_feats + ["gex_pctl", "cor1m", "gold_spx_20", "apc"],
            cols, "+ all four")

        print(f"=== {idx_name} | Experiment B (common sample: VX era, 2013+) ===")
        colsb = list(dict.fromkeys(base_feats + ["gex_pctl", "cor1m", "vx_f2f1"]))
        run(f, target, base_feats + ["gex_pctl", "cor1m"], colsb, "best-A on VX-era sample")
        run(f, target, base_feats + ["gex_pctl", "cor1m", "vx_f2f1"], colsb, "+ vx_f2f1")


if __name__ == "__main__":
    main()
