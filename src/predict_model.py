"""Walk-forward test: can a large drawdown be predicted one day ahead?

Setup:
  - Target: next-day return < -1.3% (SPX / NDX separately).
  - Features: only information available at the close of day T-1.
  - Models: logistic regression and gradient boosting.
  - Validation: expanding-window walk-forward, first 5 years as burn-in,
    refit every 60 trading days. No lookahead.

Reports ROC AUC, Brier score, and the precision/recall trade-off, plus a
simple transparent "regime rule" any practitioner could replicate.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from precursor_analysis import build_features

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "output")

FEATS = ["vix", "vix_chg5", "ts_ratio", "vvix", "move", "skew",
         "rv21", "vrp", "dd_52w", "ret_5d", "hyg_ret20", "breadth_20"]


def walk_forward(X: pd.DataFrame, y: pd.Series, model_factory, burn_in=1260, step=60):
    preds = pd.Series(np.nan, index=y.index)
    for start in range(burn_in, len(y), step):
        end = min(start + step, len(y))
        Xtr, ytr = X.iloc[:start], y.iloc[:start]
        scaler = StandardScaler().fit(Xtr)
        model = model_factory()
        model.fit(scaler.transform(Xtr), ytr)
        preds.iloc[start:end] = model.predict_proba(scaler.transform(X.iloc[start:end]))[:, 1]
    return preds.dropna()


def evaluate(name, y, p):
    y = y.loc[p.index]
    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    base = y.mean()
    print(f"\n=== {name} ===  AUC={auc:.3f}  Brier={brier:.4f}  base_rate={base:.2%}  n={len(y)}")
    rows = []
    for q in [0.80, 0.90, 0.95]:
        th = p.quantile(q)
        alarm = p >= th
        precision = y[alarm].mean()
        recall = y[alarm].sum() / y.sum()
        rows.append({"model": name, "alarm_top_pct": f"{(1-q):.0%} of days",
                     "threshold": round(float(th), 4),
                     "precision_P(event|alarm)": round(float(precision), 3),
                     "recall_share_events_caught": round(float(recall), 3),
                     "lift_vs_base": round(float(precision / base), 2),
                     "false_alarm_rate": round(float(1 - precision), 3)})
        print(f"  alarm on top {(1-q):.0%} days: precision={precision:.1%} "
              f"recall={recall:.1%} lift={precision/base:.1f}x")
    return auc, rows


def main():
    f = build_features()
    f = f[f.index >= "2011-06-09"]
    all_rows = []
    summary = []
    for idx_name, retcol in [("SPX", "spx_ret"), ("NDX", "ndx_ret")]:
        target = (f[retcol].shift(-1) < -0.013).astype(int)
        data = f[FEATS].copy()
        ok = data.notna().all(axis=1) & target.notna()
        # drop last row (target unknown)
        ok.iloc[-1] = False
        X, y = data[ok], target[ok]

        p_lr = walk_forward(X, y, lambda: LogisticRegression(max_iter=2000, C=0.5))
        auc_lr, rows = evaluate(f"{idx_name} logistic", y, p_lr)
        all_rows += rows

        p_gb = walk_forward(X, y, lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0))
        auc_gb, rows = evaluate(f"{idx_name} gradboost", y, p_gb)
        all_rows += rows

        # transparent regime rule: VIX>20 AND term structure ratio>0.95
        rule = (f["vix"] > 20) & (f["ts_ratio"] > 0.95)
        rule = rule.loc[p_lr.index]
        yr = y.loc[p_lr.index]
        prec = yr[rule].mean() if rule.sum() else float("nan")
        rec = yr[rule].sum() / yr.sum()
        print(f"  RULE vix>20 & vix/vix3m>0.95: active {rule.mean():.1%} of days, "
              f"precision={prec:.1%}, recall={rec:.1%}, lift={prec/yr.mean():.1f}x")
        all_rows.append({"model": f"{idx_name} rule(vix>20 & ts>0.95)",
                         "alarm_top_pct": f"{rule.mean():.1%} of days",
                         "threshold": "-",
                         "precision_P(event|alarm)": round(float(prec), 3),
                         "recall_share_events_caught": round(float(rec), 3),
                         "lift_vs_base": round(float(prec / yr.mean()), 2),
                         "false_alarm_rate": round(float(1 - prec), 3)})
        summary.append({"index": idx_name, "auc_logistic": round(auc_lr, 3),
                        "auc_gradboost": round(auc_gb, 3),
                        "oos_base_rate": round(float(y.loc[p_lr.index].mean()), 4)})

        # save probabilities for charting
        pd.DataFrame({"p_lr": p_lr, "p_gb": p_gb.reindex(p_lr.index),
                      "event_next_day": y.loc[p_lr.index]}).to_csv(
            os.path.join(OUT, f"oos_predictions_{idx_name.lower()}.csv"))

    pd.DataFrame(all_rows).to_csv(os.path.join(OUT, "prediction_tradeoffs.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(OUT, "prediction_summary.csv"), index=False)


if __name__ == "__main__":
    main()
