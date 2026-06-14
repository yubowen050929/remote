"""Analog (K-line shape) forecasting test.

Hypothesis (user): matching the current price SHAPE against historical
rebound shapes may re-enter better than VIX/DIX-based signals.

Method (strictly causal kNN analog):
  - feature = last L-day return shape, z-scored (pure shape, level removed)
  - for a decision at day t, search ONLY windows whose window AND their
    forward H-day outcome both ended before t (no lookahead)
  - prediction = mean forward H-day return of the K nearest analogs
Skill measured by Spearman corr and directional hit-rate vs actual forward
return, and quintile forward returns. Baseline: VIX level (a state variable)
predicting the same forward return.

Then: does conditioning the analog on the recent shape add skill over VIX
specifically on risk-off (post-exit) days, which is where re-entry happens?
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA = "../data"
L = 10      # shape window
H = 5       # forecast horizon (trading days)
K = 50      # neighbors


def load(n):
    return pd.read_csv(f"{DATA}/{n}.csv", index_col=0, parse_dates=True)


def main():
    c = load("spx")["Close"]
    vix = load("vix")["Close"].reindex(c.index).ffill()
    r = c.pct_change().fillna(0).values
    n = len(c)
    fwd = pd.Series(c.shift(-H).values / c.values - 1, index=c.index)  # forward H-day return

    # precompute z-scored shape vectors ending at each index i (uses r[i-L+1..i])
    shapes = np.full((n, L), np.nan)
    for i in range(L, n):
        w = r[i - L + 1:i + 1]
        s = w.std()
        shapes[i] = (w - w.mean()) / s if s > 0 else 0

    preds = np.full(n, np.nan)
    start = 760  # burn-in ~3y of history to search
    for t in range(start, n - H):
        q = shapes[t]
        if np.isnan(q).any():
            continue
        # candidate analog windows must end at j with j+H < t (outcome known, no lookahead)
        jmax = t - H - 1
        cand = shapes[L:jmax]
        # distance
        d = np.sqrt(((cand - q) ** 2).sum(axis=1))
        idx = np.argsort(d)[:K] + L
        preds[t] = np.nanmean(c.values[idx + H] / c.values[idx] - 1)

    pred = pd.Series(preds, index=c.index)
    mask = pred.notna() & fwd.notna() & (c.index >= "2014-06-09")
    p, a = pred[mask], fwd[mask]
    rho = spearmanr(p, a).correlation
    hit = ((p > 0) == (a > 0)).mean()
    print(f"analog kNN (L={L},H={H},K={K}): Spearman={rho:.3f}  dir-hit={hit:.1%}  n={mask.sum()}")
    # quintiles
    q = pd.qcut(p, 5, labels=False)
    tab = a.groupby(q).mean()
    print("  mean actual fwd-5d return by analog-prediction quintile (low->high):")
    print("   " + "  ".join(f"Q{int(k)+1}={v:+.2%}" for k, v in tab.items()))

    # baseline: VIX predicting forward return (inverse - high vix -> low fwd? test)
    vp = (-vix)[mask]
    rho_v = spearmanr(vp, a).correlation
    print(f"\nbaseline VIX (−level): Spearman={rho_v:.3f}")

    # on risk-off days specifically (VIX>22), does analog separate recoveries?
    ro = mask & (vix > 22)
    if ro.sum() > 50:
        rho_ro = spearmanr(pred[ro], fwd[ro]).correlation
        qro = pd.qcut(pred[ro], 3, labels=False, duplicates="drop")
        tro = fwd[ro].groupby(qro).mean()
        print(f"\nrisk-off days only (VIX>22, n={ro.sum()}): analog Spearman={rho_ro:.3f}")
        print("  fwd-5d by analog tercile: " + "  ".join(f"T{int(k)+1}={v:+.2%}" for k, v in tro.items()))


if __name__ == "__main__":
    main()
