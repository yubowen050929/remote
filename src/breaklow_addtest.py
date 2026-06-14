"""'Add only on a break of the prior low' test.

Trigger: close makes a NEW low over a trailing window W (breaks prior
support). This is rarer and more capitulation-concentrated than the X%-step
ladder. Two variants:
  - hold_boost : while at/near a new W-day low, hold full; trim on recovery
  - pyramid    : add one tranche on EACH new W-day low (average down)

High base (few reserve) to minimise drag, since adds are selective. T+2 +
FIFO fees. Decomposed into drag vs covariance, and split by episode type
(does the covariance come from V-bottoms or get eaten by 2022-style grinds).
"""
import numpy as np
import pandas as pd

DATA = "../data"


def load(n):
    return pd.read_csv(f"{DATA}/{n}.csv", index_col=0, parse_dates=True)


def backtest(target, ret, lag=2, band=0.05, fees=True, fee_u7=0.015, fee_a7=0.005):
    eff = target.shift(lag).bfill()
    port, position = 1.0, float(eff.iloc[0])
    lots = [[position, 0]]
    curve = []
    for i in range(len(ret)):
        port *= (1 + position * ret.iloc[i])
        for lot in lots:
            lot[1] += 1
        tgt = float(eff.iloc[i])
        delta = tgt - position
        if abs(delta) >= band:
            if delta > 0:
                lots.append([delta, 0])
            else:
                to_sell = -delta
                while to_sell > 1e-9 and lots:
                    lot = lots[0]
                    take = min(lot[0], to_sell)
                    if fees:
                        port -= take * (fee_u7 if lot[1] < 7 else fee_a7) * port
                    lot[0] -= take
                    to_sell -= take
                    if lot[0] <= 1e-9:
                        lots.pop(0)
            position = tgt
        curve.append(port)
    return pd.Series(curve, index=ret.index), eff


def target_breaklow(c, W, base=0.85, full=1.0, variant="hold_boost", n_tr=3):
    new_low = c <= c.shift(1).rolling(W).min()    # today breaks the prior W-day low
    pos = np.full(len(c), base)
    if variant == "hold_boost":
        # boosted to full for a holding period after each break, decaying back
        boosted = new_low.rolling(10).max().fillna(0).astype(bool)  # full for ~10d after a break
        pos = np.where(boosted, full, base)
    else:  # pyramid: each new low in a cluster adds a tranche, reset on recovery to 20d high
        step = (full - base) / n_tr
        state = base
        recov = c >= c.shift(1).rolling(20).max()
        out = []
        for i in range(len(c)):
            if bool(new_low.iloc[i]):
                state = min(full, state + step)
            elif bool(recov.iloc[i]):
                state = base
            out.append(state)
        pos = np.array(out)
    return pd.Series(pos, index=c.index)


def decomp(eff, ret, Er):
    w = eff.loc[ret.index]
    cov = np.cov(w, ret)[0, 1] * 252
    drag = (1 - w.mean()) * Er * 252
    return w.mean(), drag * 100, cov * 100


def main():
    c = load("spx")["Close"]
    c = c[c.index >= "2011-06-09"]
    ret = c.pct_change().dropna()
    cc = c.loc[ret.index]
    bh = (1 + ret).cumprod()
    yrs = len(ret) / 252
    bc = bh.iloc[-1] ** (1 / yrs) - 1
    Er = ret.mean()
    print(f"buy & hold CAGR = {bc:+.2%}\n")

    print(f"{'strategy':32s} {'CAGR':>7} {'exΔ':>7} {'MDD':>7} {'E[w]':>6} {'drag':>6} {'cov':>6}")
    for variant in ["hold_boost", "pyramid"]:
        for W in [20, 60, 120, 252]:
            t = target_breaklow(cc, W, variant=variant)
            cv, eff = backtest(t, ret)
            cg = cv.iloc[-1] ** (1 / yrs) - 1
            mdd = (cv / cv.cummax() - 1).min()
            Ew, drag, cov = decomp(eff, ret, Er)
            print(f"break {W:>3}d-low / {variant:11s}  {cg:+6.1%} {cg-bc:+6.1%} {mdd:6.0%} "
                  f"{Ew:6.2f} {drag:+5.1f} {cov:+5.1f}")

    # where does the covariance come from? split the best variant's added-weight days
    t = target_breaklow(cc, 60, variant="hold_boost")
    _, eff = backtest(t, ret)
    added = eff.loc[ret.index] > 0.85 + 1e-6
    fwd20 = (cc.shift(-20) / cc - 1).loc[ret.index]
    print(f"\nwhen boosted (break of 60d low), forward-20d outcome:")
    print(f"  share of days boosted: {added.mean():.1%}")
    print(f"  median fwd-20d return WHEN boosted: {fwd20[added].median():+.2%}  "
          f"vs unboosted {fwd20[~added].median():+.2%}")
    # by year: did boost help or hurt
    yr = pd.DataFrame({"boost": added.astype(int), "f20": fwd20})
    print("  boosted-day fwd20 by year (where the bet landed):")
    g = yr[yr.boost == 1].groupby(yr[yr.boost == 1].index.year)["f20"].median()
    print("   " + "  ".join(f"{y}:{v:+.0%}" for y, v in g.items()))


if __name__ == "__main__":
    main()
