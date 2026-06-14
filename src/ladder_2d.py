"""2D dip-ladder test: base position x deploy step.

Addresses "2% is too shallow, use 5%" AND the deeper question: the ladder's
drag comes from the cash reserve held while waiting for dips. So sweep BOTH
the base position (how little reserve you keep) and the drawdown step that
triggers each add. For every cell, report excess CAGR vs B&H and the
decomposition E[w*r] = E[w]E[r] + Cov(w,r), so the drag-vs-covariance
tradeoff is visible.

Ladder: position = base while at highs; deploy one tranche per `step`
drawdown from the 252d high, up to 100%; trim back on recovery. T+2 + fees.
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


def ladder(c, step, base, full=1.0, n=4):
    dd = c / c.rolling(252, min_periods=60).max() - 1
    tr = (full - base) / n
    return pd.Series(base + np.clip(np.floor(-dd / step), 0, n) * tr, index=c.index)


def cagr(curve, ret):
    return curve.iloc[-1] ** (252 / len(ret)) - 1


def main():
    c = load("spx")["Close"]
    c = c[c.index >= "2011-06-09"]
    ret = c.pct_change().dropna()
    cc = c.loc[ret.index]
    bh = (1 + ret).cumprod()
    bc = cagr(bh, ret)
    Er = ret.mean()
    print(f"buy & hold CAGR = {bc:+.2%}   (daily drift E[r] = {Er*252:+.1%}/yr)\n")

    print("=== excess CAGR vs B&H : base position (rows) x deploy step (cols) ===")
    print("            step=3%    5%      7%     10%")
    for base in [0.60, 0.70, 0.80, 0.90]:
        row = []
        for step in [0.03, 0.05, 0.07, 0.10]:
            t = ladder(cc, step, base)
            cv, _ = backtest(t, ret)
            row.append(f"{cagr(cv, ret)-bc:+.1%}")
        print(f"base={base:.0%}:   " + "  ".join(f"{x:>6s}" for x in row))

    print("\n=== decomposition for the user's case (base=90%, step=5%) and extremes ===")
    for base, step, label in [(0.90, 0.05, "base90 step5 (your idea)"),
                              (0.60, 0.05, "base60 step5"),
                              (0.95, 0.05, "base95 step5 (tiny reserve)")]:
        t = ladder(cc, step, base)
        cv, eff = backtest(t, ret)
        w = eff.loc[ret.index]
        Ew = w.mean()
        cov = np.cov(w, ret)[0, 1] * 252
        drag = (1 - Ew) * Er * 252
        print(f"{label:30s} E[w]={Ew:.3f}  drag={drag:+.2%}  cov(timing)={cov:+.2%}  "
              f"net={cov-drag:+.2%}  realized exΔ={cagr(cv,ret)-bc:+.2%}")
    print("\n  As reserve shrinks (base->100%), drag falls but the dip-buying")
    print("  covariance falls too — the strategy just converges to B&H from below.")


if __name__ == "__main__":
    main()
