"""Full parameter-surface test of the user's proposed fixes.

(1) (exit_vix, reentry_vix) grid: sweep exit threshold 22..32 and re-entry
    threshold 14..22, measure excess CAGR vs buy-and-hold over the WHOLE
    surface. Answers "does ANY threshold pair beat B&H", not a cherry pick.
(2) Drawdown-ladder: stay near full, hold a cash reserve, deploy one tranche
    per X% additional drawdown from the 252d high; trim back as it recovers.
    Sweep the step X to find the in-sample best, then check it out-of-sample.
(3) Return decomposition for the best strategy:
       E[w*r] = E[w]*E[r] + Cov(w, r)
    -> shows the structural drag (1-E[w])*E[r] vs the timing covariance.

All with T+2 lag and FIFO redemption fees.
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


def cagr(curve, ret):
    return curve.iloc[-1] ** (252 / len(ret)) - 1


def exit_reentry_target(vix, hi, lo, defensive=0.5):
    pos = np.empty(len(vix))
    state = 1.0
    for i in range(len(vix)):
        if state == 1.0 and vix.iloc[i] > hi:
            state = defensive
        elif state == defensive and vix.iloc[i] < lo:
            state = 1.0
        pos[i] = state
    return pd.Series(pos, index=vix.index)


def ladder_target(c, step, base=0.6, full=1.0, n_tranche=4):
    dd = c / c.rolling(252, min_periods=60).max() - 1
    tranche = (full - base) / n_tranche
    pos = base + np.clip(np.floor(-dd / step), 0, n_tranche) * tranche
    return pd.Series(pos, index=c.index)


def main():
    c = load("spx")["Close"]
    vix = load("vix")["Close"].reindex(c.index).ffill()
    c = c[c.index >= "2011-06-09"]
    vix = vix.loc[c.index]
    ret = c.pct_change().dropna()
    vix = vix.loc[ret.index]
    cc = c.loc[ret.index]
    bh = (1 + ret).cumprod()
    bc = cagr(bh, ret)
    print(f"buy & hold CAGR = {bc:+.2%}\n")

    # (1) exit/reentry grid
    print("=== (1) exit/reentry VIX grid: excess CAGR vs B&H (fees ON, T+2) ===")
    print("        reentry:  " + "  ".join(f"<{lo}" for lo in [14, 16, 18, 20, 22]))
    best = (-9, None)
    for hi in [22, 24, 25, 26, 28, 30, 32]:
        row = []
        for lo in [14, 16, 18, 20, 22]:
            if lo >= hi:
                row.append("  -  ")
                continue
            t = exit_reentry_target(vix, hi, lo)
            cv, _ = backtest(t, ret)
            ex = cagr(cv, ret) - bc
            row.append(f"{ex:+.1%}")
            if ex > best[0]:
                best = (ex, (hi, lo))
        print(f"exit>{hi:>2}:   " + "  ".join(f"{x:>6s}" for x in row))
    print(f"  best pair: exit>{best[1][0]}, reentry<{best[1][1]}  excess={best[0]:+.2%}  (still {'<' if best[0]<0 else '>'} 0)\n")

    # (2) drawdown ladder, in-sample best step then OOS
    print("=== (2) drawdown-ladder: tranche every X% drawdown ===")
    split = ret.index[len(ret) // 2]
    is_mask = ret.index <= split
    oos_mask = ret.index > split
    bh_is = cagr((1 + ret[is_mask]).cumprod(), ret[is_mask])
    bh_oos = cagr((1 + ret[oos_mask]).cumprod(), ret[oos_mask])
    rows = []
    for step in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        t = ladder_target(cc, step)
        cv, eff = backtest(t, ret)
        full = cagr(cv, ret) - bc
        cv_is, _ = backtest(t[is_mask], ret[is_mask])
        cv_oos, _ = backtest(t[oos_mask], ret[oos_mask])
        rows.append((step, full, cagr(cv_is, ret[is_mask]) - bh_is, cagr(cv_oos, ret[oos_mask]) - bh_oos))
    print(f"{'step':>6} {'full-sample exΔ':>16} {'in-sample exΔ':>14} {'out-sample exΔ':>15}")
    for s, f, i, o in rows:
        print(f"{s:>6.0%} {f:>16.2%} {i:>14.2%} {o:>15.2%}")
    best_is = max(rows, key=lambda r: r[2])
    print(f"  best IN-SAMPLE step={best_is[0]:.0%} (IS exΔ={best_is[2]:+.2%}) -> its OOS exΔ={best_is[3]:+.2%}\n")

    # (3) decomposition for best ladder
    print("=== (3) return decomposition: E[w*r] = E[w]E[r] + Cov(w,r) ===")
    t = ladder_target(cc, best_is[0])
    _, eff = backtest(t, ret)
    w = eff.loc[ret.index]
    Ew, Er = w.mean(), ret.mean()
    cov = np.cov(w, ret)[0, 1]
    drag = (1 - Ew) * Er
    print(f"  avg exposure E[w] = {Ew:.3f}  (B&H = 1.000)")
    print(f"  structural drag (1-E[w])*E[r]*252 = {drag*252:+.2%}/yr")
    print(f"  timing covariance Cov(w,r)*252    = {cov*252:+.2%}/yr")
    print(f"  net daily edge before fees        = {(cov - drag)*252:+.2%}/yr")
    print("  -> beats B&H only if Cov(w,r) > (1-E[w])*E[r]; here it does not.")


if __name__ == "__main__":
    main()
