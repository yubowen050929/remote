"""Is the root cause over-triggering of the EXIT (drawdown-prediction) signal?

Sweep exit selectivity from loose to very strict, hold a fixed sensible
re-entry (time-DCA), and measure:
  - how many days the exit is active
  - precision of the exit: P(forward 20d return < 0 | exit fires) vs base rate
  - excess CAGR vs buy-and-hold

If the user is right (too many false exits), tightening the exit should
shrink the gap. If even the strictest exit still trails B&H, the problem is
not trigger frequency but exit precision at the needed level.
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
                        rate = fee_u7 if lot[1] < 7 else fee_a7
                        port -= take * rate * port
                    lot[0] -= take
                    to_sell -= take
                    if lot[0] <= 1e-9:
                        lots.pop(0)
            position = tgt
        curve.append(port)
    return pd.Series(curve, index=ret.index)


def dca_target(exit_sig, confirm, defensive=0.5, full=1.0, K=4, D=3):
    pos = np.empty(len(exit_sig))
    state = full
    since = 0
    step = (full - defensive) / K
    for i in range(len(exit_sig)):
        if exit_sig.iloc[i]:
            state = min(state, defensive)
            since = 0
        elif state < full:
            since += 1
            if since >= D:
                state = min(full, state + step)
                since = 0
        pos[i] = state
    return pd.Series(pos, index=exit_sig.index)


def main():
    c = load("spx")["Close"]
    vix = load("vix")["Close"].reindex(c.index).ffill()
    vix3m = load("vix3m")["Close"].reindex(c.index).ffill()
    e8, e21, e34 = [c.ewm(span=s, adjust=False).mean() for s in (8, 21, 34)]
    bear = (c <= e8) & (e8 <= e21) & (e21 <= e34)
    confirm = (vix < vix.shift(3))
    ret = c.pct_change().loc[c.index >= "2011-06-09"].dropna()
    fwd20 = (c.shift(-20) / c - 1).loc[ret.index]
    base_neg = (fwd20 < 0).mean()

    bh = (1 + ret).cumprod()
    yrs = len(ret) / 252
    bc = bh.iloc[-1] ** (1 / yrs) - 1
    print(f"buy & hold CAGR={bc:+.1%}   base rate P(fwd20<0)={base_neg:.1%}\n")
    print(f"{'exit rule':34s} {'%days':>6s} {'precision':>9s} {'CAGR':>7s} {'exΔ':>7s} {'MDD':>7s}")

    exits = {
        "VIX>18 (very loose)": vix > 18,
        "VIX>20": vix > 20,
        "VIX>22 (current)": vix > 22,
        "VIX>26": vix > 26,
        "VIX>30 (strict)": vix > 30,
        "VIX>35 (very strict)": vix > 35,
        "bear stack only": bear,
        "bear AND VIX>22": bear & (vix > 22),
        "bear AND VIX>26 AND inv": bear & (vix > 26) & (vix / vix3m > 1.0),
    }
    for name, ex in exits.items():
        ex = ex.loc[ret.index].fillna(False)
        pct = ex.mean()
        prec = (fwd20[ex] < 0).mean() if ex.sum() else float("nan")
        t = dca_target(ex, confirm.loc[ret.index].fillna(False))
        cv = backtest(t, ret)
        cg = cv.iloc[-1] ** (1 / yrs) - 1
        mdd = (cv / cv.cummax() - 1).min()
        print(f"{name:34s} {pct:6.0%} {prec:8.0%} {cg:+7.1%} {cg-bc:+7.1%} {mdd:7.0%}")


if __name__ == "__main__":
    main()
