"""Round-trip timing test: exit signal + EARLY re-entry signal.

The first feasibility test only de-risked and re-entered mechanically via
the same slow EMA stack, which buys back AFTER the recovery -> structural
underperformance. This test adds dedicated, LEADING re-entry signals and
asks whether any exit+reentry round-trip beats buy-and-hold on return.

Exit (reduce to defensive weight): EMA bearish stack OR VIX>22.
Re-entry (back to full) candidates, each tested separately:
  R1 vix_rollover : VIX was >22 and is now below its value 3 days ago
  R2 dix_high     : DIX in top 40% (passive dip absorption underway)
  R3 ts_normalize : VIX/VIX3M back below 0.95 after inversion
  R4 reclaim_ema  : close back above 21 EMA after being below
  R5 combo        : vix_rollover OR dix_high (either leading sign)

All with T+2 lag, FIFO redemption fees. Compared to buy-and-hold and to the
exit-only (slow re-entry) baseline.
"""
import numpy as np
import pandas as pd

DATA = "../data"


def load(n):
    return pd.read_csv(f"{DATA}/{n}.csv", index_col=0, parse_dates=True)


def build(idx_name):
    c = load(idx_name)["Close"]
    vix = load("vix")["Close"].reindex(c.index).ffill()
    vix3m = load("vix3m")["Close"].reindex(c.index).ffill()
    dg = pd.read_csv(f"{DATA}/dix_gex.csv", parse_dates=["date"]).set_index("date")
    dix = dg["dix"].reindex(c.index).ffill()
    e21 = c.ewm(span=21, adjust=False).mean()
    e8, e34 = c.ewm(span=8, adjust=False).mean(), c.ewm(span=34, adjust=False).mean()

    bear = (c <= e8) & (e8 <= e21) & (e21 <= e34)
    exit_sig = bear | (vix > 22)

    reentry = {
        "R1 vix_rollover": (vix.shift(0) < vix.shift(3)) & (vix.rolling(5).max() > 22),
        "R2 dix_high": dix > dix.rolling(252).quantile(0.60),
        "R3 ts_normalize": (vix / vix3m) < 0.95,
        "R4 reclaim_ema": c > e21,
        "R5 combo": ((vix < vix.shift(3)) & (vix.rolling(5).max() > 22)) | (dix > dix.rolling(252).quantile(0.60)),
    }
    return c, exit_sig, reentry


def target_from_state(exit_sig, reentry_sig, defensive=0.5, full=1.0):
    """Stateful target: drop to defensive on exit, return to full on re-entry."""
    pos = np.empty(len(exit_sig))
    state = full
    for i in range(len(exit_sig)):
        if exit_sig.iloc[i]:
            state = defensive
        elif reentry_sig.iloc[i] and state < full:
            state = full
        pos[i] = state
    return pd.Series(pos, index=exit_sig.index)


def backtest(target, ret, lag=2, band=0.10, fees=True, fee_u7=0.015, fee_a7=0.005):
    eff = target.shift(lag).bfill()
    port, position = 1.0, float(eff.iloc[0])
    lots = [[position, 0]]
    fees_paid = 0.0
    trades = 0
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
            trades += 1
        curve.append(port)
    return pd.Series(curve, index=ret.index), trades


def stats(curve, ret):
    yrs = len(ret) / 252
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    mdd = (curve / curve.cummax() - 1).min()
    d = curve.pct_change().dropna()
    return cagr, mdd, d.mean() / d.std() * np.sqrt(252)


def main():
    for idx_name in ["spx", "ndx"]:
        c, exit_sig, reentry = build(idx_name)
        ret = c.pct_change().loc[c.index >= "2011-06-09"].dropna()
        exit_sig = exit_sig.loc[ret.index]
        bh = (1 + ret).cumprod()
        bc, bm, bs = stats(bh, ret)
        print(f"\n=== {idx_name.upper()} ===")
        print(f"{'buy & hold':30s} CAGR={bc:+.1%}  MDD={bm:.1%}  Sharpe={bs:.2f}")
        # exit-only baseline (slow EMA re-entry == never re-enter early)
        never = pd.Series(False, index=ret.index)
        t = target_from_state(exit_sig, never)
        cv, tr = backtest(t, ret)
        cg, md, sh = stats(cv, ret)
        print(f"{'exit-only (no smart reentry)':30s} CAGR={cg:+.1%}  MDD={md:.1%}  Sharpe={sh:.2f}  exΔ={cg-bc:+.2%}")
        for name, rsig in reentry.items():
            t = target_from_state(exit_sig, rsig.loc[ret.index].fillna(False))
            cv, tr = backtest(t, ret)
            cg, md, sh = stats(cv, ret)
            print(f"{name:30s} CAGR={cg:+.1%}  MDD={md:.1%}  Sharpe={sh:.2f}  exΔ={cg-bc:+.2%}  trades={tr}")


if __name__ == "__main__":
    main()
