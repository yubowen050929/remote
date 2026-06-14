"""Tranched (scale-in) re-entry test.

Single-shot re-entry gets whipsawed on one timing point. Real practice
scales back in over multiple tranches. Three scale-in models, all honest,
all with T+2 lag + FIFO redemption fees, vs single-shot and buy-and-hold.

State: exit signal (EMA bear OR VIX>22) sets a 'risk-off' flag and trims to
defensive. Re-entry rebuilds the position toward full in TRANCHES:

  M1 time_dca   : while risk-off, add (full-defensive)/K of position every D days
  M2 dip_avg    : add a tranche each time price prints a new 10-day low (average down)
  M3 confirm    : add a tranche each day a recovery sign holds
                  (VIX falling OR DIX>60pct OR close>21EMA), capped at full

Risk-off is cleared (stop adding rule resets) once back to full.
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
    dix = pd.read_csv(f"{DATA}/dix_gex.csv", parse_dates=["date"]).set_index("date")["dix"].reindex(c.index).ffill()
    e8, e21, e34 = [c.ewm(span=s, adjust=False).mean() for s in (8, 21, 34)]
    bear = (c <= e8) & (e8 <= e21) & (e21 <= e34)
    exit_sig = bear | (vix > 22)
    new_low = c <= c.rolling(10).min()
    confirm = (vix < vix.shift(3)) | (dix > dix.rolling(252).quantile(0.60)) | (c > e21)
    return c, exit_sig, new_low, confirm


def make_target(exit_sig, new_low, confirm, model, defensive=0.5, full=1.0,
                K=4, D=3):
    pos = np.empty(len(exit_sig))
    state = full
    days_since_add = 0
    step = (full - defensive) / K
    for i in range(len(exit_sig)):
        if exit_sig.iloc[i]:
            state = min(state, defensive)   # trim to defensive (scale-out to floor)
            days_since_add = 0
        elif state < full:
            if model == "single":
                if confirm.iloc[i]:
                    state = full
            elif model == "time_dca":
                days_since_add += 1
                if days_since_add >= D:
                    state = min(full, state + step)
                    days_since_add = 0
            elif model == "dip_avg":
                if new_low.iloc[i]:
                    state = min(full, state + step)
            elif model == "confirm":
                if confirm.iloc[i]:
                    state = min(full, state + step)
        pos[i] = state
    return pd.Series(pos, index=exit_sig.index)


def backtest(target, ret, lag=2, band=0.05, fees=True, fee_u7=0.015, fee_a7=0.005):
    eff = target.shift(lag).bfill()
    port, position = 1.0, float(eff.iloc[0])
    lots = [[position, 0]]
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
        c, exit_sig, new_low, confirm = build(idx_name)
        ret = c.pct_change().loc[c.index >= "2011-06-09"].dropna()
        e, nl, cf = (s.loc[ret.index] for s in (exit_sig, new_low, confirm))
        bh = (1 + ret).cumprod()
        bc, bm, bs = stats(bh, ret)
        print(f"\n=== {idx_name.upper()} ===")
        print(f"{'buy & hold':22s} CAGR={bc:+.1%}  MDD={bm:.1%}  Sharpe={bs:.2f}")
        for model in ["single", "time_dca", "dip_avg", "confirm"]:
            t = make_target(e, nl.fillna(False), cf.fillna(False), model)
            cv, tr = backtest(t, ret)
            cg, md, sh = stats(cv, ret)
            print(f"{model:22s} CAGR={cg:+.1%}  MDD={md:.1%}  Sharpe={sh:.2f}  "
                  f"exΔ={cg-bc:+.2%}  trades={tr}")


if __name__ == "__main__":
    main()
