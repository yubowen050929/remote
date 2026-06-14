"""QDII feasibility backtest (the decisive pre-build test).

Question this answers BEFORE building GPT's 15-module system:
  Does our validated regime signal, once degraded by T+2 settlement lag and
  1.5%/0.5% redemption fees with real FIFO lot accounting, still beat
  buy-and-hold net of costs?

Strategy: a simple, fully explainable regime-aware target position computed
at the close of day T (available before China 15:00 next day), no lookahead:
  - EMA 8>21>34 bullish stack and VIX<=22  -> 100%
  - EMA bearish stack or VIX>22            -> 50%
  - otherwise                              -> 70%

Execution realism:
  - position_effective_lag (T+2 default): signal at T affects holdings at T+lag
  - FIFO lot ledger; redemption fee 1.5% if lot age <7 calendar-ish days else 0.5%
  - no-trade band so tiny signal changes don't churn
Compared against buy-and-hold (always 100%) and fixed 70%.
"""
import numpy as np
import pandas as pd

DATA = "../data"


def load(n):
    return pd.read_csv(f"{DATA}/{n}.csv", index_col=0, parse_dates=True)


def regime_target(c, vix):
    e8, e21, e34 = [c.ewm(span=s, adjust=False).mean() for s in (8, 21, 34)]
    bull = (c >= e8) & (e8 >= e21) & (e21 >= e34)
    bear = (c <= e8) & (e8 <= e21) & (e21 <= e34)
    pos = pd.Series(0.7, index=c.index)
    pos[bull & (vix <= 22)] = 1.0
    pos[bear | (vix > 22)] = 0.5
    return pos


def backtest(target, ret, lag=2, band=0.10, fees=True,
             fee_u7=0.015, fee_a7=0.005):
    """Lot-based backtest. target/ret aligned, ret has no NaN."""
    eff = target.shift(lag).bfill()
    port = 1.0
    position = float(eff.iloc[0])
    lots = [[position, 0]]          # [fraction_of_portfolio_in_this_lot, age]
    fees_paid = 0.0
    trades = sub7 = 0
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
                position = tgt
                trades += 1
            else:
                to_sell = -delta
                while to_sell > 1e-9 and lots:
                    lot = lots[0]
                    take = min(lot[0], to_sell)
                    if fees:
                        rate = fee_u7 if lot[1] < 7 else fee_a7
                        fees_paid += take * rate
                        port *= (1 - 0)  # fee applied to redeemed notional below
                        port -= take * rate * port
                        if lot[1] < 7:
                            sub7 += 1
                    lot[0] -= take
                    to_sell -= take
                    if lot[0] <= 1e-9:
                        lots.pop(0)
                position = tgt
                trades += 1
        curve.append(port)
    return pd.Series(curve, index=ret.index), fees_paid, trades, sub7


def stats(curve, ret):
    yrs = len(ret) / 252
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    mdd = (curve / curve.cummax() - 1).min()
    daily = curve.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252)
    return cagr, mdd, sharpe


def main():
    for idx_name in ["spx", "ndx"]:
        c = load(idx_name)["Close"]
        vix = load("vxn" if idx_name == "ndx" else "vix")["Close"].reindex(c.index).ffill()
        target = regime_target(c, vix)
        ret = c.pct_change().loc[c.index >= "2011-06-09"].dropna()
        target = target.loc[ret.index]

        bh = (1 + ret).cumprod()
        fixed70 = (1 + 0.7 * ret).cumprod()
        print(f"\n=== {idx_name.upper()} ===")
        bc, bm, bs = stats(bh, ret)
        print(f"{'buy & hold':28s} CAGR={bc:+.1%}  MDD={bm:.1%}  Sharpe={bs:.2f}")
        fc, fm, fs = stats(fixed70, ret)
        print(f"{'fixed 70%':28s} CAGR={fc:+.1%}  MDD={fm:.1%}  Sharpe={fs:.2f}")
        for lag in [0, 2]:
            for fees in [False, True]:
                curve, fp, tr, s7 = backtest(target, ret, lag=lag, fees=fees)
                cg, md, sh = stats(curve, ret)
                tag = f"regime lag={lag} fee={'ON ' if fees else 'OFF'}"
                print(f"{tag:28s} CAGR={cg:+.1%}  MDD={md:.1%}  Sharpe={sh:.2f}  "
                      f"excessCAGR={cg-bc:+.2%}  trades={tr} sub7={s7} totfee={fp:.2f}")


if __name__ == "__main__":
    main()
