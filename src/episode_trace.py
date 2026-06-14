"""Trace exactly what a strict timing rule DOES during each big drawdown.

Strict rule:
  EXIT  when VIX crosses above 28 (a high-conviction panic threshold)
  RE-ENTER when VIX falls back below 22 (panic subsiding)
  T+2 execution lag on both.

For each episode it records: exit price, the lowest close while out, the
re-entry price, and the round-trip vs just holding over the same window.
This makes visible WHERE the rule sells and WHERE it buys back.
"""
import numpy as np
import pandas as pd

DATA = "../data"


def load(n):
    return pd.read_csv(f"{DATA}/{n}.csv", index_col=0, parse_dates=True)


def main():
    c = load("spx")["Close"]
    vix = load("vix")["Close"].reindex(c.index).ffill()
    c = c[c.index >= "2011-06-09"]
    vix = vix.loc[c.index]

    # build exit/re-entry state with T+2 lag
    raw = pd.Series(index=c.index, dtype=float)
    state = 1  # 1 = in, 0 = out
    states = []
    for i in range(len(c)):
        if state == 1 and vix.iloc[i] > 28:
            state = 0
        elif state == 0 and vix.iloc[i] < 22:
            state = 1
        states.append(state)
    raw = pd.Series(states, index=c.index)
    eff = raw.shift(2).fillna(1)  # T+2

    # find each out-episode
    out = eff == 0
    episodes = []
    i = 0
    n = len(c)
    while i < n:
        if out.iloc[i]:
            j = i
            while j < n and out.iloc[j]:
                j += 1
            seg = c.iloc[i:j]
            exit_px = c.iloc[i - 1] if i > 0 else c.iloc[i]
            reentry_px = c.iloc[j] if j < n else c.iloc[-1]
            low_px = seg.min()
            # hold return over the same window (exit day to re-entry day)
            hold = reentry_px / exit_px - 1
            roundtrip = 0.0  # out of market = 0% return on that sleeve
            episodes.append({
                "out_start": c.index[i].date(),
                "out_end": (c.index[j].date() if j < n else c.index[-1].date()),
                "days_out": j - i,
                "exit_px": round(exit_px, 0),
                "low_while_out": round(low_px, 0),
                "reentry_px": round(reentry_px, 0),
                "draw_avoided_to_low": round(low_px / exit_px - 1, 3),
                "reentry_vs_exit": round(reentry_px / exit_px - 1, 3),
                "hold_would_be": round(hold, 3),
                "timing_edge": round(0 - hold, 3),  # being out vs holding
            })
            i = j
        else:
            i += 1

    df = pd.DataFrame(episodes)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print()
    wins = (df["timing_edge"] > 0).sum()
    print(f"episodes where being OUT beat holding: {wins}/{len(df)}")
    print(f"sum of timing edge across all episodes: {df['timing_edge'].sum():+.1%}")
    print(f"of exits, how many re-entered HIGHER than they exited "
          f"(sold low, bought high): {(df['reentry_vs_exit']>0).sum()}/{len(df)}")
    print(f"median drawdown actually avoided (exit to lowest): "
          f"{df['draw_avoided_to_low'].median():.1%}")


if __name__ == "__main__":
    main()
