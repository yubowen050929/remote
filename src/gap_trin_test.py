"""Two claim checks:

1. "Gap direction predicts the close 72% of the time" - decompose the
   mechanical part (gap is already inside the close-to-close move) from
   the informative part (post-open drift, close vs OPEN).
2. TRIN-style divergence days ("advancers outnumber but dollar volume is
   in decliners") - proxied with RSP (equal-weight) vs SPX cap-weight
   daily return divergence, tested for next-day event risk.
"""
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")


def main():
    df = pd.read_csv(os.path.join(DATA, "spx.csv"), index_col=0, parse_dates=True)
    o, c = df["Open"], df["Close"]
    mask = df.index >= "2011-06-09"

    gap_up, gap_dn = o > c.shift(1), o < c.shift(1)
    up_vs_open, up_vs_pc = c > o, c > c.shift(1)
    print("gap-direction claim:")
    print(f"  P(close>prev_close | gap up) = {up_vs_pc[mask & gap_up].mean():.1%}  <- the '72%' (mechanical)")
    print(f"  P(close>OPEN | gap up)       = {up_vs_open[mask & gap_up].mean():.1%}  <- informative part")
    print(f"  P(close<prev_close | gap dn) = {(~up_vs_pc)[mask & gap_dn].mean():.1%}")
    print(f"  P(close<OPEN | gap dn)       = {(~up_vs_open)[mask & gap_dn].mean():.1%}")

    rsp = pd.read_csv(os.path.join(DATA, "rsp.csv"), index_col=0, parse_dates=True)["Close"]
    r_spx, r_rsp = c.pct_change(), rsp.pct_change().reindex(df.index)
    t1 = r_spx.shift(-1) < -0.013
    ok = mask & t1.notna()
    print("\ndivergence days:")
    for name, s in [("SPX dn & RSP up (count up, money down)", (r_spx < 0) & (r_rsp > 0)),
                    ("SPX up & RSP dn (narrow rally)", (r_spx > 0) & (r_rsp < 0))]:
        m = ok & s
        print(f"  {name:40s} P(next-day event)={t1[m].mean():.1%} (base {t1[ok].mean():.1%})")

    dn = ok & (r_spx < 0)
    q = pd.qcut((r_rsp - r_spx)[dn], 5, labels=False)
    tab = t1[dn].groupby(q).mean()
    print("  on down days, by RSP-minus-SPX spread quintile (Q1=avg stock hit hardest):")
    print("    " + "  ".join(f"Q{int(k)+1}={v:.1%}" for k, v in tab.items()))


if __name__ == "__main__":
    main()
