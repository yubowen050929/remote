"""Detect large drawdown events in SPX and NDX.

Definitions (per user request):
  - daily event : close-to-close return < -1.3%
  - weekly event: Friday-to-Friday (W-FRI resample) return < -3%

Outputs:
  output/daily_events_{spx,ndx}.csv
  output/weekly_events_{spx,ndx}.csv
  output/event_summary.txt
"""
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

DAILY_TH = -0.013
WEEKLY_TH = -0.03
START = "2011-06-09"  # exactly 15 years back from 2026-06-09


def load_close(name: str) -> pd.Series:
    df = pd.read_csv(os.path.join(DATA, f"{name}.csv"), index_col=0, parse_dates=True)
    return df["Close"].astype(float)


def detect(name: str):
    close = load_close(name)
    ret = close.pct_change()
    ret = ret[ret.index >= START]

    daily = pd.DataFrame({"close": close, "ret": ret})
    daily_events = daily[daily["ret"] < DAILY_TH].copy()

    wk_close = close.resample("W-FRI").last()
    wk_ret = wk_close.pct_change()
    wk_ret = wk_ret[wk_ret.index >= START]
    weekly = pd.DataFrame({"close": wk_close, "ret": wk_ret}).dropna()
    weekly_events = weekly[weekly["ret"] < WEEKLY_TH].copy()

    return daily_events, weekly_events, ret, wk_ret


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    for name in ["spx", "ndx"]:
        d, w, ret, wret = detect(name)
        d.to_csv(os.path.join(OUT, f"daily_events_{name}.csv"))
        w.to_csv(os.path.join(OUT, f"weekly_events_{name}.csv"))
        n_days = ret.notna().sum()
        n_weeks = wret.notna().sum()
        lines += [
            f"=== {name.upper()} ===",
            f"trading days: {n_days}, weeks: {n_weeks}",
            f"daily events (<{DAILY_TH:.1%}): {len(d)}  base rate={len(d)/n_days:.2%}",
            f"weekly events (<{WEEKLY_TH:.1%}): {len(w)}  base rate={len(w)/n_weeks:.2%}",
            f"worst day : {ret.idxmin().date()}  {ret.min():.2%}",
            f"worst week: {wret.idxmin().date()}  {wret.min():.2%}",
            "",
        ]
    text = "\n".join(lines)
    with open(os.path.join(OUT, "event_summary.txt"), "w") as f:
        f.write(text)
    print(text)

    # yearly distribution of daily events
    for name in ["spx", "ndx"]:
        d = pd.read_csv(os.path.join(OUT, f"daily_events_{name}.csv"), index_col=0, parse_dates=True)
        print(f"{name.upper()} daily events per year:")
        print(d.groupby(d.index.year).size().to_string())
        print()


if __name__ == "__main__":
    main()
