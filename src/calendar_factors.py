"""Macro-calendar attribution: do large drawdowns cluster around scheduled
macro events (FOMC decisions, NFP, CPI releases)?

Method: for each calendar type, compute P(big drop | day in event window)
vs unconditional base rate, and the share of all events that fall in each
window. FOMC dates are hardcoded (decision days). NFP = first Friday of
month (8:30 ET release, standard schedule). CPI uses the BLS-typical
release window (weekday, day-of-month 10..15) - an approximation, flagged
as such in the report.

Also computes self-clustering: P(event | event within previous 5 days).
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data")

# FOMC decision days (2nd day of scheduled meetings + emergency actions)
FOMC = [
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09",
    "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01",
    "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31",
    "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30",
    "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29",
    "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27",
    "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26",
    "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01",
    "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31",
    "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (incl. emergency cuts 3/3 and 3/15)
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (scheduled)
    "2026-01-28", "2026-03-18", "2026-04-29",
]
FOMC = pd.to_datetime(FOMC)


def nfp_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First Friday of each month within the sample."""
    dates = []
    for year in range(index.min().year, index.max().year + 1):
        for month in range(1, 13):
            d = pd.Timestamp(year=year, month=month, day=1)
            while d.dayofweek != 4:
                d += pd.Timedelta(days=1)
            dates.append(d)
    return pd.DatetimeIndex(dates)


def in_window(index: pd.DatetimeIndex, events: pd.DatetimeIndex, before: int, after: int) -> pd.Series:
    """Boolean: trading day is within [-before, +after] *trading days* of an event."""
    pos = pd.Series(np.arange(len(index)), index=index)
    mask = pd.Series(False, index=index)
    for e in events:
        # map event to nearest trading day at-or-after
        loc = index.searchsorted(e)
        if loc >= len(index):
            continue
        lo, hi = max(0, loc - before), min(len(index) - 1, loc + after)
        mask.iloc[lo:hi + 1] = True
    return mask


def cpi_window(index: pd.DatetimeIndex) -> pd.Series:
    """Approximate CPI release window: weekday with day-of-month in 10..15."""
    return pd.Series((index.day >= 10) & (index.day <= 15), index=index)


def main():
    results = []
    for name in ["spx", "ndx"]:
        close = pd.read_csv(os.path.join(DATA, f"{name}.csv"), index_col=0, parse_dates=True)["Close"]
        ret = close.pct_change()
        ret = ret[ret.index >= "2011-06-09"].dropna()
        idx = ret.index
        is_event = ret < -0.013
        base = is_event.mean()

        windows = {
            "FOMC day (T)": in_window(idx, FOMC, 0, 0),
            "FOMC T-1..T+1": in_window(idx, FOMC, 1, 1),
            "FOMC T+1..T+3 (post)": in_window(idx, FOMC, -1, 3) & ~in_window(idx, FOMC, 1, 0),
            "NFP day": in_window(idx, nfp_dates(idx), 0, 0),
            "NFP T..T+1": in_window(idx, nfp_dates(idx), 0, 1),
            "CPI window (d10-15, approx)": cpi_window(idx),
        }
        # post-FOMC needs care: define T+1..T+3 strictly after decision
        pos = pd.Series(np.arange(len(idx)), index=idx)
        post = pd.Series(False, index=idx)
        for e in FOMC:
            loc = idx.searchsorted(e)
            if loc >= len(idx):
                continue
            post.iloc[loc + 1: min(len(idx), loc + 4)] = True
        windows["FOMC T+1..T+3 (post)"] = post

        for label, mask in windows.items():
            n = int(mask.sum())
            if n == 0:
                continue
            p = is_event[mask].mean()
            share = is_event[mask].sum() / is_event.sum()
            results.append({
                "index": name.upper(), "window": label, "days_in_window": n,
                "P(event|window)": round(float(p), 4),
                "base_rate": round(float(base), 4),
                "lift": round(float(p / base), 2),
                "share_of_all_events": round(float(share), 3),
            })

        # self-clustering
        ev_prev5 = is_event.rolling(5).sum().shift(1).fillna(0) > 0
        p_cluster = is_event[ev_prev5].mean()
        p_calm = is_event[~ev_prev5].mean()
        results.append({
            "index": name.upper(), "window": "event in prior 5 days (self-clustering)",
            "days_in_window": int(ev_prev5.sum()),
            "P(event|window)": round(float(p_cluster), 4),
            "base_rate": round(float(base), 4),
            "lift": round(float(p_cluster / base), 2),
            "share_of_all_events": round(float(is_event[ev_prev5].sum() / is_event.sum()), 3),
        })
        results.append({
            "index": name.upper(), "window": "NO event in prior 5 days",
            "days_in_window": int((~ev_prev5).sum()),
            "P(event|window)": round(float(p_calm), 4),
            "base_rate": round(float(base), 4),
            "lift": round(float(p_calm / base), 2),
            "share_of_all_events": round(float(is_event[~ev_prev5].sum() / is_event.sum()), 3),
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, "calendar_attribution.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
