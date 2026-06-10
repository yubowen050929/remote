"""Tier-1 free data acquisition.

Downloads:
  - CBOE index histories: COR1M, COR3M, DSPX (implied correlation, dispersion)
  - CBOE VX futures, contract by contract, and builds a continuous
    front-month / second-month term-structure series (vx_curve.csv)
  - VXN / VIX9D are handled in download_data-style yfinance calls elsewhere

FRED series (NFCI, HY OAS, JGB10) are fetched separately via fredgraph CSV;
this environment truncates some daily FRED series - documented limitation.

VX expiration rule: the Wednesday 30 days before the 3rd Friday of the
following month (CBOE standard). On 404 we probe +/- a few days for
holiday-shifted contracts.
"""
import os
import time
import urllib.request

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")

CBOE_IDX = {
    "COR1M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/COR1M_History.csv",
    "COR3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/COR3M_History.csv",
    "DSPX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/DSPX_History.csv",
}
VX_URL = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{date}.csv"


def fetch(url, path):
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except Exception:
        return False


def third_friday(year, month):
    d = pd.Timestamp(year, month, 1)
    fridays = pd.date_range(d, d + pd.offsets.MonthEnd(0), freq="W-FRI")
    return fridays[2]


def vx_expiry(year, month):
    """Settlement of the contract expiring in (year, month)."""
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    return third_friday(ny, nm) - pd.Timedelta(days=30)


def download_cboe_indices():
    for name, url in CBOE_IDX.items():
        path = os.path.join(DATA, f"{name.lower()}.csv")
        if fetch(url, path):
            df = pd.read_csv(path)
            print(f"{name}: {len(df)} rows, {df.iloc[0, 0]} .. {df.iloc[-1, 0]}")


def download_vx_contracts():
    vxdir = os.path.join(DATA, "vx_contracts")
    os.makedirs(vxdir, exist_ok=True)
    got = []
    for year in range(2011, 2027):
        for month in range(1, 13):
            base = vx_expiry(year, month)
            path = os.path.join(vxdir, f"VX_{year}-{month:02d}.csv")
            if os.path.exists(path):
                got.append((year, month, path))
                continue
            ok = False
            for shift in [0, -1, 1, -2, 2, -7, 7]:
                d = (base + pd.Timedelta(days=shift)).strftime("%Y-%m-%d")
                if fetch(VX_URL.format(date=d), path):
                    try:
                        if len(pd.read_csv(path)) > 5:
                            ok = True
                            break
                    except Exception:
                        pass
            if ok:
                got.append((year, month, path))
            time.sleep(0.2)
    print(f"VX contracts downloaded: {len(got)}")
    return got


def build_vx_curve(contracts):
    frames = []
    for year, month, path in contracts:
        df = pd.read_csv(path)
        df["Trade Date"] = pd.to_datetime(df["Trade Date"])
        df = df[df["Settle"] > 0][["Trade Date", "Settle"]]
        df["expiry"] = vx_expiry(year, month)
        frames.append(df)
    allc = pd.concat(frames)
    out = []
    for date, g in allc.groupby("Trade Date"):
        g = g[g["expiry"] > date].sort_values("expiry")
        if len(g) >= 2:
            f1, f2 = g["Settle"].iloc[0], g["Settle"].iloc[1]
            dte1 = (g["expiry"].iloc[0] - date).days
            out.append({"date": date, "f1": f1, "f2": f2,
                        "f2_f1": f2 / f1, "f1_dte": dte1})
    curve = pd.DataFrame(out).set_index("date").sort_index()
    curve.to_csv(os.path.join(DATA, "vx_curve.csv"))
    print(f"vx_curve.csv: {len(curve)} days, {curve.index.min().date()} .. {curve.index.max().date()}")
    print(f"  backwardation (f2/f1<1) share: {(curve['f2_f1'] < 1).mean():.1%}")


if __name__ == "__main__":
    download_cboe_indices()
    contracts = download_vx_contracts()
    build_vx_curve(contracts)
