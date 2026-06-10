"""CFTC Commitments of Traders (Traders in Financial Futures, futures-only).

Downloads annual archives 2011-2026 and extracts weekly net positioning of
Asset Managers and Leveraged Funds in ES, NQ and VIX futures.
Output: data/cot_positioning.csv
"""
import io
import os
import urllib.request
import zipfile

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"

MARKETS = {
    "E-MINI S&P 500": "es",
    "NASDAQ-100 CONSOLIDATED": "nq",
    "NASDAQ MINI": "nq_mini",
    "VIX FUTURES": "vix",
}


def main():
    frames = []
    for year in range(2011, 2027):
        try:
            req = urllib.request.Request(URL.format(year=year),
                                         headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=60).read()
        except Exception as e:
            print(f"{year}: FAIL {e}")
            continue
        zf = zipfile.ZipFile(io.BytesIO(raw))
        name = zf.namelist()[0]
        df = pd.read_csv(io.BytesIO(zf.read(name)), low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        keep = df[df["Market_and_Exchange_Names"].str.contains(
            "|".join(MARKETS), case=False, na=False)].copy()
        frames.append(keep)
        print(f"{year}: {len(keep)} rows")

    allr = pd.concat(frames)
    allr["date"] = pd.to_datetime(allr["Report_Date_as_YYYY-MM-DD"])
    out = {}
    for key, tag in MARKETS.items():
        sub = allr[allr["Market_and_Exchange_Names"].str.contains(key, case=False)]
        if sub.empty:
            continue
        g = sub.groupby("date").first()
        out[f"{tag}_am_net"] = g["Asset_Mgr_Positions_Long_All"] - g["Asset_Mgr_Positions_Short_All"]
        out[f"{tag}_lev_net"] = g["Lev_Money_Positions_Long_All"] - g["Lev_Money_Positions_Short_All"]
    res = pd.DataFrame(out).sort_index()
    res.to_csv(os.path.join(DATA, "cot_positioning.csv"))
    print(f"cot_positioning.csv: {len(res)} weeks, {res.index.min().date()} .. {res.index.max().date()}")
    print(res.tail(3).to_string())


if __name__ == "__main__":
    main()
