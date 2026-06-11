"""Daily close-of-day risk monitor.

Usage:
    python3 daily_monitor.py            # report from cached data
    python3 daily_monitor.py --refresh  # re-download data first

Outputs, for the latest close:
  1. Model probability of a >1.3% down day tomorrow (logistic trained on
     full history to date, same spec as predict_model.py) + its percentile
     vs the trailing year of model outputs.
  2. The 7-item transparent checklist with current readings.
  3. GEX percentile and the conditional next-session expected range.
  4. B-tier conditional flags (yen carry, macro calendar proximity).

Honest-bounds footer is part of the output by design.
"""
import argparse
import os
import subprocess
import sys
import urllib.request
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from calendar_factors import FOMC, nfp_dates
from precursor_analysis import build_features, load
from predict_model import FEATS

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")


def refresh():
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "download_data.py")],
                   check=True)
    urllib.request.urlretrieve("https://squeezemetrics.com/monitor/static/DIX.csv",
                               os.path.join(DATA, "dix_gex.csv"))


def model_probability(f):
    target = (f["spx_ret"].shift(-1) < -0.013).astype(int)
    ok = f[FEATS].notna().all(axis=1)
    X = f.loc[ok, FEATS]
    y = target[ok]
    scaler = StandardScaler().fit(X.iloc[:-1])
    model = LogisticRegression(max_iter=2000, C=0.5).fit(
        scaler.transform(X.iloc[:-1]), y.iloc[:-1])
    probs = pd.Series(model.predict_proba(scaler.transform(X))[:, 1], index=X.index)
    p_now = probs.iloc[-1]
    pctl = (probs.iloc[-252:] < p_now).mean()
    return p_now, pctl


def checklist(f, spx):
    row = f.iloc[-1]
    ev5 = (f["spx_ret"] < -0.013).iloc[-5:].any()
    today = f.index[-1]
    cal_idx = pd.DatetimeIndex(list(FOMC) + list(nfp_dates(f.index)))
    upcoming = cal_idx[(cal_idx > today) & (cal_idx <= today + pd.Timedelta(days=4))]
    items = [
        ("VIX > 20", row["vix"] > 20, f"VIX={row['vix']:.1f}"),
        ("VIX/VIX3M > 0.95", row["ts_ratio"] > 0.95, f"ratio={row['ts_ratio']:.3f}"),
        ("VIX 5d change > +2", row["vix_chg5"] > 2, f"chg5={row['vix_chg5']:+.1f}"),
        ("below 52w high by >5%", row["dd_52w"] < -0.05, f"dd={row['dd_52w']:+.1%}"),
        ("event within last 5d", bool(ev5), ""),
        ("HYG 20d negative", row["hyg_ret20"] < 0, f"hyg20={row['hyg_ret20']:+.2%}"),
        ("FOMC/NFP within 4 days", len(upcoming) > 0,
         upcoming[0].date().isoformat() if len(upcoming) else ""),
    ]
    return items


def gex_range():
    dg = pd.read_csv(os.path.join(DATA, "dix_gex.csv"), parse_dates=["date"]).set_index("date")
    gexn = dg["gex"] / dg["price"] ** 2
    pctl = float((gexn.iloc[-252:-1] < gexn.iloc[-1]).mean())
    # historical median next-session range conditional on GEX quintile
    spx = pd.read_csv(os.path.join(DATA, "spx.csv"), index_col=0, parse_dates=True)
    rng = ((spx["High"] - spx["Low"]) / spx["Close"].shift(1)).shift(-1)
    gp = gexn.rolling(252).apply(lambda x: (x.iloc[:-1] < x.iloc[-1]).mean()).reindex(spx.index)
    ok = gp.notna() & rng.notna()
    q = pd.qcut(gp[ok], 5, labels=False)
    med = rng[ok].groupby(q).median()
    bucket = int(min(4, pctl * 5))
    return pctl, float(med.iloc[bucket]), float(dg["dix"].iloc[-1])


def b_flags(f):
    jpy = load("usdjpy").reindex(f.index).ffill()
    prox = jpy / jpy.rolling(756).max()
    zone = prox.iloc[-1] > 0.98
    snap = (prox.rolling(60).max().iloc[-1] > 0.97) and (jpy.pct_change(10).iloc[-1] < -0.03)
    return zone, snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if args.refresh:
        refresh()

    f = build_features().dropna(subset=["spx_ret"])
    spx = load("spx")
    asof = f.index[-1].date()
    p, p_pctl = model_probability(f)
    items = checklist(f, spx)
    n_on = sum(1 for _, on, _ in items if on)
    gex_pctl, exp_range, dix = gex_range()
    zone, snap = b_flags(f)

    tier = "正常" if n_on <= 1 else ("观察" if n_on == 2 else "易燃状态")
    print(f"=== 每日风险报告 | 数据截至 {asof} 收盘 ===\n")
    print(f"模型概率 P(明日 SPX < -1.3%) = {p:.1%}  （近一年分位 {p_pctl:.0%}；无条件基准 7.7%）")
    print(f"GEX 1年百分位 = {gex_pctl:.0%}  → 历史同档次日波幅中位 {exp_range:.2%}")
    print(f"DIX = {dix:.3f}\n")
    print(f"清单：{n_on}/7 亮灯 → 【{tier}】")
    for name, on, detail in items:
        print(f"  [{'X' if on else ' '}] {name:26s} {detail}")
    print(f"\nB 级旗标：日元拥挤区={'亮' if zone else '灭'}  日元急回弹={'亮' if snap else '灭'}")
    print("\n--- 诚实边界：本输出预测的是脆弱状态而非具体日期；最严报警精度约 25%，")
    print("    即报警 4 次约错 3 次；触发事件本身不可预测。 ---")


if __name__ == "__main__":
    main()
