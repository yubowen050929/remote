"""Generate the report figures."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from precursor_analysis import build_features

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(ROOT, "figures")

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def load_close(name):
    return pd.read_csv(os.path.join(DATA, f"{name}.csv"), index_col=0, parse_dates=True)["Close"]


def fig_timeline():
    spx = load_close("spx")
    vix = load_close("vix")
    spx = spx[spx.index >= "2011-06-09"]
    vix = vix[vix.index >= "2011-06-09"]
    ev = pd.read_csv(os.path.join(OUT, "daily_events_spx.csv"), index_col=0, parse_dates=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(spx.index, spx, lw=0.7, color="navy")
    ax1.scatter(ev.index, ev["close"], s=8, color="red", zorder=3, label=f"daily drop < -1.3% (n={len(ev)})")
    ax1.set_yscale("log")
    ax1.set_title("SPX 2011-2026: large single-day drawdowns cluster in volatility regimes")
    ax1.legend(loc="upper left")
    ax2.plot(vix.index, vix, lw=0.6, color="darkorange")
    ax2.axhline(20, color="gray", ls="--", lw=0.8)
    ax2.set_ylabel("VIX")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "timeline_events.png"))
    plt.close(fig)


def fig_quintiles():
    f = build_features()
    f = f[f.index >= "2011-06-09"]
    target = (f["spx_ret"].shift(-1) < -0.013)
    feats = {"vix": "VIX level", "ts_ratio": "VIX/VIX3M (term structure)",
             "rv21": "21d realized vol", "dd_52w": "distance from 52w high"}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, (col, label) in zip(axes, feats.items()):
        ok = f[col].notna() & target.notna()
        q = pd.qcut(f.loc[ok, col], 5, labels=False, duplicates="drop")
        tab = target[ok].groupby(q).mean() * 100
        ax.bar([f"Q{int(i)+1}" for i in tab.index], tab.values, color="steelblue")
        ax.axhline(target[ok].mean() * 100, color="red", ls="--", lw=0.8, label="base rate")
        ax.set_title(label, fontsize=8)
        ax.set_ylabel("P(next-day drop) %")
        ax.legend(fontsize=6)
    fig.suptitle("SPX: probability of a >1.3% down day tomorrow, by today's regime quintile")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "quintile_event_rates.png"))
    plt.close(fig)


def fig_oos():
    p = pd.read_csv(os.path.join(OUT, "oos_predictions_spx.csv"), index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(p.index, p["p_lr"], lw=0.5, color="steelblue", label="walk-forward P(event tomorrow), logistic")
    evt = p[p["event_next_day"] == 1]
    ax.scatter(evt.index, evt["p_lr"], s=8, color="red", zorder=3, label="actual event next day")
    ax.axhline(p["p_lr"].quantile(0.95), color="gray", ls="--", lw=0.8, label="top-5% alarm threshold")
    ax.set_title("SPX out-of-sample one-day-ahead probabilities (no lookahead): events are flagged but never certain")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "oos_probability.png"))
    plt.close(fig)


def fig_calendar():
    df = pd.read_csv(os.path.join(OUT, "calendar_attribution.csv"))
    spx = df[df["index"] == "SPX"].set_index("window")
    order = ["FOMC day (T)", "FOMC T-1..T+1", "FOMC T+1..T+3 (post)", "NFP day",
             "NFP T..T+1", "CPI window (d10-15, approx)",
             "event in prior 5 days (self-clustering)", "NO event in prior 5 days"]
    spx = spx.loc[order]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = ["steelblue"] * 6 + ["firebrick", "seagreen"]
    ax.barh(range(len(spx)), spx["lift"], color=colors)
    ax.set_yticks(range(len(spx)))
    ax.set_yticklabels(spx.index, fontsize=8)
    ax.axvline(1.0, color="black", lw=0.8)
    ax.set_xlabel("lift vs unconditional base rate (7.7%)")
    ax.set_title("SPX: what raises the odds of a >1.3% down day?\nVolatility clustering dominates the macro calendar")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "calendar_lift.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_timeline()
    fig_quintiles()
    fig_oos()
    fig_calendar()
    print("figures written:", os.listdir(FIG))
