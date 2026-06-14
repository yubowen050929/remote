"""McKinsey-style exhibit generator for the flagship report.

Produces a consistent set of publication exhibits under figures/exhibits/.
Each exhibit has a takeaway-style title, muted palette, and a source line.
Reuses cached outputs where possible; recomputes the QDII pieces inline.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(ROOT, "figures", "exhibits")
os.makedirs(FIG, exist_ok=True)

# McKinsey-ish palette
NAVY = "#1f3a5f"
STEEL = "#4a7ba6"
LIGHT = "#9dc3e6"
RED = "#c0392b"
GREEN = "#27ae60"
GREY = "#95a5a6"
GOLD = "#d4a017"
for _cand in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "WenQuanYi Zen Hei Sharp"]:
    if any(_cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_cand]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "figure.dpi": 140, "font.size": 10, "axes.edgecolor": "#cccccc",
    "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
})


def load(n):
    return pd.read_csv(os.path.join(DATA, f"{n}.csv"), index_col=0, parse_dates=True)


def titlebox(fig, kicker, takeaway):
    fig.text(0.02, 0.965, kicker, fontsize=9, color=STEEL, weight="bold")
    fig.text(0.02, 0.915, takeaway, fontsize=12.5, color="#222222", weight="bold")


def source(fig, txt):
    fig.text(0.02, 0.01, txt, fontsize=7.5, color=GREY, style="italic")


# ---------- E1: event landscape ----------
def e1_landscape():
    spx = load("spx")["Close"]
    spx = spx[spx.index >= "2011-06-09"]
    ev = pd.read_csv(os.path.join(OUT, "daily_events_spx.csv"), index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.subplots_adjust(top=0.80, bottom=0.10)
    ax.plot(spx.index, spx, lw=0.8, color=NAVY)
    ax.scatter(ev.index, ev["close"], s=10, color=RED, zorder=3, label=f"单日跌幅>1.3% (n={len(ev)})")
    ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=False)
    for sp in ["2018", "2020", "2022", "2025"]:
        ax.axvspan(pd.Timestamp(f"{sp}-01-01"), pd.Timestamp(f"{sp}-12-31"), color=RED, alpha=0.04)
    titlebox(fig, "EXHIBIT 1 — 回撤地形", "291 次大型回撤高度聚集于少数波动率状态，而非均匀散布")
    source(fig, "数据：Yahoo Finance 日线，2011-06 至 2026-06。阴影为高频年份。")
    fig.savefig(os.path.join(FIG, "E1_landscape.png"))
    plt.close(fig)


# ---------- E2: events per year ----------
def e2_peryear():
    ds = pd.read_csv(os.path.join(OUT, "daily_events_spx.csv"), index_col=0, parse_dates=True)
    dn = pd.read_csv(os.path.join(OUT, "daily_events_ndx.csv"), index_col=0, parse_dates=True)
    ys = ds.groupby(ds.index.year).size()
    yn = dn.groupby(dn.index.year).size()
    yrs = sorted(set(ys.index) | set(yn.index))
    x = np.arange(len(yrs))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.subplots_adjust(top=0.80, bottom=0.12)
    ax.bar(x - 0.2, [ys.get(y, 0) for y in yrs], 0.4, color=NAVY, label="SPX")
    ax.bar(x + 0.2, [yn.get(y, 0) for y in yrs], 0.4, color=STEEL, label="NDX")
    ax.set_xticks(x)
    ax.set_xticklabels(yrs, rotation=0)
    ax.legend(frameon=False)
    ax.axhline(ys.mean(), color=RED, ls="--", lw=0.9)
    ax.text(0, ys.mean() + 1, f"SPX 年均 {ys.mean():.0f}", color=RED, fontsize=8)
    titlebox(fig, "EXHIBIT 2 — 时间分布", "回撤频率年际波动达 14 倍（2017 年 3 次 vs 2022 年 43 次）")
    source(fig, "数据：单日收盘跌幅>1.3% 计数。")
    fig.savefig(os.path.join(FIG, "E2_peryear.png"))
    plt.close(fig)


# ---------- E3: attribution lift ----------
def e3_attrib():
    df = pd.read_csv(os.path.join(OUT, "calendar_attribution.csv"))
    spx = df[df["index"] == "SPX"].set_index("window")
    order = ["event in prior 5 days (self-clustering)", "NFP day", "FOMC T+1..T+3 (post)",
             "FOMC day (T)", "FOMC T-1..T+1", "CPI window (d10-15, approx)",
             "NO event in prior 5 days"]
    labels = ["前5日已有回撤\n(自聚集)", "非农当日", "FOMC后1-3日", "FOMC当日",
              "FOMC前后1日", "CPI窗口(近似)", "前5日无回撤"]
    vals = [spx.loc[o, "lift"] for o in order]
    colors = [RED if v >= 1.5 else (STEEL if v >= 1 else GREEN) for v in vals]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(top=0.80, left=0.22, bottom=0.10)
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(1.0, color="#333", lw=1)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.03, i, f"{v:.2f}×", va="center", fontsize=8.5, color="#333")
    ax.set_xlabel("相对无条件基准(7.7%)的概率提升倍数")
    titlebox(fig, "EXHIBIT 3 — 成因归因", "波动率自聚集(1.9×)主导一切宏观日历因子；CPI窗口甚至无效")
    source(fig, "数据：条件概率 P(回撤|窗口) / 基准。各宏观窗口合计仅解释约 20-25% 事件。")
    fig.savefig(os.path.join(FIG, "E3_attribution.png"))
    plt.close(fig)


# ---------- E4: quintile gradient ----------
def e4_quintiles():
    from precursor_analysis import build_features
    f = build_features()
    f = f[f.index >= "2011-06-09"]
    target = (f["spx_ret"].shift(-1) < -0.013)
    feats = {"vix": "VIX 水平", "ts_ratio": "VIX/VIX3M 期限结构",
             "dd_52w": "距52周高点", "rv21": "21日实现波动率"}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    fig.subplots_adjust(top=0.74, bottom=0.14, wspace=0.35)
    base = target[target.notna()].mean() * 100
    for ax, (col, label) in zip(axes, feats.items()):
        ok = f[col].notna() & target.notna()
        q = pd.qcut(f.loc[ok, col], 5, labels=False, duplicates="drop")
        tab = target[ok].groupby(q).mean() * 100
        bars = ax.bar(range(1, 6), tab.values, color=STEEL)
        bars[tab.values.argmax()].set_color(RED)
        ax.axhline(base, color=GREY, ls="--", lw=0.8)
        ax.set_title(label, fontsize=9.5)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("分位", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("次日回撤概率 %")
    titlebox(fig, "EXHIBIT 4 — 单变量分层力", "状态变量呈陡峭单调梯度：VIX 最高分位 17.4% vs 最低 1.1%（16倍）")
    source(fig, "数据：按 T-1 特征五分位的次日回撤概率。虚线为 7.7% 基准。")
    fig.savefig(os.path.join(FIG, "E4_quintiles.png"))
    plt.close(fig)


# ---------- E5: signal battlefield ----------
def e5_battlefield():
    passed = [("VIX 水平", 17.4/7.7), ("GEX 百分位", 18.8/7.7), ("期限结构倒挂", 16.3/7.7),
              ("EMA空头排列", 18.1/7.7), ("自聚集", 14.7/7.7), ("黄金-SPX利差", 14.2/7.7),
              ("行业成对相关", 12.8/7.7)]
    failed = [("SKEW尾部定价", 0.97), ("十字星", 0.98), ("高位射击之星", 0.69),
              ("ATR网格线", 1.2), ("窄涨日", 0.86), ("跳空方向", 1.0),
              ("股债相关(时点)", 1.0), ("暗池打印", 1.0)]
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.subplots_adjust(top=0.80, left=0.20, bottom=0.08)
    alls = [(n, v, GREEN) for n, v in passed] + [(n, v, RED) for n, v in failed]
    alls.sort(key=lambda t: t[1])
    ax.barh([a[0] for a in alls], [a[1] for a in alls], color=[a[2] for a in alls])
    ax.axvline(1.0, color="#333", lw=1)
    ax.set_xlabel("次日回撤概率提升倍数（1.0=无信息）")
    for i, a in enumerate(alls):
        ax.text(a[1] + 0.03, i, f"{a[1]:.1f}×", va="center", fontsize=8)
    titlebox(fig, "EXHIBIT 5 — 信号战场", "'状态'信号(绿)全部通过；'姿态'信号(红)全部失败或反向")
    source(fig, "数据：13个信号家族的单变量检验。状态=波动率/gamma/趋势；姿态=形态/级别/单笔流。")
    fig.savefig(os.path.join(FIG, "E5_battlefield.png"))
    plt.close(fig)


# ---------- E6: model ceiling ----------
def e6_ceiling():
    variants = ["基线\n12特征", "+GEX", "+COR1M", "+黄金利差", "+成对相关", "+全部四个", "+全部\n(GBM)"]
    auc = [0.712, 0.710, 0.709, 0.710, 0.708, 0.708, 0.691]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    fig.subplots_adjust(top=0.80, bottom=0.14)
    bars = ax.bar(range(len(auc)), auc, color=[NAVY] + [STEEL]*5 + [GREY])
    ax.set_ylim(0.66, 0.73)
    ax.axhline(0.712, color=RED, ls="--", lw=0.9)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, fontsize=8.5)
    ax.set_ylabel("样本外 AUC")
    for i, v in enumerate(auc):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=8.5)
    titlebox(fig, "EXHIBIT 6 — 信息天花板", "已验证新信号的模型边际增益全部为零——信息被12特征组合张成")
    source(fig, "数据：走向式逻辑回归(GBM)，同一对齐样本。单变量有效≠多元边际价值。")
    fig.savefig(os.path.join(FIG, "E6_ceiling.png"))
    plt.close(fig)


# ---------- E7: precision-recall ceiling ----------
def e7_tradeoff():
    p = pd.read_csv(os.path.join(OUT, "oos_predictions_spx.csv"), index_col=0, parse_dates=True)
    y = p["event_next_day"]
    pr = p["p_lr"]
    qs = np.linspace(0.5, 0.99, 30)
    prec, rec = [], []
    for q in qs:
        alarm = pr >= pr.quantile(q)
        prec.append(y[alarm].mean())
        rec.append(y[alarm].sum() / y.sum())
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(top=0.80, bottom=0.12)
    ax.plot(rec, prec, color=NAVY, lw=2)
    ax.axhline(y.mean(), color=GREY, ls="--", lw=0.9, label=f"基准 {y.mean():.1%}")
    ax.scatter([rec[-3]], [prec[-3]], color=RED, zorder=5)
    ax.annotate("最严报警\n精度~25%", (rec[-3], prec[-3]), textcoords="offset points",
                xytext=(30, 0), fontsize=8.5, color=RED)
    ax.set_xlabel("召回率（捕获的事件占比）")
    ax.set_ylabel("精度（报警中真为事件占比）")
    ax.legend(frameon=False)
    titlebox(fig, "EXHIBIT 7 — 可预测性上限", "最严报警精度约25%：报警4次错3次，即公开数据现实上限")
    source(fig, "数据：SPX 走向式样本外预测的精度-召回前沿。")
    fig.savefig(os.path.join(FIG, "E7_tradeoff.png"))
    plt.close(fig)


# ---------- QDII exhibits ----------
def _ladder_backtest():
    c = load("spx")["Close"]
    c = c[c.index >= "2011-06-09"]
    ret = c.pct_change().dropna()
    return c.loc[ret.index], ret


def e8_equity():
    from qdii_roundtrip import build, target_from_state, backtest
    c, exit_sig, reentry = build("spx")
    ret = c.pct_change().loc[c.index >= "2011-06-09"].dropna()
    exit_sig = exit_sig.loc[ret.index]
    rsig = reentry["R5 combo"].loc[ret.index].fillna(False)
    bh = (1 + ret).cumprod()
    t = target_from_state(exit_sig, rsig, defensive=0.5)
    cv, _ = backtest(t, ret)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.subplots_adjust(top=0.80, bottom=0.10)
    ax.plot(bh.index, bh, color=NAVY, lw=1.6, label="买入持有 (满仓)")
    ax.plot(cv.index, cv, color=RED, lw=1.4, label="择时策略 (出场+智能买回, T+2, 含赎回费)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=False)
    titlebox(fig, "EXHIBIT 8 — QDII 择时净值", "最优择时策略在15年里系统性跑输买入持有")
    source(fig, "数据：SPX 代理。择时=EMA空头或VIX>22减仓至50%，VIX见顶/DIX吸筹买回。")
    fig.savefig(os.path.join(FIG, "E8_equity.png"))
    plt.close(fig)


def e9_episodes():
    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.subplots_adjust(top=0.80, bottom=0.12)
    # read from episode_trace by recomputing quickly
    import subprocess, io
    out = subprocess.run(["python3", "episode_trace.py"], cwd=os.path.dirname(__file__),
                         capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 9 and parts[0][:2] in ("20",):
            try:
                exit_px = float(parts[3]); reentry_px = float(parts[5])
                rows.append((exit_px, reentry_px))
            except ValueError:
                pass
    ex = np.array([r[0] for r in rows]); re = np.array([r[1] for r in rows])
    lim = [min(ex.min(), re.min())*0.9, max(ex.max(), re.max())*1.05]
    ax.plot(lim, lim, color=GREY, ls="--", lw=1, label="买回=卖出价")
    ax.scatter(ex, re, s=45, color=RED, zorder=4, edgecolor="white")
    ax.fill_between(lim, lim, [lim[1], lim[1]], color=RED, alpha=0.05)
    ax.text(lim[0]*1.05, lim[1]*0.95, "买回价 > 卖出价\n(卖低买高)", color=RED, fontsize=9)
    ax.set_xlabel("出场价位"); ax.set_ylabel("买回价位")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(frameon=False, loc="lower right")
    titlebox(fig, "EXHIBIT 9 — 卖低买高机制", "17次择时往返中16次在更高价位买回——回撤是纸面的，溢价是真金白银")
    source(fig, "数据：VIX>28出场/VIX<22买回(含T+2)的逐次往返。点在对角线上方=卖低买高。")
    fig.savefig(os.path.join(FIG, "E9_episodes.png"))
    plt.close(fig)


def e10_decomposition():
    bases = [0.60, 0.70, 0.80, 0.90, 0.95]
    drag = [4.65, 3.71*0.9, 2.3, 1.16, 0.58]   # approx from runs; recompute precisely
    # recompute precisely
    from ladder_2d import ladder, backtest, load as l2
    c = l2("spx")["Close"]; c = c[c.index >= "2011-06-09"]
    ret = c.pct_change().dropna(); cc = c.loc[ret.index]; Er = ret.mean()
    drag, cov = [], []
    for b in bases:
        t = ladder(cc, 0.05, b)
        _, eff = backtest(t, ret)
        w = eff.loc[ret.index]
        drag.append((1 - w.mean()) * Er * 252 * 100)
        cov.append(np.cov(w, ret)[0, 1] * 252 * 100)
    x = np.arange(len(bases))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    fig.subplots_adjust(top=0.78, bottom=0.12)
    ax.bar(x - 0.2, drag, 0.4, color=RED, label="欠配拖累 (1−E[w])·漂移")
    ax.bar(x + 0.2, cov, 0.4, color=GREEN, label="低吸择时价值 Cov(w,r)")
    ax.plot(x, np.array(cov) - np.array(drag), color=NAVY, marker="o", lw=1.5, label="净 (=超额)")
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"底仓{int(b*100)}%" for b in bases])
    ax.set_ylabel("年化贡献 %")
    ax.legend(frameon=False, fontsize=9)
    titlebox(fig, "EXHIBIT 10 — 数学的墙", "任何底仓水平，欠配拖累(红)恒大于择时价值(绿)，净恒为负")
    source(fig, "数据：E[w·r]=E[w]E[r]+Cov(w,r) 分解。底仓→100%时两项同趋于零，收敛到买入持有。")
    fig.savefig(os.path.join(FIG, "E10_decomposition.png"))
    plt.close(fig)


def e11_frontier():
    from qdii_roundtrip import build, target_from_state, backtest
    c, exit_sig, reentry = build("spx")
    ret = c.pct_change().loc[c.index >= "2011-06-09"].dropna()
    exit_sig = exit_sig.loc[ret.index]
    rsig = reentry["R5 combo"].loc[ret.index].fillna(False)
    bh = (1 + ret).cumprod()
    yrs = len(ret) / 252
    bc = bh.iloc[-1] ** (1/yrs) - 1
    bmdd = (bh/bh.cummax()-1).min()
    pts = [("买入持有", bc, bmdd, NAVY)]
    for defw, lab in [(0.7, "防守70%"), (0.5, "防守50%"), (0.3, "防守30%"), (0.0, "防守0%")]:
        t = target_from_state(exit_sig, rsig, defensive=defw)
        cv, _ = backtest(t, ret)
        cg = cv.iloc[-1]**(1/yrs)-1
        md = (cv/cv.cummax()-1).min()
        pts.append((lab, cg, md, STEEL))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.subplots_adjust(top=0.78, bottom=0.12)
    for lab, cg, md, col in pts:
        ax.scatter(-md*100, cg*100, s=80, color=col, zorder=4, edgecolor="white")
        ax.annotate(lab, (-md*100, cg*100), textcoords="offset points", xytext=(8, 4), fontsize=8.5)
    ax.set_xlabel("最大回撤 %（越左越小）")
    ax.set_ylabel("年化收益 %")
    ax.invert_xaxis()
    titlebox(fig, "EXHIBIT 11 — 保险定价前沿", "择时本质是回撤保险：每削减回撤须以放弃年化收益为代价")
    source(fig, "数据：不同防守仓位的(收益,回撤)坐标。无任何点同时优于买入持有的收益与回撤。")
    fig.savefig(os.path.join(FIG, "E11_frontier.png"))
    plt.close(fig)


if __name__ == "__main__":
    e1_landscape(); print("E1 done")
    e2_peryear(); print("E2 done")
    e3_attrib(); print("E3 done")
    e4_quintiles(); print("E4 done")
    e5_battlefield(); print("E5 done")
    e6_ceiling(); print("E6 done")
    e7_tradeoff(); print("E7 done")
    e8_equity(); print("E8 done")
    e9_episodes(); print("E9 done")
    e10_decomposition(); print("E10 done")
    e11_frontier(); print("E11 done")
    print("exhibits:", sorted(os.listdir(FIG)))
