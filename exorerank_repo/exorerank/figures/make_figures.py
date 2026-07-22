#!/usr/bin/env python3
"""
make_figures.py — Publication figures from the benchmark summary spreadsheets.

Reads the per-condition ranking summaries (produced by analysis/rank_summary_*.py)
and renders publication-ready figures at 300 dpi:
  • Rank@1 grouped bars (RAG vs No-RAG)
  • Stacked rank-distribution bars
  • Cost-efficiency scatter (Rank@1 vs API cost)
  • RAG vs No-RAG slopegraph
  • A composite 4-panel main figure

Edit the DATA block below (or wire it to your .xlsx files) then run:
    python make_figures.py
Outputs are written to ./figures_out/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "figures_out"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#444", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#E0E0E0", "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

# ── DATA (measured, with-HPO cohort) ─────────────────────────────────────────
MODELS = ["llama-3.1-8b", "Llama-3.3-70B", "qwen3-32b", "gpt-oss-120b",
          "gpt-oss-20b", "llama-4-scout", "GPT-4.1", "kimi-k2"]
N_RAG = N_NORAG = 158
N_EXO = 172
# [Rank1, 2-5, 6-10, 11-50, >50, gene_found]
RAG = {"llama-3.1-8b":[88,23,16,24,1,152],"Llama-3.3-70B":[91,39,15,7,1,153],
       "qwen3-32b":[91,41,12,8,1,153],"gpt-oss-120b":[118,26,6,4,1,155],
       "gpt-oss-20b":[72,15,2,2,1,92],"llama-4-scout":[109,23,16,6,0,150],
       "GPT-4.1":[118,16,9,3,0,146],"kimi-k2":[128,19,6,5,0,158]}
NORAG = {"llama-3.1-8b":[79,6,3,4,0,92],"Llama-3.3-70B":[36,3,1,4,0,44],
         "qwen3-32b":[27,18,18,55,6,124],"gpt-oss-120b":[60,13,10,27,9,109],
         "gpt-oss-20b":[15,7,14,20,3,44],"llama-4-scout":[63,6,0,2,0,71],
         "GPT-4.1":[70,12,17,27,4,130],"kimi-k2":[75,13,9,6,0,103]}
EXO = [70,11,18,49,18,166]
COST = {"llama-3.1-8b":0.14,"Llama-3.3-70B":1.63,"qwen3-32b":0.88,"gpt-oss-120b":0.63,
        "gpt-oss-20b":0.31,"llama-4-scout":0.31,"GPT-4.1":6.00,"kimi-k2":2.74}
OPEN = {"llama-3.1-8b":1,"Llama-3.3-70B":1,"qwen3-32b":1,"gpt-oss-120b":1,
        "gpt-oss-20b":1,"llama-4-scout":1,"GPT-4.1":0,"kimi-k2":1}
MCOLORS = {"llama-3.1-8b":"#5B8FF9","Llama-3.3-70B":"#3A6FD8","qwen3-32b":"#9C6ADE",
           "gpt-oss-120b":"#17A589","gpt-oss-20b":"#5DB9A8","llama-4-scout":"#E67E22",
           "GPT-4.1":"#C0392B","kimi-k2":"#16A085"}
BIN_COLORS = ["#1A7A3C","#73C66B","#FBC02D","#F57C00","#C62828","#9E9E9E"]
BIN_LABELS = ["Rank 1","Ranks 2-5","Ranks 6-10","Ranks 11-50","Beyond 50","Not found"]


def fig_rank1(path):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(MODELS)); w = 0.38
    a = [NORAG[m][0]/N_NORAG*100 for m in MODELS]
    b = [RAG[m][0]/N_RAG*100 for m in MODELS]
    ax.bar(x-w/2, a, w, label="No-RAG (HPO)", color="#B0BEC5", edgecolor="#607D8B")
    ax.bar(x+w/2, b, w, label="RAG (HPO)", color="#1A7A3C", edgecolor="#0E5226")
    exo = EXO[0]/N_EXO*100
    ax.axhline(exo, ls="--", lw=1.5, color="#C62828")
    ax.text(len(MODELS)-0.5, exo+1.5, f"Exomiser baseline ({exo:.1f}%)",
            ha="right", fontsize=8.5, color="#C62828", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Rank@1 rate (%)"); ax.set_ylim(0, 90)
    ax.set_title("Rank@1 performance: effect of RAG per model", fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()


def fig_stacked(data, n, title, path):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(MODELS))
    mat = np.array([data[m][:5]+[n-data[m][5]] for m in MODELS], float)/n*100
    bottom = np.zeros(len(MODELS))
    for i in range(6):
        ax.bar(x, mat[:, i], bottom=bottom, color=BIN_COLORS[i], edgecolor="white", label=BIN_LABELS[i])
        bottom += mat[:, i]
    ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Cases (%)"); ax.set_ylim(0, 100)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    ax.grid(axis="x", visible=False)
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()


def fig_cost(path):
    fig, ax = plt.subplots(figsize=(8.4, 6))
    for m in MODELS:
        ax.scatter(COST[m], RAG[m][0]/N_RAG*100, s=180, color=MCOLORS[m],
                   marker="o" if OPEN[m] else "s", edgecolor="#222", zorder=5)
        ax.annotate(m, (COST[m], RAG[m][0]/N_RAG*100), xytext=(7, 5),
                    textcoords="offset points", fontsize=8.5, fontweight="bold")
    ax.axhline(EXO[0]/N_EXO*100, ls="--", lw=1.3, color="#C62828")
    ax.set_xscale("log"); ax.set_xlim(0.1, 8); ax.set_ylim(35, 88)
    ax.set_xticks([0.1,0.3,0.5,1,2,5]); ax.set_xticklabels(["0.1","0.3","0.5","1","2","5"])
    ax.set_xlabel("Cost USD / 158 cases (log)"); ax.set_ylabel("Rank@1 rate (%)")
    ax.set_title("Cost-efficiency: Rank@1 vs API cost (RAG)", fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()


def fig_slope(path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, metric, title in [(axes[0],"r1","Rank@1 rate (%)"),(axes[1],"gf","Gene-detection rate (%)")]:
        idx = 0 if metric == "r1" else 5
        ends = sorted([(m, RAG[m][idx]/N_RAG*100) for m in MODELS], key=lambda t: t[1])
        nud, last = {}, -100
        for m, y in ends:
            ly = y if y-last >= 3.4 else last+3.4
            nud[m] = ly; last = ly
        for m in MODELS:
            y0 = NORAG[m][idx]/N_NORAG*100; y1 = RAG[m][idx]/N_RAG*100
            ax.plot([0,1],[y0,y1],"-",color=MCOLORS[m],lw=2,alpha=.85)
            ax.scatter([0,1],[y0,y1],s=55,color=MCOLORS[m],edgecolor="#222",zorder=4)
            ax.text(1.04, nud[m], m, fontsize=8, va="center", color=MCOLORS[m], fontweight="bold")
        ax.set_xlim(-.15,1.55); ax.set_ylim(0,100)
        ax.set_xticks([0,1]); ax.set_xticklabels(["No-RAG","RAG"], fontweight="bold")
        ax.set_title(title, fontweight="bold"); ax.grid(axis="x", visible=False)
    fig.suptitle("Effect of RAG on each model (HPO-informed input)", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()


if __name__ == "__main__":
    fig_rank1(f"{OUT}/Fig1_rank1.png")
    fig_stacked(RAG, N_RAG, "Rank distribution — RAG (n=158)", f"{OUT}/Fig2_stacked_RAG.png")
    fig_stacked(NORAG, N_NORAG, "Rank distribution — No-RAG (n=158)", f"{OUT}/Fig2b_stacked_NoRAG.png")
    fig_cost(f"{OUT}/Fig3_cost.png")
    fig_slope(f"{OUT}/Fig4_slope.png")
    print(f"Figures written to {OUT}/")
