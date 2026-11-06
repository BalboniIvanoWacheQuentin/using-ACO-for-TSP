'''
Figures et tableaux a partir de results/complexity.csv et results/anytime_<instance>.csv.

    python plot_accuracy_time.py

Produit :
  results/figures/complexity_all.png     temps vs n (log-log) + Held-Karp (exponentiel)
  results/figures/anytime_<inst>.png     ecart a l'optimum vs temps, tous les algorithmes
  results/complexity_slopes.csv          pentes mesurees
  results/budget_table.csv               ecart atteint par chaque algorithme pour un budget de temps donne
'''
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import style as pc               # style et palette communs

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
C, INK2, MUTED, SURF = pc.C, pc.INK2, pc.MUTED, pc.SURF

BUDGETS = [0.01, 0.1, 1, 10, 60]
TOL = 1.1          # un budget B accepte un depassement de 10 % (meme regle pour tous)
GRID = np.logspace(-3, np.log10(60), 200)
INSTANCE_ORDER = ["eil51", "kroA100", "ch150", "pcb442", "rat783"]

# ordre et couleur fixes ; les algorithmes "a un coup" sont en gris avec etiquette
STYLE = {
    "ACO AS": C[0], "ACO ACS": C[1], "ACO MMAS": C[2], "ACO MMAS + 2-opt": C[3],
    "2-opt multi-départs": C[4], "Recuit simulé": C[5], "OR-Tools": C[6],
}
SINGLE = ["Plus proche voisin", "PPV + 2-opt", "Christofides"]
ANYTIME = ["ACO AS", "ACO ACS", "ACO MMAS", "ACO MMAS + 2-opt", "2-opt multi-départs"]
BUDGETED = ["Recuit simulé", "OR-Tools"]

THEORY = {   # pente theorique attendue en log-log (temps vs n)
    "Plus proche voisin": "2", "PPV + 2-opt": "2 (PPV) ; 2-opt : O(n) mouvements", "Christofides": "3",
    "Recuit simulé": "1 à 2 (n it. x (O(1) + inversion O(n)))", "ACO AS (10 it.)": "2 (m fixe)",
    "ACO MMAS + 2-opt (10 it.)": "2 + 2-opt",
}


def best_at(times, costs, t):
    '''meilleur cout trouve avant l'instant t (courbe en escalier), NaN si rien encore'''
    ok = times <= t * TOL
    return costs[ok].min() if ok.any() else np.nan


# ------------------------------------------------------------
def complexity():
    df = pd.read_csv(os.path.join(RES, "complexity.csv"))
    poly = df[df.algo != "Held-Karp (exact)"]
    hk = df[df.algo == "Held-Karp (exact)"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(pc.SLIDE_W, 4.5), gridspec_kw={"width_ratios": [1.7, 1]})
    slopes = []
    for k, algo in enumerate(poly.algo.unique()):
        d = poly[poly.algo == algo].sort_values("n")
        x, y = d.n.values, d.time_s.values
        h = max(3, len(x) // 2)
        s = np.polyfit(np.log(x[-h:]), np.log(y[-h:]), 1)[0]
        slopes.append((algo, round(s, 2), THEORY.get(algo, "")))
        dashed = algo.startswith("ACO AS")        # presque confondu avec MMAS + 2-opt
        a1.plot(x, y, "s--" if dashed else "o-", color=C[k], markeredgecolor=SURF, markeredgewidth=1.2,
                label=f"{algo.replace(' (10 it.)', '')} ({s:.1f})".replace(".", ","),
                zorder=3 if dashed else 2)
    xs = sorted(poly.n.unique())
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xticks(xs)
    a1.set_xticklabels([str(v) for v in xs])
    a1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_xlabel("nombre de villes n")
    a1.set_ylabel("temps (s)")
    a1.set_title("Polynomiaux (log-log ; pente)")

    x, y = hk.n.values, hk.time_s.values
    big = x >= 10
    b = np.polyfit(x[big], np.log2(y[big]), 1)[0]
    a2.plot(x, y, "o-", color=C[7], markeredgecolor=SURF, markeredgewidth=1.2,
            label=f"×{2 ** b:.1f} par ville".replace(".", ","))
    a2.set_yscale("log")
    a2.set_xlabel("nombre de villes n")
    a2.set_title("Held-Karp : exponentiel")
    a2.legend(fontsize=12.5, loc="upper left", handlelength=1.4)
    slopes.append(("Held-Karp (exact)", f"x{2 ** b:.2f} par ville", "O(n² 2ⁿ) : x2,3 vers n = 15"))
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=12.5,
               handlelength=1.8, columnspacing=1.2, labelspacing=0.3)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(FIG, "complexity_all.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(slopes, columns=["algo", "pente_mesuree", "theorie"]).to_csv(
        os.path.join(RES, "complexity_slopes.csv"), index=False)
    return slopes


# ------------------------------------------------------------
def anytime_curves(df, inst):
    d = df[df.instance == inst]
    opt = d.opt.iloc[0]
    fig, ax = plt.subplots(figsize=(pc.SLIDE_W, 4.7))
    for algo in ANYTIME:
        a = d[d.algo == algo]
        curves = []
        for _, s in a.groupby("seed"):
            curves.append([best_at(s.time_s.values, s.cost.values, t) for t in GRID])
        curves = 100 * (np.array(curves) / opt - 1)
        # on ne trace qu'a partir du moment ou toutes les graines ont une solution
        curves[:, np.isnan(curves).any(axis=0)] = np.nan
        ok = ~np.isnan(curves[0])
        ax.plot(GRID[ok], np.median(curves[:, ok], axis=0), color=STYLE[algo], label=algo)
        ax.fill_between(GRID[ok], curves[:, ok].min(axis=0), curves[:, ok].max(axis=0),
                        color=STYLE[algo], alpha=0.12, lw=0)
    for algo in BUDGETED:
        a = d[d.algo == algo].copy()
        a["gap"] = 100 * (a.cost / opt - 1)
        if algo == "Recuit simulé":
            a["cfg"] = a.groupby("seed").cumcount()
            g = a.groupby("cfg").agg(t=("time_s", "mean"), gap=("gap", "median"))
        else:
            g = a.rename(columns={"time_s": "t"})[["t", "gap"]]
        g = g[g.t <= GRID[-1] * TOL]                 # hors budget : non trace
        ax.plot(g.t, g.gap, "s--", color=STYLE[algo], lw=1.6, label=algo,
                markeredgecolor=SURF, markeredgewidth=1.2)
    for algo in SINGLE:
        a = d[d.algo == algo].iloc[0]
        gap = 100 * (a.cost / opt - 1)
        ax.plot(a.time_s, gap, "D", color=pc.INK, markeredgecolor=SURF, markeredgewidth=1.2)
        ax.annotate(algo, (a.time_s, gap), xytext=(7, 5), textcoords="offset points", fontsize=12,
                    color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([0, 0.5, 1, 2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["0", "0,5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_ylim(-0.05, 150)
    ax.set_xlim(2e-4, 80)
    ax.set_xlabel("temps de calcul (s, échelle log)")
    ax.set_ylabel("écart à l'optimum (%)")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=12.5,
               handlelength=1.6, labelspacing=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(FIG, f"anytime_{inst}.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def budget_table(df):
    rows = []
    for inst, d in df.groupby("instance", sort=False):
        opt = d.opt.iloc[0]
        for algo in SINGLE + BUDGETED + ANYTIME:
            a = d[d.algo == algo]
            for B in BUDGETS:
                if algo in ANYTIME:
                    vals = [best_at(s.time_s.values, s.cost.values, B) for _, s in a.groupby("seed")]
                    v = np.nan if any(np.isnan(vals)) else np.mean(vals)
                elif algo == "Recuit simulé":
                    a2 = a.copy()
                    a2["cfg"] = a2.groupby("seed").cumcount()
                    g = a2.groupby("cfg").agg(t=("time_s", "mean"), c=("cost", "mean"))
                    g = g[g.t <= B * TOL]
                    v = g.c.min() if len(g) else np.nan
                else:
                    ok = a[a.time_s <= B * TOL]
                    v = ok.cost.min() if len(ok) else np.nan
                rows.append((inst, algo, B, np.nan if np.isnan(v) else round(100 * (v / opt - 1), 2)))
    t = pd.DataFrame(rows, columns=["instance", "algo", "budget_s", "gap_pct"])
    t.to_csv(os.path.join(RES, "budget_table.csv"), index=False)
    return t


def main():
    os.makedirs(FIG, exist_ok=True)
    if os.path.exists(os.path.join(RES, "complexity.csv")):
        for s in complexity():
            print(s)
    files = glob.glob(os.path.join(RES, "anytime_*.csv"))
    if files:
        df = pd.concat([pd.read_csv(f) for f in files])
        order = [i for i in INSTANCE_ORDER if i in set(df.instance)]
        df = df.sort_values("instance", key=lambda s: s.map(order.index), kind="stable")
        for inst in order:
            anytime_curves(df, inst)
        t = budget_table(df)
        pd.set_option("display.width", 200)
        print(t.pivot_table(index=["instance", "algo"], columns="budget_s", values="gap_pct",
                            dropna=False).to_string())
    print(f"Figures dans {FIG}")


if __name__ == "__main__":
    main()
