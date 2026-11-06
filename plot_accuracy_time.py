'''
Figures et tableaux a partir de results/complexity.csv et results/anytime.csv.

    python plot_accuracy_time.py

Produit :
  results/figures/complexity_all.png     temps vs n (log-log) + Held-Karp (exponentiel)
  results/figures/anytime_<inst>.png     ecart a l'optimum vs temps, tous les algorithmes
  results/complexity_slopes.csv          pentes mesurees
  results/budget_table.csv               ecart atteint par chaque algorithme pour un budget de temps donne
'''
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

BUDGETS = [0.01, 0.1, 1, 10, 30]
TOL = 1.1          # un budget B accepte un depassement de 10 % (meme regle pour tous)
GRID = np.logspace(-3, np.log10(30), 200)

# ordre et couleur fixes ; les algorithmes "a un coup" sont en gris avec etiquette
STYLE = {
    "ACO AS": C[0], "ACO ACS": C[1], "ACO MMAS": C[2], "ACO MMAS + 2-opt": C[3],
    "2-opt multi-départs": C[4], "Recuit simulé": C[5], "OR-Tools": C[6],
}
SINGLE = ["Plus proche voisin", "PPV + 2-opt", "Christofides"]
ANYTIME = ["ACO AS", "ACO ACS", "ACO MMAS", "ACO MMAS + 2-opt", "2-opt multi-départs"]
BUDGETED = ["Recuit simulé", "OR-Tools"]

THEORY = {   # pente theorique attendue en log-log (temps vs n)
    "Plus proche voisin": "2", "PPV + 2-opt": "2 par passe (~3 au total)", "Christofides": "3",
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
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    slopes = []
    for k, algo in enumerate(poly.algo.unique()):
        d = poly[poly.algo == algo].sort_values("n")
        x, y = d.n.values, d.time_s.values
        h = max(3, len(x) // 2)
        s = np.polyfit(np.log(x[-h:]), np.log(y[-h:]), 1)[0]
        slopes.append((algo, round(s, 2), THEORY.get(algo, "")))
        a1.plot(x, y, "o-", color=C[k], ms=5, markeredgecolor=SURF, markeredgewidth=1.2,
                label=f"{algo} : pente {s:.1f}")
    xs = sorted(poly.n.unique())
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xticks(xs)
    a1.set_xticklabels([str(v) for v in xs])
    a1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_xlabel("nombre de villes n")
    a1.set_ylabel("temps de calcul (s)")
    a1.set_title("Algorithmes polynomiaux (log-log)")
    a1.legend(fontsize=8, loc="upper left")

    x, y = hk.n.values, hk.time_s.values
    big = x >= 10
    b = np.polyfit(x[big], np.log2(y[big]), 1)[0]
    a2.plot(x, y, "o-", color=C[7], ms=5, markeredgecolor=SURF, markeredgewidth=1.2,
            label=f"mesuré : x{2 ** b:.1f} par ville ajoutée")
    a2.set_yscale("log")
    a2.set_xlabel("nombre de villes n")
    a2.set_ylabel("temps de calcul (s)")
    a2.set_title("Held-Karp (exact) : exponentiel")
    a2.legend(fontsize=8, loc="upper left")
    slopes.append(("Held-Karp (exact)", f"x{2 ** b:.2f} par ville", "O(n² 2ⁿ) : x2,3 vers n = 15"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "complexity_all.png"), dpi=170)
    plt.close(fig)
    pd.DataFrame(slopes, columns=["algo", "pente_mesuree", "theorie"]).to_csv(
        os.path.join(RES, "complexity_slopes.csv"), index=False)
    return slopes


# ------------------------------------------------------------
def anytime_curves(df, inst):
    d = df[df.instance == inst]
    opt = d.opt.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5.4))
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
        ax.plot(g.t, g.gap, "s--", color=STYLE[algo], ms=6, lw=1.2, label=algo,
                markeredgecolor=SURF, markeredgewidth=1.2)
    for algo in SINGLE:
        a = d[d.algo == algo].iloc[0]
        gap = 100 * (a.cost / opt - 1)
        ax.plot(a.time_s, gap, "D", color=pc.INK, ms=6, markeredgecolor=SURF, markeredgewidth=1.2)
        ax.annotate(algo, (a.time_s, gap), xytext=(6, 4), textcoords="offset points", fontsize=8,
                    color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([0, 0.5, 1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["0 (optimum)", "0,5", "1", "2", "5", "10", "20", "50"])
    ax.set_ylim(-0.05, 60)
    ax.set_xlim(5e-5, 40)
    ax.set_xlabel("temps de calcul (s, échelle log)")
    ax.set_ylabel("écart à l'optimum (%)")
    ax.set_title(f"{inst} (n = {d.n.iloc[0]}) : précision vs temps, un seul cœur CPU")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"anytime_{inst}.png"), dpi=170)
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
    if os.path.exists(os.path.join(RES, "anytime.csv")):
        df = pd.read_csv(os.path.join(RES, "anytime.csv"))
        for inst in df.instance.unique():
            anytime_curves(df, inst)
        t = budget_table(df)
        pd.set_option("display.width", 200)
        print(t.pivot_table(index=["instance", "algo"], columns="budget_s", values="gap_pct",
                            dropna=False).to_string())
    print(f"Figures dans {FIG}")


if __name__ == "__main__":
    main()
