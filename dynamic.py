'''
TSP dynamique : les distances changent en cours de route. Reprendre "a chaud" aide-t-il ?

    python dynamic.py          # experience (~25 min) puis figure
    python dynamic.py rat783   # une seule instance (pour lancer les instances en parallele)
    python dynamic.py plot     # figure seulement

Scenario (facteur de trafic, comme les generateurs de TSP dynamique de la litterature) :
  1. MMAS + 2-opt tourne PRE_S secondes sur l'instance d'origine -> pheromones + meilleure tournee
  2. Embouteillage : 20 % des routes (tirees au hasard) voient leur longueur multipliee par 1 a 3
  3. Chaque methode dispose de BUDGET_S secondes sur les nouvelles distances
       a chaud : ACO qui garde ses pheromones, 2-opt sur l'ancienne tournee,
                 OR-Tools qui part de l'ancienne tournee
       a froid : ACO relance de zero, OR-Tools relance de zero, 2-opt multi-departs
Reference : meilleure tournee connue sur les nouvelles distances (OR-Tools 60 s et toutes les methodes).

-> results/dynamic_<instance>.csv (instance, seed, algo, time_s, cost), results/dynamic_table.csv
-> results/figures/dynamic.png
'''
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import style as pc
import tsp_classics as tc
from aco_variants import aco
from accuracy_time import multistart_2opt

torch.set_num_threads(1)
tc.two_opt(tc.coords_to_dist(np.random.rand(8, 2)), list(range(8)))   # compile numba avant toute mesure

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
INSTANCES = ["kroA100", "rat783"]
SEEDS = [0, 1, 2]
PRE_S, BUDGET_S, REF_S = 20.0, 20.0, 60.0
ORTOOLS_LIMITS = [0.1, 0.3, 1, 3, 10, 20]
FRACTION, MAX_FACTOR = 0.2, 3.0
ACO_ARGS = dict(variant="MMAS", local_search=True, n_ants=25, max_iter=10 ** 6)


def traffic(D, seed=100):
    '''multiplie 20 % des routes (symetriques) par un facteur tire dans [1, 3]'''
    rng = np.random.default_rng(seed)
    n = len(D)
    F = np.where(rng.random((n, n)) < FRACTION, rng.uniform(1, MAX_FACTOR, (n, n)), 1.0)
    F = np.triu(F, 1)
    F = F + F.T + np.eye(n)
    return np.round(D * F)


def anytime_rows(h):
    '''points (temps, meilleur cout) ou l'ACO s'ameliore'''
    out, last = [], float("inf")
    for t, b in zip(h["time"], h["best"]):
        if b < last - 1e-9:
            out.append((t, b))
            last = b
    out.append((h["time"][-1], h["best"][-1]))
    return out


def run(instances=None):
    for inst in instances or INSTANCES:
        run_instance(inst)


def run_instance(inst):
    '''un fichier par instance (results/dynamic_<inst>.csv) : on peut lancer les instances en parallele'''
    path = os.path.join(RES, f"dynamic_{inst}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance", "seed", "algo", "time_s", "cost"])

        def row(inst, seed, algo, t, cost):
            w.writerow([inst, seed, algo, f"{t:.5f}", round(cost, 3)])
            f.flush()

        if True:
            D0 =tc.coords_to_dist(tc.load_tsplib(os.path.join(HERE, "data", "tsplib", f"{inst}.tsp")),
                                   tsplib_round=True)
            D1 = traffic(D0)

            # reference et methodes deterministes a froid
            _, ref = tc.ortools_solve(D1, time_limit_s=REF_S)
            row(inst, -1, "reference", REF_S, ref)
            for lim in ORTOOLS_LIMITS:
                t0 = time.perf_counter()
                _, c = tc.ortools_solve(D1, time_limit_s=lim)
                row(inst, 0, "OR-Tools (à froid)", time.perf_counter() - t0, c)

            for seed in SEEDS:
                # phase 1 : la colonie apprend l'instance d'origine
                old_tour, old_cost, h0 = aco(D0, seed=seed, time_limit=PRE_S, **ACO_ARGS)
                print(f"[{inst}] graine {seed} : avant changement {old_cost:.0f}, "
                      f"ancienne tournee apres changement {tc.tour_length(D1, old_tour):.0f}")
                row(inst, seed, "ancienne tournée (sans rien faire)", 0.0, tc.tour_length(D1, old_tour))

                # a chaud
                _, _, h = aco(D1, seed=seed, time_limit=BUDGET_S, init_tau=h0["tau"], **ACO_ARGS)
                for t, b in anytime_rows(h):
                    row(inst, seed, "ACO MMAS + 2-opt (à chaud)", t, b)
                t0 = time.perf_counter()
                _, c = tc.two_opt(D1, old_tour)
                row(inst, seed, "2-opt sur l'ancienne tournée", time.perf_counter() - t0, c)
                for lim in ORTOOLS_LIMITS:
                    t0 = time.perf_counter()
                    _, c = tc.ortools_solve(D1, time_limit_s=lim, init_tour=old_tour)
                    row(inst, seed, "OR-Tools (à chaud)", time.perf_counter() - t0, c)

                # a froid
                _, _, h = aco(D1, seed=seed, time_limit=BUDGET_S, **ACO_ARGS)
                for t, b in anytime_rows(h):
                    row(inst, seed, "ACO MMAS + 2-opt (à froid)", t, b)
                for t, b in multistart_2opt(D1, BUDGET_S, seed):
                    row(inst, seed, "2-opt multi-départs (à froid)", t, b)
                print(f"[{inst}] graine {seed} : fait")


STYLE = {  # couleur, trait (plein = a chaud, pointille = a froid), libelle court pour la legende
    "ACO MMAS + 2-opt (à chaud)": (pc.C[3], "-", "ACO à chaud"),
    "ACO MMAS + 2-opt (à froid)": (pc.C[3], "--", "ACO à froid"),
    "OR-Tools (à chaud)": (pc.C[6], "-", "OR-Tools à chaud"),
    "OR-Tools (à froid)": (pc.C[6], "--", "OR-Tools à froid"),
    "2-opt multi-départs (à froid)": (pc.C[4], "--", "2-opt multi-départs"),
}
POINTS = [("ancienne tournée (sans rien faire)", "x", "ancienne tournée"),
          ("2-opt sur l'ancienne tournée", "D", "2-opt sur l'ancienne")]
GRID = np.logspace(-3, np.log10(BUDGET_S), 150)


def load():
    import glob
    return pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "dynamic_*.csv"))
                      if not f.endswith("dynamic_table.csv")])


def plot():
    df = load()
    fig, axes = plt.subplots(1, len(INSTANCES), figsize=(pc.SLIDE_W, 3.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, inst in zip(axes, INSTANCES):
        d = df[df.instance == inst]
        ref = min(d.cost)                         # meilleure tournee connue
        gap = lambda c: 100 * (np.asarray(c) / ref - 1)
        for algo, (col, ls, lab) in STYLE.items():
            a = d[d.algo == algo]
            if algo.startswith("OR-Tools"):       # un point par limite de temps
                a = a.assign(lim=a.groupby("seed").cumcount())
                g = a.groupby("lim").agg(t=("time_s", "mean"), c=("cost", "median"))
                ax.plot(g.t, gap(g.c), ls, marker="s", color=col, ms=6, lw=1.8, label=lab)
                continue
            curves = []
            for _, s in a.groupby("seed"):
                curves.append([s.cost[s.time_s <= t].min() if (s.time_s <= t).any() else np.nan for t in GRID])
            curves = gap(np.array(curves, dtype=float))
            curves[:, np.isnan(curves).any(axis=0)] = np.nan
            ax.plot(GRID, np.median(curves, axis=0), ls, color=col, label=lab)
        for algo, mk, lab in POINTS:
            a = d[d.algo == algo]
            ax.plot(max(a.time_s.mean(), 1.2e-3), np.median(gap(a.cost)), mk, color=pc.INK, ms=9,
                    mew=2, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_yticks([0, 0.5, 1, 2, 5, 10, 20, 50])
        ax.set_yticklabels(["0", "0,5", "1", "2", "5", "10", "20", "50"])
        ax.set_ylim(-0.05, 80)
        ax.set_xlabel("temps après le changement (s)")
        ax.set_title(f"{inst} (n = {len(tc.load_tsplib(os.path.join(HERE, 'data', 'tsplib', inst + '.tsp')))})")
    axes[0].set_ylabel("écart à la meilleure (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=13.5, handlelength=1.8, columnspacing=1.2,
               labelspacing=0.3)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    fig.savefig(os.path.join(RES, "figures", "dynamic.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # tableau : ecart median a 0,1 s, 1 s et BUDGET_S
    rows = []
    for inst in INSTANCES:
        d = df[df.instance == inst]
        ref = min(d.cost)
        for algo in d.algo.unique():
            if algo == "reference":
                continue
            a = d[d.algo == algo]
            vals = []
            for B in [0.1, 1, BUDGET_S]:
                per_seed = [s.cost[s.time_s <= B * 1.1].min() for _, s in a.groupby("seed")
                            if (s.time_s <= B * 1.1).any()]
                vals.append(round(100 * (np.median(per_seed) / ref - 1), 2) if per_seed else np.nan)
            rows.append([inst, algo] + vals)
    t = pd.DataFrame(rows, columns=["instance", "algo", "gap_0.1s", "gap_1s", f"gap_{BUDGET_S:g}s"])
    t.to_csv(os.path.join(RES, "dynamic_table.csv"), index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    if not (len(sys.argv) > 1 and sys.argv[1] == "plot"):
        run(sys.argv[1:])                  # ex. : python dynamic.py rat783
    plot()
