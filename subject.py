'''
Ce que demande le sujet : l'Ant System (celui de l'application) face à l'optimum exact,
et sa vitesse de convergence selon la taille.

    python subject.py          # ~5 min, puis figure
    python subject.py plot     # figure seulement

Petites instances aléatoires (8 à 16 villes) : optimum exact par Held-Karp.
Instances TSPLIB (51 à 150 villes) : optimum prouvé connu.
Ant System : 20 fourmis, 200 itérations, alpha = 1, beta = 2, rho = 0,2 ; 5 graines.

-> results/subject.csv
-> results/figures/subject_size.png
'''
import csv
import os
import sys
import time

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

import style as pc
import tsp_classics as tc
from algorithms import ACO_search

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SMALL = [8, 10, 12, 14, 16]
TSPLIB = {"eil51": 426, "berlin52": 7542, "st70": 675, "kroA100": 21282, "ch150": 6528}
SEEDS = range(5)
ACO = dict(N_ants=20, max_iter=200, alpha=1.0, beta=2.0, rho=0.2, Q=100.0)


def graph(coords):
    G = nx.Graph()
    for i, c in enumerate(coords):
        G.add_node(i, pos=np.asarray(c, dtype=float))
    return G


def run():
    instances = []
    for n in SMALL:
        coords = np.random.default_rng(n).uniform(0, 100, size=(n, 2))
        D = tc.coords_to_dist(coords)
        t0 = time.perf_counter()
        opt = tc.held_karp(D)[1]
        instances.append((f"aléatoire {n}", coords, D, opt, time.perf_counter() - t0))
    for name, opt in TSPLIB.items():
        coords = tc.load_tsplib(os.path.join(HERE, "data", "tsplib", f"{name}.tsp"))
        instances.append((name, coords, tc.coords_to_dist(coords, tsplib_round=True), opt, np.nan))

    with open(os.path.join(RES, "subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance", "n", "opt", "hk_time_s", "seed", "cost", "gap_pct",
                    "reached_opt", "it_opt", "it_1pct", "time_s"])
        for name, coords, D, opt, hk_t in instances:
            for seed in SEEDS:
                tour, _, h = ACO_search(graph(coords), seed=seed, **ACO)
                cost = tc.tour_length(D, tour)            # distances TSPLIB arrondies si besoin
                reached = cost <= opt + 1e-6
                best = np.array(h["best"])
                it_opt = int(np.argmax(best <= best[-1] + 1e-9)) + 1 if reached else ""
                it_1pct = int(np.argmax(best <= 1.01 * best[-1])) + 1
                w.writerow([name, len(D), round(opt, 3), hk_t, seed, round(cost, 3),
                            round(100 * (cost / opt - 1), 3), reached, it_opt, it_1pct,
                            round(h["time"][-1], 3)])
                f.flush()
            print(f"{name:14} n={len(D):3}  fait")


def plot():
    df = pd.read_csv(os.path.join(RES, "subject.csv"))
    g = df.groupby(["instance", "n"], sort=False)
    s = g.agg(gap=("gap_pct", "median"), gmin=("gap_pct", "min"), gmax=("gap_pct", "max"),
              it=("it_1pct", "median"), imin=("it_1pct", "min"), imax=("it_1pct", "max"),
              ok=("reached_opt", "sum"), t=("time_s", "median")).reset_index()
    small = s.n <= max(SMALL)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(pc.SLIDE_W, 3.6))
    for mask, col, lab in [(small, pc.C[0], "aléatoire (Held-Karp)"), (~small, pc.C[1], "TSPLIB")]:
        d = s[mask]
        a1.errorbar(d.n, d.gap, yerr=[d.gap - d.gmin, d.gmax - d.gap], fmt="o", color=col,
                    capsize=3, label=lab)
        a2.errorbar(d.n, d.it, yerr=[d.it - d.imin, d.imax - d.it], fmt="o", color=col, capsize=3)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks([8, 16, 51, 100, 150])
        ax.set_xticklabels(["8", "16", "51", "100", "150"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("nombre de villes n")
    a1.set_ylabel("écart à l'optimum (%)")
    a1.set_title("Qualité (200 itérations)")
    a1.legend(fontsize=12, loc="upper left")
    a2.set_ylabel("itérations jusqu'au plateau")
    a2.set_title("Vitesse de convergence")
    fig.tight_layout()
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    fig.savefig(os.path.join(RES, "figures", "subject_size.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(s.to_string(index=False))
    print(df[df.n <= max(SMALL)].groupby("n").agg(hk_s=("hk_time_s", "first"),
          atteint=("reached_opt", "sum"), it_opt=("it_opt", "median"),
          aco_s=("time_s", "median")).to_string())


if __name__ == "__main__":
    if not (len(sys.argv) > 1 and sys.argv[1] == "plot"):
        run()
    plot()
