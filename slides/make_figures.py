'''
Figures supplementaires pour la presentation (a partir de runs.csv et history.csv, a la racine).

    python slides/make_figures.py

Produit dans slides/figures/ :
  params_effect.png      effet de alpha, beta, rho et du nombre de fourmis (30 villes, 10 graines)
  premature.png          convergence alpha=1 vs alpha=3 : cout et facteur de branchement
  aco_vs_random.png      ACO vs recherche aleatoire a budget egal
  tours_kroA100.png      tournee du plus proche voisin vs tournee MMAS + 2-opt (optimale)
'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import style as pc                   # noqa: E402  (style et palette communs)
import tsp_classics as tc            # noqa: E402
from aco_variants import aco         # noqa: E402

RES = os.path.join(ROOT, "results")
OUT = os.path.join(HERE, "figures")
C, INK2, MUTED, SURF = pc.C, pc.INK2, pc.MUTED, pc.SURF
BASE = dict(alpha=1.0, beta=2.0, rho=0.2, N_ants=20)


def params_effect(runs):
    aco_runs = runs[runs.algo == "ACO"]
    rnd = runs[runs.algo == "random"]["best_cost"].mean()
    labels = {"alpha": r"$\alpha$ (phéromones)", "beta": r"$\beta$ (distance)",
              "rho": r"$\rho$ (évaporation)", "N_ants": "$m$ (fourmis)"}
    fig, axes = plt.subplots(1, 4, figsize=(pc.SLIDE_W, 3.2), sharey=True)
    for ax, p in zip(axes, labels):
        others = {k: v for k, v in BASE.items() if k != p}
        mask = np.ones(len(aco_runs), dtype=bool)
        for k, v in others.items():
            mask &= aco_runs[k].values == v
        d = aco_runs[mask]
        g = d.groupby(p)["best_cost"].agg(["mean", "std"]).reset_index()
        x = np.arange(len(g))
        ax.errorbar(x, g["mean"], yerr=g["std"], fmt="o-", color=C[0], ms=7, capsize=3,
                    markeredgecolor=SURF, markeredgewidth=1.5)
        ref = list(g[p]).index(BASE[p])
        ax.plot(x[ref], g["mean"][ref], "o", ms=11, mfc="none", mec=C[1], mew=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in g[p]])
        ax.set_xlabel(labels[p])
        ax.set_yscale("log")
    axes[0].set_ylabel("meilleure tournée")
    for ax in axes:
        ax.axhline(rnd, color=MUTED, ls="--", lw=1.2)
    axes[0].text(0.03, rnd * 0.94, "hasard", color=MUTED, fontsize=13,
                 transform=axes[0].get_yaxis_transform(), va="top")
    axes[0].set_yticks([450, 600, 800, 1200])
    axes[0].set_yticklabels(["450", "600", "800", "1200"])
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(OUT, "params_effect.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def premature(runs, hist):
    aco_runs = runs[(runs.algo == "ACO") & (runs.beta == 2) & (runs.rho == 0.2) & (runs.N_ants == 20)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(pc.SLIDE_W, 3.4))
    for k, alpha in enumerate([1.0, 3.0]):
        ids = aco_runs[aco_runs.alpha == alpha]["run_id"]
        h = hist[hist.run_id.isin(ids)].groupby("iteration")
        best, mean, br = h["best"].mean(), h["mean"].mean(), h["branching"].mean()
        a1.plot(best.index, best, color=C[k], label=rf"$\alpha={alpha:g}$ : meilleure")
        a1.plot(mean.index, mean, color=C[k], lw=1.4, ls=":", label=rf"$\alpha={alpha:g}$ : moyenne")
        a2.plot(br.index, br, color=C[k], label=rf"$\alpha={alpha:g}$")
    a1.set_xlabel("itération")
    a1.set_ylabel("longueur de tournée")
    a1.set_ylim(430, 700)
    a1.legend(fontsize=12, labelspacing=0.3)
    a2.axhline(2, color=MUTED, ls="--", lw=1.2)
    a2.set_ylim(0, 15.5)
    a2.set_yticks([2, 5, 10, 15])
    a2.text(100, 1.6, "toutes les fourmis sur la même piste", ha="right", va="top", fontsize=12,
            color=MUTED)
    a2.set_xlabel("itération")
    a2.set_ylabel("facteur de branchement")
    a2.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "premature.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def aco_vs_random(runs, hist):
    base = runs[(runs.algo == "ACO")]
    for k, v in BASE.items():
        base = base[base[k] == v]
    rnd = runs[runs.algo == "random"]
    fig, ax = plt.subplots(figsize=(pc.SLIDE_W / 2, 3.4))
    for col, ids, name in [(C[0], base.run_id, "ACO"), (C[1], rnd.run_id, "hasard")]:
        h = hist[hist.run_id.isin(ids)].groupby("iteration")["best"]
        m, s = h.mean(), h.std()
        x = m.index * 20
        ax.plot(x, m, color=col, label=name)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.15, lw=0)
    ax.set_xlabel("tournées évaluées")
    ax.set_ylabel("meilleure tournée")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "aco_vs_random.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def tours():
    coords = tc.load_tsplib(os.path.join(ROOT, "data", "tsplib", "kroA100.tsp"))
    D = tc.coords_to_dist(coords, tsplib_round=True)
    nn_tour, nn_cost = tc.nearest_neighbor(D)
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    best_tour, best_cost, _ = aco(D, variant="MMAS", device=dev, local_search=True, max_iter=200, seed=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, tour, cost, title, col in [
            (axes[0], nn_tour, nn_cost, "Plus proche voisin", C[1]),
            (axes[1], best_tour, best_cost, "MMAS + 2-opt", C[0])]:
        t = list(tour) + [tour[0]]
        ax.plot(coords[t, 0], coords[t, 1], "-", color=col, lw=1.5)
        ax.plot(coords[:, 0], coords[:, 1], "o", color=pc.INK, ms=3)
        gap = 100 * (cost / 21282 - 1)
        ax.set_title(f"{title} : {cost:.0f} (+{gap:.1f} %)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_aspect("equal")
    fig.suptitle("kroA100 (optimum prouvé : 21 282)", fontsize=11, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tours_kroA100.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return nn_cost, best_cost


def pheromone_maps():
    '''carte des pheromones (epaisseur = pheromone, normalisee a chaque iteration), alpha = 1 vs 3'''
    import algorithms as alg
    G = alg.generate_tsp(20, p_edges=0.3, seed=42)
    xy = np.array([G.nodes[i]["pos"] for i in range(20)])
    fig, axes = plt.subplots(2, 3, figsize=(pc.SLIDE_W, 5.6))
    for r, alpha in enumerate([1.0, 3.0]):
        _, cost, hist = alg.ACO_search(G, alpha=alpha, max_iter=100)
        for ax, it in zip(axes[r], [1, 10, 100]):
            w = hist["tau"][it - 1] / hist["tau"][it - 1].max()
            for i in range(20):
                for j in range(i + 1, 20):
                    if w[i, j] > 0.05:
                        ax.plot(*xy[[i, j]].T, color=C[1], lw=5 * w[i, j], alpha=w[i, j])
            ax.scatter(*xy.T, s=30, color=C[0], zorder=3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f"itération {it}")
        axes[r][0].set_ylabel(rf"$\alpha = {alpha:g}$ (coût {cost:.0f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pheromones.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    # runs.csv et history.csv sont a la racine (c'est la que la page Analyse les lit)
    runs = pd.read_csv(os.path.join(ROOT, "runs.csv"))
    hist = pd.read_csv(os.path.join(ROOT, "history.csv"))
    params_effect(runs)
    premature(runs, hist)
    aco_vs_random(runs, hist)
    pheromone_maps()
    print("tournees kroA100 :", tours())
    print(f"Figures ecrites dans {OUT}")


if __name__ == "__main__":
    main()
