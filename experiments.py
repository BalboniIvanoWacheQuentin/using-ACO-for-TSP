'''
Banc d'essai : fait varier les parametres de l'ACO et enregistre les resultats en CSV
pour faire des statistiques ensuite (pandas, R, Excel...).

    python experiments.py

Produit a la racine du projet (lus par la page Analyse de l'application) :
  runs.csv    : une ligne par execution (parametres + resultat final)
  history.csv : une ligne par iteration de chaque execution (courbes de convergence)

Chaque execution est repetee sur plusieurs graines pour pouvoir calculer moyenne / ecart-type.
Une recherche aleatoire avec le meme budget (N_ants * max_iter tournees) sert de reference.
'''
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor

from algorithms import generate_tsp, ACO_search, random_search

# ------------------------------------------------------------
#  Configuration de l'experience
# ------------------------------------------------------------
N_CITIES = [30]          # tailles d'instance
INSTANCE_SEEDS = [42]    # une graine = une instance (positions des villes)
RUN_SEEDS = range(10)    # repetitions de l'ACO sur chaque instance
MAX_ITER = 100

# Parametres de reference ; on fait varier un parametre a la fois autour d'eux
BASE = dict(N_ants=20, alpha=1.0, beta=2.0, rho=0.2, Q=100.0)
SWEEP = {
    "alpha":  [0.0, 0.5, 1.0, 2.0, 3.0],
    "beta":   [0.0, 1.0, 2.0, 3.0, 5.0],
    "rho":    [0.05, 0.1, 0.2, 0.5, 0.9],
    "N_ants": [5, 10, 20, 50],
}
# Mettre FULL_GRID = True pour tester toutes les combinaisons (beaucoup plus long)
FULL_GRID = False

RUN_RANDOM_BASELINE = True
OUT_DIR = os.path.dirname(os.path.abspath(__file__))   # racine : la page Analyse lit les CSV ici


def parameter_sets():
    '''Liste des combinaisons de parametres a tester (sans doublons)'''
    if FULL_GRID:
        import itertools
        keys = list(SWEEP)
        combos = [dict(BASE, **dict(zip(keys, vals)))
                  for vals in itertools.product(*SWEEP.values())]
    else:
        combos = [dict(BASE, **{k: v}) for k, values in SWEEP.items() for v in values]
    unique = []
    for c in combos:
        if c not in unique:
            unique.append(c)
    return unique


def run_one(job):
    '''Execute une seule experience (appelee dans un processus separe)'''
    algo, n, inst_seed, run_seed, params = job
    G = generate_tsp(n, seed=inst_seed)
    t0 = time.perf_counter()
    if algo == "ACO":
        _, cost, hist = ACO_search(G, max_iter=MAX_ITER, seed=run_seed, **params)
    else:
        # meme budget que l'ACO : N_ants * MAX_ITER tournees evaluees
        n_tours = params["N_ants"] * MAX_ITER
        _, cost, rand_hist = random_search(G, max_iter=n_tours, seed=run_seed)
        # on ramene l'historique a une valeur par "iteration" de N_ants tournees
        step = params["N_ants"]
        best = rand_hist[step - 1::step]
        hist = {"best": best, "iter_best": [None] * len(best), "mean": [None] * len(best),
                "branching": [None] * len(best), "time": [None] * len(best)}
    elapsed = time.perf_counter() - t0
    return algo, n, inst_seed, run_seed, params, cost, elapsed, hist


def first_iter_within(history_best, final, tol):
    '''Premiere iteration (1-indexee) ou le meilleur cout est a moins de tol (relatif) du cout final'''
    for i, c in enumerate(history_best):
        if c <= final * (1 + tol):
            return i + 1
    return len(history_best)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    params_list = parameter_sets()

    jobs = []
    for n in N_CITIES:
        for inst_seed in INSTANCE_SEEDS:
            for run_seed in RUN_SEEDS:
                for p in params_list:
                    jobs.append(("ACO", n, inst_seed, run_seed, p))
                if RUN_RANDOM_BASELINE:
                    jobs.append(("random", n, inst_seed, run_seed, dict(BASE)))

    print(f"{len(jobs)} executions ({len(params_list)} jeux de parametres)...")

    run_fields = ["run_id", "algo", "n_cities", "instance_seed", "run_seed",
                  "N_ants", "alpha", "beta", "rho", "Q", "max_iter",
                  "best_cost", "time_s", "iter_best_found",
                  "iter_within_1pct", "iter_within_5pct", "final_branching"]
    hist_fields = ["run_id", "iteration", "best", "iter_best", "mean", "branching", "time_s"]

    with open(os.path.join(OUT_DIR, "runs.csv"), "w", newline="") as f_runs, \
         open(os.path.join(OUT_DIR, "history.csv"), "w", newline="") as f_hist, \
         ProcessPoolExecutor() as pool:
        w_runs = csv.DictWriter(f_runs, fieldnames=run_fields)
        w_hist = csv.writer(f_hist)
        w_runs.writeheader()
        w_hist.writerow(hist_fields)

        for run_id, res in enumerate(pool.map(run_one, jobs)):
            algo, n, inst_seed, run_seed, p, cost, elapsed, hist = res
            best = hist["best"]
            w_runs.writerow({
                "run_id": run_id, "algo": algo, "n_cities": n,
                "instance_seed": inst_seed, "run_seed": run_seed,
                **p, "max_iter": MAX_ITER,
                "best_cost": round(cost, 4), "time_s": round(elapsed, 4),
                "iter_best_found": best.index(min(best)) + 1,
                "iter_within_1pct": first_iter_within(best, cost, 0.01),
                "iter_within_5pct": first_iter_within(best, cost, 0.05),
                "final_branching": hist["branching"][-1],
            })
            for i in range(len(best)):
                w_hist.writerow([run_id, i + 1, best[i], hist["iter_best"][i],
                                 hist["mean"][i], hist["branching"][i], hist["time"][i]])
            print(f"[{run_id + 1}/{len(jobs)}] {algo} n={n} seed={run_seed} {p} -> {cost:.2f}")

    print(f"Resultats ecrits dans {OUT_DIR}/runs.csv et {OUT_DIR}/history.csv")


if __name__ == "__main__":
    main()
