'''
Precision vs temps de calcul : ACO compare aux algorithmes classiques.
Tout tourne sur UN SEUL coeur CPU pour que les temps soient comparables.

    python accuracy_time.py              # les deux parties (~35 min)
    python accuracy_time.py complexity   # partie 1 seulement
    python accuracy_time.py anytime      # partie 2 seulement

Partie 1 - complexite : temps de calcul en fonction de n (instances aleatoires)
    -> results/complexity.csv   (algo, n, time_s, cost)

Partie 2 - precision vs temps sur des instances TSPLIB a optimum connu
    -> results/anytime.csv      (instance, n, opt, algo, seed, time_s, cost)
    Algorithmes "a un coup" (un seul point) : plus proche voisin, PPV + 2-opt, Christofides
    Algorithmes a budget reglable (un point par budget) : recuit simule, OR-Tools
    Algorithmes "anytime" (une courbe : meilleur cout trouve au cours du temps) :
        variantes ACO, 2-opt multi-departs (temoin : 2-opt sans apprentissage)
'''
import csv
import os
import sys
import time

import numpy as np
import torch

from aco_variants import aco
import tsp_classics as tc

torch.set_num_threads(1)
tc.two_opt(tc.coords_to_dist(np.random.rand(8, 2)), list(range(8)))   # compile numba avant toute mesure

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# ---- partie 1
COMPLEXITY_N = [25, 50, 100, 200, 400, 800, 1600]
HELD_KARP_N = list(range(6, 17))
ACO_ANTS = 25

# ---- partie 2
INSTANCES = {"eil51": 426, "kroA100": 21282, "ch150": 6528, "pcb442": 50778, "rat783": 8806}
BUDGET_S = 60.0
SEEDS = [0, 1, 2]
ORTOOLS_LIMITS = [0.1, 0.3, 1, 3, 10, 30, 60]
SA_FACTORS = [30, 100, 300, 1000, 3000, 10000, 30000, 100000]   # iterations = facteur * n
SA_US_PER_IT = 2.5     # ~2,5 microsecondes par iteration : on saute les reglages qui depassent le budget
ACO_VARIANTS = [("AS", False), ("ACS", False), ("MMAS", False), ("MMAS", True)]


def aco_name(variant, ls):
    return f"ACO {variant}" + (" + 2-opt" if ls else "")


def random_instance(n, seed=0):
    rng = np.random.default_rng(seed)
    return tc.coords_to_dist(rng.uniform(0, 100, size=(n, 2)))


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    t = time.perf_counter() - t0
    if t < 0.5:                       # algorithmes rapides : meilleur de 3 mesures
        for _ in range(2):
            t0 = time.perf_counter()
            fn()
            t = min(t, time.perf_counter() - t0)
    return t, out


# ------------------------------------------------------------
#  Partie 1 : complexite
# ------------------------------------------------------------
def part_complexity():
    algos = {
        "Plus proche voisin": (lambda D: tc.nearest_neighbor(D), None),
        "PPV + 2-opt": (lambda D: tc.two_opt(D, tc.nearest_neighbor(D)[0]), None),
        "Christofides": (lambda D: tc.christofides(D), 400),
        "Recuit simulé": (lambda D: tc.simulated_annealing(D, seed=0), None),
        "ACO AS (10 it.)": (lambda D: aco(D, "AS", n_ants=ACO_ANTS, max_iter=10)[:2], None),
        "ACO MMAS + 2-opt (10 it.)": (lambda D: aco(D, "MMAS", n_ants=ACO_ANTS, max_iter=10,
                                                     local_search=True)[:2], None),
    }
    path = os.path.join(RES, "complexity.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo", "n", "time_s", "cost"])
        for n in COMPLEXITY_N:
            D = random_instance(n)
            for name, (fn, max_n) in algos.items():
                if max_n and n > max_n:
                    continue
                t, (tour, cost) = timed(lambda: fn(D))
                w.writerow([name, n, f"{t:.6f}", round(cost, 3)])
                f.flush()
                print(f"[complexite] {name:<28} n={n:4}  {t:9.4f} s")
        for n in HELD_KARP_N:
            D = random_instance(n)
            t, (tour, cost) = timed(lambda: tc.held_karp(D))
            w.writerow(["Held-Karp (exact)", n, f"{t:.6f}", round(cost, 3)])
            f.flush()
            print(f"[complexite] Held-Karp                    n={n:4}  {t:9.4f} s")


# ------------------------------------------------------------
#  Partie 2 : precision vs temps
# ------------------------------------------------------------
def multistart_2opt(D, budget, seed, batch=8):
    '''2-opt depuis des tournees aleatoires, recommence jusqu'a epuisement du budget'''
    g = torch.Generator().manual_seed(seed)
    Dt = torch.tensor(D, dtype=torch.float64)
    n = len(D)
    best, points = float("inf"), []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < budget:
        tours = torch.stack([torch.randperm(n, generator=g) for _ in range(batch)])
        tours = tc.two_opt_batch(Dt, tours)
        L = Dt[tours, tours.roll(-1, dims=1)].sum(dim=1)
        c = float(L.min())
        if c < best - 1e-9:
            best = c
            points.append((time.perf_counter() - t0, best))
    points.append((time.perf_counter() - t0, best))
    return points


def part_anytime(instances=None):
    for inst in instances or INSTANCES:
        anytime_instance(inst, INSTANCES[inst])


def anytime_instance(inst, opt):
    '''un fichier par instance (results/anytime_<inst>.csv) : on peut lancer les instances en parallele'''
    path = os.path.join(RES, f"anytime_{inst}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance", "n", "opt", "algo", "seed", "time_s", "cost"])

        def row(inst, n, opt, algo, seed, t, cost):
            if np.isfinite(cost):
                w.writerow([inst, n, opt, algo, seed, f"{t:.5f}", round(cost, 3)])
                f.flush()

        if True:
            D = tc.coords_to_dist(tc.load_tsplib(os.path.join(HERE, "data", "tsplib", f"{inst}.tsp")),
                                  tsplib_round=True)
            n = len(D)

            # algorithmes a un coup
            for name, fn in [("Plus proche voisin", lambda: tc.nearest_neighbor(D)),
                             ("PPV + 2-opt", lambda: tc.two_opt(D, tc.nearest_neighbor(D)[0])),
                             ("Christofides", lambda: tc.christofides(D))]:
                t, (_, cost) = timed(fn)
                row(inst, n, opt, name, 0, t, cost)
                print(f"[{inst}] {name:<22} {cost:10.0f}  {t:8.3f} s")

            # algorithmes a budget
            for lim in ORTOOLS_LIMITS:
                t0 = time.perf_counter()
                _, cost = tc.ortools_solve(D, time_limit_s=lim)
                row(inst, n, opt, "OR-Tools", 0, time.perf_counter() - t0, cost)
                print(f"[{inst}] OR-Tools ({lim:>4} s)        {cost:10.0f}")
            for k in SA_FACTORS:
                if k * n * SA_US_PER_IT * 1e-6 > BUDGET_S:
                    continue
                for seed in SEEDS:
                    t0 = time.perf_counter()
                    _, cost = tc.simulated_annealing(D, n_iter=k * n, seed=seed)
                    row(inst, n, opt, "Recuit simulé", seed, time.perf_counter() - t0, cost)
                print(f"[{inst}] Recuit simule ({k} n it.) {cost:10.0f}")

            # algorithmes anytime
            for seed in SEEDS:
                for t, cost in multistart_2opt(D, BUDGET_S, seed):
                    row(inst, n, opt, "2-opt multi-départs", seed, t, cost)
                print(f"[{inst}] 2-opt multi-departs graine {seed} {cost:10.0f}")
                for variant, ls in ACO_VARIANTS:
                    _, _, h = aco(D, variant, device="cpu", n_ants=ACO_ANTS, max_iter=10 ** 6,
                                  local_search=ls, seed=seed, time_limit=BUDGET_S)
                    last = float("inf")
                    for t, b in zip(h["time"], h["best"]):
                        if b < last - 1e-9:
                            row(inst, n, opt, aco_name(variant, ls), seed, t, b)
                            last = b
                    row(inst, n, opt, aco_name(variant, ls), seed, h["time"][-1], h["best"][-1])
                    print(f"[{inst}] {aco_name(variant, ls):<22} graine {seed} {last:10.0f} "
                          f"({len(h['best'])} it.)")


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("all", "complexity"):
        part_complexity()
    if part in ("all", "anytime"):
        part_anytime(sys.argv[2:])          # ex. : python accuracy_time.py anytime rat783
