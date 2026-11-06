'''
Algorithmes "classiques" du TSP, pour comparer avec l'ACO.
Tous prennent une matrice de distances D (numpy, n x n) et renvoient (tour, cout).

  nearest_neighbor     glouton du plus proche voisin            O(n^2)
  two_opt              recherche locale 2-opt (meilleure amelioration)   O(n^2) par passe
  christofides         approximation garantie <= 1.5 x optimum   O(n^3)
  simulated_annealing  recuit simule avec mouvements 2-opt
  held_karp            programmation dynamique EXACTE            O(n^2 2^n)  (n <= ~15)
  ortools_solve        solveur Google OR-Tools (si installe)
'''
import math
import time

import numpy as np
import networkx as nx
import torch


# ------------------------------------------------------------
#  Outils
# ------------------------------------------------------------
def tour_length(D, tour):
    tour = np.asarray(tour)
    return float(D[tour, np.roll(tour, -1)].sum())


def coords_to_dist(coords, tsplib_round=False):
    '''Distances euclidiennes ; tsplib_round=True arrondit a l'entier le plus proche (convention TSPLIB EUC_2D)'''
    coords = np.asarray(coords, dtype=float)
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return np.floor(D + 0.5) if tsplib_round else D


def load_tsplib(path):
    '''Lit un fichier TSPLIB de type EUC_2D, renvoie le tableau des coordonnees (n, 2)'''
    coords, reading = [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                reading = True
            elif line.startswith("EOF"):
                break
            elif reading and line:
                _, x, y = line.split()[:3]
                coords.append((float(x), float(y)))
    return np.array(coords)


# ------------------------------------------------------------
#  Plus proche voisin
# ------------------------------------------------------------
def nearest_neighbor(D, start=0):
    n = len(D)
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        d = np.where(visited, np.inf, D[tour[-1]])
        nxt = int(np.argmin(d))
        tour.append(nxt)
        visited[nxt] = True
    return tour, tour_length(D, tour)


# ------------------------------------------------------------
#  2-opt (version "lot" : ameliore plusieurs tournees en parallele, CPU ou GPU)
# ------------------------------------------------------------
def two_opt_batch(D, tours, max_passes=None, tol=1e-9):
    '''
    2-opt meilleure amelioration applique en parallele a un lot de tournees.
    D : tenseur torch (n, n) ; tours : tenseur long (m, n). Renvoie les tournees ameliorees.
    A chaque passe, chaque tournee applique son meilleur echange 2-opt (inversion d'un segment).
    '''
    m, n = tours.shape
    if n < 5:
        return tours
    dev = tours.device
    idx = torch.arange(n, device=dev)
    valid = idx[None, :] > idx[:, None] + 1          # paires (i, j) avec j >= i + 2
    valid[0, n - 1] = False                           # aretes adjacentes (bouclage)
    for _ in range(max_passes or 10 * n):
        A, B = tours, tours.roll(-1, dims=1)          # arete i = (A[i], B[i])
        d_ab = D[A, B]
        # gain de remplacer (a,b),(c,d) par (a,c),(b,d)
        delta = (D[A[:, :, None], A[:, None, :]] + D[B[:, :, None], B[:, None, :]]
                 - d_ab[:, :, None] - d_ab[:, None, :])
        delta = delta.masked_fill(~valid, 0.0)
        best = delta.reshape(m, -1).min(dim=1)
        improve = best.values < -tol
        if not bool(improve.any()):
            break
        i = (best.indices // n)[:, None]
        j = (best.indices % n)[:, None]
        pos = idx[None, :]
        rev = (pos > i) & (pos <= j) & improve[:, None]
        tours = torch.gather(tours, 1, torch.where(rev, i + j + 1 - pos, pos))
    return tours


def two_opt(D, tour):
    Dt = torch.tensor(D, dtype=torch.float64)
    t = torch.tensor([tour], dtype=torch.long)
    out = two_opt_batch(Dt, t)[0].tolist()
    return out, tour_length(D, out)


# ------------------------------------------------------------
#  Christofides (networkx)
# ------------------------------------------------------------
def christofides(D):
    n = len(D)
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=float(D[i, j]))
    cycle = nx.approximation.christofides(G, weight="weight")
    tour = cycle[:-1]
    return tour, tour_length(D, tour)


# ------------------------------------------------------------
#  Recuit simule (mouvements 2-opt, refroidissement geometrique)
# ------------------------------------------------------------
def simulated_annealing(D, n_iter=None, T_end_ratio=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    n = len(D)
    n_iter = n_iter or 3000 * n
    tour = rng.permutation(n)
    cost = tour_length(D, tour)
    best_tour, best_cost = tour.copy(), cost

    # temperature initiale : ecart moyen d'un mouvement 2-opt aleatoire
    T0 = float(np.mean(D[D > 0])) * 0.5
    T_end = T0 * T_end_ratio
    cool = (T_end / T0) ** (1.0 / n_iter)
    T = T0

    chunk = 100_000
    for start in range(0, n_iter, chunk):
        k = min(chunk, n_iter - start)
        I = rng.integers(0, n, size=k)
        J = rng.integers(0, n, size=k)
        U = rng.random(k)
        for i, j, u in zip(I, J, U):
            if i > j:
                i, j = j, i
            if j - i < 2 or (i == 0 and j == n - 1):
                T *= cool
                continue
            a, b = tour[i], tour[i + 1]
            c, d = tour[j], tour[(j + 1) % n]
            delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
            if delta < 0 or u < math.exp(-delta / T):
                tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                cost += delta
                if cost < best_cost:
                    best_cost, best_tour = cost, tour.copy()
            T *= cool
    best_tour = best_tour.tolist()
    return best_tour, tour_length(D, best_tour)


# ------------------------------------------------------------
#  Held-Karp : programmation dynamique exacte, O(n^2 2^n) en temps, O(n 2^n) en memoire
# ------------------------------------------------------------
def held_karp(D):
    n = len(D)
    if n > 16:
        raise ValueError("Held-Karp limite a n <= 16")
    N = n - 1                         # la ville 0 est le depart fixe
    full = 1 << N
    dp = np.full((full, N), np.inf)
    parent = np.full((full, N), -1, dtype=np.int64)
    for k in range(N):
        dp[1 << k, k] = D[0, k + 1]
    Dsub = D[1:, 1:]
    for mask in range(1, full):
        row = dp[mask]
        if not np.isfinite(row).any():
            continue
        # etendre vers chaque ville j absente de mask : dp[mask|j, j] = min_k dp[mask,k] + D[k,j]
        cand = row[:, None] + Dsub                    # (k, j)
        best_k = np.argmin(cand, axis=0)
        best_v = cand[best_k, np.arange(N)]
        for j in range(N):
            if mask & (1 << j):
                continue
            nm = mask | (1 << j)
            if best_v[j] < dp[nm, j]:
                dp[nm, j] = best_v[j]
                parent[nm, j] = best_k[j]
    last = dp[full - 1] + D[1:, 0]
    k = int(np.argmin(last))
    tour, mask = [], full - 1
    while k != -1:
        tour.append(k + 1)
        pk = parent[mask, k]
        mask ^= 1 << k
        k = int(pk)
    tour = [0] + tour[::-1]
    return tour, tour_length(D, tour)


# ------------------------------------------------------------
#  Google OR-Tools (optionnel) : heuristique industrielle, recherche locale guidee
# ------------------------------------------------------------
def ortools_solve(D, time_limit_s=10):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    n = len(D)
    scale = 1000.0 / max(float(D.max()), 1e-9)       # OR-Tools travaille en entiers
    Di = np.round(D * scale * 1000).astype(np.int64)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    cb = routing.RegisterTransitCallback(
        lambda a, b: int(Di[manager.IndexToNode(a), manager.IndexToNode(b)]))
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))
    sol = routing.SolveWithParameters(params)
    idx, tour = routing.Start(0), []
    while not routing.IsEnd(idx):
        tour.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    return tour, tour_length(D, tour)
