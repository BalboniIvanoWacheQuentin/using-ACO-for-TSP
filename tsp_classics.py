'''
Algorithmes "classiques" du TSP, pour comparer avec l'ACO.
Tous prennent une matrice de distances D (numpy, n x n) et renvoient (tour, cout).

  nearest_neighbor     glouton du plus proche voisin            O(n^2)
  two_opt              recherche locale 2-opt, listes des k plus proches voisins et
                       don't-look bits (comme ACOTSP)           O(k) par mouvement teste
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
from numba import njit


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
#  2-opt avec listes de voisins et "don't-look bits" (comme ACOTSP, l'implementation de
#  reference de Dorigo & Stutzle), compile en code machine avec numba
# ------------------------------------------------------------
@njit(cache=True)
def _two_opt_nb(D, tour, knn, tol):
    '''
    Premiere amelioration : pour chaque ville a, on essaie de creer une arete (a, c) vers l'un
    de ses k plus proches voisins c. Les listes sont triees : des que d(a, c) >= d(a, b), aucun
    gain n'est possible et on s'arrete. Une ville dont aucun voisin n'ameliore la tournee est
    "eteinte" (don't-look bit) jusqu'a ce qu'une de ses aretes change.
    '''
    n = tour.shape[0]
    pos = np.empty(n, np.int64)
    for i in range(n):
        pos[tour[i]] = i
    dlb = np.zeros(n, np.bool_)
    improved = True
    while improved:
        improved = False
        for a in range(n):
            if dlb[a]:
                continue
            found = False
            for succ in (True, False):
                i = pos[a]
                b = tour[(i + 1) % n] if succ else tour[(i - 1) % n]
                d_ab = D[a, b]
                for kk in range(knn.shape[1]):
                    c = knn[a, kk]
                    if D[a, c] >= d_ab:
                        break
                    j = pos[c]
                    d = tour[(j + 1) % n] if succ else tour[(j - 1) % n]
                    if c == b or d == a:
                        continue
                    delta = D[a, c] + D[b, d] - d_ab - D[c, d]
                    if delta < -tol:
                        # aretes aux positions lo et hi : on inverse le segment lo+1 .. hi
                        lo, hi = (i, j) if succ else ((i - 1) % n, (j - 1) % n)
                        if lo > hi:
                            lo, hi = hi, lo
                        x, y = lo + 1, hi
                        while x < y:
                            tour[x], tour[y] = tour[y], tour[x]
                            pos[tour[x]] = x
                            pos[tour[y]] = y
                            x += 1
                            y -= 1
                        dlb[a] = dlb[b] = dlb[c] = dlb[d] = False
                        found = improved = True
                        break
                if found:
                    break
            if not found:
                dlb[a] = True
    return tour


_KNN_CACHE = [None, None, None]              # (D, D en numpy, listes de voisins) du dernier appel


def two_opt_batch(D, tours, max_passes=None, tol=1e-9, k=20):
    '''
    2-opt applique a un lot de tournees (tenseurs torch : D (n, n), tours (m, n)).
    k : taille des listes de voisins (20 comme dans ACOTSP). Chaque mouvement coute O(k) a trouver
    et O(n) a appliquer (inversion d'un segment).
    max_passes : conserve pour compatibilite, non utilise.
    '''
    m, n = tours.shape
    if n < 5:
        return tours
    if _KNN_CACHE[0] is not D:
        Dn = D.detach().cpu().numpy().astype(np.float64)
        k_eff = min(k, n - 1)
        knn = np.argsort(Dn + np.diag(np.full(n, np.inf)), axis=1)[:, :k_eff].astype(np.int64)
        _KNN_CACHE[:] = [D, Dn, knn]
    _, Dn, knn = _KNN_CACHE
    out = tours.detach().cpu().numpy().astype(np.int64)
    for r in range(m):
        out[r] = _two_opt_nb(Dn, out[r].copy(), knn, tol)
    return torch.from_numpy(out).to(tours.device)


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
def ortools_solve(D, time_limit_s=10, init_tour=None):
    '''init_tour : tournee de depart (reprise "a chaud"), sinon OR-Tools construit la sienne'''
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
    if init_tour is None:
        sol = routing.SolveWithParameters(params)
    else:
        routing.CloseModelWithParameters(params)
        k = list(init_tour).index(0)                  # la tournee doit partir du depot 0
        route = [int(v) for v in list(init_tour[k + 1:]) + list(init_tour[:k])]
        start = routing.ReadAssignmentFromRoutes([route], True)
        sol = routing.SolveFromAssignmentWithParameters(start, params)
    if sol is None:                                   # aucune solution dans le temps imparti
        return None, float("inf")
    idx, tour = routing.Start(0), []
    while not routing.IsEnd(idx):
        tour.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    return tour, tour_length(D, tour)
