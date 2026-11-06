'''
ACO "optimises" : toutes les fourmis d'une iteration construisent leur tournee EN MEME TEMPS
(operations vectorisees sur des matrices m x n avec PyTorch). Le meme code tourne sur CPU ou GPU.

Variantes :
  AS    Ant System (Dorigo 1996)            : toutes les fourmis deposent Q / L
  MMAS  MAX-MIN Ant System (Stutzle & Hoos 2000) : seule la meilleure fourmi depose,
        pheromones bornees dans [tau_min, tau_max], reinitialisation en cas de stagnation
  ACS   Ant Colony System (Dorigo & Gambardella 1997) : regle pseudo-aleatoire (q0),
        mise a jour locale pendant la construction, mise a jour globale par la meilleure tournee
  + local_search=True : 2-opt applique aux meilleures fourmis de chaque iteration

Note : dans ACS "classique" les fourmis avancent l'une apres l'autre ; ici elles avancent en
parallele et la mise a jour locale est appliquee apres chaque pas de toutes les fourmis.
'''
import time

import numpy as np
import torch

from tsp_classics import nearest_neighbor, two_opt_batch, tour_length

DEFAULTS = {
    # rho = evaporation
    "AS":   dict(alpha=1.0, beta=2.0, rho=0.5),
    "MMAS": dict(alpha=1.0, beta=2.0, rho=0.02),
    "ACS":  dict(alpha=1.0, beta=2.0, rho=0.1),
}


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def branching_factor(tau, lam=0.05):
    '''Facteur de lambda-branchement moyen (cf. algorithms.branching_factor)'''
    n = tau.shape[0]
    diag = torch.eye(n, dtype=torch.bool, device=tau.device)
    lo = tau.masked_fill(diag, float("inf")).min(dim=1, keepdim=True).values
    hi = tau.masked_fill(diag, float("-inf")).max(dim=1, keepdim=True).values
    return float(((tau >= lo + lam * (hi - lo)) & ~diag).sum(dim=1).float().mean())


def aco(D_np, variant="MMAS", device="cpu", n_ants=None, max_iter=200,
        alpha=None, beta=None, rho=None, q0=0.9, xi=0.1, p_best=0.05,
        local_search=False, ls_top=8, stagnation_reset=50, seed=0, time_limit=None,
        init_tau=None):
    '''
    D_np : matrice des distances (numpy n x n).
    init_tau : pheromones de depart (reprise apres un changement des distances), sinon tau0.
    Renvoie (best_tour, best_cost, history) ; history a une valeur par iteration :
      best, iter_best, mean, branching, time (secondes depuis le debut)
      et history["tau"] = pheromones finales.
    '''
    p = DEFAULTS[variant]
    alpha = p["alpha"] if alpha is None else alpha
    beta = p["beta"] if beta is None else beta
    rho = p["rho"] if rho is None else rho
    if variant == "MMAS" and local_search and rho == DEFAULTS["MMAS"]["rho"]:
        rho = 0.2                                       # valeur conseillee avec 2-opt

    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    n = len(D_np)
    m = n_ants or n

    # distances normalisees (moyenne = 1) : evite les debordements en float32, ne change pas les choix
    scale = float(D_np[D_np > 0].mean())
    D = torch.tensor(D_np / scale, dtype=torch.float32, device=dev)
    eta = torch.where(D > 0, 1.0 / D.clamp_min(1e-12), torch.zeros_like(D))
    eta_beta = eta ** beta

    L_nn = nearest_neighbor(D_np)[1] / scale
    if variant == "AS":
        tau0 = m / L_nn
    elif variant == "MMAS":
        tau0 = 1.0 / (rho * L_nn)
    else:
        tau0 = 1.0 / (n * L_nn)
    tau = torch.full((n, n), tau0, device=dev) if init_tau is None else init_tau.clone().to(dev)

    rows = torch.arange(m, device=dev)
    best_tour, best_len = None, float("inf")
    since_improve = 0
    history = {"best": [], "iter_best": [], "mean": [], "branching": [], "time": []}
    _sync(dev)
    t0 = time.perf_counter()

    for it in range(max_iter):
        weights = (tau ** alpha) * eta_beta

        # --- construction des m tournees en parallele ---
        cur = torch.randint(0, n, (m,), generator=gen, device=dev)
        tours = torch.empty((m, n), dtype=torch.long, device=dev)
        tours[:, 0] = cur
        visited = torch.zeros((m, n), dtype=torch.bool, device=dev)
        visited[rows, cur] = True
        for k in range(1, n):
            w = weights[cur].masked_fill(visited, 0.0)
            w = w + 1e-30 * (~visited)                 # evite une ligne entierement nulle
            nxt = torch.multinomial(w, 1, generator=gen).squeeze(1)
            if variant == "ACS":
                greedy = w.argmax(dim=1)
                u = torch.rand(m, generator=gen, device=dev)
                nxt = torch.where(u < q0, greedy, nxt)
            tours[:, k] = nxt
            visited[rows, nxt] = True
            if variant == "ACS":                        # mise a jour locale
                tau[cur, nxt] = (1 - xi) * tau[cur, nxt] + xi * tau0
                tau[nxt, cur] = tau[cur, nxt]
                weights[cur, nxt] = (tau[cur, nxt] ** alpha) * eta_beta[cur, nxt]
                weights[nxt, cur] = weights[cur, nxt]
            cur = nxt

        # --- recherche locale sur les meilleures fourmis ---
        L = D[tours, tours.roll(-1, dims=1)].sum(dim=1)
        if local_search:
            top = L.topk(min(ls_top, m), largest=False).indices
            tours[top] = two_opt_batch(D, tours[top], tol=1e-6)
            L = D[tours, tours.roll(-1, dims=1)].sum(dim=1)

        ib = int(L.argmin())
        ib_len = float(L[ib])
        if ib_len < best_len - 1e-9:
            best_len, best_tour = ib_len, tours[ib].clone()
            since_improve = 0
        else:
            since_improve += 1

        # --- mise a jour des pheromones ---
        if variant == "AS":
            tau *= (1 - rho)
            u, v = tours.flatten(), tours.roll(-1, dims=1).flatten()
            dep = (1.0 / L)[:, None].expand(m, n).flatten()
            tau.index_put_((u, v), dep, accumulate=True)
            tau.index_put_((v, u), dep, accumulate=True)
        else:
            if variant == "MMAS":
                # meilleure de l'iteration, et meilleure globale une iteration sur 5
                dep_tour, dep_len = (best_tour, best_len) if it % 5 == 4 else (tours[ib], ib_len)
                tau *= (1 - rho)
                amount = 1.0 / dep_len
            else:  # ACS : seule la meilleure globale, avec evaporation sur ses aretes uniquement
                dep_tour, dep_len = best_tour, best_len
                amount = rho / dep_len
            u, v = dep_tour, dep_tour.roll(-1)
            if variant == "ACS":
                tau[u, v] *= (1 - rho)
                tau[v, u] = tau[u, v]
            tau[u, v] += amount
            tau[v, u] = tau[u, v]
            if variant == "MMAS":
                tau_max = 1.0 / (rho * best_len)
                pdec = p_best ** (1.0 / n)
                tau_min = tau_max * (1 - pdec) / ((n / 2 - 1) * pdec)
                tau.clamp_(tau_min, tau_max)
                # reinitialisation seulement si la colonie a converge (branchement ~ 2)
                # ET ne progresse plus depuis longtemps (critere de Stutzle & Hoos)
                if (stagnation_reset and since_improve >= stagnation_reset
                        and branching_factor(tau) < 2.2):
                    tau.fill_(tau_max)                 # on repart (en gardant la meilleure tournee)
                    since_improve = 0

        _sync(dev)
        history["best"].append(best_len * scale)
        history["iter_best"].append(ib_len * scale)
        history["mean"].append(float(L.mean()) * scale)
        history["branching"].append(branching_factor(tau))
        history["time"].append(time.perf_counter() - t0)
        if time_limit and history["time"][-1] > time_limit:
            break

    history["tau"] = tau
    tour = best_tour.cpu().tolist()
    return tour, tour_length(D_np, tour), history
