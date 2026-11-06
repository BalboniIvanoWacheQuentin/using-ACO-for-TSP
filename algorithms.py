import numpy as np
import matplotlib.pyplot as plt
import itertools 
import random 
import networkx as nx

import time

np.random.seed(42)
random.seed(42)

def generate_graph(N, p_edges=0.25, min_length=1.00, max_length=10.0, seed=42):
    '''
    genere un arbre aleatoire de N N, ajoute ensuite les aretes avec une probabilite de p_edges
    le graphe aura donc un nombre d'aretes proche de p * (arretes d'un graphe complet).
    On assigne ensuite une distance a chaque arete dans [min_length, max_length]
    '''

    random.seed(seed)
    np.random.seed(seed)

    #Generation de l'arbre de depart
    tsp = nx.random_labeled_tree(N, seed=seed)

    #Addition d'aretes potentielles
    possible_edges = list(itertools.combinations(range(N), 2))
    for i, j in possible_edges:
        if not tsp.has_edge(i, j) and random.random() < p_edges:
            tsp.add_edge(i, j)
    
    #Assigne les poids aux aretes 
    for i, j in tsp.edges():
        #dist aleatoire
        val = round(random.uniform(min_length, max_length), 2)

        #assigne le poids
        tsp[i][j]["weight"] = val

        # assigne la distance pour graphviz
        tsp[i][j]["len"] = val
    
    return tsp

def generate_tsp(N, p_edges=0.1, bounds=(0,100), seed=42 ):
    '''
    genere un arbre aleatoire de N sommets, ajoute ensuite les aretes avec une probabilite de p_edges
    le graphe aura donc un nombre d'aretes proche de p * (arretes d'un graphe complet).
    La distance entre les sommets est issue d'un nuage de points aleatoire dans un carre de taille bounds(a, b).
    '''
    random.seed(seed)
    np.random.seed(seed)

    # Generation du nuage de points
    coords = np.random.uniform(bounds[0], bounds[1], size=(N, 2))

    #Generation de l'arbre de depart
    tsp = nx.random_labeled_tree(N, seed=seed)

    #Attribue les coordonees au tsp
    for i in range(N):
        tsp.nodes[i]["pos"] = coords[i]

    #Addition d'aretes potentielles
    possible_edges = list(itertools.combinations(range(N), 2))
    for i, j in possible_edges:
        if not tsp.has_edge(i, j) and random.random() < p_edges:
            tsp.add_edge(i, j)

    #Calcul des distances
    for u, v in tsp.edges():
        dist = float(np.linalg.norm(coords[u] - coords[v]))
        val = round(dist, 2)

        tsp[u][v]["weight"] = val
    
    return tsp
    
def tour_cost(tour, pos):
    '''Calcule la longueur de la tournee'''
    n = len(tour)
    return sum(
        float(np.linalg.norm(pos[tour[i]] - pos[tour[(i+1)%n]]))
        for i in range(n)
    )

def distance_matrix(G):
    '''
    Matrice (n x n) des distances euclidiennes entre toutes les villes.
    NB : l'ACO et la recherche aleatoire travaillent sur le graphe COMPLET
    (TSP euclidien classique) ; les aretes de G ne servent qu'a l'affichage.
    '''
    coords = np.array([G.nodes[i]["pos"] for i in range(G.number_of_nodes())])
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

def random_search(G, max_iter=2000, seed=42):
    '''
    Recherche la tournee la plus courte de facon aleatoire.
    Renvoie (best_tour, best_cost, history) ; history[k] = meilleur cout apres k+1 tournees evaluees.
    '''
    np.random.seed(seed)
    dist = distance_matrix(G)
    n = len(dist)
    best_tour, best_cost = None, float("inf")
    history = []

    for _ in range(max_iter):
        tour = list(np.random.permutation(n))
        cost = float(dist[tour, np.roll(tour, -1)].sum())
        if cost < best_cost:
            best_cost, best_tour = cost, tour
        history.append(best_cost)

    return best_tour, best_cost, history

def branching_factor(pheromones, lam=0.05):
    '''
    Facteur de lambda-branchement moyen (Dorigo & Gambardella) : nombre moyen d'aretes
    par ville dont la pheromone depasse tau_min + lam * (tau_max - tau_min).
    Vaut n-1 au depart (tout est egal) et tend vers 2 quand la colonie a converge (stagnation).
    '''
    tau = pheromones.astype(float).copy()
    np.fill_diagonal(tau, np.nan)
    lo = np.nanmin(tau, axis=1, keepdims=True)
    hi = np.nanmax(tau, axis=1, keepdims=True)
    return float(np.mean((tau >= lo + lam * (hi - lo)).sum(axis=1)))

def ACO_search(G, N_ants=20, max_iter=100, alpha=1.0, beta=2.0, rho=0.2, Q=100.0, seed=42):
    '''
    Ant System sur le graphe complet.
    Renvoie (best_tour, best_cost, history) ou history contient une valeur par iteration :
      best      : meilleur cout trouve depuis le debut
      iter_best : meilleur cout parmi les fourmis de l'iteration
      mean      : cout moyen des fourmis de l'iteration
      branching : facteur de lambda-branchement (diversite de la recherche)
      time      : temps ecoule depuis le debut (s)
      tau       : matrice des pheromones apres l'iteration
    '''
    random.seed(seed)
    np.random.seed(seed)
    dist = distance_matrix(G)
    n = len(dist)

    # Heuristique eta = 1/d calculee une seule fois (0 si d == 0, comme avant)
    with np.errstate(divide="ignore"):
        eta = np.where(dist > 0, 1.0 / dist, 0.0)
    eta_beta = eta ** beta

    pheromones = np.ones((n, n)) * 0.1
    best_tour, best_cost = None, float("inf")
    history = {"best": [], "iter_best": [], "mean": [], "branching": [], "time": [], "tau": []}
    t0 = time.perf_counter()

    for _ in range(max_iter):
        ant_tours, ant_costs = [], []
        # Attractivite de chaque arete pour cette iteration : tau^alpha * eta^beta
        weights = (pheromones ** alpha) * eta_beta

        for _ in range(N_ants):
            curr = random.randrange(n)
            tour = [curr]
            unvisited = np.ones(n, dtype=bool)
            unvisited[curr] = False

            for _ in range(n - 1):
                targets = np.flatnonzero(unvisited)
                probs = weights[curr, targets]
                s = probs.sum()
                nxt = int(
                    np.random.choice(targets)
                    if s == 0
                    else np.random.choice(targets, p=probs / s)
                )

                tour.append(nxt)
                unvisited[nxt] = False
                curr = nxt

            cost = float(dist[tour, np.roll(tour, -1)].sum())
            ant_tours.append(tour)
            ant_costs.append(cost)

            if cost < best_cost:
                best_cost, best_tour = cost, tour

        pheromones *= 1.0 - rho
        for tour, cost in zip(ant_tours, ant_costs):
            deposit = Q / cost
            for i in range(n):
                u, v = tour[i], tour[(i + 1) % n]
                pheromones[u][v] += deposit
                pheromones[v][u] += deposit

        history["best"].append(best_cost)
        history["iter_best"].append(min(ant_costs))
        history["mean"].append(float(np.mean(ant_costs)))
        history["branching"].append(branching_factor(pheromones))
        history["time"].append(time.perf_counter() - t0)
        history["tau"].append(pheromones.copy())      # pour visualiser les pistes au fil des iterations

    return best_tour, best_cost, history

def plot_convergence(history, filename, title="ACO - convergence"):
    '''Trace l'evolution du cout et du facteur de branchement au fil des iterations'''
    it = range(1, len(history["best"]) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(it, history["mean"], label="cout moyen des fourmis", alpha=0.6)
    ax1.plot(it, history["iter_best"], label="meilleur de l'iteration", alpha=0.6)
    ax1.plot(it, history["best"], label="meilleur global", linewidth=2)
    ax1.set_ylabel("longueur de tournee")
    ax1.set_title(title)
    ax1.legend()
    ax2.plot(it, history["branching"], color="tab:purple")
    ax2.set_ylabel("facteur de branchement")
    ax2.set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

def save_tour(G, pos, tour, cost, exec_time, algo_name, filename):
    ''' Enregistre le tour dans un fichier'''
    plt.figure(figsize=(9, 9))

    # Base graph
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="#3498db")
    nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white")

    # Tour overlay
    tour_edges = [(tour[i], tour[(i + 1) % len(tour)]) for i in range(len(tour))]
    nx.draw_networkx_edges(G, pos, edgelist=tour_edges, edge_color="#e74c3c", width=2.5)

    # Timing header
    time_str = f"{exec_time * 1000:.1f} ms" if exec_time < 1.0 else f"{exec_time:.3f} s"
    plt.title(f"{algo_name}\nTour Distance: {cost:.2f} | Execution Time: {time_str}")

    plt.axis("off")
    plt.savefig(filename, format="svg", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    N = 80
    tsp = generate_tsp(N, p_edges=0.3, seed=42)
    pos = nx.get_node_attributes(tsp, "pos")

    #Random Search
    t0 = time.perf_counter()
    rand_tour, rand_cost, _ = random_search(tsp, max_iter=150000)
    t_rand = time.perf_counter() - t0
    save_tour(tsp, pos, rand_tour, rand_cost, t_rand, "Random Search", "random_search_tour.svg")

    #ACO
    t0 = time.perf_counter()
    aco_tour, aco_cost, aco_history = ACO_search(tsp)
    t_aco = time.perf_counter() - t0
    save_tour(tsp, pos, aco_tour, aco_cost, t_aco, "ACO Search", "aco_search_tour.svg")
    plot_convergence(aco_history, "aco_convergence.svg")