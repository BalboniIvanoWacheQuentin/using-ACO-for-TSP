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

def random_search(G, max_iter=2000):
    '''Recherche la tournee la plus courte de facon aleatoire'''
    pos = nx.get_node_attributes(G, "pos")
    nodes = list(G.nodes())
    best_tour, best_cost = None, float("inf")

    for _ in range(max_iter):
        tour = list(np.random.permutation(nodes))
        cost = tour_cost(tour, pos)
        if cost < best_cost:
            best_cost, best_tour = cost, tour

    return best_tour, best_cost

def ACO_search(G, N_ants=20, max_iter=100, alpha=1.0, beta=2.0, rho=0.2, Q=100.0, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    pos = nx.get_node_attributes(G, "pos")
    nodes = list(G.nodes())
    n = len(nodes)

    pheromones = np.ones((n, n)) * 0.1
    best_tour, best_cost = None, float("inf")

    for _ in range(max_iter):
        ant_tours, ant_costs = [], []
        for _ in range(N_ants):
            curr = random.choice(nodes)
            tour, unvisited = [curr], set(nodes) - {curr}

            while unvisited:
                targets = list(unvisited)
                probs = []

                for nxt in targets:
                    d = float(np.linalg.norm(pos[curr] - pos[nxt]))
                    eta = 1.0 / d if d > 0 else 0
                    probs.append(
                        (pheromones[curr][nxt] ** alpha) * (eta**beta)
                    )

                s = sum(probs)
                nxt = (
                    random.choice(targets)
                    if s == 0
                    else np.random.choice(targets, p=[p / s for p in probs])
                )

                tour.append(nxt)
                unvisited.remove(nxt)
                curr = nxt

            cost = tour_cost(tour, pos)
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

    return best_tour, best_cost

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
    rand_tour, rand_cost = random_search(tsp, max_iter=150000)
    t_rand = time.perf_counter() - t0
    save_tour(tsp, pos, rand_tour, rand_cost, t_rand, "Random Search", "random_search_tour.svg")

    #ACO
    t0 = time.perf_counter()
    aco_tour, aco_cost = ACO_search(tsp)
    t_aco = time.perf_counter() - t0
    save_tour(tsp, pos, aco_tour, aco_cost, t_aco, "ACO Search", "aco_search_tour.svg")