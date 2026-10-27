
from algorithms import*
import streamlit as st




# ============================================================
# TP 6 — Colonies de fourmis
# ============================================================

st.set_page_config(page_title="Colonies de fourmis", page_icon="🐜", layout="wide")

# ------------------------------------------------------------
#  Barre latérale — Paramètres de l'instance
# ------------------------------------------------------------
with st.sidebar:
    st.header("Instance")



    n_villes = st.slider(
        "Nombre de villes",
        min_value=5, max_value=30, value=12, step=1,
        help="Taille du problème.",
    )

    graine = st.number_input(
        "Graine aléatoire (reproductibilité)",
        min_value=0, max_value=999, value=42, step=1,
    )


    p_edges = st.slider(
        "Densité d'arêtes",
        min_value=0.1, max_value=1.0, value=0.3, step=0.1,
        help="Probabilité d'ajouter chaque arête manquante. "
             "p=1.0 → graphe complet (TSP classique).",
    )

    col_bmin, col_bmax = st.columns(2)
    bound_min = col_bmin.number_input(
        "Zone : min",
        min_value=0.0, max_value=100.0, value=0.0, step=1.0,
    )
    bound_max = col_bmax.number_input(
        "Zone : max",
        min_value=0.0, max_value=200.0, value=100.0, step=1.0,
    )
    bounds = (bound_min, bound_max)


    # ------------------------------------------------------------
    #  Instance : génération / régénération du graphe
    # ------------------------------------------------------------
    if "tsp" not in st.session_state:
        st.session_state["tsp"] = generate_tsp(
            n_villes, p_edges=p_edges, bounds=bounds, seed=graine
        )

    if st.session_state.get("regenerer", False):
        st.session_state["tsp"] = generate_tsp(
            n_villes, p_edges=p_edges, bounds=bounds, seed=graine
        )
        st.session_state["regenerer"] = False
        st.session_state.pop("resultat", None)


    if st.button("Nouvelle instance"):
        st.session_state["regenerer"] = True
        st.rerun()
 

    # --------------------------------------------------------
    #  Barre latérale — Paramètres de l'algorithme 
    # --------------------------------------------------------
    st.header("Paramètres")

    nb_fourmis = st.slider(
        "Nombre de fourmis",
        min_value=1, max_value=100, value=20, step=1,
        help="Plus de fourmis = plus de solutions explorées par itération (coût + élevé).",
    )

    nb_iterations = st.number_input(
        "Nombre d'itérations",
        min_value=100, max_value=1000, value=200, step=100,
    )

 
    # --------------------------------------------------------
    #  Barre latérale — Paramètres ACO
    # --------------------------------------------------------
    st.subheader("Paramètres ACO")

    col_a, col_b = st.columns(2)
    alpha = col_a.slider(
        "α — poids des phéromones",
        min_value=0.0, max_value=3.0, value=1.0, step=0.1,
        help="Influence des phéromones sur le choix de la ville suivante. "
             "Élevé = exploitation des pistes existantes.",
    )
    beta = col_b.slider(
        "β — poids de la distance",
        min_value=0.0, max_value=5.0, value=2.0, step=0.1,
        help="Influence de la proximité (1/distance). "
             "Élevé = comportement glouton, fourmis attirées par les villes proches.",
    )

    col_r, col_q = st.columns(2)
    rho = col_r.slider(
        "ρ — évaporation",
        min_value=0.0, max_value=1.0, value=0.2, step=0.1,
        help="Taux d'évaporation par itération. Élevé = mémoire courte.",
    )
    Q = col_q.number_input(
        "Q — dépôt de phéromone",
        min_value=1.0, max_value=1000.0, value=100.0, step=1.0,
        help="Quantité déposée, proportionnelle à Q / coût de la tournée.",
    )


    # --------------------------------------------------------
    #  Barre latérale — Préréglages et lancement
    # --------------------------------------------------------
    st.header("Lancement")

    col_run, col_reset = st.columns(2)

    lancer = col_run.button("▶ Lancer")
    reinit = col_reset.button("↺ Réinitialiser")

    if reinit:
        # TODO : remettre la matrice de phéromones à zéro / régénérer l'instance
        st.rerun()

# ------------------------------------------------------------
#  Zone principale — Réservée à l'algorithme 
# ------------------------------------------------------------
st.title("🐜 Colonies de fourmis & problème du voyageur de commerce")

st.markdown(
    "Dépôt/évaporation de phéromones, compromis exploration/exploitation, "
)


# ------------------------------------------------------------
#  Lancement de l'ACO
# ------------------------------------------------------------
if lancer:
    G = st.session_state["tsp"]

    with st.status("🐜 Exécution de l'ACO en cours…", expanded=True) as status:
        t0 = time.perf_counter()
        tour, cost = ACO_search(
            G,
            N_ants=nb_fourmis,
            max_iter=nb_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            Q=Q,
        )
        t_exec = time.perf_counter() - t0

        status.update(
            label=f"✅ Terminé en {t_exec:.2f} s",
            state="complete",
        )

    st.session_state["resultat"] = {
        "tour": tour,
        "cost": cost,
        "temps": t_exec,
        "algo": "ACO",
        "params": {"nb_fourmis": nb_fourmis, "nb_iterations": nb_iterations,
                   "alpha": alpha, "beta": beta, "rho": rho, "Q": Q},
    }

# ------------------------------------------------------------
#  Disposition principale : visualisation + statistiques
# ------------------------------------------------------------
col_carte, col_stats = st.columns([3, 2])



with col_carte:
    st.subheader("Carte des phéromones & meilleure tournée")

    G = st.session_state["tsp"]
    pos = nx.get_node_attributes(G, "pos")

    if "resultat" in st.session_state:
        res = st.session_state["resultat"]
        # Sauvegarde la carte via la fonction du fichier algorithms.py
        save_tour(
            G, pos,
            res["tour"], res["cost"],
            res["temps"], res["algo"],
            "meilleure_tournee.svg",
        )
        # Affichage dans Streamlit
        st.image("meilleure_tournee.svg")

        st.write(f"Vérification (tour_cost) : {tour_cost(res['tour'], pos):.2f}")
    else:
        st.info("Cliquez sur ▶ Lancer pour afficher la meilleure tournée.")

 
  
with col_stats:
    st.subheader("Statistiques de l'instance")

    if "resultat" in st.session_state:
        res = st.session_state["resultat"]
        G = st.session_state["tsp"]

        st.metric("Longueur de la tournée", f"{res['cost']:.2f}")
        st.metric("Temps d'exécution", f"{res['temps']:.2f} s")

        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Villes", G.number_of_nodes())
        col_s2.metric("Arêtes", G.number_of_edges())
        col_s1.metric("Temps / itération", f"{1000 * res['temps'] / nb_iterations:.1f} ms")
        col_s2.metric("Longueur / ville", f"{res['cost'] / G.number_of_nodes():.2f}")

        with st.expander("Détails de l'instance"):
            poids = [d["weight"] for _, _, d in G.edges(data=True)]
            st.json({
                "densité": round(nx.density(G), 3),
                "distance_min": min(poids),
                "distance_max": max(poids),
                "distance_moyenne": round(sum(poids) / len(poids), 2),
                "connexe": nx.is_connected(G),
            })
    else:
        st.info("Aucun résultat. Cliquez sur ▶ Lancer.")

 
 



# ------------------------------------------------------------
#  Bloc de vérification : valeurs récupérées de l'UI
# ------------------------------------------------------------
with st.expander("🔎 Valeurs récupérées de l'interface (debug)"):
    valeurs = {
        "n_villes": n_villes,
        "graine": graine,
        "p_edges": p_edges,
        "bounds": list(bounds),
        "nb_fourmis": nb_fourmis,
        "nb_iterations": nb_iterations,
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "Q": Q,
        "lancer": lancer,
        "reinit": reinit,
    }
    st.json(valeurs, expanded=True)
    
