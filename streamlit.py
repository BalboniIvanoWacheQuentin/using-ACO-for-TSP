
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
        "Nombre de villes (n)",
        min_value=5, max_value=30, value=12, step=1,
        help="Taille du problème.",
    )

    graine = st.number_input(
        "Graine aléatoire (reproductibilité)",
        min_value=0, max_value=999, value=42, step=1,
    )

    # ------------------------------------------------------------
    #  Instance : génération / régénération du graphe
    # ------------------------------------------------------------
    if "tsp" not in st.session_state:
        st.session_state["tsp"] = generate_tsp(n_villes, p_edges=0.3,bounds=(0,100), seed=graine)

    if st.session_state.get("regenerer", False):
        st.session_state["tsp"] = generate_tsp(n_villes, p_edges=0.3, bounds=(0,100), seed=graine)
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


if lancer:
    st.info("▶ Lancement demandé")
 
    G = st.session_state["tsp"]

    t0 = time.perf_counter()
    tour, cost = ACO_search(G, N_ants=nb_fourmis, max_iter=nb_iterations)
    t_exec = time.perf_counter() - t0

    st.session_state["resultat"] = {
        "tour": tour,
        "cost": cost,
        "temps": t_exec,
        "algo": "ACO",
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
    st.subheader("Statistiques")

st.subheader("Convergence")
 



# ------------------------------------------------------------
#  Bloc de vérification : valeurs récupérées de l'UI
# ------------------------------------------------------------
with st.expander("🔎 Valeurs récupérées de l'interface (debug)"):
    valeurs = {
        # Instance
        "n_villes": n_villes,
        "graine": graine,
        # Paramètres 
        "nb_fourmis": nb_fourmis,
        "nb_iterations": nb_iterations,
        # Contrôles
        "lancer": lancer,
        "reinit": reinit,
    }
    st.json(valeurs, expanded=True)
    
