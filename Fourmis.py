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


    if st.button("Nouvelle instance"):
        # TODO : régénérer les coordonnées des villes avec la graine
        # (ex. inst.villes = generer_villes(n_villes, graine, topologie))
        st.success("Instance à (re)générer.")
 

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


# ------------------------------------------------------------
#  Disposition principale : visualisation + statistiques
# ------------------------------------------------------------
col_carte, col_stats = st.columns([3, 2])

with col_carte:
    st.subheader("Carte des phéromones & meilleure tournée")
    # tracer les villes + arêtes (épaisseur et phéromone)
    # via st.pyplot() ou st.plotly_chart() une fois l'algo branché
 

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
    
