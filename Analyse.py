"""
Analyse des expériences ACO (colonies de fourmis) sur le TSP.

Lancer :  streamlit run app.py
Dépendances : streamlit (idéalement >= 1.50, compatible avec les anciennes versions), pandas, numpy, plotly

Par défaut, l'app cherche runs.csv et history.csv dans le même dossier que app.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ACO & TSP – Analyse des résultats", page_icon="🐜", layout="wide")

BASE_DIR = Path(__file__).parent
PARAMS = ["N_ants", "alpha", "beta", "rho"]
PARAM_LABELS = {
    "N_ants": "N_ants (nombre de fourmis)",
    "alpha": "alpha (poids des phéromones)",
    "beta": "beta (poids de l'heuristique 1/d)",
    "rho": "rho (évaporation)",
}
RUN_METRICS = {
    "best_cost": "Meilleur coût (longueur du tour)",
    "time_s": "Temps d'exécution (s)",
    "iter_best_found": "Itération du meilleur tour",
    "iter_within_1pct": "Itération à moins de 1 % du meilleur",
    "iter_within_5pct": "Itération à moins de 5 % du meilleur",
    "final_branching": "Facteur de branchement final",
}
HIST_METRICS = {
    "best": "Meilleur coût depuis le début",
    "iter_best": "Meilleur coût de l'itération",
    "mean": "Coût moyen des fourmis",
    "branching": "Facteur de branchement",
    "time_s": "Temps cumulé (s)",
}


# --------------------------------------------------------------------------- #
# Compatibilité avec les anciennes versions de Streamlit
# --------------------------------------------------------------------------- #
def _version():
    out = []
    for x in st.__version__.split(".")[:2]:
        out.append(int("".join(ch for ch in x if ch.isdigit()) or 0))
    return tuple(out)


_NEW_API = _version() >= (1, 50)

# st.cache_data a remplacé st.cache à partir de Streamlit 1.18
if hasattr(st, "cache_data"):
    cache_data = st.cache_data
else:
    cache_data = st.cache(allow_output_mutation=True, show_spinner=False)


def make_tabs(labels):
    if hasattr(st, "tabs"):
        return st.tabs(labels)
    out = []  # très ancienne version : sections empilées au lieu d'onglets
    for lab in labels:
        c = st.container()
        c.markdown("## " + lab)
        out.append(c)
    return out


def columns(spec):
    if hasattr(st, "columns"):
        return st.columns(spec)
    return st.beta_columns(spec)


def plot(fig):
    if _NEW_API:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def table(df, hide_index=False):
    if _NEW_API:
        st.dataframe(df, width="stretch", hide_index=hide_index)
        return
    try:
        st.dataframe(df, use_container_width=True, hide_index=hide_index)
    except TypeError:  # très ancienne version : pas de hide_index
        st.dataframe(df.reset_index(drop=True) if hide_index else df)


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
@cache_data
def read_path(path):
    return pd.read_csv(path)


def load(name):
    p = BASE_DIR / f"{name}.csv"
    return read_path(str(p)) if p.exists() else None


def baseline_of(runs):
    """Valeur la plus fréquente de chaque paramètre parmi les runs ACO (= config de référence)."""
    aco = runs[runs["algo"] == "ACO"]
    return {p: aco[p].mode().iloc[0] for p in PARAMS}


def at_baseline(df, baseline, vary):
    """Garde les runs ACO où tous les paramètres sauf `vary` sont à leur valeur de référence."""
    mask = df["algo"] == "ACO"
    for p in PARAMS:
        if p != vary:
            mask &= df[p] == baseline[p]
    return df[mask]


def fmt_baseline(baseline, vary):
    return ", ".join(f"{p} = {v:g}" for p, v in baseline.items() if p != vary)


# --------------------------------------------------------------------------- #
# Page RUNS
# --------------------------------------------------------------------------- #
def page_runs(runs):
    st.header("📊 Statistiques sur les runs")

    aco = runs[runs["algo"] == "ACO"]
    rnd = runs[runs["algo"] != "ACO"]
    best = runs["best_cost"].min()
    baseline = baseline_of(runs)

    c = columns(5)
    c[0].metric("Runs", len(runs))
    c[1].metric("Runs ACO", len(aco))
    c[2].metric("Runs aléatoires", len(rnd))
    c[3].metric("Meilleur coût", f"{best:,.1f}")
    if len(rnd):
        gain = 1 - aco["best_cost"].median() / rnd["best_cost"].median()
        c[4].metric("Gain médian ACO vs aléatoire", f"{gain:.0%}")

    t_over, t_sens, t_corr, t_data = make_tabs(
        ["Vue d'ensemble", "Sensibilité aux paramètres", "Corrélations & coût/temps", "Données"]
    )

    # ---- Vue d'ensemble -------------------------------------------------- #
    with t_over:
        left, right = columns(2)
        with left:
            st.subheader("ACO vs recherche aléatoire")
            fig = px.box(runs, x="algo", y="best_cost", color="algo", points="all",
                         labels={"best_cost": "Meilleur coût", "algo": "Algorithme"})
            fig.update_layout(showlegend=False)
            plot(fig)
        with right:
            st.subheader("Distribution du meilleur coût (ACO)")
            fig = px.histogram(aco, x="best_cost", nbins=40, labels={"best_cost": "Meilleur coût"})
            fig.update_layout(yaxis_title="Nombre de runs")
            plot(fig)

        st.subheader("Résumé par algorithme")
        summary = runs.groupby("algo")[["best_cost", "time_s"]].agg(["count", "mean", "std", "min", "median", "max"])
        table(summary.round(3))

        st.subheader("Meilleures configurations ACO (moyenne sur les seeds)")
        top = (aco.groupby(PARAMS)
               .agg(runs=("run_id", "count"), cout_moyen=("best_cost", "mean"), cout_std=("best_cost", "std"),
                    cout_min=("best_cost", "min"), temps_moyen=("time_s", "mean"),
                    iter_meilleur=("iter_best_found", "mean"))
               .sort_values("cout_moyen").reset_index())
        top["ecart_au_meilleur_%"] = (top["cout_moyen"] / best - 1) * 100
        n_top = st.slider("Nombre de configurations affichées", 3, min(30, len(top)), min(10, len(top)))
        table(top.head(n_top).round(3), hide_index=True)

    # ---- Sensibilité ------------------------------------------------------ #
    with t_sens:
        st.subheader("Effet d'un paramètre, les autres étant fixés à leur valeur de référence")
        c1, c2 = columns(2)
        param = c1.selectbox("Paramètre à faire varier", PARAMS, format_func=PARAM_LABELS.get)
        metric = c2.selectbox("Métrique", list(RUN_METRICS), format_func=RUN_METRICS.get)
        st.caption(f"Valeurs de référence fixées : {fmt_baseline(baseline, param)}")

        sub = at_baseline(runs, baseline, param)
        agg = (sub.groupby(param)[metric]
               .agg(["count", "mean", "std", "min", "median", "max"]).reset_index())

        fig = go.Figure()
        fig.add_trace(go.Box(x=sub[param].astype(str), y=sub[metric], name="Runs (seeds)",
                             boxpoints="all", jitter=0.4, marker_color="#888", showlegend=False))
        fig.add_trace(go.Scatter(x=agg[param].astype(str), y=agg["mean"], mode="lines+markers",
                                 error_y=dict(type="data", array=agg["std"].fillna(0)),
                                 name="Moyenne ± écart-type", line=dict(color="#d62728", width=3)))
        if metric in ("best_cost", "time_s") and len(rnd):
            fig.add_hline(y=rnd[metric].mean(), line_dash="dash", line_color="#1f77b4",
                          annotation_text="moyenne aléatoire")
        fig.update_layout(xaxis_title=PARAM_LABELS[param], yaxis_title=RUN_METRICS[metric],
                          legend=dict(orientation="h", y=1.08))
        plot(fig)
        table(agg.round(3), hide_index=True)

        with st.expander("Voir les 4 paramètres côte à côte (moyenne du meilleur coût)"):
            cols = columns(2)
            for i, p in enumerate(PARAMS):
                s = at_baseline(runs, baseline, p).groupby(p)[metric].agg(["mean", "std"]).reset_index()
                f = go.Figure(go.Scatter(x=s[p].astype(str), y=s["mean"], mode="lines+markers",
                                         error_y=dict(type="data", array=s["std"].fillna(0))))
                f.update_layout(title=PARAM_LABELS[p], yaxis_title=RUN_METRICS[metric], height=300,
                                margin=dict(t=40, b=20))
                plot(f)

    # ---- Corrélations ------------------------------------------------------ #
    with t_corr:
        left, right = columns(2)
        with left:
            st.subheader("Coût vs temps")
            color = st.selectbox("Couleur", ["algo"] + PARAMS, key="scatter_color")
            d = runs.copy()
            if color != "algo":
                d[color] = d[color].astype(str)
            fig = px.scatter(d, x="time_s", y="best_cost", color=color, hover_data=["run_id"] + PARAMS,
                             labels={"time_s": "Temps (s)", "best_cost": "Meilleur coût"})
            plot(fig)
        with right:
            st.subheader("Matrice de corrélation (ACO)")
            cols = [c for c in PARAMS + list(RUN_METRICS) if aco[c].nunique() > 1]
            corr = aco[cols].corr(method="spearman")
            fig = px.imshow(corr.round(2), text_auto=True, zmin=-1, zmax=1,
                            color_continuous_scale="RdBu_r", aspect="auto")
            plot(fig)
            st.caption("Corrélation de Spearman. Attention : les paramètres sont variés un par un autour "
                       "de la référence, la corrélation mesure donc surtout des effets marginaux.")

    # ---- Données ------------------------------------------------------------ #
    with t_data:
        st.subheader("Filtrer les données")
        f = runs
        cols = columns(5)
        algos = cols[0].multiselect("algo", sorted(runs["algo"].unique()), default=sorted(runs["algo"].unique()))
        f = f[f["algo"].isin(algos)]
        for col, p in zip(cols[1:], PARAMS):
            vals = sorted(runs[p].unique())
            sel = col.multiselect(p, vals, default=vals)
            f = f[f[p].isin(sel)]
        st.write(f"{len(f)} runs sélectionnés")
        table(f, hide_index=True)
        table(f.describe().T.round(3))


# --------------------------------------------------------------------------- #
# Page HISTORY
# --------------------------------------------------------------------------- #
def page_history(hist, runs):
    st.header("📈 Statistiques sur l'historique de convergence")

    n_runs = hist["run_id"].nunique()
    c = columns(4)
    c[0].metric("Lignes", f"{len(hist):,}".replace(",", " "))
    c[1].metric("Runs", n_runs)
    c[2].metric("Itérations / run", int(hist.groupby("run_id").size().max()))
    c[3].metric("Meilleur coût final", f"{hist.groupby('run_id')['best'].min().min():,.1f}")

    # Jointure avec les paramètres si runs.csv est disponible
    if runs is not None:
        h = hist.merge(runs[["run_id", "algo", "run_seed"] + PARAMS], on="run_id", how="left")
    else:
        h = hist.copy()

    modes = ["Par run_id"] + (["Par paramètre (moyenne sur les seeds)"] if runs is not None else [])
    mode = st.radio("Mode d'analyse", modes, horizontal=True)

    metric = st.selectbox("Métrique", list(HIST_METRICS), format_func=HIST_METRICS.get)
    log_y = st.checkbox("Axe Y logarithmique", value=False)

    t_curve, t_final, t_raw = make_tabs(["Courbes", "Statistiques finales par run", "Données"])

    with t_curve:
        if mode == "Par run_id":
            ids = sorted(h["run_id"].unique())
            sel = st.multiselect("Runs à afficher", ids, default=ids[:3])
            d = h[h["run_id"].isin(sel)].copy()
            d["run_id"] = d["run_id"].astype(str)
            fig = px.line(d, x="iteration", y=metric, color="run_id",
                          labels={"iteration": "Itération", metric: HIST_METRICS[metric]})
        else:
            baseline = baseline_of(runs)
            c1, c2, c3 = columns(3)
            param = c1.selectbox("Paramètre à faire varier", PARAMS, format_func=PARAM_LABELS.get)
            band = c2.checkbox("Afficher ± écart-type", value=True)
            with_random = c3.checkbox("Ajouter la recherche aléatoire", value=True,
                                      disabled=metric != "best")
            st.caption(f"Valeurs de référence fixées : {fmt_baseline(baseline, param)}")

            d = at_baseline(h, baseline, param)
            g = d.groupby([param, "iteration"])[metric].agg(["mean", "std"]).reset_index()
            fig = go.Figure()
            palette = px.colors.qualitative.Plotly
            for i, (val, gg) in enumerate(g.groupby(param)):
                col = palette[i % len(palette)]
                fig.add_trace(go.Scatter(x=gg["iteration"], y=gg["mean"], name=f"{param} = {val:g}",
                                         line=dict(color=col)))
                if band:
                    sd = gg["std"].fillna(0)
                    fig.add_trace(go.Scatter(
                        x=list(gg["iteration"]) + list(gg["iteration"])[::-1],
                        y=list(gg["mean"] + sd) + list(gg["mean"] - sd)[::-1],
                        fill="toself", fillcolor=col, opacity=0.15, line=dict(width=0),
                        hoverinfo="skip", showlegend=False))
            if with_random and metric == "best":
                r = h[h["algo"] == "random"].groupby("iteration")["best"].mean().reset_index()
                if len(r):
                    fig.add_trace(go.Scatter(x=r["iteration"], y=r["best"], name="aléatoire",
                                             line=dict(color="black", dash="dash")))
            fig.update_layout(xaxis_title="Itération", yaxis_title=HIST_METRICS[metric])
        if log_y:
            fig.update_yaxes(type="log")
        plot(fig)

    with t_final:
        last = h.sort_values("iteration").groupby("run_id").tail(1).set_index("run_id")
        first = h.sort_values("iteration").groupby("run_id").head(1).set_index("run_id")
        stats = pd.DataFrame({
            "cout_initial": first["best"],
            "cout_final": last["best"],
            "amelioration_%": (1 - last["best"] / first["best"]) * 100,
            "temps_total_s": last["time_s"],
        })
        # première itération où le coût final est atteint
        stats["iter_plateau"] = h.merge(last[["best"]].rename(columns={"best": "final"}), on="run_id") \
            .query("best <= final").groupby("run_id")["iteration"].min()
        if runs is not None:
            stats = stats.join(runs.set_index("run_id")[["algo"] + PARAMS])
        table(stats.round(3))
        table(stats.describe().T.round(3))

    with t_raw:
        table(hist.head(1000), hide_index=True)
        st.caption("1000 premières lignes affichées.")
        table(hist.describe().T.round(3))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("🐜 Colonies de fourmis & problème du voyageur de commerce")

with st.sidebar:
    st.header("Données")
    choice = st.radio("Fichier à analyser", ["runs.csv", "history.csv"], key="dataset").replace(".csv", "")

runs_df = load("runs")
hist_df = load("history")

if choice == "runs":
    if runs_df is None:
        st.warning("runs.csv introuvable : place-le dans le même dossier que le script.")
    else:
        page_runs(runs_df)
else:
    if hist_df is None:
        st.warning("history.csv introuvable : place-le dans le même dossier que le script.")
    else:
        if runs_df is None:
            st.info("runs.csv absent : l'analyse par paramètre est désactivée (seul le mode « Par run_id » est disponible).")
        page_history(hist_df, runs_df)