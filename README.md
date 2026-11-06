# Colonies de fourmis et problème du voyageur de commerce

Précision contre temps de calcul : quand utiliser l'ACO (Ant Colony Optimization) pour le TSP ?

Application en ligne : https://using-aco-for-tsp-faqcu8kd7htkggervtpugk.streamlit.app

## Application Streamlit

```bash
pip install -r requirements.txt
streamlit run streamlit.py
```

| Page | Fichier | Contenu |
|---|---|---|
| Démonstrateur | `Fourmis.py` | ACO (Ant System) sur 5 à 30 villes : tournée, statistiques, convergence et facteur de branchement |
| Analyse | `Analyse.py` | Étude des paramètres α, β, ρ et du nombre de fourmis (`runs.csv`, `history.csv`) |
| Présentation | `presentation.py` | Diapos de la soutenance (`presentation.pdf`) |

## Reproduire les résultats de la présentation

```bash
pip install -r requirements.txt -r requirements-experiences.txt
```

| Commande | Résultat | Durée |
|---|---|---|
| `python experiments.py` | Étude des paramètres (30 villes, 10 graines) : `runs.csv`, `history.csv` | quelques minutes |
| `python accuracy_time.py` | Complexité (`results/complexity.csv`) et précision vs temps sur eil51, kroA100, ch150 (`results/anytime.csv`) | ≈ 35 min |
| `python plot_accuracy_time.py` | Figures `results/figures/` et tableau par budget `results/budget_table.csv` | quelques secondes |
| `python slides/make_figures.py` | Figures des paramètres dans `slides/figures/` | ≈ 1 min |
| `tectonic slides/presentation.tex` | Diapos (Beamer, thème Metropolis) | ≈ 1 min |

Tous les temps sont mesurés sur **un seul cœur CPU** pour que les algorithmes soient comparables.

## Fichiers

| Fichier | Rôle |
|---|---|
| `algorithms.py` | Ant System et recherche aléatoire de l'application, avec l'historique par itération |
| `aco_variants.py` | AS, ACS, MMAS et MMAS + 2-opt (toutes les fourmis en parallèle, PyTorch) |
| `tsp_classics.py` | Plus proche voisin, 2-opt, Christofides, recuit simulé, Held-Karp, OR-Tools |
| `style.py` | Couleurs et style communs des figures |
| `data/tsplib/` | Instances TSPLIB à optimum prouvé (eil51 : 426, berlin52 : 7542, st70 : 675, kroA100 : 21282, ch150 : 6528) |
| `results/` | Résultats bruts et figures de la comparaison précision vs temps |
| `slides/` | Source des diapos et figures |

## Principaux résultats

- La complexité mesurée suit la théorie pour Christofides (pente 2,9 pour 3), le 2-opt et Held-Karp (×2,1 par ville ajoutée).
- Sous la seconde, plus proche voisin + 2-opt est imbattable (1,4 à 3 % de l'optimum en quelques ms).
- Avec quelques secondes, MMAS + 2-opt atteint 0 à 0,4 % de l'optimum, au niveau d'OR-Tools.
- À recherche locale égale, partir des tournées des fourmis plutôt que de tournées aléatoires divise l'écart par 7 à 10.
- L'ACO sans recherche locale est toujours battu par une méthode qui utilise le 2-opt.
