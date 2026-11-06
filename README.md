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
| `python subject.py` | Ant System face à l'optimum exact (Held-Karp, 8 à 16 villes) et selon la taille, jusqu'à 150 villes (`results/subject.csv`, `results/figures/subject_size.png`) | ≈ 5 min |
| `python accuracy_time.py complexity` | Complexité, n = 25 à 1 600 (`results/complexity.csv`) | ≈ 5 min |
| `python accuracy_time.py anytime rat783` | Précision vs temps (budget 60 s) sur une instance (`results/anytime_rat783.csv`) ; sans nom, les 5 instances à la suite | ≈ 20 min par instance |
| `python dynamic.py rat783` | TSP dynamique (20 % des routes ralenties en cours de calcul) : reprise à chaud ou à froid (`results/dynamic_rat783.csv`) ; sans nom, kroA100 et rat783 | ≈ 7 min par instance |
| `python plot_accuracy_time.py` | Figures `results/figures/` et tableau par budget `results/budget_table.csv` | quelques secondes |
| `python slides/make_figures.py` | Figures des paramètres dans `slides/figures/` | ≈ 1 min |
| `tectonic slides/presentation.tex` | Diapos (Beamer, thème Metropolis) | ≈ 1 min |


## Fichiers

| Fichier | Rôle |
|---|---|
| `algorithms.py` | Ant System et recherche aléatoire de l'application, avec l'historique par itération |
| `aco_variants.py` | AS, ACS, MMAS et MMAS + 2-opt (toutes les fourmis en parallèle, PyTorch) |
| `tsp_classics.py` | Plus proche voisin, 2-opt (listes de 20 voisins, compilé avec numba), Christofides, recuit simulé, Held-Karp, OR-Tools |
| `style.py` | Couleurs et style communs des figures (tailles pensées pour des diapos) |
| `data/tsplib/` | Instances TSPLIB à optimum prouvé (eil51 : 426, berlin52 : 7542, st70 : 675, kroA100 : 21282, ch150 : 6528, pcb442 : 50778, rat783 : 8806) |
| `results/` | Résultats bruts et figures de la comparaison précision vs temps |
| `slides/` | Source des diapos et figures |

## Principaux résultats

- La complexité mesurée suit la théorie : pentes 1,8 à 2,8 pour 2 à 3, Held-Karp ×2,1 par ville ; seul le plus proche voisin s'en écarte (coûts fixes).
- Un 2-opt rapide relancé depuis des tournées aléatoires gagne presque partout, de 51 à 783 villes.
- MMAS + 2-opt atteint l'optimum jusqu'à 150 villes et gagne sur ch150 à 10 s (0,02 % contre 0,45 %), mais il lui faut des centaines d'itérations : en 60 s, il n'en fait que 123 sur pcb442 et 45 sur rat783, où il reste à 3 à 5 %.
- L'ACO sans recherche locale est toujours battu par une méthode qui utilise le 2-opt.
- TSP dynamique : garder ses phéromones aide un peu (rat783 à 20 s : 1,84 % contre 2,34 %), mais un 2-opt multi-départs relancé de zéro fait mieux (1,17 %).
