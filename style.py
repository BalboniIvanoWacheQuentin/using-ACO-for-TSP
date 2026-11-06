'''Style commun des figures (palette lisible par les daltoniens, ordre fixe).'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "legend.frameon": False,
    # tailles pensees pour une figure de ~9 pouces de large affichee sur toute la largeur d'une diapo
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "lines.linewidth": 2.6, "lines.markersize": 8,
})

SLIDE_W = 9.0          # largeur (pouces) d'une figure pleine largeur ; demi-largeur : SLIDE_W / 2
