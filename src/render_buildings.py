"""Plot house plans from above and check that every room is reachable.

Usage:
    python render_buildings.py --level 2 --n 12 --seed 100      # writes ../diagnostics/buildings_level2.png
    python render_buildings.py --level 1 2 3 4 --stats 300      # statistics over 300 houses per level, no image
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from house_generator import check_layout, generate_house

ap = argparse.ArgumentParser()
ap.add_argument("--level", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--stats", type=int, default=0, help="N bâtiments par niveau : statistiques et vérification, pas d'image")
args = ap.parse_args()

for lvl in args.level:
    if args.stats:
        bad, nr, nd, area = 0, [], [], []
        for k in range(args.stats):
            lay = generate_house(lvl, random.Random(args.seed + k))
            pb = check_layout(lay)
            if pb:
                bad += 1
                if bad <= 3:
                    print(f"  niveau {lvl} seed {args.seed + k} : {pb}")
            nr.append(len(lay.rooms)); nd.append(len(lay.doors) - (len(lay.rooms) - 1)); area.append(lay.footprint[0] * lay.footprint[1])
        print(f"niveau {lvl} : {args.stats} bâtiments, {bad} invalides | pièces {min(nr)}-{max(nr)} (moy {sum(nr)/len(nr):.1f}) "
              f"| portes en plus (boucles) moy {sum(nd)/len(nd):.2f} | surface moy {sum(area)/len(area):.0f} m2")
        continue
    cols = 4
    rows = (args.n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows))
    for k, ax in enumerate(axs.flat):
        if k >= args.n:
            ax.axis("off"); continue
        lay = generate_house(lvl, random.Random(args.seed + k))
        pb = check_layout(lay)
        for name, cx, cy, hx, hy, z0, z1, rgba in lay.boxes:
            if name == "plafond":
                continue
            if name.startswith("sol"):
                ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, color=rgba[:3], alpha=0.35, lw=0))
            elif name.startswith("fenetre"):
                ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, color="tab:blue", alpha=0.8, lw=0))
            elif z0 < 1.25 < z1:
                ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, color="k", lw=0))
        ax.plot(*lay.spawn_xy, "r*", ms=12)
        for i, r in enumerate(lay.rooms):
            ax.text(*r.center, str(i), ha="center", va="center", color="gray", fontsize=8)
        ax.set_xlim(*lay.world_x_range); ax.set_ylim(*lay.world_y_range); ax.set_aspect("equal")
        loops = len(lay.doors) - (len(lay.rooms) - 1)
        ax.set_title(f"seed {args.seed + k} | {len(lay.rooms)} pièces, {loops} boucle(s) | {lay.footprint[0]:.0f}x{lay.footprint[1]:.0f} m" + (" | PROBLÈME" if pb else ""),
                     fontsize=9, color="red" if pb else "black")
        ax.grid(alpha=0.2)
    fig.suptitle(f"Niveau {lvl} — étoile rouge = départ (cour), bleu = fenêtre")
    fig.tight_layout()
    out = os.path.join("..", "diagnostics", f"buildings_level{lvl}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)
    print("écrit", os.path.abspath(out))
