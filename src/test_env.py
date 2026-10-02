"""Validation de la mécanique de base, sans RL branché (plan section 12).

Deux modes, via un seul switch :
- rapide (par défaut) : pas de fenêtre, boucle à pleine vitesse, pour vérifier
  que rien ne plante sur beaucoup de steps.
- visuel (--visual) : ouvre le viewer MuJoCo, ralentit la boucle, affiche les
  rayons LiDAR (rouge=proche, vert=loin) et les frontières détectées (points
  bleus) par-dessus la scène.

Usage :
    python test_env.py                # rapide, 300 steps
    python test_env.py --visual        # visuel, 100 steps par défaut
    python test_env.py --visual --steps 500
"""

import argparse
import time

import numpy as np

from explorer_env import ExplorerEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual", "-v", action="store_true",
                         help="Ouvre le viewer MuJoCo et ralentit la boucle pour observer.")
    parser.add_argument("--steps", type=int, default=None,
                         help="Nombre de steps à jouer (défaut : 100 en visuel, 300 en rapide).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-lidar", action="store_true", help="Cache les rayons LiDAR en mode visuel.")
    parser.add_argument("--no-frontiers", action="store_true", help="Cache les marqueurs de frontière en mode visuel.")
    parser.add_argument("--static", action="store_true",
                         help="Action nulle à chaque step (le drone ne bouge pas) — pour juste regarder "
                              "le bâtiment, le LiDAR et les frontières. Active --visual automatiquement "
                              "si non précisé.")
    parser.add_argument("--slow", action="store_true",
                         help="Ralentit encore plus la boucle visuelle (0.15s/step au lieu de 0.02s) — "
                              "pratique pour observer un comportement précis image par image.")
    args = parser.parse_args()

    if args.static and not args.visual:
        args.visual = True
        print("(--static implique --visual, activé automatiquement)")

    n_steps = args.steps or (100 if args.visual else 300)
    print_every = 1 if args.slow else (10 if args.visual else 50)

    env = ExplorerEnv(n_rooms=(2, 4), seed=args.seed)
    obs, info = env.reset(seed=args.seed)

    print(f"Mode : {'VISUEL' if args.visual else 'rapide'}")
    print(f"Bâtiment généré : {len(env.layout.rooms)} pièces")
    print(f"Grille : {env.grid.width}x{env.grid.height} cellules "
          f"({env.grid_resolution} m/cellule)")
    print(f"Formes d'observation : "
          f"{ {k: v.shape for k, v in obs.items()} }")

    total_reward = 0.0
    for i in range(n_steps):
        action = np.zeros(4, dtype=np.float32) if args.static else env.action_space.sample() * 0.3
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if args.visual:
            env.render(show_lidar=not args.no_lidar, show_frontiers=not args.no_frontiers)
            time.sleep(0.15 if args.slow else 0.02)

        if i % print_every == 0:
            print(f"step {i:4d} | couverture={info['coverage']*100:5.1f}% "
                  f"| reward={reward:+.3f} "
                  f"| frontières valides={sum(f['valid'] for f in env._last_frontiers)}")

        if terminated or truncated:
            print(f"Épisode terminé au step {i} "
                  f"(collision={info['collision']}, couverture={info['coverage']*100:.1f}%)")
            obs, info = env.reset()

    print(f"\nReward cumulé sur {n_steps} steps : {total_reward:.2f}")
    print("Mécanique de base OK : bâtiment + LiDAR + grille + frontières "
          "fonctionnent ensemble.")


if __name__ == "__main__":
    main()