"""Diagnostic visuel : carte 2D de la grille d'occupation, avec les rayons
LiDAR de la dernière frame superposés (en bleu), la zone atteignable
entourée (contour noir), et la fenêtre exacte donnée au CNN encadrée
(pointillés violets). Sert à vérifier si les cellules marquées "libres"
correspondent bien à des rayons qui les ont réellement traversées, ou s'il y
a un vrai bug de marquage — pas juste de la ligne de vue à travers une porte
(légitime, un vrai LiDAR ferait pareil).

Usage :
    python debug_grid.py --seed 9002 --steps 60                        # actions aléatoires
    python debug_grid.py --seed 9002 --steps 60 --name explorer --version v4  # modèle entraîné
"""

import argparse

import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

from explorer_env import ExplorerEnv
from occupancy_grid import FREE, OCCUPIED, UNKNOWN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=9002)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--n-rooms", type=int, nargs=2, default=[2, 4])
    parser.add_argument("--name", type=str, default=None,
                         help="Modèle entraîné à utiliser (sinon actions aléatoires x0.3).")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--out", type=str, default="debug_grid.png")
    args = parser.parse_args()

    env_kwargs = {}
    model = None
    vec_env = None

    if args.name:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import model_manager as mm

        version = args.version or mm.latest_version(args.name)
        paths = mm.checkpoint_paths(args.name, version)
        meta = mm.read_metadata(args.name, version)
        env_kwargs = meta.get("env_kwargs", {})
        print(f"Modèle {args.name}/{version}, réglages : {env_kwargs}")

        vec_env = DummyVecEnv([lambda: ExplorerEnv(n_rooms=tuple(args.n_rooms), seed=args.seed, **env_kwargs)])
        vec_env = VecNormalize.load(paths["vecnormalize"], vec_env)
        vec_env.training = False
        model = PPO.load(paths["model"], env=vec_env)
        vec_env.seed(args.seed)
        obs = vec_env.reset()
        env = vec_env.envs[0]
    else:
        env = ExplorerEnv(n_rooms=tuple(args.n_rooms), seed=args.seed, **env_kwargs)
        obs, _ = env.reset(seed=args.seed)

    n_steps_done = 0
    for i in range(args.steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            n_steps_done = i + 1
            if dones[0]:
                break
        else:
            action = env.action_space.sample() * 0.3
            obs, r, term, trunc, info = env.step(action)
            n_steps_done = i + 1
            if term or trunc:
                break

    grid = env.grid

    img = np.full(grid.grid.shape, 2)
    img[grid.grid == FREE] = 0
    img[grid.grid == OCCUPIED] = 1
    img[grid.grid == UNKNOWN] = 2
    cmap = matplotlib.colors.ListedColormap(["#8fd19e", "#e07a5f", "#cccccc"])

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap=cmap, origin="lower",
              extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max])

    xs = np.linspace(grid.x_min, grid.x_max, grid.width)
    ys = np.linspace(grid.y_min, grid.y_max, grid.height)
    ax.contour(xs, ys, grid.reachable.astype(float), levels=[0.5], colors="black", linewidths=1.2)

    origin = env._last_lidar_origin
    if origin is not None:
        for direction, dist in zip(env._last_lidar_dirs, env._last_lidar_dists):
            end = origin + direction * dist
            ax.plot([origin[0], end[0]], [origin[1], end[1]],
                    color="#3d5a80", linewidth=0.4, alpha=0.5, zorder=3)
        ax.plot(origin[0], origin[1], marker="o", color="black", markersize=9, zorder=5)

        half_size_m = (env.crop_size // 2) * env.grid_resolution
        rect = matplotlib.patches.Rectangle(
            (origin[0] - half_size_m, origin[1] - half_size_m),
            2 * half_size_m, 2 * half_size_m,
            fill=False, edgecolor="purple", linewidth=2, linestyle="--", zorder=4)
        ax.add_patch(rect)

    ax.set_title(
        f"seed {args.seed}, step {n_steps_done} — vert=libre, corail=occupé, gris=inconnu\n"
        f"contour noir=zone atteignable, bleu=rayons LiDAR de la dernière frame, "
        f"pointillés violets=fenêtre du CNN"
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Sauvegardé : {args.out} (couverture actuelle : {grid.coverage_ratio()*100:.1f}%)")


if __name__ == "__main__":
    main()