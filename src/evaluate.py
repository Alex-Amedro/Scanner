"""Evaluate a trained model on fixed seeds and append the result to eval_logs/journal.csv.

The evaluation houses are always the same (seeds 9000+ by default, outside the training seeds), so runs can be compared.

Usage:
    python evaluate.py --name explorer --version v1
    python evaluate.py --name explorer --n-episodes 20 --building house --house-level 2 --max-steps 3000
"""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np


def _in_corridor(env, x, y):
    """True si (x, y) tombe dans un rectangle de couloir (pas de pièce) du
    layout couramment chargé dans env. Les couloirs sont ajoutés après les
    pièces par compute_navigable_rects, d'où le découpage par n_rooms."""
    return env._in_corridor(x, y)


def run_episode(model, vec_env, seed, max_steps=2000, visual=False, slow=False, debug_angles=False):
    vec_env.seed(seed)
    obs = vec_env.reset()
    raw_env = vec_env.venv.envs[0]  # VecNormalize enveloppe le DummyVecEnv

    traj_length = 0.0
    prev_pos = None
    total_reward = 0.0
    collision = False
    retourne = False
    truncated = False
    final_coverage = 0.0
    step_count = 0
    ended = False  # False si la boucle sort sur --max-steps sans que l'env ait terminé l'épisode
    collision_location = None  # "couloir" ou "pièce", fixé à la 1ère collision de l'épisode
    minlid_by_loc = {"couloir": [], "pièce": []} if debug_angles else None

    if debug_angles:
        print(f"{'step':>4} {'roll°':>7} {'pitch°':>7} {'yaw°':>7} {'up_z':>6} "
              f"{'v_ang':>6} {'align':>6} {'frnt_d':>6} {'speed':>6} {'minLid':>6} {'loc':>7} {'reward':>7}")

    for step_count in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        info = infos[0]
        total_reward += float(reward[0])
        final_coverage = info.get("coverage", final_coverage)

        if debug_angles:
            pos = info.get("pos")
            loc = ("couloir" if _in_corridor(raw_env, pos[0], pos[1]) else "pièce") if pos else "?"
            if loc in minlid_by_loc:
                minlid_by_loc[loc].append(info.get("min_lidar_dist", 0))
            print(f"{step_count:>4} {info.get('roll_deg', 0):>7.1f} {info.get('pitch_deg', 0):>7.1f} "
                  f"{info.get('yaw_deg', 0):>7.1f} {info.get('up_z', 0):>6.2f} "
                  f"{info.get('vel_ang_norm', 0):>6.2f} {info.get('alignement', 0):>6.2f} "
                  f"{info.get('nearest_frontier_dist', -1):>6.2f} "
                  f"{info.get('speed_horiz', 0):>6.2f} {info.get('min_lidar_dist', 0):>6.2f} "
                  f"{loc:>7} {float(reward[0]):>7.2f}")

        pos = info.get("pos")
        if pos is not None:
            if prev_pos is not None:
                traj_length += float(np.linalg.norm(np.array(pos) - np.array(prev_pos)))
            prev_pos = pos

        if info.get("collision"):
            collision = True
            if collision_location is None:
                x, y = info["pos"][0], info["pos"][1]
                collision_location = "couloir" if _in_corridor(raw_env, x, y) else "pièce"
        if info.get("retourne"):
            retourne = True
        if info.get("truncated"):
            truncated = True

        if visual:
            vec_env.envs[0].render()
            time.sleep(0.15 if slow else 0.03)

        if dones[0]:
            ended = True
            break

    if retourne:
        outcome = "retourné"
    elif collision:
        outcome = "mort"
    elif truncated or not ended:
        outcome = "timeout"
    else:
        outcome = "victoire"

    return {
        "outcome": outcome,
        "coverage": final_coverage,
        "steps": step_count,
        "traj_length": traj_length,
        "total_reward": total_reward,
        "collision_location": collision_location,
        "minlid_by_loc": minlid_by_loc,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="explorer")
    parser.add_argument("--version", type=str, default=None, help="Défaut : dernière version disponible.")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=9000,
                         help="Seeds de test fixes, hors de la plage d'entraînement (0-7 par défaut).")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--logdir", type=str, default="eval_logs")
    parser.add_argument("--visual", action="store_true",
                         help="Ouvre le viewer MuJoCo et rejoue chaque épisode visuellement "
                              "(pense à réduire --n-episodes, ex: 3).")
    parser.add_argument("--slow", action="store_true", help="Ralentit encore plus la boucle visuelle.")
    parser.add_argument("--debug-angles", action="store_true",
                         help="Imprime roll/pitch/yaw/uprightness/vitesse angulaire/alignement/distance "
                              "à la frontière la plus proche à chaque step — utile pour analyser "
                              "précisément un comportement de rotation, sans dépendre du visuel. "
                              "Combinable avec --visual, ou seul (limite --n-episodes à 2-3, ça imprime "
                              "beaucoup de lignes).")
    parser.add_argument("--building", choices=["chain", "house"], default=None, help="Force le type de bâtiment (maison = house_generator).")
    parser.add_argument("--house-level", type=int, nargs="+", default=None, help="Niveau 1-4 (ou min max) des maisons.")
    parser.add_argument("--yaw-follow", action="store_true", help="(inutile : actif par défaut quand le modèle le permet)")
    parser.add_argument("--no-yaw-follow", action="store_true",
                        help="Désactive le lacet réaliste (par défaut le nez suit la direction de déplacement pour les modèles à cap fixe --no-yaw).")
    parser.add_argument("--n-rooms-min", type=int, default=None,
                        help="Surcharge le nombre min de pièces (défaut : celui de l'entraînement).")
    parser.add_argument("--n-rooms-max", type=int, default=None)
    parser.add_argument("--up-z-min", type=float, default=None,
                         help="Surcharge le seuil de retournement pour CETTE évaluation, sans "
                              "réentraîner (par défaut : reprend celui utilisé à l'entraînement, lu "
                              "dans metadata.json). Utile pour tester à coût nul si desserrer la marge "
                              "change l'issue des épisodes sur un modèle déjà entraîné, avant de décider "
                              "si ça vaut le coup de relancer un entraînement avec ce nouveau seuil.")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    import model_manager as mm
    from explorer_env import ExplorerEnv

    version = args.version or mm.latest_version(args.name)
    if version is None:
        raise ValueError(f"Aucune version trouvée pour '{args.name}'.")
    paths = mm.checkpoint_paths(args.name, version)
    meta = mm.read_metadata(args.name, version)

    print(f"Évaluation de {args.name}/{version} "
          f"({meta.get('total_timesteps', '?')} steps entraînés) "
          f"sur {args.n_episodes} épisodes (seeds {args.seed_start}-{args.seed_start + args.n_episodes - 1})\n")

    env_kwargs = meta.get("env_kwargs", {})
    if args.up_z_min is not None:
        env_kwargs = dict(env_kwargs)
        env_kwargs["up_z_min"] = args.up_z_min
    if args.building or args.house_level:
        env_kwargs = dict(env_kwargs)
        env_kwargs["building"] = args.building or "house"
        if args.house_level:
            env_kwargs["house_level"] = args.house_level[0] if len(args.house_level) == 1 else tuple(args.house_level)
    if not args.no_yaw_follow and env_kwargs.get("no_yaw") and env_kwargs.get("action_mode", "velocity") == "velocity":
        env_kwargs = dict(env_kwargs)   # le réseau est aveugle au cap : mêmes entrées, rendu réaliste
        env_kwargs["yaw_follow"] = True
    if env_kwargs:
        print(f"Réglages repris de l'entraînement : {env_kwargs}")
    env_kwargs = dict(env_kwargs)
    env_kwargs.setdefault("n_rooms", (2, 4))  # modèles historiques : 2 à 4 pièces
    if args.n_rooms_min is not None or args.n_rooms_max is not None:
        lo, hi = env_kwargs["n_rooms"]
        env_kwargs["n_rooms"] = (args.n_rooms_min or lo, args.n_rooms_max or hi)
    print(f"Pièces par bâtiment : {tuple(env_kwargs['n_rooms'])}")
    vec_env = DummyVecEnv([lambda: ExplorerEnv(seed=0, **env_kwargs)])
    try:
        vec_env = VecNormalize.load(paths["vecnormalize"], vec_env)
    except AssertionError as e:
        raise SystemExit(
            f"'{args.name}/{version}' a été entraîné avec un espace d'observation "
            f"différent de celui du code actuel (probablement un changement dans "
            f"explorer_env.py depuis) — incompatible, impossible à évaluer tel quel. "
            f"Détail : {e}\nRéentraîne une nouvelle version avec le code actuel."
        )
    vec_env.training = False  # ne pas continuer à mettre à jour les stats de normalisation
    model = PPO.load(paths["model"], env=vec_env)

    header = f"{'#':>3} {'issue':<9} {'couverture':>10} {'steps':>7} {'trajectoire':>11} {'reward':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for i in range(args.n_episodes):
        seed = args.seed_start + i
        r = run_episode(model, vec_env, seed=seed, max_steps=args.max_steps,
                         visual=args.visual, slow=args.slow, debug_angles=args.debug_angles)
        results.append(r)
        print(f"{i:>3} {r['outcome']:<9} {r['coverage']*100:>9.1f}% {r['steps']:>7} "
              f"{r['traj_length']:>10.1f}m {r['total_reward']:>9.1f}")

    n = len(results)
    victoire_rate = sum(r["outcome"] == "victoire" for r in results) / n
    mort_rate = sum(r["outcome"] == "mort" for r in results) / n
    retourne_rate = sum(r["outcome"] == "retourné" for r in results) / n
    timeout_rate = sum(r["outcome"] == "timeout" for r in results) / n
    couverture_moy = np.mean([r["coverage"] for r in results])
    couverture_std = np.std([r["coverage"] for r in results])
    traj_moy = np.mean([r["traj_length"] for r in results])
    steps_victoire = [r["steps"] for r in results if r["outcome"] == "victoire"]
    steps_victoire_moy = np.mean(steps_victoire) if steps_victoire else None

    print("\n--- Résumé ---")
    print(f"Victoire : {victoire_rate*100:.0f}%   Mort (collision) : {mort_rate*100:.0f}%   "
          f"Retourné : {retourne_rate*100:.0f}%   Timeout : {timeout_rate*100:.0f}%")
    collisions_couloir = sum(1 for r in results if r["outcome"] == "mort" and r.get("collision_location") == "couloir")
    collisions_piece = sum(1 for r in results if r["outcome"] == "mort" and r.get("collision_location") == "pièce")
    total_c = collisions_couloir + collisions_piece
    if total_c:
        print(f"Collisions en couloir : {collisions_couloir}/{total_c} ({100*collisions_couloir/total_c:.0f}%)   "
              f"en pièce : {collisions_piece}/{total_c} ({100*collisions_piece/total_c:.0f}%)")
    if args.debug_angles:
        all_couloir = [v for r in results if r.get("minlid_by_loc") for v in r["minlid_by_loc"]["couloir"]]
        all_piece = [v for r in results if r.get("minlid_by_loc") for v in r["minlid_by_loc"]["pièce"]]
        if all_couloir:
            print(f"minLid en couloir : médiane {np.median(all_couloir):.2f}, min {min(all_couloir):.2f} "
                  f"(sur {len(all_couloir)} steps)")
        if all_piece:
            print(f"minLid en pièce   : médiane {np.median(all_piece):.2f}, min {min(all_piece):.2f} "
                  f"(sur {len(all_piece)} steps)")
    print(f"Couverture moyenne : {couverture_moy*100:.1f}% (± {couverture_std*100:.1f})")
    print(f"Trajectoire moyenne : {traj_moy:.1f} m")
    if steps_victoire_moy:
        print(f"Steps moyens jusqu'à victoire (sur les épisodes gagnés) : {steps_victoire_moy:.0f}")

    os.makedirs(args.logdir, exist_ok=True)
    journal_path = os.path.join(args.logdir, "journal.csv")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": args.name, "version": version,
        "total_timesteps_entraine": meta.get("total_timesteps"),
        "n_episodes": n, "seed_start": args.seed_start,
        "victoire_rate": round(victoire_rate, 3), "mort_rate": round(mort_rate, 3),
        "retourne_rate": round(retourne_rate, 3),
        "timeout_rate": round(timeout_rate, 3),
        "couverture_moyenne": round(couverture_moy, 3), "couverture_std": round(couverture_std, 3),
        "traj_moyenne_m": round(traj_moy, 1),
        "steps_victoire_moyen": round(steps_victoire_moy, 0) if steps_victoire_moy else "",
    }
    write_header = not os.path.exists(journal_path)
    with open(journal_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nLigne ajoutée à {journal_path} — historique de toutes tes évaluations, à comparer entre runs.")
    vec_env.close()


if __name__ == "__main__":
    main()