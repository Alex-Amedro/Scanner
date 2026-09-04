"""Évaluation d'un modèle sur plusieurs épisodes, avec un journal persistant
pour comparer les runs entre eux au fil du temps.

Les bâtiments évalués sont toujours les MÊMES (seeds fixes, par défaut
9000-9009, en dehors de la plage utilisée à l'entraînement) — pour que
comparer v1 à v2 dans 3 semaines soit une vraie comparaison, pas du bruit dû
à des bâtiments différents.

Usage :
    python evaluate.py --name explorer --version v1
    python evaluate.py --name explorer --version v1 --n-episodes 20
"""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np


def run_episode(model, vec_env, seed, max_steps=2000, visual=False, slow=False):
    vec_env.seed(seed)
    obs = vec_env.reset()

    traj_length = 0.0
    prev_pos = None
    total_reward = 0.0
    survivor_found = False
    step_survivor = None
    collision = False
    retourne = False
    truncated = False
    final_coverage = 0.0
    step_count = 0

    for step_count in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        info = infos[0]
        total_reward += float(reward[0])
        final_coverage = info.get("coverage", final_coverage)

        pos = info.get("pos")
        if pos is not None:
            if prev_pos is not None:
                traj_length += float(np.linalg.norm(np.array(pos) - np.array(prev_pos)))
            prev_pos = pos

        if info.get("survivant_repere") and not survivor_found:
            survivor_found = True
            step_survivor = step_count

        if info.get("collision"):
            collision = True
        if info.get("retourne"):
            retourne = True
        if info.get("truncated"):
            truncated = True

        if visual:
            vec_env.envs[0].render()
            time.sleep(0.15 if slow else 0.03)

        if dones[0]:
            break

    if retourne:
        outcome = "retourné"
    elif collision:
        outcome = "mort"
    elif truncated:
        outcome = "timeout"
    else:
        outcome = "victoire"

    return {
        "outcome": outcome,
        "coverage": final_coverage,
        "steps": step_count,
        "traj_length": traj_length,
        "survivor_found": survivor_found,
        "step_survivor": step_survivor,
        "total_reward": total_reward,
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
    if env_kwargs:
        print(f"Réglages repris de l'entraînement : {env_kwargs}")
    vec_env = DummyVecEnv([lambda: ExplorerEnv(n_rooms=(2, 4), seed=0, **env_kwargs)])
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

    header = f"{'#':>3} {'issue':<9} {'couverture':>10} {'steps':>7} {'trajectoire':>11} {'survivant':>10} {'reward':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for i in range(args.n_episodes):
        seed = args.seed_start + i
        r = run_episode(model, vec_env, seed=seed, max_steps=args.max_steps,
                         visual=args.visual, slow=args.slow)
        results.append(r)
        survivant_str = f"oui @{r['step_survivor']}" if r["survivor_found"] else "non"
        print(f"{i:>3} {r['outcome']:<9} {r['coverage']*100:>9.1f}% {r['steps']:>7} "
              f"{r['traj_length']:>10.1f}m {survivant_str:>10} {r['total_reward']:>9.1f}")

    n = len(results)
    victoire_rate = sum(r["outcome"] == "victoire" for r in results) / n
    mort_rate = sum(r["outcome"] == "mort" for r in results) / n
    retourne_rate = sum(r["outcome"] == "retourné" for r in results) / n
    timeout_rate = sum(r["outcome"] == "timeout" for r in results) / n
    couverture_moy = np.mean([r["coverage"] for r in results])
    couverture_std = np.std([r["coverage"] for r in results])
    traj_moy = np.mean([r["traj_length"] for r in results])
    survivant_rate = sum(r["survivor_found"] for r in results) / n
    steps_victoire = [r["steps"] for r in results if r["outcome"] == "victoire"]
    steps_victoire_moy = np.mean(steps_victoire) if steps_victoire else None

    print("\n--- Résumé ---")
    print(f"Victoire : {victoire_rate*100:.0f}%   Mort (collision) : {mort_rate*100:.0f}%   "
          f"Retourné : {retourne_rate*100:.0f}%   Timeout : {timeout_rate*100:.0f}%")
    print(f"Couverture moyenne : {couverture_moy*100:.1f}% (± {couverture_std*100:.1f})")
    print(f"Trajectoire moyenne : {traj_moy:.1f} m")
    print(f"Survivant repéré : {survivant_rate*100:.0f}% des épisodes")
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
        "survivant_repere_rate": round(survivant_rate, 3),
        "steps_victoire_moyen": round(steps_victoire_moy, 0) if steps_victoire_moy else "",
    }
    write_header = not os.path.exists(journal_path)
    with open(journal_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nLigne ajoutée à {journal_path} — historique de toutes tes évaluations, à comparer entre runs.")


if __name__ == "__main__":
    main()