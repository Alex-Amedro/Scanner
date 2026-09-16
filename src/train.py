"""Entraînement PPO avec système de versions (plan section 7 + demande de
reprise/versioning) :

- Sans option : crée une nouvelle version fraîche (v1, puis v2, v3...).
- --resume : reprend la dernière version de --name là où elle s'est arrêtée
  (même modèle, même normalisation, le compteur de steps continue).
- --from-version v1 : repart des poids de v1 mais écrit dans une nouvelle
  version dérivée (v1.1, puis v1.2 si on refait --from-version v1 encore).

Usage :
    python train.py --name explorer --total-timesteps 2000000          # v1
    python train.py --name explorer --resume --total-timesteps 2000000 # continue v1
    python train.py --name explorer --from-version v1 --total-timesteps 500000  # v1.1
"""

import argparse
import os

from explorer_env import ExplorerEnv

NORM_OBS_KEYS = ["coverage", "frontier_vector", "relief_rays", "kinematics"]  # pas local_crop (déjà 0/1)


def make_env_fn(seed, env_kwargs):
    def _init():
        return ExplorerEnv(n_rooms=(2, 4), seed=seed, **env_kwargs)
    return _init


def build_vec_env(n_envs, no_subproc, env_kwargs):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    vec_env_cls = DummyVecEnv if no_subproc else SubprocVecEnv
    env_fns = [make_env_fn(seed=i, env_kwargs=env_kwargs) for i in range(n_envs)]
    return vec_env_cls(env_fns)


def main():
    # Imports lourds (torch, SB3, l'extracteur) déplacés ICI plutôt qu'en haut
    # du fichier : sur Windows, SubprocVecEnv relance ce script depuis le début
    # dans chaque worker, qui exécute donc aussi tout le code au niveau module.
    # Si ces imports étaient en haut, chaque worker chargerait torch pour rien
    # (8 copies en mémoire) — c'est ce qui a fait planter l'allocation.
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    import model_manager as mm
    from feature_extractor import ExplorerFeaturesExtractor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="explorer", help="Nom du modèle (dossier models/<name>/).")
    parser.add_argument("--resume", action="store_true", help="Reprend la dernière version de --name.")
    parser.add_argument("--from-version", type=str, default=None,
                         help="Repart des poids de cette version, dans une nouvelle version dérivée (ex: v1 -> v1.1).")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000,
                         help="Nombre de steps à jouer CETTE session (s'ajoute à ce qui a déjà été fait si --resume).")
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--no-subproc", action="store_true")
    parser.add_argument("--task", type=str, default="explore", choices=["explore", "hover"],
                         help="'explore' = tâche complète, 'hover' = juste apprendre à rester stable "
                              "(curriculum, pas de couverture/frontières dans le reward).")
    parser.add_argument("--gear-roll-pitch", type=float, default=0.5,
                         help="Validé par test_gear.py sur le projet précédent : 0.5 au lieu de "
                              "l'original 2.0 (97-100%% de succès contre 20-33%% avant).")
    parser.add_argument("--gear-yaw", type=float, default=0.25,
                         help="Validé par test_gear.py : 0.25 au lieu de l'original 1.0.")
    parser.add_argument("--up-z-min", type=float, default=0.3,
                         help="Seuil d'uprightness sous lequel l'épisode se termine (retournement). "
                              "0.3 ≈ 70° d'inclinaison.")
    parser.add_argument("--coverage-target", type=float, default=0.9,
                         help="Pourcentage de couverture (sur la zone atteignable) déclenchant la victoire.")
    parser.add_argument("--coeff-align", type=float, default=0.0,
                         help="Bonus d'alignement entre le cap et la direction de déplacement horizontal "
                              "(cos de l'angle entre les deux, actif seulement si vitesse horizontale > 0.3 m/s). "
                              "Basé sur le 'tangent path reward' documenté pour la navigation par capteurs de "
                              "proximité (ScienceDirect 2025) — plus ciblé qu'une pénalité générale sur la "
                              "rotation. DÉSACTIVÉ par défaut (0.0), à tester et comparer via evaluate.py.")
    parser.add_argument("--coeff-spin", type=float, default=0.0,
                         help="Pénalité sur la norme de la vitesse angulaire, pour décourager le "
                              "comportement de 'toupie'. DÉSACTIVÉ par défaut (0.0) : le projet "
                              "précédent avait testé une pénalité similaire (coeff_agressivite) et "
                              "observé une RÉGRESSION nette (40%%->20%% de succès) — à tester prudemment, "
                              "pas à activer par défaut sans comparaison via evaluate.py.")
    args = parser.parse_args()

    if args.resume and args.from_version:
        raise ValueError("--resume et --from-version sont exclusifs (soit on continue une version, soit on en dérive une nouvelle).")

    env_kwargs = dict(
        task=args.task, gear_roll_pitch=args.gear_roll_pitch, gear_yaw=args.gear_yaw,
        up_z_min=args.up_z_min, coverage_target=args.coverage_target, coeff_spin=args.coeff_spin,
        coeff_align=args.coeff_align,
    )
    env = build_vec_env(args.n_envs, args.no_subproc, env_kwargs)

    base_timesteps = 0
    if args.resume:
        version = mm.latest_version(args.name)
        if version is None:
            raise ValueError(f"Pas de version existante pour '{args.name}' à reprendre.")
        paths = mm.checkpoint_paths(args.name, version)
        print(f"Reprise de {args.name}/{version}")
        env = VecNormalize.load(paths["vecnormalize"], env)
        model = PPO.load(paths["model"], env=env)
        base_timesteps = mm.read_metadata(args.name, version).get("total_timesteps", 0)

    elif args.from_version:
        version = mm.next_minor_version(args.name, args.from_version)
        src_paths = mm.checkpoint_paths(args.name, args.from_version)
        if not os.path.exists(src_paths["model"]):
            raise ValueError(f"Version source introuvable : {args.name}/{args.from_version}")
        print(f"Nouvelle version {args.name}/{version}, dérivée de {args.from_version}")
        env = VecNormalize.load(src_paths["vecnormalize"], env)
        model = PPO.load(src_paths["model"], env=env)

    else:
        version = mm.next_major_version(args.name)
        print(f"Nouvelle version {args.name}/{version}, entraînement depuis zéro")
        env = VecNormalize(env, norm_obs=True, norm_obs_keys=NORM_OBS_KEYS,
                            norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        policy_kwargs = dict(
            features_extractor_class=ExplorerFeaturesExtractor,
            features_extractor_kwargs=dict(cnn_out_dim=128, mlp_out_dim=64),
            net_arch=dict(pi=[128], vf=[128]),
        )
        model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs,
                     n_steps=args.n_steps, batch_size=args.batch_size,
                     verbose=1, tensorboard_log=args.logdir)

    checkpoint_callback = mm.RotatingCheckpointCallback(
        name=args.name, version=version, vec_normalize_env=env,
        save_freq_steps=args.checkpoint_freq,
        extra_meta={"base_timesteps": base_timesteps, "env_kwargs": env_kwargs}, verbose=1,
    )
    metrics_callback = mm.EpisodeMetricsCallback()
    from stable_baselines3.common.callbacks import CallbackList
    callback = CallbackList([checkpoint_callback, metrics_callback])

    model.learn(total_timesteps=args.total_timesteps,
                callback=callback,
                tb_log_name=f"{args.name}_{version}",
                reset_num_timesteps=not args.resume)

    total = base_timesteps + args.total_timesteps
    mm.save_checkpoint(args.name, version, model, env, total,
                        extra_meta={"n_envs": args.n_envs, "n_steps": args.n_steps,
                                    "batch_size": args.batch_size, "env_kwargs": env_kwargs})
    print(f"Entraînement terminé. {args.name}/{version} sauvegardé ({total} steps au total) "
          f"dans models/{args.name}/{version}/")


if __name__ == "__main__":
    main()