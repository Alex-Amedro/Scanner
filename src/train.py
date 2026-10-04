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

import sys



import ram_guard

from explorer_env import ExplorerEnv



NORM_OBS_KEYS = ["coverage", "frontier_vector", "relief_rays", "kinematics", "proximity_rays", "velocity"]  # pas local_crop (déjà 0/1)





def make_env_fn(seed, env_kwargs):

    def _init():

        return ExplorerEnv(seed=seed, **env_kwargs)

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

    # Console Windows en cp1252 : un emoji ou un "≈" dans un message faisait planter le programme
    # (UnicodeEncodeError), y compris dans le chemin d'erreur qui doit justement rester propre.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    from stable_baselines3 import PPO
    from caps_ppo import PPOCaps

    from stable_baselines3.common.vec_env import VecNormalize

    import torch
    import model_manager as mm

    from feature_extractor import ExplorerFeaturesExtractor



    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--name", type=str, default="explorer", help="Nom du modèle (dossier models/<name>/).")

    parser.add_argument("--resume", action="store_true", help="Reprend la dernière version de --name.")

    parser.add_argument("--from-version", type=str, default=None,

                         help="Repart des poids de cette version, dans une nouvelle version dérivée (ex: v1 -> v1.1).")

    parser.add_argument("--n-envs", type=int, default=16,
                        help="Environnements parallèles DEMANDÉS. Réduit automatiquement si la RAM libre "
                             "ne suffit pas (cf. --ram-reserve-gb), sauf avec --no-ram-guard.")
    parser.add_argument("--ram-reserve-gb", type=float, default=1.0,
                        help="RAM libre minimale à laisser au système/aux autres programmes. Sert à "
                             "dimensionner --n-envs au lancement ; l'arrêt d'urgence propre (sauvegarde "
                             "puis sortie) se déclenche si la RAM libre passe sous la moitié de cette valeur.")
    parser.add_argument("--max-ram-gb", type=float, default=None,
                        help="Plafond optionnel de RAM pour l'entraînement (processus principal + workers). "
                             "Dépassé = arrêt propre avec sauvegarde. Utile pour détecter une fuite.")
    parser.add_argument("--no-ram-guard", action="store_true",
                        help="Désactive l'adaptation de --n-envs et la surveillance de la RAM.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Device PyTorch/SB3 : auto (= cpu, aussi rapide ici et ~2,6 Go de RAM en moins), cuda ou cpu.")

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

    parser.add_argument("--death-penalty", type=float, default=50.0,

                         help="Pénalité de collision/retournement, en dur à -50 jusqu'ici. À monter "

                              "si le reward total ne distingue pas assez mort et victoire (cf. diagnostic "

                              "ret_rms.std).")

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

    parser.add_argument("--damping", type=float, default=0.0,

                         help="Amortissement du joint libre du corps du drone (freinage physique de "

                              "la vitesse angulaire/linéaire, indépendant de la politique — contrairement "

                              "à coeff_spin qui est une incitation de reward). Testé seul en v4 (0.05, "

                              "AVANT l'existence de coeff_spin) : insuffisant isolément, jamais testé "

                              "en même temps que coeff_spin. DÉSACTIVÉ par défaut (0.0).")

    parser.add_argument("--coeff-proximity", type=float, default=0.0,

                         help="Pénalité de proximité à l'obstacle le plus proche (LiDAR), "

                              "indépendante de la vitesse — contrairement à une pénalité "

                              "'vitesse x proximité', pour ne pas décourager les manœuvres "

                              "d'évitement réussies qui gardent une vitesse élevée (observé sur "

                              "v12 : cf. journal). Monte linéairement sous --proximity-threshold. "

                              "DÉSACTIVÉ par défaut (0.0).")

    parser.add_argument("--proximity-threshold", type=float, default=0.6,

                         help="Distance sous laquelle la pénalité de proximité s'active, en PIÈCE uniquement " \

                              "(exclut les couloirs, cf. journal). "

                              "Sans effet si --coeff-proximity vaut 0.")

    parser.add_argument("--coeff-speed", type=float, default=0.0,

                         help="Pénalité de vitesse horizontale générale, à CHAQUE step, pas "

                              "seulement près d'un obstacle (contrairement à --coeff-proximity en "

                              "v13, qui a créé un réflexe de panique/bank agressif près des murs). "

                              "Vise à décourager la vitesse de croisière élevée en général plutôt "

                              "que de réagir localement au danger. DÉSACTIVÉ par défaut (0.0).")

    parser.add_argument("--max-speed", type=float, default=6.0,

                         help="Plafond DUR de vitesse horizontale (m/s), appliqué directement sur "

                              "la physique à chaque sous-step — pas une incitation de reward "

                              "négociable comme --coeff-speed (v15 a montré que ça ne suffit pas : "

                              "la politique paie le coût et accélère quand même).")

    parser.add_argument("--action-smoothing-alpha", type=float, default=1.0,
                        help="Lissage exponentiel de l'action brute PPO (1.0 = aucun lissage, DÉFAUT). "
                             "0.3 a cassé l'apprentissage en v2.0 (3%% de victoires contre 60%% sans lissage).")
    parser.add_argument("--coeff-potential", type=float, default=0.0,
                        help="Poids du terme de potentiel de frontière (phi - last_phi) dans le reward. "
                             "À 1.0 il est ~200-1000x plus petit que n_new_cells et ne guide rien.")
    parser.add_argument("--substeps", type=int, default=10,
                        help="Sous-steps physiques (2 ms) par action RL. 10 = 50Hz. Plus grand = "
                             "horizon de décision plus long pour un gamma donné.")
    parser.add_argument("--no-ego-crop", action="store_true",
                        help="Garde le crop de grille aligné sur les axes du monde (ancien comportement) "
                             "au lieu de le tourner dans le repère du drone.")
    parser.add_argument("--time-penalty", type=float, default=0.0,
                        help="Pénalité par step (non terminal). Évite l'optimum \"rester en vie sans explorer\".")
    parser.add_argument("--obs-keys", nargs="+", default=["proximity_rays", "frontier_vector", "velocity"],
                        help="Entrées données à la politique. Défaut = le MINIMUM (plan de reprise) : rayons de "
                             "proximité, frontières, vitesse du drone. À rajouter une par une : last_action, "
                             "local_crop, relief_rays, coverage, kinematics.")
    parser.add_argument("--k-frontiers", type=int, default=2, help="Nombre de frontières les plus proches observées.")
    parser.add_argument("--free-altitude", action="store_true",
                        help="4 sorties avec vz libre. Par défaut : altitude bloquée, 3 sorties (vx, vy, lacet).")
    parser.add_argument("--n-rooms-min", type=int, default=3)
    parser.add_argument("--n-rooms-max", type=int, default=3,
                        help="Nombre de pièces tiré entre min et max (défaut 3-3 : on commence direct à 3 pièces).")
    parser.add_argument("--action-mode", choices=["velocity", "accel", "direct"], default="velocity",
                        help="velocity = la sortie du réseau est une consigne de vitesse (historique) ; "
                             "accel = une accélération/inclinaison directe, avec frottement virtuel qui plafonne à --max-speed.")
    parser.add_argument("--start-inside", action="store_true",
                        help="Démarre au milieu de la 1re pièce (ancien comportement). Par défaut : à l'EXTÉRIEUR, "
                             "dans un porche devant la porte d'entrée, la couverture comptée depuis presque zéro.")
    parser.add_argument("--drag", type=float, default=0.0,
                        help="Frottement de l'air linéaire (1/s) : le drone freine tout seul au lieu de glisser. "
                             "0 = aucun (historique), 0,2 à 0,5 = réaliste.")
    parser.add_argument("--action-deadzone", type=float, default=0.0,
                        help="Zone morte sur les sorties du réseau (0,1 à 0,3 conseillé) : sous ce seuil la sortie vaut 0. Coupe les petites "
                             "oscillations et le bruit d'exploration. 0 = désactivée.")
    parser.add_argument("--direct-tau", type=float, default=0.15,
                        help="Mode --action-mode direct : constante de temps (s) du suivi de vitesse (plus petit = plus instantané).")
    parser.add_argument("--direct-a-max", type=float, default=12.0,
                        help="Mode --action-mode direct : accélération horizontale maximale (m/s^2), freinages compris.")
    parser.add_argument("--caps-lambda", type=float, default=1.0,
                        help="CAPS (Mysore et al., ICRA 2021), terme temporel : poids de la perte ||mu(s_t) - mu(s_t+1)||^2 sur l'action MOYENNE "
                             "de la politique (jamais l'action échantillonnée), ajoutée à la perte de PPO. Lisse le pilotage sans toucher au "
                             "reward. 1.0 par défaut (validé : 100 %% de victoires sur 2 seeds) ; 0 = PPO standard.")
    parser.add_argument("--ref-accel-limit", type=float, default=0.0,
                        help="Rampe de consigne de vitesse (m/s^2) dans le contrôleur : la consigne ne peut pas sauter, elle monte avec cette "
                             "accélération maximale (3 à 5 conseillé). Garde l'inclinaison proportionnelle au lieu de saturée. 0 = désactivée.")
    parser.add_argument("--coeff-tilt-rate", type=float, default=0.0,
                        help="Pénalité de basculement physique : coeff * (variation de roulis^2 + de tangage^2) par step, "
                             "en radians. Calme les à-coups d'inclinaison sans punir le bruit d'exploration (5 à 10 testés). 0 = désactivée.")
    parser.add_argument("--coeff-action-rate", type=float, default=0.0,
                        help="Pénalité de jitter : coeff * (changement d'action entre deux steps)^2, somme sur les 2 sorties. "
                             "Décourage les retournements brusques (de +1 à -1 d'un coup). 0 = désactivée.")
    parser.add_argument("--no-yaw", action="store_true",
                        help="2 sorties (vx, vy), pas de lacet (suppose l'altitude bloquée).")
    parser.add_argument("--building", choices=["chain", "house"], default="chain",
                        help="chain = pièces alignées (historique) ; house = maison à portes multiples, entrée par une fenêtre (house_generator).")
    parser.add_argument("--house-level", type=int, nargs="+", default=[2],
                        help="Niveau 1-4 des maisons ; deux valeurs (min max) = niveau tiré au hasard à chaque bâtiment.")
    parser.add_argument("--yaw-follow", action="store_true",
                        help="Le nez du drone suit sa direction de déplacement (réalisme) ; le réseau reste dans le repère du monde. Suppose --no-yaw.")
    parser.add_argument("--no-last-action", action="store_true",
                        help="Met l'observation last_action à zéro (ablation).")
    parser.add_argument("--gamma", type=float, default=0.99, help="Facteur d'actualisation PPO.")
    parser.add_argument("--max-climb-rate", type=float, default=2.0)
    parser.add_argument("--max-yaw-rate", type=float, default=2.0)
    parser.add_argument("--max-tilt-angle-deg", type=float, default=35.0)
    parser.add_argument("--kp-vel", type=float, default=4.0)
    parser.add_argument("--ki-vel", type=float, default=0.1)
    parser.add_argument("--kp-att", type=float, default=20.0)
    parser.add_argument("--kp-rate", type=float, default=0.4)
    parser.add_argument("--ki-rate", type=float, default=0.20)
    parser.add_argument("--kd-rate", type=float, default=0.003)
    parser.add_argument("--kp-alt", type=float, default=0.15)
    parser.add_argument("--ki-alt", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=None,

                         help="Seed pour l'initialisation du modèle et la stochastique d'entraînement "

                              "(PPO/torch). Absent jusqu'ici : chaque run repartait d'une politique "

                              "initialisée différemment même à hyperparamètres identiques, ce qui rend "

                              "une comparaison entre deux versions ambiguë (écart dû au changement testé, "

                              "ou juste au tirage d'init ?). Ne fixe PAS les seeds des environnements "

                              "d'entraînement, déjà fixes (0 à n_envs-1) indépendamment de ce paramètre.")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé mais indisponible (vérifier torch.cuda.is_available()).")
    device = args.device
    if device == "auto":
        # Mesuré : 814 fps en CUDA (10 envs) contre 807 fps en CPU (8 envs) : le goulot est la physique
        # des environnements, pas le réseau (petit). CUDA coûte en plus ~2,6 Go de RAM système.
        device = "cpu"
        print("Device auto -> cpu (le GPU n'accélère pas ce petit réseau et coûte ~2,6 Go de RAM ; "
              "--device cuda pour le forcer).")
    print(f"Device sélectionné : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    n_envs = args.n_envs
    ram_guard_on = not args.no_ram_guard and ram_guard.available()
    if not args.no_ram_guard and not ram_guard.available():
        print("⚠️ psutil absent (pip install psutil) : surveillance de la RAM désactivée.")
    if ram_guard_on:
        n_envs, est_gb, budget_gb = ram_guard.choose_n_envs(
            args.n_envs, args.n_steps, ExplorerEnv(obs_keys=args.obs_keys, k_frontiers=args.k_frontiers).observation_space, device == "cuda", args.ram_reserve_gb)
        print(f"RAM : {ram_guard.available_gb():.1f} Go libres sur {ram_guard.total_gb():.1f} Go, "
              f"réserve {args.ram_reserve_gb:.1f} Go -> budget {max(budget_gb, 0):.1f} Go ; "
              f"estimation pour {n_envs} env(s) : {est_gb:.1f} Go.")
        if est_gb > budget_gb:
            print(f"Erreur : même avec 1 environnement, l'entraînement demande ~{est_gb:.1f} Go alors "
                  f"que le budget est de {max(budget_gb, 0):.1f} Go. Ferme d'autres programmes, baisse "
                  f"--n-steps / --ram-reserve-gb, ou passe --no-ram-guard (à tes risques).")
            return 2
        if n_envs < args.n_envs:
            print(f"⚠️ --n-envs réduit de {args.n_envs} à {n_envs} pour tenir en RAM "
                  f"(un rollout est donc plus petit : ajuste --n-steps si tu veux garder la même "
                  f"taille de rollout).")

    n_cpu = os.cpu_count() or 1
    print(f"CPU logiques : {n_cpu}, environnements parallèles : {n_envs}")
    if n_envs > n_cpu:
        print(f"⚠️ --n-envs ({n_envs}) dépasse le nombre de CPU logiques ({n_cpu}) : "
              "les workers vont se disputer les cœurs.")



    if args.resume and args.from_version:

        raise ValueError("--resume et --from-version sont exclusifs (soit on continue une version, soit on en dérive une nouvelle).")



    base_timesteps = 0

    if args.resume:

        # CRITIQUE : ne PAS construire env_kwargs à partir des args CLI ici — tout flag non

        # retapé sur la commande --resume retombe silencieusement sur son défaut argparse

        # (ex: --coeff-spin, --damping repassent à 0.0), ce qui revient à continuer

        # l'entraînement dans un environnement différent de celui d'origine sans le moindre

        # avertissement. Un vrai incident a eu lieu pour cette raison précise (v11 : reprise

        # avec coeff_spin/joint_damping remis à 0 sans le vouloir, 500k steps corrompus).

        # env_kwargs vient donc TOUJOURS du metadata.json de la version reprise, jamais des args.

        version = mm.latest_version(args.name)

        if version is None:

            raise ValueError(f"Pas de version existante pour '{args.name}' à reprendre.")

        meta = mm.read_metadata(args.name, version)

        env_kwargs = meta.get("env_kwargs")

        if not env_kwargs:

            raise ValueError(

                f"metadata.json de {args.name}/{version} ne contient pas d'env_kwargs — impossible "

                f"de reprendre sans risquer de changer la physique/le reward en cours de route. "

                f"Entraîne une nouvelle version from scratch à la place."

            )

        print(f"Reprise de {args.name}/{version} ({meta.get('total_timesteps', '?')} steps déjà "

              f"faits) — env_kwargs repris tel quel du metadata.json : {env_kwargs}")

        print("Note : les flags --coeff-spin/--coeff-align/--damping/--gear-*/--up-z-min/"

              "--coverage-target passés en ligne de commande sont IGNORÉS pour un --resume, "

              "précisément pour éviter ce genre d'incident.")

        env = build_vec_env(n_envs, args.no_subproc, env_kwargs)

        paths = mm.checkpoint_paths(args.name, version)

        env = VecNormalize.load(paths["vecnormalize"], env)

        model = PPO.load(paths["model"], env=env, device=device)

        base_timesteps = meta.get("total_timesteps", 0)



    elif args.from_version:

        env_kwargs = dict(

            task=args.task, gear_roll_pitch=args.gear_roll_pitch, gear_yaw=args.gear_yaw,

            up_z_min=args.up_z_min, coverage_target=args.coverage_target, death_penalty=args.death_penalty,

            coeff_spin=args.coeff_spin,

            coeff_align=args.coeff_align, joint_damping=args.damping,

            coeff_proximity=args.coeff_proximity, proximity_threshold=args.proximity_threshold,

            coeff_speed=args.coeff_speed, max_speed=args.max_speed,
            max_climb_rate=args.max_climb_rate, max_yaw_rate=args.max_yaw_rate,
            max_tilt_angle_deg=args.max_tilt_angle_deg,
            kp_vel=args.kp_vel, ki_vel=args.ki_vel, kp_att=args.kp_att,
            kp_rate=args.kp_rate, ki_rate=args.ki_rate, kd_rate=args.kd_rate,
            kp_alt=args.kp_alt, ki_alt=args.ki_alt,
            action_smoothing_alpha=args.action_smoothing_alpha, coeff_potential=args.coeff_potential,
            substeps=args.substeps, ego_crop=not args.no_ego_crop,
            use_last_action=not args.no_last_action, time_penalty=args.time_penalty,
            obs_keys=args.obs_keys, fixed_altitude=not args.free_altitude, k_frontiers=args.k_frontiers,
            no_yaw=args.no_yaw, action_mode=args.action_mode, start_outside=not args.start_inside, drag=args.drag, coeff_action_rate=args.coeff_action_rate, coeff_tilt_rate=args.coeff_tilt_rate, ref_accel_limit=args.ref_accel_limit, direct_tau=args.direct_tau, direct_a_max=args.direct_a_max, action_deadzone=args.action_deadzone, yaw_follow=args.yaw_follow, building=args.building, house_level=(args.house_level[0] if len(args.house_level) == 1 else tuple(args.house_level)),
            n_rooms=(args.n_rooms_min, args.n_rooms_max),

        )

        print(f"env_kwargs pour cette nouvelle version dérivée : {env_kwargs} — vérifie que ça "

              f"correspond bien à ce que tu voulais tester (rien n'est repris automatiquement de "

              f"la version source ici, contrairement à --resume).")

        env = build_vec_env(n_envs, args.no_subproc, env_kwargs)

        version = mm.next_minor_version(args.name, args.from_version)

        src_paths = mm.checkpoint_paths(args.name, args.from_version)

        if not os.path.exists(src_paths["model"]):

            raise ValueError(f"Version source introuvable : {args.name}/{args.from_version}")

        print(f"Nouvelle version {args.name}/{version}, dérivée de {args.from_version}")

        env = VecNormalize.load(src_paths["vecnormalize"], env)

        model = PPO.load(src_paths["model"], env=env, device=device)



    else:

        env_kwargs = dict(

            task=args.task, gear_roll_pitch=args.gear_roll_pitch, gear_yaw=args.gear_yaw,

            up_z_min=args.up_z_min, coverage_target=args.coverage_target, death_penalty=args.death_penalty,

            coeff_spin=args.coeff_spin,

            coeff_align=args.coeff_align, joint_damping=args.damping,

            coeff_proximity=args.coeff_proximity, proximity_threshold=args.proximity_threshold,

            coeff_speed=args.coeff_speed, max_speed=args.max_speed,
            max_climb_rate=args.max_climb_rate, max_yaw_rate=args.max_yaw_rate,
            max_tilt_angle_deg=args.max_tilt_angle_deg,
            kp_vel=args.kp_vel, ki_vel=args.ki_vel, kp_att=args.kp_att,
            kp_rate=args.kp_rate, ki_rate=args.ki_rate, kd_rate=args.kd_rate,
            kp_alt=args.kp_alt, ki_alt=args.ki_alt,
            action_smoothing_alpha=args.action_smoothing_alpha, coeff_potential=args.coeff_potential,
            substeps=args.substeps, ego_crop=not args.no_ego_crop,
            use_last_action=not args.no_last_action, time_penalty=args.time_penalty,
            obs_keys=args.obs_keys, fixed_altitude=not args.free_altitude, k_frontiers=args.k_frontiers,
            no_yaw=args.no_yaw, action_mode=args.action_mode, start_outside=not args.start_inside, drag=args.drag, coeff_action_rate=args.coeff_action_rate, coeff_tilt_rate=args.coeff_tilt_rate, ref_accel_limit=args.ref_accel_limit, direct_tau=args.direct_tau, direct_a_max=args.direct_a_max, action_deadzone=args.action_deadzone, yaw_follow=args.yaw_follow, building=args.building, house_level=(args.house_level[0] if len(args.house_level) == 1 else tuple(args.house_level)),
            n_rooms=(args.n_rooms_min, args.n_rooms_max),

        )

        print(f"env_kwargs pour ce nouvel entraînement : {env_kwargs}")

        env = build_vec_env(n_envs, args.no_subproc, env_kwargs)

        version = mm.next_major_version(args.name)

        print(f"Nouvelle version {args.name}/{version}, entraînement depuis zéro")

        env = VecNormalize(env, norm_obs=True, norm_obs_keys=[k for k in NORM_OBS_KEYS if k in args.obs_keys],

                            norm_reward=True, clip_obs=10.0, clip_reward=10.0)

        policy_kwargs = dict(

            features_extractor_class=ExplorerFeaturesExtractor,

            features_extractor_kwargs=dict(cnn_out_dim=128, mlp_out_dim=64),

            net_arch=dict(pi=[128], vf=[128]),

        )

        caps_kw = {"caps_lambda": args.caps_lambda} if args.caps_lambda > 0 else {}
        model = (PPOCaps if args.caps_lambda > 0 else PPO)("MultiInputPolicy", env, policy_kwargs=policy_kwargs, **caps_kw,

                     n_steps=args.n_steps, batch_size=args.batch_size,

                     seed=args.seed, device=device, gamma=args.gamma,

                     verbose=1, tensorboard_log=args.logdir)



    checkpoint_callback = mm.RotatingCheckpointCallback(

        name=args.name, version=version, vec_normalize_env=env,

        save_freq_steps=args.checkpoint_freq,

        extra_meta={"base_timesteps": base_timesteps, "env_kwargs": env_kwargs}, verbose=1,

    )

    metrics_callback = mm.EpisodeMetricsCallback()

    from stable_baselines3.common.callbacks import CallbackList

    callbacks = [checkpoint_callback, metrics_callback]
    ram_callback = None
    if ram_guard_on:
        ram_callback = mm.RamGuardCallback(min_free_gb=args.ram_reserve_gb * 0.5, max_tree_gb=args.max_ram_gb)
        callbacks.append(ram_callback)
        print(f"RAM au démarrage de l'entraînement : {ram_guard.tree_rss_gb():.2f} Go "
              f"(processus principal + {n_envs} workers).")
    callback = CallbackList(callbacks)

    # Tout ce qui suit doit finir par libérer les workers et la mémoire, quelle que soit la sortie
    # (fin normale, Ctrl+C, manque de RAM, worker mort, bug) — d'où le try/finally.
    stop_reason, exit_code = None, 0
    try:
        try:
            model.learn(total_timesteps=args.total_timesteps,
                        callback=callback,
                        tb_log_name=f"{args.name}_{version}",
                        reset_num_timesteps=not args.resume)
            if ram_callback is not None and ram_callback.stop_reason:
                stop_reason, exit_code = ram_callback.stop_reason, 3
        except KeyboardInterrupt:
            stop_reason, exit_code = "interruption clavier (Ctrl+C)", 130
        except Exception as e:  # noqa: BLE001
            if not _is_memory_failure(e):
                raise
            stop_reason, exit_code = f"{type(e).__name__}: {e}", 2
            print(f"\n❌ Manque de mémoire (ou worker tué par le système) : {stop_reason}")
            print("   Les workers sont fermés et la RAM libérée ; relance avec moins d'envs "
                  "(--n-envs) ou après avoir fermé d'autres programmes.")

        # Sauvegarde dans tous les cas (fin normale ou arrêt propre) : le total est le VRAI nombre de
        # steps faits, pas args.total_timesteps, et --resume repart de là.
        total = model.num_timesteps
        try:
            mm.save_checkpoint(args.name, version, model, env, total,
                               extra_meta={"n_envs": n_envs, "n_steps": args.n_steps,
                                           "batch_size": args.batch_size, "env_kwargs": env_kwargs,
                                           "seed": args.seed, "gamma": args.gamma,
                                           "stopped_early": stop_reason})
            saved = True
        except Exception as e:  # noqa: BLE001 - déjà en train de s'arrêter : on le dit sans masquer la cause
            saved = False
            print(f"❌ Sauvegarde finale impossible : {type(e).__name__}: {e}")
            exit_code = exit_code or 2

        if stop_reason:
            print(f"Entraînement ARRÊTÉ : {stop_reason}.")
            if saved:
                print(f"{args.name}/{version} sauvegardé à {total} steps — reprise possible avec "
                      f"--resume (env_kwargs relus du metadata.json).")
        else:
            print(f"Entraînement terminé. {args.name}/{version} sauvegardé ({total} steps au total) "
                  f"dans models/{args.name}/{version}/")
    finally:
        model = None
        ram_guard.release(env)
        if ram_guard_on:
            print(f"RAM après nettoyage : {ram_guard.available_gb():.1f} Go libres, "
                  f"{ram_guard.tree_rss_gb():.2f} Go pour ce processus et ses enfants.")
    return exit_code


def _is_memory_failure(e):
    """Erreur qui ressemble à un manque de mémoire : MemoryError, OOM CUDA, ou un worker SubprocVecEnv
    disparu (le système l'a tué, ce qui ressort côté parent en EOFError/BrokenPipeError)."""
    if isinstance(e, (MemoryError, EOFError, BrokenPipeError, ConnectionError)):
        return True
    msg = str(e).lower()
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in msg or "unable to allocate" in msg


if __name__ == "__main__":

    try:
        code = main()
    finally:
        # Filet de sécurité si une exception sort de main() avant le nettoyage normal (ex : échec
        # pendant la création du modèle, alors que les workers tournent déjà).
        ram_guard.kill_leftover_children()
    sys.exit(code)
