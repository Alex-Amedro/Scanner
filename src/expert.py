"""Scripted explorer used as the teacher for imitation (no learning).

It decides from the same inputs as the network: the 16 distance sectors and the BFS frontier vector.
  - heading: first step of the BFS path to the nearest frontier
  - speed: speed_frac x 6 m/s, lower near walls
  - avoidance: pushed away from sectors closer than 0.8 m
Fixed heading, so the body frame is the world frame.
"""
import numpy as np

N_BINS = 16
PORTEE = 15.0
_ANG = np.linspace(-np.pi, np.pi, 180, endpoint=False)
BIN_DIRS = np.array([[np.cos(a.mean()), np.sin(a.mean())] for a in np.array_split(_ANG, N_BINS)])


def expert_action(obs, speed_frac=0.6, rep_gain=0.4):
    """obs : dict d'observations BRUTES (proximity_rays, frontier_vector). Renvoie une action (vx, vy) dans [-1, 1]."""
    bins = np.asarray(obs["proximity_rays"], dtype=np.float64) * PORTEE
    f = np.asarray(obs["frontier_vector"], dtype=np.float64)
    if f[3] > 0:                                         # frontière 1 valide
        a = f[1] * np.pi
        v = speed_frac * float(np.clip((bins.min() - 0.3) / 0.9, 0.25, 1.0))
        vec = v * np.array([np.cos(a), np.sin(a)])
    else:
        vec = np.zeros(2)
    w = np.clip((0.8 - bins) / 0.8, 0.0, 1.0)
    vec = vec - rep_gain * (w[:, None] * BIN_DIRS).sum(0) / N_BINS * 6.0
    return np.clip(vec, -1.0, 1.0).astype(np.float32)


# Réglages d'environnement de l'expert et de l'imitation (maisons, compas BFS, marge 0,45 m) : tout est ici, aucun modèle sur disque n'est nécessaire.
EXPERT_ENV_KW = {"task": "explore", "gear_roll_pitch": 0.5, "gear_yaw": 0.25, "up_z_min": 0.3, "coverage_target": 0.9, "death_penalty": 50.0,
                 "coeff_spin": 0.0, "coeff_align": 0.0, "joint_damping": 0.0, "coeff_proximity": 0.0, "proximity_threshold": 0.6, "coeff_speed": 0.0,
                 "max_speed": 6.0, "max_climb_rate": 2.0, "max_yaw_rate": 2.0, "max_tilt_angle_deg": 35.0, "kp_vel": 4.0, "ki_vel": 0.1, "kp_att": 20.0,
                 "kp_rate": 0.4, "ki_rate": 0.2, "kd_rate": 0.003, "kp_alt": 0.15, "ki_alt": 0.05, "action_smoothing_alpha": 1.0, "coeff_potential": 0.0,
                 "substeps": 10, "ego_crop": True, "use_last_action": True, "time_penalty": 0.0,
                 "obs_keys": ["proximity_rays", "frontier_vector", "velocity", "local_crop"], "fixed_altitude": True, "k_frontiers": 2, "no_yaw": True,
                 "action_mode": "velocity", "start_outside": True, "frontier_mode": "bfs", "bfs_clearance_m": 0.45, "building": "house",
                 "n_rooms": (3, 3), "max_steps": 3000}
