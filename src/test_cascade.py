"""Smoke test of the PID cascade (velocity -> attitude -> rates) and of the egocentric observations.

Contacts are disabled so the drone can fly through walls: only the dynamics are tested.

Usage:
    python test_cascade.py
"""

import mujoco
import numpy as np

from explorer_env import ExplorerEnv, _quat_roll_pitch_upz, _quat_to_yaw
from occupancy_grid import OccupancyGrid

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'ÉCHEC'}] {name} {detail}")
    if not ok:
        FAILED.append(name)


def fresh_env(**kw):
    env = ExplorerEnv(seed=0, **kw)
    env.reset(seed=0)
    env.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return env


def run(env, action, n):
    """Joue n steps ; renvoie les listes (vitesse corps, roll/pitch max, z)."""
    vb, tilt, zs = [], 0.0, []
    for _ in range(n):
        obs, _, term, _, info = env.step(action)
        vb.append(obs["kinematics"][:3].copy())
        tilt = max(tilt, abs(info["roll_deg"]), abs(info["pitch_deg"]))
        zs.append(info["pos"][2])
        assert np.isfinite(obs["kinematics"]).all(), "observation non finie"
    return np.array(vb), tilt, np.array(zs)


def main():
    # 1. hover : action nulle
    env = fresh_env()
    vb, tilt, zs = run(env, np.zeros(4), 300)
    check("hover : altitude tenue", abs(zs[-1] - zs[0]) < 0.15, f"(dz={zs[-1]-zs[0]:+.3f} m)")
    check("hover : pas de dérive", np.linalg.norm(vb[-1][:2]) < 0.1, f"(|v|={np.linalg.norm(vb[-1][:2]):.3f})")
    check("hover : à plat", tilt < 3.0, f"(tilt max={tilt:.2f}°)")

    # 2. avance à vitesse constante : 0.5 * max_speed
    env = fresh_env()
    cmd = np.array([0.5, 0, 0, 0], dtype=np.float32)
    vb, tilt, zs = run(env, cmd, 250)
    target = 0.5 * env.max_speed
    check("avance : vitesse corps atteinte", abs(vb[-1][0] - target) < 0.3, f"(vx={vb[-1][0]:.2f}, cible {target})")
    check("avance : pas de latéral", abs(vb[-1][1]) < 0.3, f"(vy={vb[-1][1]:.2f})")
    check("avance : inclinaison bornée", tilt <= env.max_tilt_angle_deg + 5, f"(tilt max={tilt:.1f}°)")
    check("avance : altitude tenue (poussée compensée)", abs(zs[-1] - zs[0]) < 0.3, f"(dz={zs[-1]-zs[0]:+.3f} m)")
    t63 = int(np.argmax(vb[:, 0] > 0.63 * target)) * 0.02
    print(f"      temps de réponse (63%) : {t63:.2f} s")

    # 3. freinage depuis cette vitesse
    vb, tilt, zs = run(env, np.zeros(4), 150)
    check("freinage : arrêt", np.linalg.norm(vb[-1][:2]) < 0.3, f"(|v|={np.linalg.norm(vb[-1][:2]):.2f})")
    check("freinage : altitude tenue", abs(zs[-1] - zs[0]) < 0.4, f"(dz={zs[-1]-zs[0]:+.3f} m)")

    # 4. repère du corps : tourner de ~90° puis avancer -> vitesse monde le long du nouveau cap,
    #    vitesse observée en repère corps toujours le long de x
    env = fresh_env()
    for _ in range(40):
        env.step(np.array([0, 0, 0, 1.0], dtype=np.float32))
    for _ in range(60):
        env.step(np.zeros(4, dtype=np.float32))
    yaw = _quat_to_yaw(env.data.body("drone_body").xquat)
    obs = None
    for _ in range(250):
        obs, *_ = env.step(np.array([0.5, 0, 0, 0], dtype=np.float32))
    v_world = env.data.qvel[0:2]
    heading = np.array([np.cos(_quat_to_yaw(env.data.body("drone_body").xquat)),
                        np.sin(_quat_to_yaw(env.data.body("drone_body").xquat))])
    check("repère : yaw non nul avant l'avance", abs(np.degrees(yaw)) > 20, f"(yaw={np.degrees(yaw):.0f}°)")
    check("repère : vitesse monde alignée sur le cap",
          np.dot(v_world, heading) > 0.9 * np.linalg.norm(v_world) and np.linalg.norm(v_world) > 2.0,
          f"(|v|={np.linalg.norm(v_world):.2f})")
    check("repère : obs vitesse en repère corps (vx>0, vy~0)",
          obs["kinematics"][0] > 2.0 and abs(obs["kinematics"][1]) < 0.4,
          f"(vx={obs['kinematics'][0]:.2f}, vy={obs['kinematics'][1]:.2f})")

    # 5. latéral gauche (vy > 0 corps)
    env = fresh_env()
    vb, tilt, _ = run(env, np.array([0, 0.5, 0, 0], dtype=np.float32), 250)
    check("latéral : vy corps atteinte", abs(vb[-1][1] - 0.5 * env.max_speed) < 0.3, f"(vy={vb[-1][1]:.2f})")

    # 6. vertical + lacet
    env = fresh_env()
    vb, tilt, zs = run(env, np.array([0, 0, 0.5, 0], dtype=np.float32), 100)
    check("montée : vz atteinte", abs(vb[-1][2] - 0.5 * env.max_climb_rate) < 0.2, f"(vz={vb[-1][2]:.2f})")
    env = fresh_env()
    for _ in range(100):
        obs, *_ = env.step(np.array([0, 0, 0, 0.5], dtype=np.float32))
    check("lacet : taux atteint", abs(obs["kinematics"][5] - 0.5 * env.max_yaw_rate) < 0.15,
          f"(r={obs['kinematics'][5]:.2f})")

    # 7. norme de consigne plafonnée (diagonale)
    env = fresh_env()
    vb, *_ = run(env, np.array([1, 1, 0, 0], dtype=np.float32), 400)
    v = np.linalg.norm(vb[-1][:2])
    check("diagonale pleine : |v| <= max_speed", v <= env.max_speed + 0.3, f"(|v|={v:.2f}, max={env.max_speed})")

    # 8. lissage + last_action
    env = fresh_env(action_smoothing_alpha=0.3)
    obs, *_ = env.step(np.array([1, 0, 0, 0], dtype=np.float32))
    check("last_action = EMA de l'action", abs(obs["last_action"][0] - env.action_smoothing_alpha) < 1e-6,
          f"({obs['last_action']})")

    # 9. crop égocentrique : un mur marqué DEVANT le drone doit apparaître en haut de l'image,
    #    quel que soit le yaw ; un mur à GAUCHE, à gauche de l'image
    g = OccupancyGrid((-10, 10), (-10, 10), resolution=0.15)
    g.grid[:] = 0
    for yaw in (0.0, np.pi / 2, 1.0, -2.0):
        g.grid[:] = 0
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-np.sin(yaw), np.cos(yaw)])
        for d in np.arange(2.0, 2.3, 0.05):                         # mur à 2 m devant
            cx, cy = g.world_to_cell(*(fwd * d))
            g.grid[cy, cx] = 1
        for d in np.arange(-0.5, 0.5, 0.05):
            cx, cy = g.world_to_cell(*(fwd * 2.0 + left * d))
            g.grid[cy, cx] = 1
        crop = g.local_crop_onehot_ego(0.0, 0.0, yaw, 64)
        ys, xs = np.nonzero(crop[1])
        row_ok = len(ys) > 0 and abs(np.mean(ys) - (32 - 2.0 / 0.15)) < 2.0
        col_ok = len(xs) > 0 and abs(np.mean(xs) - 32) < 2.0
        check(f"crop égocentrique (yaw={np.degrees(yaw):.0f}°) : mur devant en haut, centré", row_ok and col_ok,
              f"(ligne moy.={np.mean(ys) if len(ys) else float('nan'):.1f}, col moy.={np.mean(xs) if len(xs) else float('nan'):.1f})")
    g.grid[:] = 0
    left = np.array([0.0, 1.0])  # yaw=0 : gauche = +y
    for d in np.arange(-0.5, 0.5, 0.05):
        cx, cy = g.world_to_cell(d, 2.0)
        g.grid[cy, cx] = 1
    crop = g.local_crop_onehot_ego(0.0, 0.0, 0.0, 64)
    ys, xs = np.nonzero(crop[1])
    check("crop égocentrique : mur à gauche = colonnes de gauche", np.mean(xs) < 32 - 8, f"(col moy.={np.mean(xs):.1f})")

    print("\n" + ("TOUS LES TESTS PASSENT" if not FAILED else f"ÉCHECS : {FAILED}"))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
