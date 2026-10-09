"""Step responses of the three controller loops, each tested alone, without RL: rate, attitude, velocity.

Contacts are disabled (the drone flies through walls): only the dynamics matter. Writes a PNG and prints rise time, overshoot and settling time.

Usage:
    python step_response.py
    python step_response.py --kp-vel 3 --ki-vel 0 --kp-att 6 --kp-rate 0.15 --ki-rate 0.2 --kd-rate 0.003
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from explorer_env import ExplorerEnv, _quat_roll_pitch_upz

ap = argparse.ArgumentParser()
for k, d in dict(kp_vel=1.5, ki_vel=0.1, kp_att=6.0, kp_rate=0.15, ki_rate=0.20, kd_rate=0.003, max_tilt=35.0, damping=0.05).items():
    ap.add_argument("--" + k.replace("_", "-"), type=float, default=d)
ap.add_argument("--out", default=os.path.join("..", "diagnostics", "step_response.png"))
args = ap.parse_args()


def fresh():
    e = ExplorerEnv(n_rooms=(3, 3), seed=0, obs_keys=("proximity_rays", "frontier_vector", "velocity"),
                    fixed_altitude=True, no_yaw=True, k_frontiers=2, joint_damping=args.damping, max_speed=6.0,
                    kp_vel=args.kp_vel, ki_vel=args.ki_vel, kp_att=args.kp_att, kp_rate=args.kp_rate,
                    ki_rate=args.ki_rate, kd_rate=args.kd_rate, max_tilt_angle_deg=args.max_tilt)
    e.reset(seed=1)
    e.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return e


def metrics(t, y, target, tol=0.05):
    """temps de montée 10-90 %, dépassement %, temps d'établissement (+-tol), nombre de pics (oscillation)."""
    y = np.asarray(y)
    lo, hi = 0.1 * target, 0.9 * target
    t10 = t[np.argmax(y >= lo)] if (y >= lo).any() else np.nan
    t90 = t[np.argmax(y >= hi)] if (y >= hi).any() else np.nan
    over = max(0.0, (y.max() - target) / target * 100)
    out = np.abs(y - target) > tol * abs(target)
    settle = t[np.where(out)[0][-1] + 1] if out.any() and np.where(out)[0][-1] + 1 < len(t) else (np.nan if out.any() else 0.0)
    d = np.diff(y)
    peaks = int(((d[1:] < 0) & (d[:-1] >= 0)).sum())
    return t90 - t10, over, settle, peaks


# 1. BOUCLE DE TAUX seule : formules identiques à _inner_attitude_rate_loop, sans le terme d'attitude
e = fresh()
dt = e._dt_sub
T = 0.6
rate_des = 1.0
ts, ps = [], []
int_p = int_q = int_r = 0.0
prev_p = prev_q = 0.0
for k in range(int(T / dt)):
    p, q, r = e.data.qvel[3:6]
    err_p, err_q, err_r = rate_des - p, 0.0 - q, 0.0 - r
    int_p = np.clip(int_p + err_p * dt, -1, 1); int_q = np.clip(int_q + err_q * dt, -1, 1); int_r = np.clip(int_r + err_r * dt, -1, 1)
    d_p = -(p - prev_p) / dt; d_q = -(q - prev_q) / dt; prev_p, prev_q = float(p), float(q)
    e.data.ctrl[:] = [e._hover_thrust_ctrl,
                      np.clip(e.kp_rate * err_p + e.ki_rate * int_p + e.kd_rate * d_p, -1, 1),
                      np.clip(e.kp_rate * err_q + e.ki_rate * int_q + e.kd_rate * d_q, -1, 1),
                      np.clip(e.kp_rate * err_r + e.ki_rate * int_r, -1, 1)]
    mujoco.mj_step(e.model, e.data)
    ts.append((k + 1) * dt); ps.append(e.data.qvel[3])
t1, p1 = np.array(ts), np.array(ps)
m1 = metrics(t1, p1, rate_des)

# 2. BOUCLE D'ATTITUDE seule : consigne d'angle fixe
res2 = {}
for deg in (10.0, 25.0):
    e = fresh()
    des = np.radians(deg)
    tt, rr = [], []
    for k in range(int(1.2 / dt)):
        e.data.ctrl[:] = e._inner_attitude_rate_loop((des, 0.0, e._hover_thrust_ctrl, 0.0))
        mujoco.mj_step(e.model, e.data)
        roll, pitch, up = _quat_roll_pitch_upz(e.data.body("drone_body").xquat)
        tt.append((k + 1) * dt); rr.append(np.degrees(roll))
    res2[deg] = (np.array(tt), np.array(rr), metrics(np.array(tt), rr, deg))

# 3. BOUCLE DE VITESSE seule : consigne de vitesse via l'environnement complet mais SANS politique
res3 = {}
for v in (2.0, 4.0, 6.0):
    e = fresh()
    tt, vv, pp = [], [], []
    for k in range(int(3.0 / 0.02)):
        o, r, te, tr, info = e.step(np.array([v / 6.0, 0.0], dtype=np.float32))
        tt.append((k + 1) * 0.02); vv.append(info["speed_horiz"]); pp.append(info["pitch_deg"])
    res3[v] = (np.array(tt), np.array(vv), np.array(pp), metrics(np.array(tt), vv, v, tol=0.05))

print(f"Gains : amortissement articulation={args.damping} | kp_rate={args.kp_rate} ki_rate={args.ki_rate} kd_rate={args.kd_rate} | kp_att={args.kp_att} | kp_vel={args.kp_vel} ki_vel={args.ki_vel} | tilt max {args.max_tilt} deg")
print(f"{'boucle':34s} {'montee 10-90%':>14s} {'depassement':>12s} {'etabli (+-5%)':>14s} {'pics':>5s}")
fmt = lambda m: f"{m[0]*1000:11.0f} ms {m[1]:10.1f} % {('%.2f s' % m[2]) if not np.isnan(m[2]) else '  jamais':>13s} {m[3]:5d}"
print(f"{'1. taux (consigne 1 rad/s)':34s} {fmt(m1)}")
for deg, (t, y, m) in res2.items():
    print(f"{'2. attitude (consigne %d deg)' % deg:34s} {fmt(m)}")
for v, (t, y, pp, m) in res3.items():
    print(f"{'3. vitesse (consigne %.0f m/s)' % v:34s} {fmt(m)}")

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0, 0].plot(t1, p1, label="vitesse de roulis réelle"); ax[0, 0].axhline(rate_des, color="r", ls="--", label="consigne")
ax[0, 0].set_title("1. Boucle de TAUX seule (consigne 1 rad/s)"); ax[0, 0].set_xlabel("s"); ax[0, 0].set_ylabel("rad/s"); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)
for deg, (t, y, m) in res2.items():
    ax[0, 1].plot(t, y, label=f"réel (consigne {deg:.0f}°)"); ax[0, 1].axhline(deg, color="r", ls="--", lw=.8)
ax[0, 1].set_title("2. Boucle d'ATTITUDE seule (angle de roulis)"); ax[0, 1].set_xlabel("s"); ax[0, 1].set_ylabel("deg"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)
for v, (t, y, pp, m) in res3.items():
    ax[1, 0].plot(t, y, label=f"réel (consigne {v:.0f} m/s)"); ax[1, 0].axhline(v, color="r", ls="--", lw=.8)
ax[1, 0].set_title("3. Boucle de VITESSE seule"); ax[1, 0].set_xlabel("s"); ax[1, 0].set_ylabel("m/s"); ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)
for v, (t, y, pp, m) in res3.items():
    ax[1, 1].plot(t, pp, label=f"inclinaison (consigne {v:.0f} m/s)")
ax[1, 1].set_title("3b. Inclinaison pendant l'échelon de vitesse"); ax[1, 1].set_xlabel("s"); ax[1, 1].set_ylabel("deg"); ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)
fig.suptitle(f"Réponses indicielles — kp_rate={args.kp_rate} ki_rate={args.ki_rate} kd_rate={args.kd_rate}, kp_att={args.kp_att}, kp_vel={args.kp_vel}, ki_vel={args.ki_vel}")
fig.tight_layout()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out, dpi=110)
print("Courbes écrites dans", os.path.abspath(args.out))
