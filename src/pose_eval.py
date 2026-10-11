"""Measure what pose errors do to the scripted explorer.

For each (level, noise, seed) the explorer flies one house with the pose source chosen, and the script records:
  - pose error: RMSE, maximum and final error in metres, yaw error in degrees, final drift in % of the distance flown
  - map quality: the estimated map against the reference map (same scans placed with the exact pose), F1 on occupied cells with a 1-cell tolerance
  - task result: success (true coverage >= 90%, no collision), collision, timeout; true coverage
  - cost: wall-clock time per simulation step (ms)

Usage:
    python pose_eval.py --levels 1 2 3 4 --noise none low medium high --seeds 9000 9020 --out ../docs/results/gate1_dead_reckoning.csv
    python pose_eval.py --table ../docs/results/gate1_dead_reckoning.csv
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert import EXPERT_ENV_KW, expert_action                    # noqa: E402
from explorer_env import ExplorerEnv                                # noqa: E402
from occupancy_grid import OCCUPIED                                 # noqa: E402

FIELDS = ["level", "noise", "pose_source", "seed", "outcome", "steps", "true_coverage", "path_m", "pose_rmse_m", "pose_max_m",
          "pose_final_m", "yaw_rmse_deg", "yaw_final_deg", "drift_pct", "map_f1", "map_precision", "map_recall", "ms_per_step", "match_ms", "corrected_pct", "keyframes", "loops", "loops_false", "loops_rejected",
          "loop_err_before_m", "loop_err_after_m", "backend_ms", "odom_only_rmse_m", "odom_only_max_m", "odom_only_yaw_rmse_deg", "odom_only_map_f1"]


def _dilate(m, r=1):
    out = m
    for _ in range(r):
        a = out.copy()
        a[1:, :] |= out[:-1, :]; a[:-1, :] |= out[1:, :]
        b = a.copy()
        b[:, 1:] |= a[:, :-1]; b[:, :-1] |= a[:, 1:]
        out = b
    return out


def map_f1(est_grid, ref_grid, tol=1):
    e, r = est_grid.grid == OCCUPIED, ref_grid.grid == OCCUPIED
    prec = (e & _dilate(r, tol)).sum() / max(int(e.sum()), 1)
    rec = (r & _dilate(e, tol)).sum() / max(int(r.sum()), 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return float(f1), float(prec), float(rec)


def run_episode(task):
    level, noise, seed, pose_source = task
    kw = dict(EXPERT_ENV_KW)
    kw.update(house_level=level, obs_keys=["proximity_rays", "frontier_vector", "velocity"], max_steps=5000)
    if noise != "none":
        kw.update(pose_source=pose_source, odom_noise=noise, shadow_odom=pose_source in ("slam", "slam_lc"))
    env = ExplorerEnv(seed=seed, **kw)
    obs, _ = env.reset(seed=seed)
    t0 = time.perf_counter()
    term = trunc = False
    for t in range(env.max_steps):
        obs, r, term, trunc, info = env.step(expert_action(obs))
        if term or trunc:
            break
    ms = (time.perf_counter() - t0) / (t + 1) * 1000.0
    cov = env.grid.coverage_ratio()
    outcome = "success" if (term and cov >= env.coverage_target) else ("collision" if term else "timeout")
    row = dict(level=level, noise=noise, pose_source=pose_source if noise != "none" else "oracle", seed=seed, outcome=outcome, steps=t + 1,
               true_coverage=round(cov, 4), ms_per_step=round(ms, 2), path_m=round(env.path_length, 2) if env._odom else "")
    if env._odom is not None:
        pe, ye = np.array(env.pose_err), np.degrees(np.array(env.yaw_err))
        f1, p, rc = map_f1(env.est_grid, env.grid)
        row.update(pose_rmse_m=round(float(np.sqrt(np.mean(pe ** 2))), 3), pose_max_m=round(float(pe.max()), 3), pose_final_m=round(float(pe[-1]), 3),
                   yaw_rmse_deg=round(float(np.sqrt(np.mean(ye ** 2))), 2), yaw_final_deg=round(float(abs(ye[-1])), 2),
                   drift_pct=round(100.0 * float(pe[-1]) / max(env.path_length, 1e-6), 2), map_f1=round(f1, 3), map_precision=round(p, 3),
                   map_recall=round(rc, 3))
    if env._odom is not None and getattr(env._odom, "match_ms", None):
        row["match_ms"] = round(float(np.mean(env._odom.match_ms)), 2)
        row["corrected_pct"] = round(100.0 * env._odom.n_corrected / len(env._odom.match_ms), 1)
    if env._slam is not None:
        sl = env._slam
        row.update(keyframes=len(sl.kfs), loops=sl.loops_accepted, loops_false=sl.loops_false, loops_rejected=sl.loops_rejected,
                   backend_ms=round(float(np.mean(sl.kf_ms)), 2) if sl.kf_ms else "")
        if sl.events:
            row["loop_err_before_m"] = round(float(np.mean([ev["err_before"] for ev in sl.events])), 3)
            row["loop_err_after_m"] = round(float(np.mean([ev["err_after"] for ev in sl.events])), 3)
    if env._shadow is not None and env.shadow_err:
        se, sy = np.array(env.shadow_err), np.degrees(np.array(env.shadow_yaw_err))
        row.update(odom_only_rmse_m=round(float(np.sqrt(np.mean(se ** 2))), 3), odom_only_max_m=round(float(se.max()), 3),
                   odom_only_yaw_rmse_deg=round(float(np.sqrt(np.mean(sy ** 2))), 2), odom_only_map_f1=round(map_f1(env.shadow_grid, env.grid)[0], 3))
    env.close()
    return row


def table(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    groups = {}
    for r in rows:
        groups.setdefault((int(r["level"]), r["noise"]), []).append(r)
    order = {"none": 0, "low": 1, "medium": 2, "high": 3, "extreme": 4}
    f = lambda rs, k: np.mean([float(x[k]) for x in rs if x[k] != ""]) if any(x[k] != "" for x in rs) else float("nan")
    print(f"{'niveau':>6} {'bruit':>7} {'n':>3} | {'succès':>6} {'collis.':>7} {'timeout':>7} {'couv.vraie':>10} | {'RMSE m':>6} {'max m':>6} {'lacet °':>7} {'dérive %':>8} | {'F1 carte':>8} | {'ms/pas':>6}")
    for (lvl, nz), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], order[kv[0][1]])):
        n = len(rs)
        pct = lambda o: 100.0 * sum(1 for x in rs if x["outcome"] == o) / n
        print(f"{lvl:>6} {nz:>7} {n:>3} | {pct('success'):5.0f}% {pct('collision'):6.0f}% {pct('timeout'):6.0f}% {100 * f(rs, 'true_coverage'):9.1f}% | "
              f"{f(rs, 'pose_rmse_m'):6.2f} {f(rs, 'pose_max_m'):6.2f} {f(rs, 'yaw_rmse_deg'):7.1f} {f(rs, 'drift_pct'):8.1f} | {f(rs, 'map_f1'):8.2f} | {f(rs, 'ms_per_step'):6.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--noise", nargs="+", default=["none", "low", "medium", "high"], help="none = exact pose")
    ap.add_argument("--pose-source", default="odom", choices=["odom", "slam", "slam_lc"])
    ap.add_argument("--seeds", type=int, nargs=2, default=[9000, 9020], help="seeds a..b-1")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join("..", "docs", "results", "gate1_dead_reckoning.csv"))
    ap.add_argument("--table", default=None, help="Only print the summary table of an existing csv.")
    a = ap.parse_args()
    if a.table:
        table(a.table)
        sys.exit(0)
    tasks = [(lvl, nz, s, a.pose_source) for lvl in a.levels for nz in a.noise for s in range(*a.seeds)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    with Pool(a.procs) as pool, open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_episode, tasks), 1):
            w.writerow({k: row.get(k, "") for k in FIELDS})
            fh.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(tasks)} épisodes ({time.time() - t0:.0f} s)", flush=True)
    print("écrit", os.path.abspath(a.out))
    table(a.out)
