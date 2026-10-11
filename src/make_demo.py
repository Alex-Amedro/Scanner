"""Split-screen demo videos and gifs. It only observes the environment, never changes its behaviour.

Left: MuJoCo view (top view or 3D follow camera). Right: the occupancy grid built live, with the LiDAR rays, the drone and its heading,
the target frontier, the BFS path and the coverage. The controller is the scripted explorer (default) or the learned policy (--policy clone).
The nose follows the direction of motion.

Usage:
    python make_demo.py --list --level 1 --seed-range 9000 9016          # which seeds finish
    python make_demo.py --level 1 --seeds 9007 --gif --out ../demo
    python make_demo.py --policy clone --model bc2_s42 --level 2 --seeds 9002 --out ../demo
    python make_demo.py --view follow --level 1 --seeds 9007 --out ../demo   # 3D camera
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

from expert import expert_action
from explorer_env import ExplorerEnv, _quat_to_yaw
from occupancy_grid import FREE, OCCUPIED, UNKNOWN, OccupancyGrid

sys.stdout.reconfigure(errors="replace")
OccupancyGrid.record_path = True      # garde le chemin BFS pour le dessiner (aucun effet sur les décisions)

# Réglages d'environnement de l'expert (identiques à ceux de l'imitation) : la vidéo ne dépend d'aucun modèle sur disque.
EXPERT_KW = {"task": "explore", "gear_roll_pitch": 0.5, "gear_yaw": 0.25, "up_z_min": 0.3, "coverage_target": 0.9, "death_penalty": 50.0,
             "coeff_spin": 0.0, "coeff_align": 0.0, "joint_damping": 0.0, "coeff_proximity": 0.0, "proximity_threshold": 0.6, "coeff_speed": 0.0,
             "max_speed": 6.0, "max_climb_rate": 2.0, "max_yaw_rate": 2.0, "max_tilt_angle_deg": 35.0, "kp_vel": 4.0, "ki_vel": 0.1, "kp_att": 20.0,
             "kp_rate": 0.4, "ki_rate": 0.2, "kd_rate": 0.003, "kp_alt": 0.15, "ki_alt": 0.05, "action_smoothing_alpha": 1.0, "coeff_potential": 0.0,
             "substeps": 10, "ego_crop": True, "use_last_action": True, "time_penalty": 0.0,
             "obs_keys": ["proximity_rays", "frontier_vector", "velocity"], "fixed_altitude": True, "k_frontiers": 2, "no_yaw": True,
             "action_mode": "velocity", "start_outside": True, "frontier_mode": "bfs", "bfs_clearance_m": 0.45, "building": "house",
             "n_rooms": (3, 3), "max_steps": 3000}
OPT = dict(compare=False, pose_source="oracle", odom_noise="medium", full=True, stag=600, yaw_rate=1.2, yaw_min_speed=0.8, view="top")   # modifiées par la ligne de commande
COLORS = np.array([[0x8F, 0xD1, 0x9E], [0xE0, 0x7A, 0x5F], [0xCC, 0xCC, 0xCC]], dtype=np.uint8)   # free, wall, unknown


def make_font(size):
    for path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


class LeftPanel:
    """Rendu MuJoCo hors écran, vue du dessus, avec un repère rouge sous le drone (il ne mesure que 0,2 m)."""

    def __init__(self, env, size, azimuth=90.0, view="top"):
        self.view = view
        m = env.model
        ceiling = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "plafond")
        if ceiling >= 0:
            m.geom_rgba[ceiling, 3] = 0.0           # invisible seulement dans l'image ; collisions et LiDAR inchangés
        if view == "follow":      # murs semi-transparents dans l'image (rendu seulement), sinon un mur cache le drone
            for gid in range(m.ngeom):
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if nm.startswith(("mur", "ext_", "cour_")):
                    m.geom_rgba[gid, 3] = 0.28
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, size)      # taille du tampon de rendu : réglage d'image seulement
        m.vis.global_.offheight = max(m.vis.global_.offheight, size)
        m.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]       # éclairage de l'image seulement
        m.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
        self.renderer = mujoco.Renderer(m, size, size)
        x0, x1 = env.layout.world_x_range
        y0, y1 = env.layout.world_y_range
        self.cam = mujoco.MjvCamera()
        if view == "follow":      # vue 3D : caméra de suivi derrière le drone (le nez et sa rotation se voient)
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "drone_body")
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 4.8, azimuth, -48.0
        else:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.lookat[:] = [(x0 + x1) / 2, (y0 + y1) / 2, 0.0]
            self.cam.distance = 0.5 * max(x1 - x0, y1 - y0) / np.tan(np.radians(22.5)) * 1.22
            self.cam.azimuth, self.cam.elevation = azimuth, -90.0

    def frame(self, env):
        self.renderer.update_scene(env.data, camera=self.cam)
        scn = self.renderer.scene
        if self.view == "top" and scn.ngeom < scn.maxgeom:
            pos = env.data.body("drone_body").xpos.copy()
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.28, 0, 0]), pos,
                                np.eye(3).flatten(), np.array([1.0, 0.15, 0.1, 0.9], dtype=np.float32))
            scn.ngeom += 1
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


class RightPanel:
    """Grille d'occupation en direct (matplotlib) : carte connue, rayons LiDAR, drone + cap, frontière visée, chemin BFS, couverture."""

    def __init__(self, env, size):
        g = env.grid
        self.size = size
        self.fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
        self.ax = self.fig.add_axes([0.03, 0.06, 0.94, 0.83])
        self.img = self.ax.imshow(self._rgb(g), origin="lower", extent=[g.x_min, g.x_max, g.y_min, g.y_max], interpolation="nearest")
        self.odom = env._odom is not None      # pose estimée : on montre la carte ESTIMÉE, les vrais murs et les deux trajectoires
        if self.odom:
            h, w = g.grid.shape
            self.true_img = self.ax.imshow(np.zeros((h, w, 4), dtype=np.uint8), origin="lower", extent=[g.x_min, g.x_max, g.y_min, g.y_max],
                                           interpolation="nearest", zorder=2)
            self.path_true, = self.ax.plot([], [], color="#0b6e2e", lw=2.4, zorder=5)
            self.path_est, = self.ax.plot([], [], color="#d62828", lw=2.4, zorder=5)
            self.true_marker, = self.ax.plot([], [], marker="o", color="#0b6e2e", ms=10, mec="white", ls="", zorder=9)
            self.xy_true, self.xy_est = [], []
        self.slam = env._slam is not None          # pose graph overlay: keyframes, odometry edges (grey), loop edges (red)
        self.n_frames, self.last_loops, self.flash_until = 0, 0, -1
        if self.slam:
            self.loop_seen = {}                # loop edge index -> frame at which it first appeared (loop edges are drawn only briefly)
            self.g_odom = LineCollection([], colors="#b0b0b0", linewidths=0.8, zorder=4)
            self.g_loop = LineCollection([], colors="#d62828", linewidths=2.2, zorder=6)
            self.ax.add_collection(self.g_odom)
            self.ax.add_collection(self.g_loop)
            self.nodes, = self.ax.plot([], [], marker="o", color="#222222", ms=3.2, ls="", zorder=6)
            self.flash = self.ax.text(0.5, 0.985, "", transform=self.ax.transAxes, ha="center", va="top", fontsize=12, weight="bold", color="white",
                                      bbox=dict(facecolor="#d62828", alpha=0.92, edgecolor="none", pad=4), zorder=20)
        self.rays = LineCollection([], colors="#3d5a80", linewidths=0.4, alpha=0.45, zorder=3)
        self.ax.add_collection(self.rays)
        self.path, = self.ax.plot([], [], color="#1f5fd1", lw=2.2, zorder=4)
        self.target, = self.ax.plot([], [], marker="*", color="gold", mec="black", ms=18, ls="", zorder=6)
        self.drone, = self.ax.plot([], [], marker="o", color="black", ms=9, ls="", zorder=7)
        self.arrow = self.ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2), zorder=8)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(g.x_min, g.x_max)
        self.ax.set_ylim(g.y_min, g.y_max)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.fig.text(0.5, 0.965, ("SLAM: scan matching + loop closure" if env.pose_source == "slam_lc" else "SLAM: scan matching"
                       if env.pose_source == "slam" else "Odometry only (no correction)")
                      if self.odom else "Carte d'occupation construite par le drone",
                      ha="center", va="center", fontsize=13 if self.odom else 15, weight="bold")
        self.bar_bg = Rectangle((0.08, 0.915), 0.84, 0.028, transform=self.fig.transFigure, facecolor="#dddddd", edgecolor="black", lw=0.8)
        self.bar = Rectangle((0.08, 0.915), 0.0, 0.028, transform=self.fig.transFigure, facecolor="#2a9d8f")
        self.fig.add_artist(self.bar_bg); self.fig.add_artist(self.bar)
        self.fig.add_artist(plt.Line2D([0.08 + 0.84 * 0.9] * 2, [0.908, 0.950], transform=self.fig.transFigure, color="black", lw=2))
        self.cov_text = self.fig.text(0.5, 0.929, "", ha="center", va="center", fontsize=11, weight="bold")
        self.fig.text(0.5, 0.025, ((("black: true walls | green: true path | red: estimated path | dots: keyframes | thick red: loop")
                                   if self.slam else "black: true walls | green: true path | red: estimated path") if self.odom
                                   else "vert = libre   corail = mur   gris = inconnu   bleu = rayons LiDAR / chemin   étoile = frontière visée"),
                      ha="center", va="center", fontsize=9.0 if self.odom else 9.5)

    @staticmethod
    def _rgb(g):
        idx = np.full(g.grid.shape, 2, dtype=np.uint8)
        idx[g.grid == FREE] = 0
        idx[g.grid == OCCUPIED] = 1
        return COLORS[idx]

    def frame(self, env, coverage):
        g = env.est_grid if self.odom else env.grid
        self.img.set_data(self._rgb(g))
        o, dirs, dists = env._last_lidar_origin, env._last_lidar_dirs, env._last_lidar_dists
        pos = env.data.body("drone_body").xpos[:2]
        nose = _quat_to_yaw(env.data.body("drone_body").xquat) + np.pi / 2     # le côté orange (+y du corps) est l'avant
        if self.odom:
            ex, ey, eyaw = env._odom.pose
            dpsi = eyaw - _quat_to_yaw(env.data.body("drone_body").xquat)
            c, s_ = np.cos(dpsi), np.sin(dpsi)
            occ = env.grid.grid == OCCUPIED
            rgba = np.zeros(occ.shape + (4,), dtype=np.uint8)
            rgba[occ] = (0, 0, 0, 215)
            self.true_img.set_data(rgba)
            self.rays.set_segments([[(ex, ey), (ex + (c * d[0] - s_ * d[1]) * r, ey + (s_ * d[0] + c * d[1]) * r)] for d, r in zip(dirs, dists)])
            self.xy_true.append((pos[0], pos[1])); self.xy_est.append((ex, ey))
            self.path_true.set_data([p[0] for p in self.xy_true], [p[1] for p in self.xy_true])
            self.path_est.set_data([p[0] for p in self.xy_est], [p[1] for p in self.xy_est])
            self.true_marker.set_data([pos[0]], [pos[1]])
            pos = np.array([ex, ey])          # le drone estimé est le point noir ; le vrai est le point vert
            nose = eyaw + np.pi / 2
        else:
            self.rays.set_segments([[(o[0], o[1]), (o[0] + d[0] * r, o[1] + d[1] * r)] for d, r in zip(dirs, dists)])
        fr = env._last_frontiers[0] if env._last_frontiers else None
        if fr is not None and fr["valid"] > 0 and fr.get("world_xy"):
            self.target.set_data([fr["world_xy"][0]], [fr["world_xy"][1]])
            pts = fr.get("path_xy") or []
            self.path.set_data([p[0] for p in pts], [p[1] for p in pts])
        else:
            self.target.set_data([], []); self.path.set_data([], [])
        self.drone.set_data([pos[0]], [pos[1]])
        self.arrow.xy = (pos[0] + 0.9 * np.cos(nose), pos[1] + 0.9 * np.sin(nose))
        self.arrow.xyann = (pos[0], pos[1])
        self.bar.set_width(0.84 * min(coverage, 1.0))
        self.n_frames += 1
        if self.slam:
            sl = env._slam
            pts = np.array([k.pose[:2] for k in sl.kfs]) if sl.kfs else np.zeros((0, 2))
            self.nodes.set_data(pts[:, 0], pts[:, 1])
            self.g_odom.set_segments([[tuple(sl.kfs[i].pose[:2]), tuple(sl.kfs[j].pose[:2])] for i, j, m, kd in sl.edges if kd == "odom"])
            loops = [e for e in sl.edges if e[3] == "loop"]
            for idx in range(len(loops)):
                self.loop_seen.setdefault(idx, self.n_frames)
            self.g_loop.set_segments([[tuple(sl.kfs[e[0]].pose[:2]), tuple(sl.kfs[e[1]].pose[:2])] for idx, e in enumerate(loops)
                                      if self.n_frames - self.loop_seen[idx] <= 12])
            if sl.loops_accepted > self.last_loops:
                self.last_loops = sl.loops_accepted
                ev = sl.events[-1]
                if ev["max_shift"] > 0.02:       # only announce loops that really moved the map
                    self.flash_until = self.n_frames + 8
                    self.flash.set_text(f"Loop closure #{sl.loops_accepted}: map rebuilt")
            if self.n_frames > self.flash_until:
                self.flash.set_text("")
        if self.odom and self.slam:
            self.cov_text.set_text(f"Coverage {coverage * 100:.0f}%  |  position error {(env.pose_err[-1] if env.pose_err else 0.0):.2f} m  |  loops {env._slam.loops_accepted}")
        elif self.odom:
            self.cov_text.set_text(f"Coverage {coverage * 100:.0f}%  |  position error {(env.pose_err[-1] if env.pose_err else 0.0):.2f} m")
        else:
            self.cov_text.set_text(f"Couverture {coverage * 100:.0f} %   (objectif 90 %)")
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        plt.close(self.fig)


def _dilate1(m):
    a = m.copy()
    a[1:, :] |= m[:-1, :]; a[:-1, :] |= m[1:, :]
    b = a.copy()
    b[:, 1:] |= a[:, :-1]; b[:, :-1] |= a[:, 1:]
    return b


def _wall_match(est_grid, ref_grid):
    """F1 of the occupied cells of a map against the reference map (same scans, exact pose), one cell of tolerance."""
    e, r = est_grid.grid == OCCUPIED, ref_grid.grid == OCCUPIED
    prec = (e & _dilate1(r)).sum() / max(int(e.sum()), 1)
    rec = (r & _dilate1(e)).sum() / max(int(r.sum()), 1)
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


class ComparePanel:
    """One map of the side-by-side comparison, built from the SAME flight, the SAME noisy measurements and the SAME scans.
    kind 'odom': pure dead reckoning (what the drone would believe with the odometer alone). kind 'slam': scan matching + loop closure,
    the estimate the drone actually flies with. Black = true walls, green = true path, red = estimated path."""

    def __init__(self, env, size, kind):
        g = env.grid
        self.kind = kind
        self.fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
        self.ax = self.fig.add_axes([0.03, 0.06, 0.94, 0.80])
        ext = [g.x_min, g.x_max, g.y_min, g.y_max]
        self.img = self.ax.imshow(RightPanel._rgb(g), origin="lower", extent=ext, interpolation="nearest")
        h, w = g.grid.shape
        self.true_img = self.ax.imshow(np.zeros((h, w, 4), dtype=np.uint8), origin="lower", extent=ext, interpolation="nearest", zorder=2)
        self.path_true, = self.ax.plot([], [], color="#0b6e2e", lw=2.4, zorder=5)
        self.path_est, = self.ax.plot([], [], color="#d62828", lw=2.4, zorder=5)
        self.true_marker, = self.ax.plot([], [], marker="o", color="#0b6e2e", ms=10, mec="white", ls="", zorder=9)
        self.est_marker, = self.ax.plot([], [], marker="o", color="black", ms=9, ls="", zorder=8)
        self.xy_true, self.xy_est = [], []
        self.n_frames, self.last_loops, self.flash_until = 0, 0, -1
        if kind == "slam":
            self.nodes, = self.ax.plot([], [], marker="o", color="#222222", ms=3.0, ls="", zorder=6)
            self.flash = self.ax.text(0.5, 0.985, "", transform=self.ax.transAxes, ha="center", va="top", fontsize=12, weight="bold", color="white",
                                      bbox=dict(facecolor="#d62828", alpha=0.92, edgecolor="none", pad=4), zorder=20)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(g.x_min, g.x_max)
        self.ax.set_ylim(g.y_min, g.y_max)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.fig.text(0.5, 0.965, "Odometry only (no correction)" if kind == "odom" else "SLAM: scan matching + loop closure",
                      ha="center", va="center", fontsize=15, weight="bold", color="#b00020" if kind == "odom" else "#0b6e2e")
        self.stat_text = self.fig.text(0.5, 0.905, "", ha="center", va="center", fontsize=12, weight="bold")
        self.fig.text(0.5, 0.025, "black: true walls | green: true path | red: estimated path", ha="center", va="center", fontsize=9.5)

    def frame(self, env, coverage):
        grid = env.shadow_grid if self.kind == "odom" else env.est_grid
        est = (env._shadow if self.kind == "odom" else env._odom).pose
        errs = env.shadow_err if self.kind == "odom" else env.pose_err
        self.img.set_data(RightPanel._rgb(grid))
        occ = env.grid.grid == OCCUPIED
        rgba = np.zeros(occ.shape + (4,), dtype=np.uint8)
        rgba[occ] = (0, 0, 0, 215)
        self.true_img.set_data(rgba)
        pos = env.data.body("drone_body").xpos[:2]
        self.xy_true.append((pos[0], pos[1])); self.xy_est.append((est[0], est[1]))
        self.path_true.set_data([p[0] for p in self.xy_true], [p[1] for p in self.xy_true])
        self.path_est.set_data([p[0] for p in self.xy_est], [p[1] for p in self.xy_est])
        self.true_marker.set_data([pos[0]], [pos[1]])
        self.est_marker.set_data([est[0]], [est[1]])
        self.n_frames += 1
        if self.kind == "slam":
            sl = env._slam
            pts = np.array([k.pose[:2] for k in sl.kfs]) if sl.kfs else np.zeros((0, 2))
            self.nodes.set_data(pts[:, 0], pts[:, 1])
            if sl.loops_accepted > self.last_loops:
                self.last_loops = sl.loops_accepted
                if sl.events[-1]["max_shift"] > 0.02:
                    self.flash_until = self.n_frames + 8
                    self.flash.set_text(f"Loop closure #{sl.loops_accepted}: map rebuilt")
            if self.n_frames > self.flash_until:
                self.flash.set_text("")
        self.stat_text.set_text(f"position error {(errs[-1] if errs else 0.0):.2f} m   |   walls match {_wall_match(grid, env.grid):.2f}"
                                f"   |   coverage {coverage * 100:.0f}%")
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        plt.close(self.fig)


def video_kw():
    """Réglages propres à la vidéo : vitesse de rotation du nez ; et, par défaut, PAS d'arrêt à 90 % (cible inatteignable) : l'épisode ne finit que
    quand le drone ne découvre plus aucune case pendant OPT['stag'] pas (ou collision, ou 6000 pas). Aucun effet sur les entrées du réseau."""
    kw = dict(yaw_follow_rate=OPT["yaw_rate"], yaw_follow_min_speed=OPT["yaw_min_speed"])
    if OPT["pose_source"] != "oracle":
        kw.update(pose_source=OPT["pose_source"], odom_noise=OPT["odom_noise"])
    if OPT["compare"]:
        kw.update(shadow_odom=True)
    if OPT["full"]:
        kw.update(coverage_target=1.01, stagnation_start=0, stagnation_limit=OPT["stag"], max_steps=6000)
    return kw


def build_env(level, policy, model_name, seed, yaw_follow=True):
    if policy == "clone":
        import model_manager as mm
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        version = mm.latest_version(model_name)
        paths = mm.checkpoint_paths(model_name, version)
        kw = dict(mm.read_metadata(model_name, version)["env_kwargs"])
        kw["n_rooms"] = tuple(kw.get("n_rooms", (3, 3)))
        kw.update(building="house", house_level=level, yaw_follow=yaw_follow, max_steps=3000)
        kw.update(video_kw())
        vec = DummyVecEnv([lambda: ExplorerEnv(seed=seed, **kw)])
        vn = VecNormalize.load(paths["vecnormalize"], vec)
        vn.training = False
        model = PPO.load(paths["model"], env=vn)
        return vec.envs[0], (vn, model)
    kw = dict(EXPERT_KW)
    kw.update(house_level=level, yaw_follow=yaw_follow)
    kw.update(video_kw())
    return ExplorerEnv(seed=seed, **kw), None


def act(policy, obs, bundle):
    if policy == "clone":
        vn, model = bundle
        o = vn.normalize_obs({k: v[None] for k, v in obs.items()})
        a, _ = model.predict(o, deterministic=True)
        return a[0]
    return expert_action(obs)


def outcome(env, term, trunc, idle_done=False):
    cov = env.grid.coverage_ratio()
    if OPT["full"]:
        if term:
            return "collision", cov
        return ("complète" if (idle_done or env._stag >= env.stagnation_limit) else "timeout"), cov
    if term and cov >= env.coverage_target:
        return "victoire", cov
    return ("collision" if term else "timeout"), cov


def run(level, seed, policy, model_name, out_dir, stride, fps, size, azimuth, gif, list_only):
    env, bundle = build_env(level, policy, model_name, seed)
    obs, _ = env.reset(seed=seed)
    if list_only:
        t, idle = 0, 0
        for t in range(env.max_steps):
            obs, r, term, trunc, info = env.step(act(policy, obs, bundle))
            idle = idle + 1 if obs["frontier_vector"][3] <= 0 else 0
            if term or trunc or (OPT["full"] and idle >= 25 and t > 200):
                break
        res, cov = outcome(env, term, trunc, idle >= 25 and t > 200)
        print(f"niveau {level} seed {seed} [{policy}] : {res:9s} {t + 1:5d} pas ({(t + 1) * 0.02:5.1f} s) couverture {cov * 100:.1f} %"
              + (f" | erreur finale {(env.pose_err[-1] if env.pose_err else 0.0):.2f} m, max {max(env.pose_err):.2f} m" if env._odom is not None else "")
              + (f" | MÊME vol, odométrie seule : finale {env.shadow_err[-1]:.2f} m, max {max(env.shadow_err):.2f} m" if env._shadow is not None and env.shadow_err else ""))
        return res, t + 1
    if OPT["compare"]:
        left, right = ComparePanel(env, size, "odom"), ComparePanel(env, size, "slam")
    else:
        left, right = LeftPanel(env, size, azimuth, OPT["view"]), RightPanel(env, size)
    font, small = make_font(26), make_font(20)
    name = (f"demo_compare_niveau{level}_seed{seed}_{OPT['odom_noise']}" if OPT["compare"] else
            f"demo_{policy}_niveau{level}_seed{seed}" + (f"_{OPT['pose_source']}_{OPT['odom_noise']}" if OPT["pose_source"] != "oracle" else ""))
    name = name + ("_3d" if OPT["view"] == "follow" else "") + (f"_lacet{OPT['yaw_rate']:g}" if OPT["yaw_rate"] != 1.2 else "") + ("_90pct" if not OPT["full"] else "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".mp4")
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16)
    gif_frames = []

    def grab(t, banner=None):
        if OPT["compare"]:
            cov_now = env.grid.coverage_ratio()
            frame = np.concatenate([left.frame(env, cov_now), right.frame(env, cov_now)], axis=1)
            if banner:
                img = Image.fromarray(frame)
                d = ImageDraw.Draw(img)
                d.rectangle([0, frame.shape[0] - 54, frame.shape[1], frame.shape[0]], fill=(20, 20, 20))
                d.text((12, frame.shape[0] - 46), banner, fill=(255, 255, 255), font=font)
                frame = np.asarray(img)
            return frame
        a = left.frame(env)
        b = right.frame(env, env.grid.coverage_ratio())
        img = Image.fromarray(a)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, size, 44], fill=(20, 20, 20))
        d.text((12, 8), (f"MuJoCo simulation   t = {t * 0.02:4.1f} s" if OPT["pose_source"] != "oracle" else f"Simulation MuJoCo ({'vue du dessus' if OPT['view'] == 'top' else 'vue 3D, caméra de suivi'})   t = {t * 0.02:5.1f} s"), fill=(255, 255, 255), font=small)
        if banner:
            d.rectangle([0, size - 54, size, size], fill=(20, 20, 20))
            d.text((12, size - 46), banner, fill=(255, 255, 255), font=font)
        frame = np.concatenate([np.asarray(img), b], axis=1)
        return frame

    def emit(frame, n=1):
        for _ in range(n):
            writer.append_data(frame)
        if gif:
            gif_frames.append(Image.fromarray(frame).resize((frame.shape[1] // 2, frame.shape[0] // 2), Image.LANCZOS))

    first = grab(0, f"Level {level}, seed {seed}: same flight, same measurements, same scans ({OPT['odom_noise']} noise)" if OPT["compare"] else (f"Level {level}, seed {seed}: " + {"odom": "odometry only", "slam": "scan matching", "slam_lc": "scan matching + loop closure"}[OPT["pose_source"]] + f" ({OPT['odom_noise']} noise)") if OPT["pose_source"] != "oracle" else f"niveau {level} - seed {seed} - {'expert scripté' if policy == 'expert' else 'clone appris'}")
    emit(first, fps)
    t, idle = 0, 0
    for t in range(env.max_steps):
        obs, r, term, trunc, info = env.step(act(policy, obs, bundle))
        idle = idle + 1 if obs["frontier_vector"][3] <= 0 else 0
        if t % stride == 0:
            emit(grab(t + 1))
        if term or trunc or (OPT["full"] and idle >= 25 and t > 200):
            break
    res, cov = outcome(env, term, trunc, idle >= 25 and t > 200)
    label = {"victoire": f"Terminé : {cov * 100:.0f} % de la maison cartographiée, sans collision",
             "complète": f"Terminé : {cov * 100:.0f} % de la maison cartographiée, sans collision",
             "collision": f"Collision à {cov * 100:.0f} % de couverture", "timeout": f"Temps écoulé à {cov * 100:.0f} % de couverture"}[res]
    if OPT["pose_source"] != "oracle":
        label = {"complète": f"Done - {cov * 100:.0f}% mapped", "victoire": f"Done - {cov * 100:.0f}% mapped",
                 "collision": f"Collision at {cov * 100:.0f}% coverage", "timeout": f"Timeout at {cov * 100:.0f}% coverage"}[res]
    emit(grab(t + 1, label), 2 * fps)
    writer.close()
    if gif:
        gif_frames[0].save(os.path.join(out_dir, name + ".gif"), save_all=True, append_images=gif_frames[1::2], duration=int(2000 / fps), loop=0, optimize=True)
    left.close(); right.close()
    print(f"écrit {path} : {res}, {t + 1} pas ({(t + 1) * 0.02:.1f} s simulées), couverture {cov * 100:.1f} %")
    return res, t + 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[9001])
    ap.add_argument("--seed-range", type=int, nargs=2, default=None, help="Replaces --seeds: every seed from a to b-1.")
    ap.add_argument("--policy", choices=["expert", "clone"], default="expert")
    ap.add_argument("--model", default="bc2_s42", help="Model used by --policy clone.")
    ap.add_argument("--out", default=os.path.join("..", "demo"))
    ap.add_argument("--stride", type=int, default=3, help="One frame every N simulation steps (50 steps = 1 s): 3 gives 1.7x real time at 30 fps.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, default=720, help="Side of one panel in pixels (the video is 2 x size wide).")
    ap.add_argument("--azimuth", type=float, default=90.0, help="Orientation of the top view (adjust if the map looks rotated).")
    ap.add_argument("--gif", action="store_true", help="Also write a lighter gif (half resolution) for the README.")
    ap.add_argument("--list", action="store_true", help="Render nothing: print the outcome of each seed, to choose the videos.")
    ap.add_argument("--view", choices=["top", "follow"], default="top", help="Left panel: top view or 3D follow camera.")
    ap.add_argument("--stop-at-target", action="store_true", help="Stop the episode at 90%% coverage (default: run until nothing new is discovered).")
    ap.add_argument("--stag", type=int, default=600, help="End the episode after N steps without a new cell (full mode). The normal end is no valid frontier for 25 steps.")
    ap.add_argument("--yaw-rate", type=float, default=1.2, help="Maximum nose rotation rate, rad/s (display only).")
    ap.add_argument("--yaw-min-speed", type=float, default=0.8, help="The nose only turns above this speed, m/s.")
    ap.add_argument("--pose-source", choices=["oracle", "odom", "slam", "slam_lc"], default="oracle",
                    help="oracle: exact pose; odom: dead reckoning; slam: + scan matching; slam_lc: + loop closure and map reconstruction.")
    ap.add_argument("--odom-noise", choices=["low", "medium", "high", "extreme"], default="medium", help="Noise level of the odometry (not used with oracle).")
    ap.add_argument("--compare", action="store_true", help="Side by side, same flight: odometry only (left) against scan matching + loop closure (right).")
    a = ap.parse_args()
    if a.compare:
        a.pose_source = "slam_lc"
    OPT.update(compare=a.compare, pose_source=a.pose_source, odom_noise=a.odom_noise, full=not a.stop_at_target, stag=a.stag, yaw_rate=a.yaw_rate, yaw_min_speed=a.yaw_min_speed, view=a.view)
    seeds = list(range(*a.seed_range)) if a.seed_range else a.seeds
    for s in seeds:
        run(a.level, s, a.policy, a.model, a.out, a.stride, a.fps, a.size, a.azimuth, a.gif, a.list)
