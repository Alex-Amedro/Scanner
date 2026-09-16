"""Environnement d'exploration/cartographie (plan sections 3 à 8).

Étape "avant de brancher PPO" : cet environnement valide que la génération de
bâtiment, le LiDAR (horizontal + vertical), la construction de la grille
d'occupation et la détection de frontières fonctionnent ensemble. Le reward
suit la formule du plan (section 6), mais rien n'indique encore qu'elle est
bien réglée — c'est justement ce qu'on va pouvoir observer une fois que ça
tourne.
"""

import random

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from building_generator import compute_navigable_rects, generate_building_xml, generate_layout
from occupancy_grid import OccupancyGrid

# --- Corps du drone : repris tel quel du projet précédent (drone-rl-speedrunner).
# Seule la position de spawn change (à l'intérieur de la première pièce).
_DRONE_XML = """
<body name="drone_body" pos="{spawn_x} {spawn_y} {spawn_z}">
    <joint type="free" damping="0.05"/>
    <inertial pos="0 0 0" mass="0.8" diaginertia="0.002 0.002 0.002"/>
    <geom type="box" size="0.1 0.1 0.05" rgba="0.18 0.18 0.22 1" mass="0.8"/>
    <geom type="box" pos="0 0.1 0" size="0.05 0.05 0.05" rgba="1 0.25 0.1 1"/>

    <geom type="capsule" fromto="0 0 0.01  0.17 0.17 0.03" size="0.012"
          rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="capsule" fromto="0 0 0.01 -0.17 0.17 0.03" size="0.012"
          rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="capsule" fromto="0 0 0.01  0.17 -0.17 0.03" size="0.012"
          rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="capsule" fromto="0 0 0.01 -0.17 -0.17 0.03" size="0.012"
          rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0" mass="0"/>

    <geom type="cylinder" pos="0.17 0.17 0.035" size="0.028 0.022"
          rgba="1 0.35 0.12 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="cylinder" pos="-0.17 0.17 0.035" size="0.028 0.022"
          rgba="1 0.35 0.12 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="cylinder" pos="0.17 -0.17 0.035" size="0.028 0.022"
          rgba="0.2 0.2 0.24 1" contype="0" conaffinity="0" mass="0"/>
    <geom type="cylinder" pos="-0.17 -0.17 0.035" size="0.028 0.022"
          rgba="0.2 0.2 0.24 1" contype="0" conaffinity="0" mass="0"/>

    <site name="center_of_mass" pos="0 0 0" size="0.01"/>
</body>
"""

_ACTUATORS_XML_TEMPLATE = """
<actuator>
    <motor name="thrust" site="center_of_mass" gear="0 0 20 0 0 0" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="roll" site="center_of_mass" gear="0 0 0 {gear_rp} 0 0" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="pitch" site="center_of_mass" gear="0 0 0 0 {gear_rp} 0" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="yaw" site="center_of_mass" gear="0 0 0 0 0 {gear_yaw}" ctrllimited="true" ctrlrange="-1 1"/>
</actuator>
"""


class ExplorerEnv(gym.Env):
    def __init__(self, n_rooms=(2, 4), grid_resolution=0.15, crop_size=64,
                 k_frontiers=5, n_lidar_horizontal=180, portee_lidar=15.0,
                 beta=0.5, r_exp=100.0, coverage_target=0.9,
                 gear_roll_pitch=0.5, gear_yaw=0.25, up_z_min=0.3, coeff_spin=0.0, coeff_align=0.0,
                 task="explore", show_survivor_marker=True,
                 max_steps=2000, seed=None):
        super().__init__()

        self.n_rooms = n_rooms
        self.grid_resolution = grid_resolution
        self.crop_size = crop_size
        self.k_frontiers = k_frontiers
        self.n_lidar_horizontal = n_lidar_horizontal
        self.portee_lidar = portee_lidar
        self.beta = beta
        self.r_exp = r_exp
        self.coverage_target = coverage_target
        self.gear_roll_pitch = gear_roll_pitch
        self.gear_yaw = gear_yaw
        self.up_z_min = up_z_min
        self.coeff_spin = coeff_spin
        self.coeff_align = coeff_align
        self.task = task
        self.show_survivor_marker = show_survivor_marker
        self.max_steps = max_steps

        self._rng = random.Random(seed)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "local_crop": spaces.Box(low=0.0, high=1.0, shape=(3, crop_size, crop_size), dtype=np.float32),
            "coverage": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "frontier_vector": spaces.Box(low=-1.0, high=1.0, shape=(k_frontiers * 4,), dtype=np.float32),
            "relief_rays": spaces.Box(low=0.0, high=1.0, shape=(18,), dtype=np.float32),
            "kinematics": spaces.Box(low=-50.0, high=50.0, shape=(10,), dtype=np.float32),
        })

        self.model = None
        self.data = None
        self.grid = None
        self.layout = None
        self.viewer = None
        self._viewer_model = None
        self._step_count = 0
        self._last_phi = 0.0
        self._last_known_cells = 0
        self._last_lidar_origin = None
        self._last_lidar_dirs = []
        self._last_lidar_dists = []
        self.survivor_xy = None

    # ------------------------------------------------------------------ #
    # Construction du monde
    # ------------------------------------------------------------------ #

    def _build_xml(self):
        self.layout = generate_layout(n_rooms=self.n_rooms, rng=self._rng)
        building_xml = generate_building_xml(self.layout)

        room0 = self.layout.rooms[0]
        spawn_x, spawn_y = room0.center
        self._spawn_xy = (spawn_x, spawn_y)
        self._spawn_z = 1.2
        drone_xml = _DRONE_XML.format(spawn_x=f"{spawn_x:.3f}", spawn_y=f"{spawn_y:.3f}", spawn_z="1.2")

        survivor_room = self.layout.rooms[-1]
        sx = self._rng.uniform(survivor_room.x_min + 0.5, survivor_room.x_max - 0.5)
        sy = self._rng.uniform(survivor_room.y_min + 0.5, survivor_room.y_max - 0.5)
        self.survivor_xy = (sx, sy)
        survivor_xml = ""
        if self.show_survivor_marker:
            survivor_xml = (f'<geom name="survivant" type="sphere" pos="{sx:.3f} {sy:.3f} 0.3" '
                             f'size="0.2" rgba="1.0 0.1 0.1 0.9" contype="0" conaffinity="0"/>')

        actuators_xml = _ACTUATORS_XML_TEMPLATE.format(gear_rp=self.gear_roll_pitch, gear_yaw=self.gear_yaw)

        full_xml = f"""
        <mujoco>
            <option timestep="0.002" gravity="0 0 -9.81"/>
            <worldbody>
                <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1"/>
                {building_xml}
                {survivor_xml}
                {drone_xml}
            </worldbody>
            {actuators_xml}
        </mujoco>
        """
        return full_xml

    # ------------------------------------------------------------------ #
    # API Gymnasium
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        xml = self._build_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.grid = OccupancyGrid(self.layout.world_x_range, self.layout.world_y_range,
                                   resolution=self.grid_resolution)
        self.grid.set_reachable_mask(compute_navigable_rects(self.layout))
        self._step_count = 0
        self._last_phi = 0.0
        self._last_known_cells = 0

        self._scan_and_update_grid()
        self._last_known_cells = int(np.count_nonzero((self.grid.grid != -1) & self.grid.reachable))
        self._last_phi = self.grid.potential(self._last_frontiers, beta=self.beta)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action_reelle = action.copy()
        action_reelle[0] = (action[0] * 0.6) + 0.392  # même recentrage de la poussée que le runner
        self.data.ctrl[:] = action_reelle

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self._scan_and_update_grid()
        obs = self._get_obs()
        self._step_count += 1

        pos = self.data.body("drone_body").xpos
        quat = self.data.body("drone_body").xquat
        en_collision = self.data.ncon > 0
        roll, pitch, up_z = _quat_roll_pitch_upz(quat)
        yaw = _quat_to_yaw(quat)
        retourne = up_z < self.up_z_min

        known_cells = int(np.count_nonzero((self.grid.grid != -1) & self.grid.reachable))
        n_new_cells = known_cells - self._last_known_cells
        self._last_known_cells = known_cells

        coverage = self.grid.coverage_ratio()
        phi = self.grid.potential(self._last_frontiers, beta=self.beta)

        vel_lin = self.data.qvel[0:3]
        vel_ang = self.data.qvel[3:6]
        vitesse_horiz = float(np.hypot(vel_lin[0], vel_lin[1]))
        alignement = 0.0
        if vitesse_horiz > 0.3:  # sous ce seuil, la direction de vitesse est trop bruitée pour être significative
            cap = np.array([np.cos(yaw), np.sin(yaw)])
            direction_vitesse = np.array([vel_lin[0], vel_lin[1]]) / vitesse_horiz
            alignement = float(np.dot(cap, direction_vitesse))  # 1 = aligné, -1 = à l'envers
        bonus_alignement = self.coeff_align * alignement if self.coeff_align > 0.0 else 0.0

        if self.task == "hover":
            derive_horizontale = float(np.hypot(pos[0] - self._spawn_xy[0], pos[1] - self._spawn_xy[1]))
            erreur_altitude = abs(pos[2] - self._spawn_z)
            reward = up_z - 0.5 * derive_horizontale - 0.5 * erreur_altitude + bonus_alignement
        elif coverage >= self.coverage_target:
            reward = self.r_exp
        else:
            reward = n_new_cells + (phi - self._last_phi) + bonus_alignement
            if self.coeff_spin > 0.0:
                reward -= self.coeff_spin * float(np.linalg.norm(vel_ang))
        self._last_phi = phi

        terminated = False
        if en_collision or retourne:
            terminated = True
            reward -= 50.0
        elif self.task != "hover" and coverage >= self.coverage_target:
            terminated = True

        truncated = self._step_count >= self.max_steps

        dist_survivant = float(np.hypot(pos[0] - self.survivor_xy[0], pos[1] - self.survivor_xy[1]))
        info = {
            "coverage": coverage,
            "n_new_cells": n_new_cells,
            "distance_survivant": dist_survivant,
            "survivant_repere": dist_survivant < 1.0,
            "collision": en_collision,
            "retourne": retourne,
            "truncated": bool(truncated),
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "roll_deg": float(np.degrees(roll)),
            "pitch_deg": float(np.degrees(pitch)),
            "yaw_deg": float(np.degrees(yaw)),
            "up_z": up_z,
            "vel_ang_norm": float(np.linalg.norm(vel_ang)),
            "alignement": alignement,
        }
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Perception
    # ------------------------------------------------------------------ #

    def _scan_and_update_grid(self):
        drone_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone_body")
        pos = self.data.body("drone_body").xpos.copy()
        quat = self.data.body("drone_body").xquat
        yaw = _quat_to_yaw(quat)

        geom_id = np.zeros(1, dtype=np.int32)
        origin_xy = (pos[0], pos[1])

        self._last_lidar_origin = pos.copy()
        self._last_lidar_dirs = []
        self._last_lidar_dists = []

        angles = np.linspace(-np.pi, np.pi, self.n_lidar_horizontal, endpoint=False)
        for angle in angles:
            a = angle + yaw
            vec = np.array([np.cos(a), np.sin(a), 0.0])
            dist = mujoco.mj_ray(self.model, self.data, pos, vec, None, 1, drone_id, geom_id)
            hit = dist >= 0
            dist = min(dist, self.portee_lidar) if hit else self.portee_lidar
            hit_xy = (pos[0] + vec[0] * dist, pos[1] + vec[1] * dist)
            self.grid.update_ray(origin_xy, hit_xy, hit)
            self._last_lidar_dirs.append(vec)
            self._last_lidar_dists.append(dist)

        # éventail de rayons inclinés (relief) — remplace les 2 rayons verticaux
        # droits, qui ne pouvaient rien voir en dehors de l'aplomb exact du
        # drone. 8 directions autour du drone, alignées sur son cap, chacune
        # avec un rayon incliné vers le bas et un vers le haut (~30°) : ça
        # permet de sentir une marche, un trou, ou un rebord qui approche,
        # pas seulement ce qui est pile au-dessus/en-dessous.
        self._relief_dists = []
        self._relief_dirs = []
        n_az = 8
        pitch = np.radians(30.0)
        for i in range(n_az):
            az = yaw + i * (2 * np.pi / n_az)
            horiz = np.array([np.cos(az), np.sin(az), 0.0])
            for sign in (-1.0, 1.0):  # bas puis haut
                vec = horiz * np.cos(pitch)
                vec[2] = np.sin(pitch) * sign
                vec = vec / np.linalg.norm(vec)
                dist = mujoco.mj_ray(self.model, self.data, pos, vec, None, 1, drone_id, geom_id)
                dist = min(dist, self.portee_lidar) if dist >= 0 else self.portee_lidar
                self._relief_dists.append(dist / self.portee_lidar)
                self._relief_dirs.append(vec)

        # 2 rayons verticaux purs, en complément de l'éventail incliné :
        # les 16 rayons ci-dessus sont tous à ±30°, donc un trou/rebord pile
        # à l'aplomb du drone (au-dessus ou en-dessous) leur échapperait.
        for vec in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])):
            dist = mujoco.mj_ray(self.model, self.data, pos, vec, None, 1, drone_id, geom_id)
            dist = min(dist, self.portee_lidar) if dist >= 0 else self.portee_lidar
            self._relief_dists.append(dist / self.portee_lidar)
            self._relief_dirs.append(vec)

        self._last_frontiers = self.grid.frontier_features(
            pos[0], pos[1], yaw, k=self.k_frontiers)

    def _get_obs(self):
        pos = self.data.body("drone_body").xpos
        quat = self.data.body("drone_body").xquat
        vel_lin = self.data.qvel[0:3].copy()
        vel_ang = self.data.qvel[3:6].copy()
        roll, pitch, up_z = _quat_roll_pitch_upz(quat)

        local_crop = self.grid.local_crop_onehot(pos[0], pos[1], self.crop_size)
        coverage = np.array([self.grid.coverage_ratio()], dtype=np.float32)

        frontier_vec = np.array([
            v for f in self._last_frontiers
            for v in (f["distance_norm"], f["angle_norm"], f["info_gain_norm"], f["valid"])
        ], dtype=np.float32)

        kinematics = np.concatenate([
            vel_lin, vel_ang, [pos[2] / 3.0, roll, pitch, up_z],
        ]).astype(np.float32)

        return {
            "local_crop": local_crop.astype(np.float32),
            "coverage": coverage,
            "frontier_vector": frontier_vec,
            "relief_rays": np.array(self._relief_dists, dtype=np.float32),
            "kinematics": kinematics,
        }

    # ------------------------------------------------------------------ #
    # Rendu (debug)
    # ------------------------------------------------------------------ #

    def render(self, show_lidar=True, show_frontiers=True):
        if self.viewer is None or self._viewer_model is not self.model:
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer_model = self.model
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = 1
            self.viewer.cam.distance = 8.0
            self.viewer.cam.elevation = -35

        scn = self.viewer.user_scn
        scn.ngeom = 0
        if show_lidar and self._last_lidar_origin is not None:
            self._draw_lidar(scn)
            self._draw_relief(scn)
        if show_frontiers:
            self._draw_frontiers(scn)
        self.viewer.sync()

    def _draw_relief(self, scn):
        origin = self._last_lidar_origin
        for direction, dist_norm in zip(self._relief_dirs, self._relief_dists):
            dist = dist_norm * self.portee_lidar
            rgba = np.array([0.75, 0.35, 0.95, 0.55], dtype=np.float32)
            self._add_segment(scn, origin, origin + direction * dist, 0.008, rgba)

    def _draw_lidar(self, scn):
        origin = self._last_lidar_origin
        dists = np.array(self._last_lidar_dists)
        dirs = self._last_lidar_dirs
        d_min, d_max = float(dists.min()), float(dists.max())
        span = max(d_max - d_min, 1e-6)
        for direction, dist in zip(dirs, dists):
            t = (dist - d_min) / span
            rgba = np.array([1.0 - t, t, 0.15, 0.35], dtype=np.float32)
            self._add_segment(scn, origin, origin + direction * dist, 0.01, rgba)

    def _draw_frontiers(self, scn):
        for f in self._last_frontiers:
            if f["valid"] <= 0 or f["world_xy"] is None:
                continue
            wx, wy = f["world_xy"]
            pos = np.array([wx, wy, 1.2], dtype=np.float32)
            rgba = np.array([0.2, 0.7, 1.0, 0.85], dtype=np.float32)
            self._add_marker(scn, pos, 0.15, rgba)

    @staticmethod
    def _add_segment(scn, start, end, width, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_LINE, size=np.zeros(3),
                             pos=np.zeros(3), mat=np.eye(3).flatten(), rgba=rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)
        scn.ngeom += 1

    @staticmethod
    def _add_marker(scn, pos, radius, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                             size=np.array([radius, 0, 0], dtype=np.float32), pos=pos,
                             mat=np.eye(3).flatten(), rgba=rgba)
        scn.ngeom += 1


def _quat_to_yaw(quat):
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _quat_roll_pitch_upz(quat):
    """Roll, pitch (rad) et 'uprightness' (1 = bien droit, 0 = sur la tranche,
    -1 = complètement à l'envers) à partir du quaternion du corps."""
    w, x, y, z = quat
    roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    up_z = float(1 - 2 * (x * x + y * y))
    return roll, pitch, up_z