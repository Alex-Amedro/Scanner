"""Back end of the 2D LiDAR SLAM: keyframes, loop closure, pose graph optimisation, map reconstruction.

The front end (scan-to-map matching, pose_estimation.py) gives a pose at every scan. Here:
  - a keyframe (pose + scan) is kept every KF_DIST metres or KF_ANGLE of motion;
  - consecutive keyframes are linked by a relative-pose constraint taken from the front end ("odometry edge");
  - when the drone comes back close to an old keyframe, the new scan is matched against a local map built from the old keyframes; if the match
    is good, a "loop edge" is added;
  - the graph (nodes = keyframe poses, edges = relative-pose constraints) is optimised by Gauss-Newton on (x, y, yaw) in plain numpy;
  - the occupancy grid is rebuilt from the corrected keyframes.
The true pose is only used for statistics (false loops, error before / after), never for the estimate.
"""
import time

import numpy as np

from occupancy_grid import OccupancyGrid, UNKNOWN
from pose_estimation import ScanMatcher


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def compose(a, b):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def relative(a, b):
    """Pose of b expressed in the frame of a."""
    c, s = np.cos(a[2]), np.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])


def insert_scan(grid, pose, b, d, h):
    ea = b + pose[2]
    ex, ey = pose[0] + np.cos(ea) * d, pose[1] + np.sin(ea) * d
    for i in range(len(d)):
        grid.update_ray((pose[0], pose[1]), (ex[i], ey[i]), h[i])


class Keyframe:
    __slots__ = ("pose", "front", "b", "d", "h", "true")

    def __init__(self, pose, front, b, d, h, true):
        self.pose, self.front, self.b, self.d, self.h, self.true = pose, front, b, d, h, true


class PoseGraphSlam:
    def __init__(self, kf_dist=0.5, kf_angle=np.radians(10.0), loop_radius=3.0, min_gap=20, submap_half=2, max_candidates=2,
                 min_inliers=0.65, min_points=60, loop_cooldown=8, odom_sigma=(0.05, np.radians(1.0)), loop_sigma=(0.10, np.radians(2.0)), max_loop_shift=0.8):
        self.kf_dist, self.kf_angle = kf_dist, kf_angle
        self.loop_radius, self.min_gap, self.submap_half, self.max_candidates = loop_radius, min_gap, submap_half, max_candidates
        self.min_inliers, self.min_points, self.max_loop_shift = min_inliers, min_points, max_loop_shift
        self.loop_cooldown = loop_cooldown     # keyframes to wait after an accepted loop before looking for the next one
        self._last_loop_kf = -10 ** 9
        self.w_odom = (1.0 / odom_sigma[0] ** 2, 1.0 / odom_sigma[1] ** 2)
        self.w_loop = (1.0 / loop_sigma[0] ** 2, 1.0 / loop_sigma[1] ** 2)
        self.matcher = ScanMatcher(coarse=(0.6, 0.1, np.radians(10.0), np.radians(2.0)), fine=(0.1, 0.025, np.radians(2.0), np.radians(0.5)),
                                   prior_w=0.005, min_gain=-1e9)
        self.kfs, self.edges = [], []          # edge: (i, j, measurement(3,), kind)
        self.loops_accepted = 0
        self.loops_false = 0
        self.loops_rejected = 0
        self.events = []                       # one dict per accepted loop: error before / after, shift, wall-clock time
        self.kf_ms = []

    # -------------------------------------------------------------------------------- graph optimisation
    def _optimise(self, iters=8):
        n = len(self.kfs)
        P = np.array([k.pose for k in self.kfs], dtype=np.float64)
        for _ in range(iters):
            H = np.zeros((3 * n, 3 * n))
            g = np.zeros(3 * n)
            for i, j, m, kind in self.edges:
                wt, wr = self.w_odom if kind == "odom" else self.w_loop
                c, s = np.cos(P[i, 2]), np.sin(P[i, 2])
                Rt = np.array([[c, s], [-s, c]])
                v = P[j, :2] - P[i, :2]
                e = np.array([*(Rt @ v - m[:2]), wrap(P[j, 2] - P[i, 2] - m[2])])
                J = np.zeros((3, 6))
                J[:2, 0:2] = -Rt
                J[:2, 2] = np.array([[-s, c], [-c, -s]]) @ v
                J[:2, 3:5] = Rt
                J[2, 2], J[2, 5] = -1.0, 1.0
                W = np.diag([wt, wt, wr])
                idx = np.r_[3 * i:3 * i + 3, 3 * j:3 * j + 3]
                H[np.ix_(idx, idx)] += J.T @ W @ J
                g[idx] += J.T @ W @ e
            H[0:3, 0:3] += np.eye(3) * 1e9      # the first keyframe is the anchor
            dx = np.linalg.solve(H, -g)
            P += dx.reshape(n, 3)
            P[:, 2] = wrap(P[:, 2])
            if np.abs(dx).max() < 1e-5:
                break
        return P

    def _edge_residual(self, P, i, j, m):
        c, s = np.cos(P[i, 2]), np.sin(P[i, 2])
        Rt = np.array([[c, s], [-s, c]])
        return np.array([*(Rt @ (P[j, :2] - P[i, :2]) - m[:2]), wrap(P[j, 2] - P[i, 2] - m[2])])

    # -------------------------------------------------------------------------------- loop detection
    def _verify(self, k_new, j):
        """Match the newest scan against a local map made of the old keyframes around j. Returns (pose in the graph frame, inlier fraction)."""
        newest = self.kfs[k_new]
        keep = newest.h & (newest.d < 12.0)
        pts = np.stack([newest.d[keep] * np.cos(newest.b[keep]), newest.d[keep] * np.sin(newest.b[keep])], axis=1)
        if len(pts) < self.min_points:
            return None, 0.0
        lo, hi = max(0, j - self.submap_half), min(len(self.kfs) - 1, j + self.submap_half)
        sub = OccupancyGrid(self._extent[0], self._extent[1], resolution=self._res)
        for i in range(lo, hi + 1):
            kf = self.kfs[i]
            insert_scan(sub, kf.pose, kf.b, kf.d, kf.h)
        if len(pts) > self.matcher.max_points:
            pts = pts[np.linspace(0, len(pts) - 1, self.matcher.max_points).astype(int)]
        field = self.matcher._field(sub)
        x0, y0, yaw0 = newest.pose
        (_, dx, dy, dyaw, _), _ = self.matcher._search(pts, x0, y0, yaw0, self.matcher.coarse, field, sub)
        (_, dx2, dy2, dyaw2, _), _ = self.matcher._search(pts, x0 + dx, y0 + dy, yaw0 + dyaw, self.matcher.fine, field, sub)
        pose = np.array([x0 + dx + dx2, y0 + dy + dy2, wrap(yaw0 + dyaw + dyaw2)])
        c, s = np.cos(pose[2]), np.sin(pose[2])
        wx = pose[0] + c * pts[:, 0] - s * pts[:, 1]
        wy = pose[1] + s * pts[:, 0] + c * pts[:, 1]
        ix = np.clip(((wx - sub.x_min) / sub.resolution).astype(int), 0, sub.width - 1)
        iy = np.clip(((wy - sub.y_min) / sub.resolution).astype(int), 0, sub.height - 1)
        inl = float(np.mean(field[iy, ix] >= 0.79))
        return pose, inl

    # -------------------------------------------------------------------------------- main entry
    def process(self, pose, b, d, h, true_pose, grid):
        """Called after each scan with the front-end pose. Returns the corrected current pose when the graph was corrected, else None."""
        t0 = time.perf_counter()
        self._extent = ((grid.x_min, grid.x_max), (grid.y_min, grid.y_max))
        self._res = grid.resolution
        pose = np.array(pose, dtype=np.float64)
        if not self.kfs:
            self.kfs.append(Keyframe(pose.copy(), pose.copy(), b.copy(), d.copy(), h.copy(), np.array(true_pose, dtype=np.float64)))
            return None
        last = self.kfs[-1]
        rel = relative(last.front, pose)
        if np.hypot(rel[0], rel[1]) < self.kf_dist and abs(rel[2]) < self.kf_angle:
            return None
        new_pose = compose(last.pose, rel)
        self.kfs.append(Keyframe(new_pose, pose.copy(), b.copy(), d.copy(), h.copy(), np.array(true_pose, dtype=np.float64)))
        k = len(self.kfs) - 1
        self.edges.append((k - 1, k, rel, "odom"))

        # loop candidates: old keyframes that are close in the current graph
        cands = []
        for j in range(0, k - self.min_gap):
            dist = float(np.hypot(*(self.kfs[j].pose[:2] - new_pose[:2])))
            if dist <= self.loop_radius:
                cands.append((dist, j))
        cands.sort()
        if k - self._last_loop_kf < self.loop_cooldown:
            cands = []
        accepted = None
        for dist, j in cands[:self.max_candidates]:
            matched, inl = self._verify(k, j)
            if matched is None or inl < self.min_inliers:
                continue
            if np.hypot(*(matched[:2] - new_pose[:2])) > self.max_loop_shift:
                self.loops_rejected += 1
                continue
            accepted = (j, relative(self.kfs[j].pose, matched), matched, inl)
            break

        if accepted is None:
            self.kf_ms.append((time.perf_counter() - t0) * 1000.0)
            return None
        j, meas, matched, inl = accepted
        before_err = float(np.hypot(*(pose[:2] - self.kfs[k].true[:2])))
        before = np.array([kf.pose for kf in self.kfs])
        self.edges.append((j, k, meas, "loop"))
        P = self._optimise()
        # a loop whose constraint stays badly violated after optimisation is dropped and the graph is optimised again without it
        r = self._edge_residual(P, j, k, meas)
        if np.hypot(r[0], r[1]) > 0.5 or abs(r[2]) > np.radians(8.0):
            self.edges.pop()
            self.loops_rejected += 1
            self.kf_ms.append((time.perf_counter() - t0) * 1000.0)
            return None
        true_rel = relative(self.kfs[j].true, self.kfs[k].true)
        false_loop = bool(np.hypot(*(meas[:2] - true_rel[:2])) > 0.5 or abs(wrap(meas[2] - true_rel[2])) > np.radians(10.0))
        for kf, p in zip(self.kfs, P):
            kf.pose = p
        shift = float(np.abs(P[:, :2] - before[:, :2]).max())
        self.loops_accepted += 1
        self._last_loop_kf = k
        self.loops_false += int(false_loop)
        if shift > 0.02:
            grid.grid[:] = UNKNOWN              # reconstruction: the map is rebuilt from the corrected keyframes
            for kf in self.kfs:
                insert_scan(grid, kf.pose, kf.b, kf.d, kf.h)
        corrected = self.kfs[k].pose.copy()
        self.kfs[k].front = corrected.copy()
        after_err = float(np.hypot(*(corrected[:2] - self.kfs[k].true[:2])))
        ms = (time.perf_counter() - t0) * 1000.0
        self.events.append(dict(kf=k, with_kf=j, inliers=inl, false=false_loop, err_before=before_err, err_after=after_err, max_shift=shift, ms=ms))
        self.kf_ms.append(ms)
        return corrected
