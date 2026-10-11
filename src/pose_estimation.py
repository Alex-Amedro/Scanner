"""Estimated drone pose, to replace the exact pose given by the simulator.

Step 1, DeadReckoning: integrate noisy velocity and yaw-rate measurements, with no correction at all.
Step 2, ScanMatchedPose: the same prediction, then each LiDAR scan is matched to the map built so far (scan-to-map) to correct the pose.
The noise parameters below are chosen by hand, they are NOT calibrated on a real sensor.
"""
import time

import numpy as np

from occupancy_grid import FREE, OCCUPIED

# scale: relative speed error (one random value per flight); v_bias: constant velocity bias per axis, body frame (m/s);
# v_noise: white noise on each velocity measurement (m/s); w_bias: constant gyro bias (rad/s); w_noise: white noise on yaw rate (rad/s)
NOISE_LEVELS = {
    "low": dict(scale=0.005, v_bias=0.01, v_noise=0.02, w_bias=np.radians(0.05), w_noise=np.radians(0.3)),
    "medium": dict(scale=0.015, v_bias=0.03, v_noise=0.05, w_bias=np.radians(0.2), w_noise=np.radians(0.6)),
    "high": dict(scale=0.03, v_bias=0.06, v_noise=0.10, w_bias=np.radians(0.5), w_noise=np.radians(1.0)),
    "extreme": dict(scale=0.06, v_bias=0.15, v_noise=0.20, w_bias=np.radians(1.5), w_noise=np.radians(2.0)),
}


class DeadReckoning:
    """Pose (x, y, yaw) obtained by integrating noisy body-frame velocity and yaw rate."""

    def __init__(self, level, rng, x, y, yaw):
        p = NOISE_LEVELS[level]
        self.p, self.rng = p, rng
        self.scale = rng.normal(0.0, p["scale"])
        self.v_bias = rng.normal(0.0, p["v_bias"], 2)
        self.w_bias = rng.normal(0.0, p["w_bias"])
        self.pose = np.array([x, y, yaw], dtype=np.float64)

    def measure(self, v_body, yaw_rate):
        """Noisy measurement of the body-frame velocity (2,) and of the yaw rate, from the true values (which never leave this method)."""
        p = self.p
        v = (1.0 + self.scale) * np.asarray(v_body) + self.v_bias + self.rng.normal(0.0, p["v_noise"], 2)
        w = yaw_rate + self.w_bias + self.rng.normal(0.0, p["w_noise"])
        return v, w

    def integrate(self, v, w, dt):
        """Dead reckoning step: move the pose by the measured velocity and rotation."""
        self.pose[2] += w * dt
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        self.pose[0] += (c * v[0] - s * v[1]) * dt
        self.pose[1] += (s * v[0] + c * v[1]) * dt
        return self.pose

    def step(self, v_body, yaw_rate, dt):
        v, w = self.measure(v_body, yaw_rate)
        return self.integrate(v, w, dt)


def _dilate3(m):
    a = m.copy()
    a[1:, :] |= m[:-1, :]; a[:-1, :] |= m[1:, :]
    b = a.copy()
    b[:, 1:] |= a[:, :-1]; b[:, :-1] |= a[:, 1:]
    return b


class ScanMatcher:
    """Correlative scan-to-map matching on a smooth likelihood field.

    The map gives a field that is 1 on occupied cells and decreases to 0 over `ramp` cells around them. A candidate correction (x, y, yaw) is scored
    by the sum of the field at the scan end points (bilinear interpolation, so the score varies smoothly inside a cell), minus a prior that prefers
    small corrections. A coarse search is followed by a fine one around the best coarse candidate."""

    def __init__(self, coarse=(0.20, 0.05, np.radians(4.0), np.radians(1.0)), fine=(0.05, 0.0125, np.radians(1.0), np.radians(0.25)),
                 max_points=120, min_points=25, prior_w=0.02, min_gain=0.01, ramp=4):
        self.coarse, self.fine = coarse, fine
        self.max_points, self.min_points = max_points, min_points
        self.prior_w, self.min_gain, self.ramp = prior_w, min_gain, ramp

    def _field(self, grid):
        occ = grid.grid == OCCUPIED
        field = occ.astype(np.float32)
        layer = occ
        for k in range(1, self.ramp + 1):
            layer = _dilate3(layer)
            field = np.maximum(field, np.where(layer, 1.0 - k / (self.ramp + 1.0), 0.0).astype(np.float32))
        return field

    def _search(self, pts, x0, y0, yaw0, win, field, grid):
        half_xy, step_xy, half_yaw, step_yaw = win
        offs = np.arange(-half_xy, half_xy + 1e-9, step_xy)
        dyaws = np.arange(-half_yaw, half_yaw + 1e-9, step_yaw)
        DX, DY = np.meshgrid(offs, offs)
        DX, DY = DX.ravel(), DY.ravel()
        n = len(pts)
        h, w = field.shape
        px, py = pts[:, 0], pts[:, 1]
        best = (-1e18, 0.0, 0.0, 0.0, 0.0)
        zero_score = None
        for dyaw in dyaws:
            c, s = np.cos(yaw0 + dyaw), np.sin(yaw0 + dyaw)
            bx = x0 + c * px - s * py
            by = y0 + s * px + c * py
            fx = (bx[None, :] + DX[:, None] - grid.x_min) / grid.resolution - 0.5
            fy = (by[None, :] + DY[:, None] - grid.y_min) / grid.resolution - 0.5
            ix = np.floor(fx).astype(np.int32)
            iy = np.floor(fy).astype(np.int32)
            ax, ay = fx - ix, fy - iy
            ix0 = np.clip(ix, 0, w - 1); ix1 = np.clip(ix + 1, 0, w - 1)
            iy0 = np.clip(iy, 0, h - 1); iy1 = np.clip(iy + 1, 0, h - 1)
            v = (field[iy0, ix0] * (1 - ax) * (1 - ay) + field[iy0, ix1] * ax * (1 - ay)
                 + field[iy1, ix0] * (1 - ax) * ay + field[iy1, ix1] * ax * ay)
            raw = v.sum(1)
            pen = self.prior_w * n * ((DX ** 2 + DY ** 2) / 0.1 ** 2 + (dyaw / np.radians(2.0)) ** 2)
            tot = raw - pen
            k = int(np.argmax(tot))
            if tot[k] > best[0]:
                best = (float(tot[k]), float(DX[k]), float(DY[k]), float(dyaw), float(raw[k]))
            if abs(dyaw) < 1e-12:
                zero_score = float(raw[int(np.argmin(DX ** 2 + DY ** 2))])
        return best, zero_score

    def match(self, pose, pts, grid):
        """pts: (N, 2) scan end points (hits only) in the body frame. Returns (new_pose, info); new_pose is None when no correction is applied."""
        if len(pts) < self.min_points:
            return None, dict(n=len(pts), gain=0.0, applied=False)
        if len(pts) > self.max_points:
            pts = pts[np.linspace(0, len(pts) - 1, self.max_points).astype(int)]
        if (grid.grid == OCCUPIED).sum() < self.min_points:
            return None, dict(n=len(pts), gain=0.0, applied=False)
        field = self._field(grid)
        x0, y0, yaw0 = pose
        (s1, dx, dy, dyaw, raw1), zero = self._search(pts, x0, y0, yaw0, self.coarse, field, grid)
        (s2, dx2, dy2, dyaw2, raw2), zero2 = self._search(pts, x0 + dx, y0 + dy, yaw0 + dyaw, self.fine, field, grid)
        dx, dy, dyaw = dx + dx2, dy + dy2, dyaw + dyaw2
        gain = (raw2 - (zero if zero is not None else raw2)) / len(pts)
        info = dict(n=len(pts), gain=float(gain), applied=False, dx=dx, dy=dy, dyaw=dyaw)
        if gain < self.min_gain:
            return None, info
        info["applied"] = True
        return np.array([x0 + dx, y0 + dy, yaw0 + dyaw]), info


class ScanMatchedPose(DeadReckoning):
    """Prediction by dead reckoning (same noisy measurements), then correction by scan-to-map matching at every scan."""

    def __init__(self, level, rng, x, y, yaw, matcher=None):
        super().__init__(level, rng, x, y, yaw)
        self.matcher = matcher or ScanMatcher()
        self.n_corrected = 0
        self.match_ms = []

    def correct(self, pts, grid):
        t0 = time.perf_counter()
        new, info = self.matcher.match(self.pose, pts, grid)
        if new is not None:
            self.pose[:] = new
            self.n_corrected += 1
        self.match_ms.append((time.perf_counter() - t0) * 1000.0)
        return info
