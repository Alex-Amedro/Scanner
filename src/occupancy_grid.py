"""Grille d'occupation 2D, mise à jour par tracé de rayons (Bresenham), et
détection de frontières par clustering (flood-fill maison, sans scipy —
volontairement pour ne pas ajouter de dépendance à l'environnement figé).
"""

import numpy as np

FREE = 0
OCCUPIED = 1
UNKNOWN = -1


class OccupancyGrid:
    def __init__(self, x_range, y_range, resolution=0.15):
        self.resolution = resolution
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.width = int(np.ceil((self.x_max - self.x_min) / resolution))
        self.height = int(np.ceil((self.y_max - self.y_min) / resolution))
        self.grid = np.full((self.height, self.width), UNKNOWN, dtype=np.int8)

    def world_to_cell(self, x, y):
        cx = int((x - self.x_min) / self.resolution)
        cy = int((y - self.y_min) / self.resolution)
        return cx, cy

    def cell_to_world(self, cx, cy):
        x = self.x_min + (cx + 0.5) * self.resolution
        y = self.y_min + (cy + 0.5) * self.resolution
        return x, y

    def in_bounds(self, cx, cy):
        return 0 <= cx < self.width and 0 <= cy < self.height

    def update_ray(self, origin_xy, hit_xy, hit):
        """Marque FREE les cellules traversées, OCCUPIED la cellule d'impact
        si le rayon a touché quelque chose (hit=True)."""
        x0, y0 = self.world_to_cell(*origin_xy)
        x1, y1 = self.world_to_cell(*hit_xy)
        for cx, cy in _bresenham(x0, y0, x1, y1):
            if not self.in_bounds(cx, cy):
                break
            if (cx, cy) == (x1, y1) and hit:
                self.grid[cy, cx] = OCCUPIED
            elif self.grid[cy, cx] != OCCUPIED:
                self.grid[cy, cx] = FREE

    def coverage_ratio(self):
        return float(np.count_nonzero(self.grid != UNKNOWN)) / self.grid.size

    def local_crop(self, x, y, crop_size=64):
        """Crop local centré sur (x, y), axis-aligned (PAS encore aligné sur
        le cap du drone — rotation laissée en TODO V2, voir note dans le
        plan section 5 : ça demande une interpolation d'image, pas nécessaire
        pour valider la mécanique de base)."""
        cx, cy = self.world_to_cell(x, y)
        half = crop_size // 2
        crop = np.full((crop_size, crop_size), UNKNOWN, dtype=np.int8)

        gy0, gy1 = cy - half, cy - half + crop_size
        gx0, gx1 = cx - half, cx - half + crop_size

        src_y0, src_y1 = max(0, gy0), min(self.height, gy1)
        src_x0, src_x1 = max(0, gx0), min(self.width, gx1)
        if src_y0 >= src_y1 or src_x0 >= src_x1:
            return crop

        dst_y0, dst_x0 = src_y0 - gy0, src_x0 - gx0
        dst_y1, dst_x1 = dst_y0 + (src_y1 - src_y0), dst_x0 + (src_x1 - src_x0)
        crop[dst_y0:dst_y1, dst_x0:dst_x1] = self.grid[src_y0:src_y1, src_x0:src_x1]
        return crop

    def local_crop_onehot(self, x, y, crop_size=64):
        """Version 3 canaux (libre / occupé / inconnu) du crop, prête à être
        passée telle quelle au CNN."""
        crop = self.local_crop(x, y, crop_size)
        onehot = np.zeros((3, crop_size, crop_size), dtype=np.float32)
        onehot[0] = (crop == FREE)
        onehot[1] = (crop == OCCUPIED)
        onehot[2] = (crop == UNKNOWN)
        return onehot

    def frontier_cells(self):
        """Cellules libres ayant au moins un voisin inconnu (4-connexe)."""
        free_mask = self.grid == FREE
        unknown_mask = self.grid == UNKNOWN
        neighbor_unknown = np.zeros_like(free_mask)
        neighbor_unknown[1:, :] |= unknown_mask[:-1, :]
        neighbor_unknown[:-1, :] |= unknown_mask[1:, :]
        neighbor_unknown[:, 1:] |= unknown_mask[:, :-1]
        neighbor_unknown[:, :-1] |= unknown_mask[:, 1:]
        return free_mask & neighbor_unknown

    def frontier_clusters(self, min_cluster_size=2):
        """Regroupe les cellules-frontières en composantes connexes (flood-fill
        itératif, pas de récursion pour éviter tout risque de RecursionError
        sur une grande grille)."""
        mask = self.frontier_cells()
        visited = np.zeros_like(mask)
        clusters = []
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        for y0, x0 in zip(ys, xs):
            if visited[y0, x0]:
                continue
            stack = [(x0, y0)]
            visited[y0, x0] = True
            cells = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if len(cells) >= min_cluster_size:
                clusters.append(cells)
        return clusters

    def frontier_features(self, drone_x, drone_y, drone_yaw, k=5, info_radius_cells=4):
        """K frontières les plus proches. Chaque frontière : distance
        normalisée [0,1], angle relatif normalisé [-1,1] (rad/pi), gain
        d'info normalisé [0,1], et un flag de validité (1.0 = réelle
        frontière, 0.0 = padding s'il y en a moins de K)."""
        clusters = self.frontier_clusters()
        diag = float(np.hypot(self.width, self.height) * self.resolution)
        max_info = (2 * info_radius_cells + 1) ** 2

        candidates = []
        for cells in clusters:
            cxs = [c[0] for c in cells]
            cys = [c[1] for c in cells]
            ccx, ccy = sum(cxs) / len(cxs), sum(cys) / len(cys)
            wx, wy = self.cell_to_world(ccx, ccy)
            dist = float(np.hypot(wx - drone_x, wy - drone_y))
            angle = np.arctan2(wy - drone_y, wx - drone_x) - drone_yaw
            angle = float((angle + np.pi) % (2 * np.pi) - np.pi)

            cxi, cyi = int(round(ccx)), int(round(ccy))
            y0, y1 = max(0, cyi - info_radius_cells), min(self.height, cyi + info_radius_cells + 1)
            x0, x1 = max(0, cxi - info_radius_cells), min(self.width, cxi + info_radius_cells + 1)
            info_gain = int(np.count_nonzero(self.grid[y0:y1, x0:x1] == UNKNOWN))

            candidates.append({
                "distance_norm": min(dist / diag, 1.0),
                "angle_norm": angle / np.pi,
                "info_gain_norm": min(info_gain / max_info, 1.0),
                "world_xy": (wx, wy),
                "size": len(cells),
                "valid": 1.0,
            })

        candidates.sort(key=lambda c: c["distance_norm"])
        top_k = candidates[:k]
        while len(top_k) < k:
            top_k.append({"distance_norm": 1.0, "angle_norm": 0.0,
                           "info_gain_norm": 0.0, "world_xy": None,
                           "size": 0, "valid": 0.0})
        return top_k

    def potential(self, frontiers, beta=0.5):
        """Champ de potentiel de frontière (plan section 6) :
        Φ_i = -d̄_i + β·Γ̄_i, renormalisé, Φ(s) = max sur les frontières valides."""
        valid = [f for f in frontiers if f["valid"] > 0]
        if not valid:
            return 0.0
        phis = [(-f["distance_norm"] + beta * f["info_gain_norm"] + 1) / (beta + 1)
                for f in valid]
        return max(phis)


def _bresenham(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield (x, y)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy