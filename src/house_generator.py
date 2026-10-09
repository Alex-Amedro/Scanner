"""House generator (levels 1-4): rooms of varied sizes, doors at varied positions, extra doors that create loops, an optional central hall,
and a single entrance: a window in the south wall, facing a closed courtyard where the drone starts.

Construction:
  1. a W x H rectangle, south wall at y = 0
  2. optional hall strip, then recursive splitting into rooms
  3. random spanning tree of doors (every room reachable) plus extra doors (loops)
  4. window with a sill and a lintel (the 1.2 m LiDAR sees it as an opening)

The wall boxes are the single source for the MuJoCo XML, the accessibility check and the plots. One floor only.
"""
import math
from dataclasses import dataclass

import numpy as np

from building_generator import Room

EPS = 1e-6
WALL_T = 0.15
WALL_H = 2.6
SILL_TOP, LINTEL_BOTTOM = 0.7, 1.7          # la fenêtre est ouverte entre ces deux hauteurs (le LiDAR est à 1,2 m)
DOOR_MARGIN = 0.3                            # dégagement mini entre une porte et un coin de pièce
MIN_ROOM = 3.0

LEVELS = {
    1: dict(W=(9.0, 11.0), H=(7.0, 9.0), n=(3, 4), hall=0.0, loop=0.0, door=(1.3, 1.5), window=(1.3, 1.5), p_end=0.2),
    2: dict(W=(12.0, 15.0), H=(9.0, 12.0), n=(5, 7), hall=1.0, loop=0.25, door=(1.2, 1.5), window=(1.2, 1.4), p_end=0.3),
    3: dict(W=(14.0, 18.0), H=(11.0, 14.0), n=(6, 8), hall=0.5, loop=0.5, door=(1.0, 1.4), window=(1.1, 1.3), p_end=0.5),
    4: dict(W=(17.0, 21.0), H=(12.0, 16.0), n=(8, 10), hall=0.5, loop=0.6, door=(1.0, 1.3), window=(1.0, 1.2), p_end=0.5),
}


@dataclass
class HouseLayout:
    rooms: list
    wall_height: float
    world_x_range: tuple
    world_y_range: tuple
    courtyard: tuple          # (x_min, x_max, y_min, y_max) de la cour fermée où démarre le drone
    window: tuple             # (x_centre, largeur) dans le mur sud y = 0
    doors: list               # (i, j, axe, coord, centre, largeur)
    boxes: list               # (nom, cx, cy, hx, hy, z0, z1, rgba)
    navigable_rects: list     # pièces (rectangles) — base du calcul de couverture
    corridor_rects: list      # halls (pas de pénalité de proximité)
    level: int
    footprint: tuple          # (W, H)
    spawn_xy: tuple = (0.0, 0.0)


# ----------------------------------------------------------------------------------------------- plan ---

def _bsp(rect, n, rng, min_dim=MIN_ROOM):
    """Découpe `rect` = (x0, x1, y0, y1) en au plus n rectangles ; coupe la plus grande pièce le long de sa plus grande dimension."""
    leaves = [rect]
    while len(leaves) < n:
        splittable = [r for r in leaves if (r[1] - r[0]) >= 2 * min_dim or (r[3] - r[2]) >= 2 * min_dim]
        if not splittable:
            break
        areas = np.array([(r[1] - r[0]) * (r[3] - r[2]) for r in splittable])
        r = splittable[int(rng.choices(range(len(splittable)), weights=areas)[0])]
        w, h = r[1] - r[0], r[3] - r[2]
        can_x, can_y = w >= 2 * min_dim, h >= 2 * min_dim
        along_x = ((w >= h) if rng.random() < 0.8 else (w < h)) if (can_x and can_y) else can_x
        if along_x:
            s = rng.uniform(max(r[0] + min_dim, r[0] + 0.35 * w), min(r[1] - min_dim, r[0] + 0.65 * w))
            a, b = (r[0], s, r[2], r[3]), (s, r[1], r[2], r[3])
        else:
            s = rng.uniform(max(r[2] + min_dim, r[2] + 0.35 * h), min(r[3] - min_dim, r[2] + 0.65 * h))
            a, b = (r[0], r[1], r[2], s), (r[0], r[1], s, r[3])
        leaves.remove(r)
        leaves += [a, b]
    return leaves


def _shared_edges(rooms):
    """Murs partagés : dicts (i, j, axe, coord, a, b). axe 'v' = mur à x constant (s'étend en y), 'h' = mur à y constant."""
    edges = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            for p, q in ((rooms[i], rooms[j]), (rooms[j], rooms[i])):
                if abs(p.x_max - q.x_min) < EPS:
                    lo, hi = max(p.y_min, q.y_min), min(p.y_max, q.y_max)
                    if hi - lo > EPS:
                        edges.append(dict(i=i, j=j, axis="v", coord=p.x_max, a=lo, b=hi))
                if abs(p.y_max - q.y_min) < EPS:
                    lo, hi = max(p.x_min, q.x_min), min(p.x_max, q.x_max)
                    if hi - lo > EPS:
                        edges.append(dict(i=i, j=j, axis="h", coord=p.y_max, a=lo, b=hi))
    return edges


def _door_center(a, b, width, p_end, rng):
    lo, hi = a + DOOR_MARGIN + width / 2, b - DOOR_MARGIN - width / 2
    if hi - lo < 0.4 or rng.random() >= p_end:
        return rng.uniform(lo, hi)
    near = min(0.5, hi - lo)                  # « à l'extrémité » : dans les 50 cm les plus proches d'un coin
    return lo + rng.uniform(0, near) if rng.random() < 0.5 else hi - rng.uniform(0, near)


def _segments(a, b, gaps, t):
    """Intervalles de mur de [a, b] une fois les ouvertures (centre, largeur) retirées ; les bouts aux coins sont prolongés de t/2."""
    spans = sorted((c - w / 2, c + w / 2) for c, w in gaps)
    out, cur = [], a
    for g0, g1 in spans:
        if g0 > cur + EPS:
            out.append((cur, g0))
        cur = g1
    if b > cur + EPS:
        out.append((cur, b))
    return [(s0 - t / 2 if abs(s0 - a) < EPS else s0, s1 + t / 2 if abs(s1 - b) < EPS else s1) for s0, s1 in out]


def _wall_boxes(axis, coord, a, b, gaps, name, rgba=(0.75, 0.72, 0.68, 1.0), t=WALL_T, z0=0.0, z1=WALL_H):
    boxes = []
    for k, (s0, s1) in enumerate(_segments(a, b, gaps, t)):
        mid, half = (s0 + s1) / 2, (s1 - s0) / 2
        if axis == "v":
            boxes.append((f"{name}_{k}", coord, mid, t / 2, half, z0, z1, rgba))
        else:
            boxes.append((f"{name}_{k}", mid, coord, half, t / 2, z0, z1, rgba))
    return boxes


def _generate_once(lv, level, rng):
    W, H = rng.uniform(*lv["W"]), rng.uniform(*lv["H"])
    n = rng.randint(*lv["n"])
    use_hall = rng.random() < lv["hall"]

    halls = []
    if use_hall:
        hw = rng.uniform(1.6, 2.0)
        hx = rng.uniform(MIN_ROOM, W - hw - MIN_ROOM)
        left, right = (0.0, hx, 0.0, H), (hx + hw, W, 0.0, H)
        n_left = max(1, round(n * (hx / (W - hw))))
        n_right = max(1, n - n_left)
        rects = [(hx, hx + hw, 0.0, H)] + _bsp(left, n_left, rng) + _bsp(right, n_right, rng)
        halls = [0]
    else:
        rects = _bsp((0.0, W, 0.0, H), n, rng)
    rooms = [Room(index=i, x_min=r[0], x_max=r[1], y_min=r[2], y_max=r[3]) for i, r in enumerate(rects)]

    # --- portes : arbre couvrant aléatoire (Kruskal) + portes en plus (boucles)
    door_w_max = lv["door"][1]
    edges = [e for e in _shared_edges(rooms) if e["b"] - e["a"] >= door_w_max + 2 * DOOR_MARGIN]
    rng.shuffle(edges)
    parent = list(range(len(rooms)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree, extra = [], []
    for e in edges:
        ri, rj = find(e["i"]), find(e["j"])
        if ri != rj:
            parent[ri] = rj
            tree.append(e)
        else:
            extra.append(e)
    if len({find(i) for i in range(len(rooms))}) != 1:
        return None
    chosen = tree + [e for e in extra if rng.random() < lv["loop"]]
    doors, gaps_by_edge = [], {}
    for e in chosen:
        w = rng.uniform(*lv["door"])
        c = _door_center(e["a"], e["b"], w, lv["p_end"], rng)
        doors.append((e["i"], e["j"], e["axis"], e["coord"], c, w))
        gaps_by_edge[id(e)] = [(c, w)]

    # --- fenêtre : un tronçon de façade sud assez long
    ww = rng.uniform(*lv["window"])
    south = [r for r in rooms if abs(r.y_min) < EPS and (r.x_max - r.x_min) >= ww + 2 * DOOR_MARGIN]
    if not south:
        return None
    wr = rng.choice(south)
    wc = rng.uniform(wr.x_min + DOOR_MARGIN + ww / 2, wr.x_max - DOOR_MARGIN - ww / 2)

    boxes = []
    for e in edges:
        boxes += _wall_boxes(e["axis"], e["coord"], e["a"], e["b"], gaps_by_edge.get(id(e), []), f"mur_{e['i']}_{e['j']}_{e['axis']}")
    for r in rooms:
        sides = []
        if abs(r.x_min) < EPS:
            sides.append(("v", 0.0, r.y_min, r.y_max, "ouest"))
        if abs(r.x_max - W) < EPS:
            sides.append(("v", W, r.y_min, r.y_max, "est"))
        if abs(r.y_min) < EPS:
            sides.append(("h", 0.0, r.x_min, r.x_max, "sud"))
        if abs(r.y_max - H) < EPS:
            sides.append(("h", H, r.x_min, r.x_max, "nord"))
        for axis, coord, a, b, tag in sides:
            gaps = [(wc, ww)] if (r is wr and tag == "sud") else []
            boxes += _wall_boxes(axis, coord, a, b, gaps, f"ext_{r.index}_{tag}", rgba=(0.62, 0.60, 0.58, 1.0))
        boxes.append((f"sol_{r.index}", (r.x_min + r.x_max) / 2, (r.y_min + r.y_max) / 2, (r.x_max - r.x_min) / 2,
                      (r.y_max - r.y_min) / 2, -0.1, 0.0, (0.35, 0.35, 0.38, 1.0) if r.index not in halls else (0.30, 0.30, 0.33, 1.0)))
    # rebord et linteau de la fenêtre (le trou est visible du LiDAR à 1,2 m)
    boxes.append(("fenetre_rebord", wc, 0.0, ww / 2, WALL_T / 2, 0.0, SILL_TOP, (0.55, 0.45, 0.35, 1.0)))
    boxes.append(("fenetre_linteau", wc, 0.0, ww / 2, WALL_T / 2, LINTEL_BOTTOM, WALL_H, (0.55, 0.45, 0.35, 1.0)))

    # --- cour fermée devant la fenêtre
    depth = rng.uniform(3.5, 5.0)
    yw = min(W, 6.0)
    yx0 = min(max(wc - yw / 2, 0.0), W - yw)
    yx1 = yx0 + yw
    boxes += _wall_boxes("v", yx0, -depth, 0.0, [], "cour_o", rgba=(0.6, 0.57, 0.52, 1.0))
    boxes += _wall_boxes("v", yx1, -depth, 0.0, [], "cour_e", rgba=(0.6, 0.57, 0.52, 1.0))
    boxes += _wall_boxes("h", -depth, yx0, yx1, [], "cour_s", rgba=(0.6, 0.57, 0.52, 1.0))
    boxes.append(("sol_cour", (yx0 + yx1) / 2, -depth / 2, yw / 2, depth / 2, -0.1, 0.0, (0.42, 0.40, 0.36, 1.0)))

    world_x, world_y = (-0.5, W + 0.5), (-depth - 0.5, H + 0.5)
    boxes.append(("plafond", (world_x[0] + world_x[1]) / 2, (world_y[0] + world_y[1]) / 2, (world_x[1] - world_x[0]) / 2,
                  (world_y[1] - world_y[0]) / 2, WALL_H, WALL_H + 0.1, (0.5, 0.5, 0.55, 0.4)))

    return HouseLayout(
        rooms=rooms, wall_height=WALL_H, world_x_range=world_x, world_y_range=world_y,
        courtyard=(yx0, yx1, -depth, 0.0), window=(wc, ww), doors=doors, boxes=boxes,
        navigable_rects=[(r.x_min, r.x_max, r.y_min, r.y_max) for r in rooms],
        corridor_rects=[(rooms[i].x_min, rooms[i].x_max, rooms[i].y_min, rooms[i].y_max) for i in halls],
        level=level, footprint=(W, H), spawn_xy=((yx0 + yx1) / 2, -depth / 2))


def generate_house(level=2, rng=None):
    """Bâtiment de niveau `level` (1-4). `level` peut être un couple (min, max) : niveau tiré au hasard à chaque bâtiment."""
    import random
    rng = rng or random
    if isinstance(level, (tuple, list)):
        level = rng.randint(int(level[0]), int(level[1]))
    lv = LEVELS[int(level)]
    for _ in range(50):
        layout = _generate_once(lv, int(level), rng)
        if layout is not None:
            return layout
    raise RuntimeError(f"generate_house: 50 tentatives sans plan valide (niveau {level})")


def generate_house_xml(layout):
    """XML MuJoCo (geoms) du bâtiment, à insérer dans <worldbody>."""
    xml = ""
    for name, cx, cy, hx, hy, z0, z1, rgba in layout.boxes:
        xml += (f'<geom name="{name}" type="box" pos="{cx:.3f} {cy:.3f} {(z0 + z1) / 2:.3f}" '
                f'size="{hx:.3f} {hy:.3f} {(z1 - z0) / 2:.3f}" rgba="{rgba[0]:.2f} {rgba[1]:.2f} {rgba[2]:.2f} {rgba[3]:.2f}"/>\n')
    return xml


# ------------------------------------------------------------------------------------------ vérification ---

def check_layout(layout, radius=0.25, res=0.05):
    """Liste de problèmes (vide = bon). Vérifie : pièces qui pavent l'empreinte sans trou ni recouvrement ; portes >= 1 m ;
    et, SURTOUT, l'accessibilité physique : un drone de rayon `radius` (carré, conservateur) part du centre de la cour et
    doit pouvoir atteindre le centre de CHAQUE pièce en passant par la fenêtre et les portes (raster de la coupe à 1,2 m)."""
    problems = []
    W, H = layout.footprint
    area = sum((r.x_max - r.x_min) * (r.y_max - r.y_min) for r in layout.rooms)
    if abs(area - W * H) > 1e-3:
        problems.append(f"pièces: aire {area:.2f} != empreinte {W * H:.2f}")
    if any(d[5] < 1.0 - 1e-9 for d in layout.doors) or layout.window[1] < 1.0 - 1e-9:
        problems.append("porte ou fenêtre < 1 m")

    x0, x1 = layout.world_x_range
    y0, y1 = layout.world_y_range
    nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
    blocked = np.zeros((ny, nx), dtype=bool)
    for _, cx, cy, hx, hy, z0, z1, _ in layout.boxes:
        if z0 < 1.25 < z1 and z0 > -0.5:     # murs visibles à la hauteur de vol (pas les sols, ni le rebord/linteau, ni le plafond)
            ia, ib = int((cx - hx - radius - x0) / res), int(math.ceil((cx + hx + radius - x0) / res))
            ja, jb = int((cy - hy - radius - y0) / res), int(math.ceil((cy + hy + radius - y0) / res))
            blocked[max(ja, 0):max(jb, 0), max(ia, 0):max(ib, 0)] = True
    sx, sy = layout.spawn_xy
    start = (int((sy - y0) / res), int((sx - x0) / res))
    if blocked[start]:
        problems.append("départ dans un mur")
        return problems
    seen = np.zeros_like(blocked)
    seen[start] = True
    stack = [start]
    while stack:
        j, i = stack.pop()
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nj, ni = j + dj, i + di
            if 0 <= nj < ny and 0 <= ni < nx and not seen[nj, ni] and not blocked[nj, ni]:
                seen[nj, ni] = True
                stack.append((nj, ni))
    for r in layout.rooms:
        cx, cy = r.center
        if not seen[int((cy - y0) / res), int((cx - x0) / res)]:
            problems.append(f"pièce {r.index} inaccessible ({r.x_max - r.x_min:.1f}x{r.y_max - r.y_min:.1f} m)")
    return problems
