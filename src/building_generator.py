"""Génération procédurale d'un bâtiment simple : pièces rectangulaires alignées
le long de Y, reliées par des couloirs. Volontairement basique pour la V1
(cf. plan section 8) — la structure (liste de pièces + couloirs) est gardée
séparée du rendu XML pour pouvoir brancher un générateur plus riche plus tard
sans toucher au reste du pipeline (LiDAR / grille / frontières).
"""

import random
from dataclasses import dataclass, field


@dataclass
class Room:
    index: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    floor: int = 0  # prêt pour le multi-étage, non utilisé en V1

    @property
    def center(self):
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


@dataclass
class BuildingLayout:
    rooms: list
    corridor_width: float
    wall_height: float
    world_x_range: tuple
    world_y_range: tuple
    corridor_x_offsets: list = field(default_factory=list)


def generate_layout(n_rooms=(2, 4), room_width_range=(4.0, 7.0),
                     room_depth_range=(4.0, 7.0), corridor_width=1.6,
                     corridor_length_range=(1.5, 3.0), wall_height=2.6,
                     rng=None):
    """Séquence de pièces alignées le long de Y, reliées par des couloirs
    centrés en x=0. Garantit la connectivité par construction (chaîne
    linéaire) — pas de topologie en boucle ou en étoile pour la V1."""
    rng = rng or random
    n = rng.randint(*n_rooms)
    rooms = []
    y_cursor = 0.0
    max_half_width = 0.0

    for i in range(n):
        w = rng.uniform(*room_width_range)
        d = rng.uniform(*room_depth_range)
        x_min, x_max = -w / 2, w / 2
        y_min, y_max = y_cursor, y_cursor + d
        rooms.append(Room(index=i, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max))
        max_half_width = max(max_half_width, w / 2)
        y_cursor = y_max
        if i < n - 1:
            y_cursor += rng.uniform(*corridor_length_range)

    world_x_range = (-max_half_width - 0.5, max_half_width + 0.5)
    world_y_range = (-0.5, y_cursor + 0.5)

    # Décalage en x de chaque couloir : contraint pour rester dans les deux
    # pièces qu'il relie, avec une marge — casse la ligne de vue rectiligne
    # qu'un couloir toujours centré en x=0 créait sur toute la longueur du
    # bâtiment (cf. journal d'expérimentation, diagnostic via debug_grid.py).
    gap_half = corridor_width / 2
    margin = 0.3
    corridor_x_offsets = []
    for i in range(n - 1):
        half_i = (rooms[i].x_max - rooms[i].x_min) / 2
        half_next = (rooms[i + 1].x_max - rooms[i + 1].x_min) / 2
        limit = min(half_i, half_next) - gap_half - margin
        limit = max(limit, 0.0)
        offset = rng.uniform(-limit, limit) if limit > 0 else 0.0
        corridor_x_offsets.append(offset)

    return BuildingLayout(rooms=rooms, corridor_width=corridor_width,
                           wall_height=wall_height, world_x_range=world_x_range,
                           world_y_range=world_y_range,
                           corridor_x_offsets=corridor_x_offsets)


def _wall_x(y, x_start, x_end, thickness, height, name):
    """Segment de mur orienté le long de X (murs avant/arrière d'une pièce)."""
    length = x_end - x_start
    if length <= 1e-6:
        return ""
    cx = (x_start + x_end) / 2
    return (f'<geom name="{name}" type="box" pos="{cx:.3f} {y:.3f} {height/2:.3f}" '
            f'size="{length/2:.3f} {thickness/2:.3f} {height/2:.3f}" '
            f'rgba="0.75 0.72 0.68 1"/>\n')


def _wall_y(x, y_start, y_end, thickness, height, name):
    """Segment de mur orienté le long de Y (murs latéraux / couloirs)."""
    length = y_end - y_start
    if length <= 1e-6:
        return ""
    cy = (y_start + y_end) / 2
    return (f'<geom name="{name}" type="box" pos="{x:.3f} {cy:.3f} {height/2:.3f}" '
            f'size="{thickness/2:.3f} {length/2:.3f} {height/2:.3f}" '
            f'rgba="0.75 0.72 0.68 1"/>\n')


def compute_navigable_rects(layout):
    """Rectangles (x_min, x_max, y_min, y_max) de toutes les zones RÉELLEMENT
    navigables (pièces + couloirs). La grille globale est un rectangle
    dimensionné sur la pièce la plus large ; les pièces plus étroites
    laissent donc des bandes d'espace mort sur les côtés (ni libres, ni
    occupées, jamais atteignables) — on les exclut du calcul de couverture."""
    rects = [(r.x_min, r.x_max, r.y_min, r.y_max) for r in layout.rooms]
    gap_half = layout.corridor_width / 2
    for i in range(len(layout.rooms) - 1):
        y0, y1 = layout.rooms[i].y_max, layout.rooms[i + 1].y_min
        cx = layout.corridor_x_offsets[i]
        rects.append((cx - gap_half, cx + gap_half, y0, y1))
    return rects


def generate_building_xml(layout, wall_thickness=0.15):
    xml = ""
    gap_half = layout.corridor_width / 2

    for i, room in enumerate(layout.rooms):
        xml += _wall_y(room.x_min, room.y_min, room.y_max, wall_thickness,
                        layout.wall_height, name=f"mur_g_{i}")
        xml += _wall_y(room.x_max, room.y_min, room.y_max, wall_thickness,
                        layout.wall_height, name=f"mur_d_{i}")

        if i == 0:
            xml += _wall_x(room.y_min, room.x_min, room.x_max, wall_thickness,
                            layout.wall_height, name=f"mur_arr_{i}")
        else:
            cx_arr = layout.corridor_x_offsets[i - 1]
            xml += _wall_x(room.y_min, room.x_min, cx_arr - gap_half, wall_thickness,
                            layout.wall_height, name=f"mur_arr_{i}_g")
            xml += _wall_x(room.y_min, cx_arr + gap_half, room.x_max, wall_thickness,
                            layout.wall_height, name=f"mur_arr_{i}_d")

        if i == len(layout.rooms) - 1:
            xml += _wall_x(room.y_max, room.x_min, room.x_max, wall_thickness,
                            layout.wall_height, name=f"mur_av_{i}")
        else:
            cx_av = layout.corridor_x_offsets[i]
            xml += _wall_x(room.y_max, room.x_min, cx_av - gap_half, wall_thickness,
                            layout.wall_height, name=f"mur_av_{i}_g")
            xml += _wall_x(room.y_max, cx_av + gap_half, room.x_max, wall_thickness,
                            layout.wall_height, name=f"mur_av_{i}_d")

        xml += (f'<geom name="sol_{i}" type="box" '
                f'pos="{room.center[0]:.3f} {room.center[1]:.3f} -0.05" '
                f'size="{(room.x_max-room.x_min)/2:.3f} {(room.y_max-room.y_min)/2:.3f} 0.05" '
                f'rgba="0.35 0.35 0.38 1"/>\n')

        if i < len(layout.rooms) - 1:
            next_room = layout.rooms[i + 1]
            y0, y1 = room.y_max, next_room.y_min
            cx = layout.corridor_x_offsets[i]
            xml += _wall_y(cx - gap_half, y0, y1, wall_thickness, layout.wall_height,
                            name=f"couloir_g_{i}")
            xml += _wall_y(cx + gap_half, y0, y1, wall_thickness, layout.wall_height,
                            name=f"couloir_d_{i}")
            xml += (f'<geom name="sol_couloir_{i}" type="box" '
                    f'pos="{cx:.3f} {(y0+y1)/2:.3f} -0.05" '
                    f'size="{gap_half:.3f} {(y1-y0)/2:.3f} 0.05" rgba="0.3 0.3 0.33 1"/>\n')

    x_min, x_max = layout.world_x_range
    y_min, y_max = layout.world_y_range
    xml += (f'<geom name="plafond" type="box" '
            f'pos="{(x_min+x_max)/2:.3f} {(y_min+y_max)/2:.3f} {layout.wall_height + 0.05:.3f}" '
            f'size="{(x_max-x_min)/2:.3f} {(y_max-y_min)/2:.3f} 0.05" '
            f'rgba="0.5 0.5 0.55 0.4"/>\n')

    return xml