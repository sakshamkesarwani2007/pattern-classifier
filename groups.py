"""The 17 wallpaper groups.

An operation is a move that leaves the pattern looking the same: a rotation, a mirror,
or a glide (mirror + slide along the mirror line). A group is the list of all such moves.
To make a pattern with a given symmetry we copy a random shape through every operation
of the group and add the copies up.

Points are in lattice coordinates: (x, y) means x*a + y*b, so one tile is 0 <= x, y < 1.
An operation is (M, t) and sends p to M @ p + t.
"""
from fractions import Fraction as F
import math

import numpy as np

half = F(1, 2)

identity = (((1, 0), (0, 1)), (0, 0))
rot2 = (((-1, 0), (0, -1)), (0, 0))         # (x, y) -> (-x, -y)
rot3 = (((0, -1), (1, -1)), (0, 0))         # (x, y) -> (-y, x-y)   hex lattice
rot4 = (((0, -1), (1, 0)), (0, 0))          # (x, y) -> (-y, x)     square lattice
rot6 = (((1, -1), (1, 0)), (0, 0))          # (x, y) -> (x-y, x)    hex lattice

mirror_x = (((-1, 0), (0, 1)), (0, 0))      # (x, y) -> (-x, y)
mirror_y = (((1, 0), (0, -1)), (0, 0))      # (x, y) -> (x, -y)
swap = (((0, 1), (1, 0)), (0, 0))           # (x, y) -> (y, x)
hex_mirror = (((-1, 1), (0, 1)), (0, 0))    # (x, y) -> (y-x, y)

glide_pg = (((-1, 0), (0, 1)), (0, half))       # (x, y) -> (-x, y+1/2)
glide_pmg = (((1, 0), (0, -1)), (half, 0))      # (x, y) -> (x+1/2, -y)
glide_pgg = (((1, 0), (0, -1)), (half, half))   # (x, y) -> (x+1/2, -y+1/2)
glide_p4gm = (((-1, 0), (0, 1)), (half, half))  # (x, y) -> (-x+1/2, y+1/2)

# name -> (lattice type, generators). The order here is the class index order.
GROUPS = {
    "p1":   ("oblique", []),
    "p2":   ("oblique", [rot2]),
    "pm":   ("rectangular", [mirror_x]),
    "pg":   ("rectangular", [glide_pg]),
    "cm":   ("rhombic", [swap]),
    "pmm":  ("rectangular", [mirror_x, mirror_y]),
    "pmg":  ("rectangular", [rot2, glide_pmg]),
    "pgg":  ("rectangular", [rot2, glide_pgg]),
    "cmm":  ("rhombic", [swap, rot2]),
    "p4":   ("square", [rot4]),
    "p4mm": ("square", [rot4, mirror_x]),
    "p4gm": ("square", [rot4, glide_p4gm]),
    "p3":   ("hexagonal", [rot3]),
    "p3m1": ("hexagonal", [rot3, hex_mirror]),
    "p31m": ("hexagonal", [rot3, swap]),
    "p6":   ("hexagonal", [rot6]),
    "p6mm": ("hexagonal", [rot6, swap]),
}

GROUP_NAMES = list(GROUPS)
NAME_TO_IDX = {n: i for i, n in enumerate(GROUP_NAMES)}
IDX_TO_NAME = {i: n for i, n in enumerate(GROUP_NAMES)}

# how many different operations each group has (ignoring whole-tile shifts)
ORDER = {"p1": 1, "p2": 2, "pm": 2, "pg": 2, "cm": 2, "pmm": 4, "pmg": 4, "pgg": 4,
         "cmm": 4, "p4": 4, "p4mm": 8, "p4gm": 8, "p3": 3, "p3m1": 6, "p31m": 6,
         "p6": 6, "p6mm": 12}


def clean(op):
    M, t = op
    M = tuple(tuple(int(v) for v in row) for row in M)
    t = tuple(F(v) % 1 for v in t)
    return M, t


def compose(op1, op2):
    """Do op2 first, then op1."""
    (M1, t1), (M2, t2) = op1, op2
    M = tuple(tuple(sum(M1[i][k] * M2[k][j] for k in range(2)) for j in range(2))
              for i in range(2))
    t = tuple(sum(M1[i][k] * t2[k] for k in range(2)) + t1[i] for i in range(2))
    return clean((M, t))


def group_ops(name):
    """All operations of the group as (M, t) numpy arrays."""
    gens = [clean(g) for g in GROUPS[name][1]]
    found = {clean(identity)}
    todo = list(found)
    while todo:
        op = todo.pop()
        for g in gens:
            new = compose(g, op)
            if new not in found:
                found.add(new)
                todo.append(new)
    return [(np.array(M, dtype=float), np.array([float(v) for v in t]))
            for M, t in sorted(found)]


def sample_lattice(lattice, rng):
    """Random lattice vectors a and b as the rows of a 2x2 array.

    Angles and lengths stay away from the special values (90, 60, 120 degrees, equal
    lengths) so the lattice doesn't end up with more symmetry than the group has.
    """
    if lattice == "square":
        la, lb, angle = 1.0, 1.0, 90.0
    elif lattice == "hexagonal":
        la, lb, angle = 1.0, 1.0, 120.0
    elif lattice == "rectangular":
        la, lb, angle = 1.0, rng.uniform(1.15, 1.5), 90.0
    elif lattice == "rhombic":
        la = lb = 1.0
        angle = rng.choice([rng.uniform(35, 55), rng.uniform(65, 85)])
    elif lattice == "oblique":
        la, lb = 1.0, rng.uniform(1.15, 1.8)
        angle = rng.choice([rng.uniform(65, 85), rng.uniform(95, 115)])
    else:
        raise ValueError("unknown lattice: " + lattice)
    t = math.radians(angle)
    return np.array([[la, 0.0], [lb * math.cos(t), lb * math.sin(t)]])


if __name__ == "__main__":
    for name in GROUP_NAMES:
        n = len(group_ops(name))
        print(name, GROUPS[name][0], n, "ok" if n == ORDER[name] else "WRONG")
