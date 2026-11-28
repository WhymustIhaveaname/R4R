"""Experiment 4b: Brookfield's ball test of tofu, in the engines, against the continuum.

A rigid frictionless sphere of radius 12.7 mm (Brookfield's TA43 probe, NOTES.md) is lowered at
2 mm/s into a 10 x 10 x 4 cm slab, about the size of a 14 oz block, its bottom face bonded, no
gravity.  2 mm/s is quasi-static (the slab's lowest mode is about 12 Hz), and the force on the
sphere is read as it passes each depth.

The depths stop at 2.5 mm, not at Brookfield's 7: at 2.9 mm the material of MuJoCo's flex,
Saint Venant-Kirchhoff, reaches its largest stress in compression under the sphere and its
continuum limit ends (axisym_nl.py).  Genesis's materials, linear corotated and neo-Hookean,
reach 7 mm, which DEEP is for.

The reference is axisym_nl.py's large-deformation FEM of a cylinder of the same area (radius
5.64 cm) in each engine's own material.  The slab's top centre is a vertex of every grid.
"""

from functools import lru_cache

import numpy as np

import common as c

SIDES = np.array([0.10, 0.10, 0.04])
RADIUS = 0.0127                                   # m, Brookfield's 25.4 mm sphere
DEPTHS = np.arange(1, 11) * 0.25e-3               # m
DEEP = np.arange(1, 15) * 0.5e-3                  # m, to Brookfield's 7 mm, for neo-Hookean
RATE = 2e-3       # m/s
WINDOW = 0.01     # s around each depth over which the force is averaged
R_EQUIVALENT = np.sqrt(SIDES[0] * SIDES[1] / np.pi)
REFERENCE_FINE = 1e-4   # m, the reference's cells under the sphere


@lru_cache(maxsize=None)
def reference(model, depths=None):
    """The continuum's force (N) at DEPTHS, or at the given tuple of depths, in the material `model`."""
    depths = DEPTHS if depths is None else np.asarray(depths)
    key = "_".join(f"{d * 1e3:g}" for d in depths)
    path = c.CACHE / f"indent_reference_{model}_{RADIUS * 1e3:.1f}mm_{key}.npy"
    if path.exists():
        return np.load(path)
    import axisym_nl
    layer = axisym_nl.LayerNL(R_EQUIVALENT, SIDES[2], c.E0, c.NU, model, REFERENCE_FINE, 0.015)
    F = np.array([layer.indent(RADIUS, d) for d in depths])
    c.CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, F)
    return F


def schedule(h, depths=DEPTHS):
    """The sphere's depth at every step, and the steps over which each depth's force is read."""
    depths = np.asarray(depths)
    n = int(round(depths[-1] / RATE / h)) + int(round(WINDOW / h))
    depth = np.minimum(RATE * h * np.arange(1, n + 1), depths[-1])
    half = max(1, int(round(WINDOW / 2 / h)))
    centres = np.round(depths / RATE / h).astype(int) - 1
    reads = [np.arange(max(0, k - half), min(n, k + half + 1)) for k in centres]
    return depth, reads
