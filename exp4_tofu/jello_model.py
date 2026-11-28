"""Step 2: the Jello cube's E / rho from its frequencies (video_modes.py), by the P1 modes.

The cube of Visual Vibration Tomography, 4.9 x 4.7 x 4.5 cm on a fixed base, with a 2.2 x 2.9 x
1.4 cm block of clay inside at a place the paper does not give.  Clay is a thousand times
stiffer than Jello and about 1.8 times as dense; it is put on the axis at the bottom, the middle
and the top, which bound the frequencies, and left out.  The modes scale as sqrt(E / rho) at a
fixed nu and fixed ratios to the clay, so one solve at E / rho = 1 gives each mode's constant C,
f = C sqrt(E / rho).  The three lowest modes: two sways, then the twist about the vertical.

  python jello_model.py
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

import p1

SIDES = np.array([0.049, 0.047, 0.045])
CLAY = np.array([0.022, 0.029, 0.014])
CLAY_STIFFER, CLAY_DENSER = 1e3, 1.8


def box_mesh(sides, n):
    """A grid of n[0] x n[1] x n[2] cells, each cut into six tetrahedra about one diagonal."""
    axes = [np.linspace(0, s, k + 1) for s, k in zip(sides, n)]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    idx = np.arange(len(X)).reshape(n[0] + 1, n[1] + 1, n[2] + 1)
    c = [idx[:-1, :-1, :-1], idx[1:, :-1, :-1], idx[1:, 1:, :-1], idx[:-1, 1:, :-1],
         idx[:-1, :-1, 1:], idx[1:, :-1, 1:], idx[1:, 1:, 1:], idx[:-1, 1:, 1:]]
    c = [a.ravel() for a in c]
    paths = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    tets = np.concatenate([np.stack([c[a] for a in p], 1) for p in paths])
    return X, tets


def modes(n, nu, clay_at=None, k=6):
    """The k lowest (f at E / rho = 1, kind) of the cube on its fixed base."""
    X, tets = box_mesh(SIDES, n)
    cen = X[tets].mean(1)
    if clay_at is None:
        clay = np.zeros(len(tets), bool)
    else:
        lo = clay_at - CLAY / 2
        clay = np.all((cen > lo) & (cen < lo + CLAY), 1)
    K1, vol = p1.assemble(X, tets[~clay], 1.0, nu)
    K = K1
    if clay.any():
        K2, _ = p1.assemble(X, tets[clay], CLAY_STIFFER, 0.3)
        K = K1 + K2
    vol_all = np.abs(np.linalg.det(np.concatenate([np.ones((len(tets), 4, 1)), X[tets]], 2))) / 6
    rho = np.where(clay, CLAY_DENSER, 1.0)
    m = np.zeros(len(X))
    np.add.at(m, tets.ravel(), np.repeat(rho * vol_all / 4, 4))
    free = np.flatnonzero(np.repeat(X[:, 2] > 1e-12, 3))
    Kf = K[free][:, free]
    Mf = diags(np.repeat(m, 3)[free])
    w2, V = eigsh(Kf, k=k, M=Mf, sigma=0.0, which="LM")
    order = np.argsort(w2)
    out = []
    for j in order:
        u = np.zeros(3 * len(X))
        u[free] = V[:, j]
        u = u.reshape(-1, 3)
        r = X[:, :2] - X[:, :2].mean(0)
        twist = np.sum(m * (r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0])) / np.sqrt(np.sum(m * (r**2).sum(1)) * np.sum(m * (u[:, :2]**2).sum(1)))
        ux, uy, uz = (np.sum(m[:, None] * u**2, 0))
        kind = "twist" if abs(twist) > 0.8 else "sway x" if ux > max(uy, uz) else "sway y" if uy > uz else "vertical"
        out.append((np.sqrt(w2[j]) / (2 * np.pi), kind))
    return out


def main():
    observed = {"sway": (7.97, 8.28), "twist": (9.99, 10.40)}    # Hz, over the three videos
    print("f at E/rho = 1 (Hz, with E in Pa and rho in kg/m^3):")
    rows = []
    for nu in (0.45, 0.49):
        for n in ((16, 16, 15), (24, 24, 22)):
            for where, at in (("none", None), ("bottom", np.array([SIDES[0] / 2, SIDES[1] / 2, CLAY[2] / 2 + 1e-3])),
                              ("middle", SIDES / 2), ("top", np.array([SIDES[0] / 2, SIDES[1] / 2, SIDES[2] - CLAY[2] / 2 - 1e-3]))):
                ms = modes(n, nu, at)
                sway = [f for f, kd in ms if kd.startswith("sway")][:2]
                twist = [f for f, kd in ms if kd == "twist"][:1]
                rows.append((nu, n, where, sway, twist))
                print(f"  nu {nu}  cells {n}  clay {where:6s}  sway {', '.join(f'{f:.4f}' for f in sway)}"
                      f"   twist {', '.join(f'{f:.4f}' for f in twist)}   "
                      + "  ".join(f"{f:.4f} {kd}" for f, kd in ms), flush=True)
    print("\nE / rho (m^2/s^2) from the observed frequencies, finest cells:")
    for nu, n, where, sway, twist in rows:
        if n != (24, 24, 22):
            continue
        s = [(f_obs / np.mean(sway)) ** 2 for f_obs in observed["sway"]]
        t = [(f_obs / twist[0]) ** 2 for f_obs in observed["twist"]] if twist else [np.nan, np.nan]
        print(f"  nu {nu}  clay {where:6s}  from the sways {s[0]:.2f} to {s[1]:.2f}   from the twist {t[0]:.2f} to {t[1]:.2f}")


if __name__ == "__main__":
    main()
