"""The spatial discretization alone: linear tetrahedra, as in MuJoCo's and Genesis's FEM.

assemble() builds the small-strain stiffness of constant-strain tetrahedra.  Its lowest elastic
mode, with either of the two lumped masses, gives how E* must flow with the cell size dx as
h -> 0, with nothing fitted:

  E*(dx) / E = (omega_continuum / omega_discrete(dx))^2.

MuJoCo gives every vertex the same mass; Genesis gives each vertex a quarter of the mass of
every tetrahedron it belongs to.
"""

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

import common as c


def mesh_from_mujoco(dx, xml=None):
    """MuJoCo's grid: six tetrahedra per cell along one body diagonal (user_mesh.cc).

    The block of experiment 4a, or the flex of any other model given as `xml`.
    """
    import mujoco
    from mujoco_sim import model_xml
    m = mujoco.MjModel.from_xml_string(xml or model_xml(dx, "discrete", 1e-3))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    X = d.flexvert_xpos.copy()
    tets = m.flex_elem.reshape(-1, 4).copy()
    return X, tets


def assemble(X, tets, E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] += 2 * mu
    C[np.arange(3, 6), np.arange(3, 6)] = mu
    P = X[tets]                                   # (ne, 4, 3)
    A = np.concatenate([np.ones((len(tets), 4, 1)), P], 2)
    Ainv = np.linalg.inv(A)                       # rows 1..3: gradients of the shape functions
    vol = np.abs(np.linalg.det(A)) / 6
    g = Ainv[:, 1:, :]                            # (ne, 3, 4): d N_a / d x_i
    B = np.zeros((len(tets), 6, 12))
    for a in range(4):
        gx, gy, gz = g[:, 0, a], g[:, 1, a], g[:, 2, a]
        B[:, 0, 3 * a] = gx
        B[:, 1, 3 * a + 1] = gy
        B[:, 2, 3 * a + 2] = gz
        B[:, 3, 3 * a + 1], B[:, 3, 3 * a + 2] = gz, gy
        B[:, 4, 3 * a], B[:, 4, 3 * a + 2] = gz, gx
        B[:, 5, 3 * a], B[:, 5, 3 * a + 1] = gy, gx
    Ke = np.einsum("eki,kl,elj,e->eij", B, C, B, vol)
    dofs = (3 * tets[:, :, None] + np.arange(3)[None, None]).reshape(len(tets), 12)
    rows = np.repeat(dofs, 12, 1).ravel()
    cols = np.tile(dofs, (1, 12)).ravel()
    n = 3 * len(X)
    K = coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsc()
    return K, vol


def masses(X, tets, vol, kind):
    total = c.RHO * np.prod(c.SIDES)
    if kind == "equal":
        return np.full(len(X), total / len(X))
    if kind == "volume":
        m = np.zeros(len(X))
        np.add.at(m, tets.ravel(), np.repeat(c.RHO * vol / 4, 4))
        return m
    raise ValueError(kind)


def discrete_mode(X, tets, kind, n_eig=8):
    """(omega, overlap) of the discrete mode closest to the continuum's MODE."""
    K, vol = assemble(X, tets, c.E0, c.NU)
    m = masses(X, tets, vol, kind)
    M = diags(np.repeat(m, 3))
    w0 = c.omega0()
    w2, V = eigsh(K, k=n_eig, M=M, sigma=w0**2, which="LM")
    phi = c.mode_shape(X).ravel()
    phi = phi / np.sqrt(phi @ (M @ phi))
    overlap = np.abs(V.T @ (M @ phi))
    k = int(np.argmax(overlap))
    return float(np.sqrt(w2[k])), float(overlap[k])


def spatial_prediction(dx, kind):
    X, tets = mesh_from_mujoco(dx)
    w, ov = discrete_mode(X, tets, kind)
    return (c.omega0() / w) ** 2, ov


def first_order_equal_mass():
    """d(E*/E)/d(dx) at dx -> 0 for equal vertex masses: the quadrature error of the kinetic energy.

    In each direction the equal-weight rule over n + 1 points is the trapezoid rule, plus dx
    times (the mean of the two end values minus the mean over the length), to first order.
    So T_discrete / T = 1 + dx * kappa, with kappa summed over the three directions from the
    mode's |phi|^2 on the faces and in the volume; omega^2 falls by the same factor, and E*
    rises by it.
    """
    from numpy.polynomial import legendre
    xs, ws = legendre.leggauss(40)
    grids = [xs * a for a in c.HALF]
    wts = [ws * a for a in c.HALF]
    G = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)
    W = np.einsum("i,j,k->ijk", *wts).ravel()
    u2 = (c.mode_shape(G) ** 2).sum(1)
    I = W @ u2
    kappa = 0.0
    for d in range(3):
        others = [e for e in range(3) if e != d]
        face = 0.0
        for sgn in (-1, 1):
            pts = np.stack(np.meshgrid(*[grids[e] for e in others], indexing="ij"), -1).reshape(-1, 2)
            P = np.zeros((len(pts), 3))
            P[:, others] = pts
            P[:, d] = sgn * c.HALF[d]
            wf = np.einsum("i,j->ij", wts[others[0]], wts[others[1]]).ravel()
            face += wf @ (c.mode_shape(P) ** 2).sum(1)
        kappa += face / 2 / I - 1 / c.SIDES[d]
    return kappa


if __name__ == "__main__":
    print(f"first-order slope, equal masses: d(E*/E)/d(dx) = {first_order_equal_mass():.3f} per m")
    for dx in (0.01, 0.005, 0.01 / 3, 0.0025, 0.002):
        out = [spatial_prediction(dx, kind) for kind in ("equal", "volume")]
        print(f"dx {dx * 1e3:5.2f} mm   E*/E equal masses {out[0][0]:.5f} (overlap {out[0][1]:.3f})"
              f"   volume masses {out[1][0]:.5f} (overlap {out[1][1]:.3f})")


def mode_vector(dx, kind):
    """The scheme's own version of the mode, at MuJoCo's vertices, cached on disk.

    The block starts with this velocity field, so that it moves in one discrete mode and the
    measured frequency is that mode's: the continuum's shape would also excite the scheme's
    other modes a little, and their leakage biases the fit by up to 1e-4 at the coarsest
    cells, as large as the smallest flows.  It is scaled so that its projection on the
    continuum's shape, in the scheme's masses, is the shape itself.
    """
    path = c.CACHE / f"mode_{dx:.6f}_{kind}.npy"
    if path.exists():
        return np.load(path)
    X, tets = mesh_from_mujoco(dx)
    K, vol = assemble(X, tets, c.E0, c.NU)
    m = masses(X, tets, vol, kind)
    M = diags(np.repeat(m, 3))
    w2, V = eigsh(K, k=8, M=M, sigma=c.omega0() ** 2, which="LM")
    phi = c.mode_shape(X).ravel()
    k = int(np.argmax(np.abs(V.T @ (M @ phi))))
    v = V[:, k]
    v = v * (phi @ (M @ phi)) / (v @ (M @ phi))
    c.CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, v.reshape(-1, 3))
    return v.reshape(-1, 3)
