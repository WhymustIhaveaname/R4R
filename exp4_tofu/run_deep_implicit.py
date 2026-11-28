"""Genesis 1.5.0's implicit FEM pressed to Brookfield's 7 mm: the run, the continuum, the statics.

The explicit FEM went to 7 mm in results/gs_deep_explicit_5mm_rigid.json; this is the implicit
one, on the same slab (indentation.py), sphere and depths (indentation.DEEP):

  run       genesis_sim.GsIndent("gs_fem_implicit"): SAP coupler, linear corotated, 5 mm cells,
            h = 1 ms, E = E0; the force read as GsIndent.forces reads it, with, per depth, its
            spread over the window, the smallest volume ratio of any tet, and how far the sphere
            is from its schedule (it leads by the step's RATE h, 2 um at 1 ms)
  F_ref     the continuum in linear corotated (indentation.reference at DEEP)
  statics   the same mesh solved for equilibrium in linear corotated, its top on SAP's compliant
            layer (p1_static, kind "winkler"), at DEEP

  python run_deep_implicit.py reference | statics | press     # each computes and caches its part
  python run_deep_implicit.py                                 # the table
"""

import json
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

import common as c  # noqa: E402
import indentation as ind  # noqa: E402

ARM, MODEL = "gs_fem_implicit", "corotated"
DX, H = 0.005, 1e-3
OUT = c.RESULTS / "gs_deep_implicit_5mm.json"


def reference():
    """The continuum's force (N) at DEEP, linear corotated, E0; cached by indentation.reference."""
    return ind.reference(MODEL, tuple(ind.DEEP))


def statics():
    """The static P1 slab on SAP's layer at DEEP, linear corotated, E0; cached by p1_static."""
    import p1_static as ps
    return ps.static_forces(DX, MODEL, "winkler", deep=True)


def press(dx=DX, h=H, E=c.E0):
    from genesis_sim import GsIndent
    t0 = time.time()
    g = GsIndent(ARM, dx, h, ind.RADIUS, E_ratio=E / c.E0)
    built = time.time() - t0
    tets = g.gradient.dofs[:, ::3] // 3
    Dm = np.stack([g.X0[tets[:, k]] - g.X0[tets[:, 0]] for k in (1, 2, 3)], 2)
    Bm = np.linalg.inv(Dm)
    depth, reads = ind.schedule(h, ind.DEEP)
    read = set(np.concatenate(reads).tolist())
    g.scene.reset()
    F = np.full(len(depth), np.nan)
    J = np.full(len(depth), np.nan)
    lag = np.full(len(depth), np.nan)
    failed = None
    for n, dep in enumerate(depth):
        g.ball.set_dofs_position(np.array([-dep]))
        g.ball.set_dofs_velocity(np.array([-ind.RATE]))
        g.scene.step()
        if n in read:
            x = g.slab.get_verts_pos().reshape(-1, 3).cpu().numpy()
            F[n] = g.gradient(x)[g.bottom_z].sum()
            xs = x[tets]
            Ds = np.stack([xs[:, k] - xs[:, 0] for k in (1, 2, 3)], 2)
            J[n] = np.linalg.det(Ds @ Bm).min()
            lag[n] = float(g.ball.get_dofs_position().cpu().numpy().ravel()[0]) + dep
            if not np.isfinite(F[n]):
                failed = float(dep)
                break
    return dict(arm=ARM, dx=dx, h=h, E=E, depths=ind.DEEP.tolist(),
                force=[float(F[r].mean()) for r in reads], force_std=[float(F[r].std()) for r in reads],
                min_J=[float(J[r].min()) for r in reads], sphere_lag=[float(np.abs(lag[r]).max()) for r in reads],
                failed_at=failed, build_seconds=built, seconds=time.time() - t0)


def table():
    run = json.loads(OUT.read_text())
    F, Fr, Fs = np.array(run["force"]), reference(), statics()
    print(f"Genesis implicit, {run['dx'] * 1e3:g} mm cells, h = {run['h'] * 1e3:g} ms, E = {run['E'] / 1e3:.1f} kPa: "
          f"{run['seconds']:.0f} s ({run['build_seconds']:.0f} s to build); failed at {run['failed_at']}")
    print(" depth   F (N)   F_ref   statics  F/F_ref-1  statics/F_ref-1  F/statics-1  window std  min J")
    for k, d in enumerate(run["depths"]):
        print(f" {d * 1e3:4.1f}  {F[k]:7.4f}  {Fr[k]:7.4f}  {Fs[k]:7.4f}  {F[k] / Fr[k] - 1:+8.2%}  {Fs[k] / Fr[k] - 1:+12.2%}"
              f"  {F[k] / Fs[k] - 1:+10.2%}  {run['force_std'][k]:9.1e}  {run['min_J'][k]:.3f}")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "table"
    t0 = time.time()
    if what == "reference":
        print("F_ref:", " ".join(f"{f:.5f}" for f in reference()))
    elif what == "statics":
        print("statics:", " ".join(f"{f:.5f}" for f in statics()))
    elif what == "press":
        OUT.write_text(json.dumps(press(), indent=1))
        print("wrote", OUT)
    else:
        table()
        return
    print(f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
