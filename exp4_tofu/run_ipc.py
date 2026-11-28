"""Genesis 1.5.0's FEM through the IPC coupler (genesis_ipc.py): experiment 4a's free vibration
and 4b's ball press, one point per parts file.

  python run_ipc.py flow [--task i]        # E* on the (h, dx) grid; all points, or one
  python run_ipc.py indent [--task i]      # the ball press
  python run_ipc.py merge

The IPC path has no damping parameter, so the free vibration fits E* to the frequency alone,
by the secant method on ln E*, and records the damping ratio the scheme then gives.  Its start
is the theory: the P1 mode's S(dx) times undamped backward Euler's T(h).
"""

import argparse
import json
import os
import time

import numpy as np

import common as c
import genesis_ipc as gi

FLOW_PARTS = c.RESULTS / "ipc_flow_parts"
INDENT_PARTS = c.RESULTS / "ipc_indent_parts"
DXS = (0.01, 0.005, 0.01 / 3, 0.0025, 0.002)
H_IMPLICIT = (0.25e-3, 0.5e-3, 0.75e-3, 1e-3) + tuple(round(0.5e-3 * k, 6) for k in range(3, 11))  # to 5 ms


def h_star():
    """The step at which undamped backward Euler gives the mode the continuum's zeta = 0.10."""
    from scipy.optimize import brentq
    return brentq(lambda h: gi.backward_euler_time_factor(h)[1] - c.ZETA0, 1e-4, 1e-2, xtol=1e-15)


def flow_tasks():
    hs = sorted(H_IMPLICIT + (round(h_star(), 9),))
    return [("gs_ipc", dx, h) for dx in DXS for h in hs]


def fit_E(block, start, tol=1e-9, max_iter=8):
    """Solve omega(E*) = omega0 by the secant method on x = ln(E*/E); returns the best point's
    (E*/E, zeta, |ln(omega / omega0)|, rollouts, q rmse)."""
    w0 = c.omega0()
    t = block.steps * block.h
    q_true = c.truth_q(t)
    pts = []
    x = start
    for k in range(max_iter):
        q = block.modal(np.exp(x))
        w, z = c.measure(t, q)
        r = np.log(w / w0) if np.isfinite(w) else np.nan
        rmse = float(np.sqrt(np.mean((q - q_true) ** 2)) / c.amplitude() * w0)
        pts.append((x, r, z, rmse))
        print(f"    E*/E {np.exp(x):.9f}: ln(w/w0) {r:+.2e}, zeta {z:.6f}", flush=True)
        if not np.isfinite(r) or abs(r) < tol:
            break
        slope = 0.5 if k == 0 else (pts[-1][1] - pts[-2][1]) / (pts[-1][0] - pts[-2][0])
        x = x - r / slope
    x, r, z, rmse = min(pts, key=lambda p: abs(p[1]) if np.isfinite(p[1]) else np.inf)
    return float(np.exp(x)), float(z), float(abs(r)), len(pts), rmse


def run_flow_point(task):
    import run_flow as rf
    arm, dx, h = task
    out = FLOW_PARTS / f"{arm}_{dx * 1e3:.3f}mm_{h * 1e6:.1f}us.json"
    if out.exists():
        return out.name
    t0 = time.time()
    S = rf.theory_table()[f"{dx:.6f}_volume"]["S"]
    T, z_theory = gi.backward_euler_time_factor(h)
    block = gi.IpcBlock(dx, h)
    print(f"{arm} dx {dx * 1e3:.3f} mm, h {h * 1e3:.4f} ms", flush=True)
    E, Z, misfit, n, rmse = fit_E(block, np.log(S * T))
    q = block.modal(1.0)
    t = block.steps * h
    w_nom, z_nom = c.measure(t, q)
    rmse_nom = float(np.sqrt(np.mean((q - c.truth_q(t)) ** 2)) / c.amplitude() * c.omega0())
    row = {"arm": arm, "dx": dx, "h": h, "E_ratio": E, "zeta": Z, "misfit": misfit, "nfev": n,
           "S_theory": S, "time_theory": [T, z_theory], "nominal": [w_nom, z_nom],
           "q_rmse": {"nominal": rmse_nom, "fitted": rmse}, "tol": block.tol, "integrator": block.integrator,
           "seconds": time.time() - t0}
    os.listdir(FLOW_PARTS.parent)
    FLOW_PARTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=1))
    return out.name


INDENT_STEPS = (5e-4, 1e-3, 2e-3)


def indent_tasks():
    """(arm, dx, h, depths): every cell size at three steps to 2.5 mm, then the 5 and 3.3 mm
    cells to Brookfield's 7 mm at 1 ms; the coarsest first."""
    out = [("gs_ipc", dx, h, "std") for dx in DXS for h in INDENT_STEPS]
    out += [("gs_ipc", dx, 1e-3, "deep") for dx in (0.005, 0.01 / 3)]
    return out


def run_indent_point(task):
    import indentation as ind
    arm, dx, h, kind = task
    out = INDENT_PARTS / f"{arm}_{kind}_{dx * 1e3:.3f}mm_{h * 1e6:.0f}us.json"
    if out.exists():
        return out.name
    t0 = time.time()
    depths = ind.DEPTHS if kind == "std" else ind.DEEP
    press = gi.IpcIndent(dx, h, ind.RADIUS)
    rec = press.forces(depths)
    press.close()
    row = {"arm": arm, "dx": dx, "h": h, "kind": kind, "depths": depths.tolist(), "force": rec[:, 1].tolist(),
           "reaction": rec[:, 0].tolist(), "depth_actual": rec[:, 2].tolist(), "gap": rec[:, 3].tolist(),
           "interior_gradient_max": rec[:, 4].tolist(), "d_hat": gi.D_HAT, "subdivisions": gi.BALL_SUBDIVISIONS,
           "seconds": time.time() - t0}
    os.listdir(INDENT_PARTS.parent)
    INDENT_PARTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=1))
    return out.name


def jello():
    """The Jello cube (jello_sim.py) in libuipc: E from the video's sway and the continuum, then
    nominal (E, h = 1 ms) and renormalized (E* = E S T(h*), h*), where h* gives the sway the
    video's damping by backward Euler's own and S = (C / C_P1)^2 comes from the P1 sway of the
    same mesh with libuipc's masses: nothing fitted.  Saves results/ipc_jello.json."""
    import jello_sim as js
    from scipy.optimize import brentq
    cube = gi.IpcJello()
    C = js.continuum_constant()
    E = js.RHO * (js.F_VIDEO / C) ** 2
    S = (C / cube.C_p1) ** 2
    w = 2 * np.pi * js.F_VIDEO
    hs = brentq(lambda h: gi.backward_euler_time_factor(h, w)[1] - js.ZETA_VIDEO, 1e-5, 1e-2, xtol=1e-15)
    T = gi.backward_euler_time_factor(hs, w)[0]
    out = {"C": C, "C_p1": cube.C_p1, "S": S, "E": E, "h_star": hs, "T": T, "runs": []}
    for name, E_run, h in (("nominal", E, 1e-3), ("renormalized", E * S * T, hs)):
        t0 = time.time()
        f, z, t, q = cube.run(E_run, h, duration=2.0)
        print(f"{name}: E {E_run:.1f} Pa, h {h * 1e3:.4f} ms: f {f:.4f} Hz, zeta {z:.5f} ({time.time() - t0:.0f} s)",
              flush=True)
        out["runs"].append({"name": name, "E": E_run, "h": h, "f": f, "zeta": z, "t": t.tolist(), "q": q.tolist()})
    (c.RESULTS / "ipc_jello.json").write_text(json.dumps(out))


def dhat_check(dx, d_hat, h=1e-3):
    """The press to 1 mm at another barrier width d_hat (m): the force's sensitivity to it.
    Saved to results/ipc_dhat/."""
    import indentation as ind
    depths = ind.DEPTHS[:4]
    press = gi.IpcIndent(dx, h, ind.RADIUS, d_hat=d_hat)
    rec = press.forces(depths)
    press.close()
    row = {"dx": dx, "h": h, "d_hat": d_hat, "depths": depths.tolist(), "force": rec[:, 1].tolist(),
           "gap": rec[:, 3].tolist()}
    out = c.RESULTS / "ipc_dhat" / f"{dx * 1e3:.3f}mm_dhat{d_hat * 1e6:g}um.json"
    os.listdir(out.parent.parent)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=1))
    print(json.dumps(row), flush=True)


def press_timing(dx, h=1e-3, warmup=10, window=50, late=450):
    """Wall time per step of the press's stepping loop alone (the sphere reset to its schedule,
    then scene.step(); no readout), over `window` steps after `warmup`, and over the same window
    from step `late` on (0.9 mm deep at 1 ms), where more of the surface is in contact.  Saved to
    results/ipc_press_timing/."""
    import socket

    import indentation as ind
    import torch
    t0 = time.time()
    press = gi.IpcIndent(dx, h, ind.RADIUS)
    build = time.time() - t0
    depth, _ = ind.schedule(h)
    press.scene.reset()
    walls = {}
    for n in range(late + window):
        if n in (warmup, late):
            torch.cuda.synchronize()
            start = time.perf_counter()
        press.ball.set_dofs_position(np.array([-depth[n]]))
        press.ball.set_dofs_velocity(np.array([0.0]))
        press.scene.step()
        if n + 1 in (warmup + window, late + window):
            torch.cuda.synchronize()
            walls["early" if n + 1 == warmup + window else "late"] = (time.perf_counter() - start) / window
    press.close()
    row = {"dx": dx, "h": h, "n_vertices": len(press.X0), "n_tets": len(press.tets), "build_s": build,
           "wall_per_step_s": walls, "speed": {k: h / v for k, v in walls.items()},
           "node": socket.gethostname(), "gpu": torch.cuda.get_device_name()}
    out = c.RESULTS / "ipc_press_timing" / f"{dx * 1e3:.3f}mm_{h * 1e6:.0f}us.json"
    os.listdir(out.parent.parent)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=1))
    print(json.dumps(row), flush=True)


def lattice_check(dx, ppe, depths=(0.25e-3, 0.5e-3, 0.75e-3, 1e-3)):
    """p1_static's statics ("neohookean", "surface") with `ppe` contact points per triangle edge
    instead of its 8, at the first depths, where the contact is a cell or two across and the
    lattice matters.  At 5 mm cells, 8, 24 and 48 per edge give
    0.02542 0.05458 0.10374 / 0.02424 0.05850 0.10103 / 0.02435 0.05788 0.10018 N at 0.25 to 0.75 mm
    (2026-11-28); at 50 per edge the contact stops settling past 1 mm."""
    from p1_static import P1Static
    P = P1Static(dx, "neohookean", "surface", points_per_edge=ppe)
    return [P.indent(d) for d in depths]


def merge():
    for parts, name in ((FLOW_PARTS, "ipc_flow.json"), (INDENT_PARTS, "ipc_indent.json")):
        rows = [json.loads(p.read_text()) for p in sorted(parts.glob("*.json"))]
        (c.RESULTS / name).write_text(json.dumps(rows, indent=1))
        print("merged", len(rows), "into", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("flow", "indent", "jello", "lattice", "press-timing", "dhat", "merge", "count"))
    ap.add_argument("--dhat", type=float)
    ap.add_argument("--dx", type=float)
    ap.add_argument("--ppe", type=int, default=24)
    ap.add_argument("--task", type=int, action="append")
    args = ap.parse_args()
    if args.what == "merge":
        merge()
        return
    if args.what == "jello":
        jello()
        return
    if args.what == "dhat":
        dhat_check(args.dx, args.dhat)
        return
    if args.what == "press-timing":
        press_timing(args.dx)
        return
    if args.what == "lattice":
        print(args.ppe, "per edge:", " ".join(f"{f:.5f}" for f in lattice_check(args.dx, args.ppe)))
        return
    if args.what in ("flow", "count"):
        tasks, run = flow_tasks(), run_flow_point
    else:
        tasks, run = indent_tasks(), run_indent_point
    if args.what == "count":
        print(len(flow_tasks()), "flow,", len(indent_tasks()), "indent")
        return
    for i in (args.task if args.task else range(len(tasks))):
        print("done", run(tasks[i]), flush=True)


if __name__ == "__main__":
    main()
