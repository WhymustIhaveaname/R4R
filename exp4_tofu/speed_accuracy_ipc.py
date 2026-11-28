"""Speed against accuracy on the Jello cube for Genesis 1.5.0's FEM through the IPC coupler (gs_ipc).

As speed_accuracy_genesis.py (the cube, MuJoCo's grid of n^3 cells, the two runs, the residuals,
the readout, the timing and its cutoff), with what IPC changes:

  - libuipc simulates the cube (genesis_ipc.py): stable neo-Hookean, masses lumped by volume,
    backward Euler, no damping parameter; the base vertices are fixed in libuipc.
  - libuipc fixes the material and the start at its world's init, so a run builds its scene with
    the mode's velocity written in; the warm-up and the timed run are separated by a reset to the
    world's frame 0.
  - E* = E S T(h): S = (C / C_sway)^2 from the mesh's own sway, T(h) undamped backward Euler's at
    the video's sway (genesis_ipc.backward_euler_time_factor).  zeta is backward Euler's own.
  - h = 0.25, 0.5, 1, 2 ms.
  - frequency residual: lam on E* fitted to the two frequencies, each predicted by undamped
    backward Euler from its mode's frequency on the mesh, corrected by the runs and refined with
    runs until it stops moving.  Damping residual: at that lam, the damping ratios as they come.

  python speed_accuracy_ipc.py --count
  python speed_accuracy_ipc.py --task K --timing-only    # on the cluster's B300: the timed sway run
  python speed_accuracy_ipc.py --task K --no-cut         # anywhere: the fit, untimed
  python speed_accuracy_ipc.py --merge
"""

import argparse
import json
import time

import numpy as np
import torch
from scipy.optimize import minimize_scalar

import genesis_ipc as gi
import speed_accuracy_genesis as sg

METHOD = "gs_ipc"
H = (0.25e-3, 0.5e-3, 1e-3, 2e-3)


def tasks():
    return [(n, h) for n in sg.CELLS for h in H]


def theory(n, h):
    """The mesh's sway and twist at E / rho = 1 and its masses (speed_accuracy_genesis.fem_theory),
    with undamped backward Euler's T(h)."""
    th = sg.fem_theory("gs_fem_implicit", n, 0)
    T, zeta = gi.backward_euler_time_factor(h, sg.W)
    th.update(h=h, T=T, E_ratio=th["S"] * T, damping={}, damping_name="none (backward Euler's own)",
              zeta_sway_theory=zeta)
    return th


class IpcModel(sg.Model):
    def __init__(self, n, th, E):
        gi.init()
        self.h, self.th, self.E = th["h"], th, E
        self.X, self.tets = sg.mesh(n)
        self.base = np.isclose(self.X[:, 2], self.X[:, 2].min())
        self.E_ratio = th["E_ratio"]
        self.scene, self.key = None, None
        self.X0_t = torch.zeros((1, 3), dtype=torch.float64, device="cuda")

    def set(self, E_ratio, damping=None):
        self.E_ratio = E_ratio             # applied by the next build

    def start(self, mode):
        if self.scene is not None and self.key == (mode, self.E_ratio):
            self.scene.reset()
            return
        self.close()
        shape = self.th["shapes"][mode]
        v = sg.gj.V_PEAK * shape / np.abs(shape).max()
        self.scene, self.ent, order = gi.build_body(
            self.X, self.tets, sg.SIDES, self.E * self.E_ratio, sg.NU, sg.RHO, self.h, lambda o: v[o],
            fixed=lambda o: self.base[o])
        dev = self.ent.get_verts_pos().device
        self.X0_t = torch.as_tensor(self.X[order], dtype=torch.float64, device=dev)
        self.prepare({k: s[order] for k, s in self.th["shapes"].items()}, self.th["masses"][order], dev)
        self.key = (mode, self.E_ratio)

    def close(self):
        if self.scene is not None:
            gi.release(self.scene)
            self.scene = None

    def pos(self):
        return self.ent.get_verts_pos().reshape(-1, 3)


def run_point(n, h, force=False, timing_only=False):
    t_start = time.time()
    cs, ct, E = sg.continuum()
    th = theory(n, h)
    name = f"{METHOD}_{n}cells_{h * 1e6:.1f}us.json"
    out_file = (sg.OUT_TIMING if timing_only else sg.OUT if sg.CUTOFF else sg.OUT_FIT) / name
    if out_file.exists() and not force:
        return out_file.name
    timing = sg.OUT_TIMING / name
    if not sg.CUTOFF and timing.exists() and json.loads(timing.read_text())["cut"]:
        return f"{name} skipped: the cluster's timing cut it"
    row = {"method": METHOD, "cells": n, "dx_m": sg.GM / n, "h_s": h, "E": E, "E_ratio": th["E_ratio"],
           "damping_name": th["damping_name"], "damping_value": None, "f_twist_ref": ct * np.sqrt(E / sg.RHO),
           "targets": sg.TARGET, "S": th["S"], "T": th["T"], "zeta_sway_theory": th["zeta_sway_theory"],
           "tolerances": gi.TOL}
    for key in ("f_sway_model_Erho1", "f_twist_model_Erho1", "n_vertices", "n_elements"):
        row[key] = th[key]
    sim = IpcModel(n, th, E)
    row["hardware"] = sg.hardware()
    row["hardware"]["libuipc"] = "pyuipc 0.0.30, CUDA"
    scale = np.sqrt(E * th["E_ratio"] / sg.RHO)
    w_star = 2 * np.pi * scale * np.array([th["f_sway_model_Erho1"], th["f_twist_model_Erho1"]])
    guess = {"sway": gi.backward_euler_undamped(h, w_star[0])[0] / (2 * np.pi),
             "twist": gi.backward_euler_undamped(h, w_star[1])[0] / (2 * np.pi)}
    n_runs = [0]
    try:
        if timing_only:
            t, q, wall, sim_s, cut = sim.rollout("sway", timed=True)
            f, z = (np.nan, np.nan) if cut else sg.measure(t, q, guess["sway"])
            row.update(wall_s=wall, sim_s=sim_s, speed=sim_s / wall, cut=cut, f_sway_timed_run=f,
                       zeta_sway_timed_run=z, seconds_total=time.time() - t_start)
            sg.save(out_file, row)
            return out_file.name

        def run_both(E_ratio, timed=False):
            sim.set(E_ratio)
            meas, timing = {}, None
            for mode in ("sway", "twist"):
                t, q, wall, sim_s, cut = sim.rollout(mode, timed=timed and mode == "sway")
                n_runs[0] += 1
                if timed and mode == "sway":
                    timing = (wall, sim_s, cut)
                    if cut:
                        return None, timing
                f, z = sg.measure(t, q, guess[mode])
                meas[f"f_{mode}"], meas[f"zeta_{mode}"] = f, z
            return meas, timing

        meas, (wall, sim_s, cut) = run_both(th["E_ratio"], timed=True)
        row.update(wall_s=wall, sim_s=sim_s, speed=sim_s / wall, cut=cut)
        if cut:
            row["seconds_total"] = time.time() - t_start
            sg.save(out_file, row)
            return out_file.name
        row["renormalized"] = meas
        print(f"{METHOD} n {n} h {h * 1e6:.1f} us: speed {row['speed']:.3g}, renormalized {meas}", flush=True)

        def predict(lam):
            """The two frequencies (Hz) undamped backward Euler gives at lam E*."""
            return np.array([gi.backward_euler_undamped(h, np.sqrt(lam) * w)[0] for w in w_star]) / (2 * np.pi)
        targets_f = np.array([sg.TARGET["f_sway"], sg.TARGET["f_twist"]])
        lam, history = 1.0, []
        for _ in range(5):
            measured = np.array([meas["f_sway"], meas["f_twist"]])
            if not np.all(np.isfinite(measured)):
                break
            corr = measured / predict(lam)

            def obj(x):
                r = corr * predict(np.exp(x)) / targets_f - 1
                return float(np.sum(r ** 2))
            x = minimize_scalar(obj, bounds=(np.log(lam) - 4, np.log(lam) + 4), method="bounded",
                                options={"xatol": 1e-10}).x
            history.append({"lam": lam, **meas})
            if abs(x - np.log(lam)) < 2e-6:
                break
            lam = float(np.exp(x))
            meas, _ = run_both(th["E_ratio"] * lam)
        row["lam"] = lam
        row["lam_history"] = history
        row.update(meas)
        row.update(sg.errors(meas) if np.all(np.isfinite(list(meas.values()))) else
                   {"err_freq_pct": np.nan, "err_damp_pct": np.nan})
        row["err_pct"] = row["err_freq_pct"]
        row["n_runs"] = n_runs[0]
        row["seconds_total"] = time.time() - t_start
        print(f"{METHOD} n {n} h {h * 1e6:.1f} us: lam {lam:.6f}, final {meas}, err_freq {row['err_freq_pct']:.4g} %, "
              f"err_damp {row['err_damp_pct']:.4g} %", flush=True)
        sg.save(out_file, row)
        return out_file.name
    finally:
        sim.close()


def merge():
    """results/speed_accuracy/gs_ipc_*.json from the fit (OUT_FIT) and the cluster's timing
    (OUT_TIMING): speed, cut, wall_s, sim_s and hardware are the timing's."""
    done = 0
    for tp in sorted(sg.OUT_TIMING.glob(f"{METHOD}_*.json")):
        out = sg.OUT / tp.name
        tim = json.loads(tp.read_text())
        if tim["cut"]:
            row = tim
        elif (sg.OUT_FIT / tp.name).exists():
            row = json.loads((sg.OUT_FIT / tp.name).read_text())
            row["fit_hardware"] = row.get("hardware")
            row["fit_speed_untimed"] = row.get("speed")
            row.update({k: tim[k] for k in ("speed", "cut", "wall_s", "sim_s", "hardware")})
            row["timing_from"] = "speed_accuracy_timing"
        else:
            continue
        sg.save(out, row)
        done += 1
    print("merged", done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--task", type=int, action="append")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--no-cut", action="store_true", help="fit without the cutoff, into OUT_FIT")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.count:
        print(len(tasks()))
        return
    if args.merge:
        merge()
        return
    sg.CUTOFF = not args.no_cut
    for k in args.task if args.task else range(len(tasks())):
        print("done", run_point(*tasks()[k], force=args.force, timing_only=args.timing_only), flush=True)


if __name__ == "__main__":
    main()
