"""The ball press: each engine's force under Brookfield's sphere against the continuum's.

The force against the depth, F / F_ref - 1 in percent, one colour per cell size dx, at one step
h per scheme (the press is quasi-static, so the step does not matter).  The rings are the
statics: the same mesh, material and contact solved for equilibrium at each depth
(p1_static.py), nothing fitted.

fig_indent_{arm}:    Genesis's FEM, explicit (h at half its stability limit), implicit and through the IPC
                     coupler (1 ms); Genesis's MPM and PBD, which have no statics: only the
                     continuum is their reference.  MuJoCo's has none: README.zh.md says why.

  python plot_indent.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as c  # noqa: E402
import indentation as ind  # noqa: E402
from figures import INK, INK2, clear_legends, style, use_house_style  # noqa: E402

# 10 mm cells are left out: one cell is wider than the whole contact (radius 1.8 to 5.6 mm)
DX_COLOR = {5.0: "#5598e7", 3.333: "#2a78d6", 2.5: "#1c5cab", 2.0: "#104281"}
H = 1e-3   # s, the one step drawn, but for Genesis's explicit FEM


def static(dx, model="stvk", kind="surface"):
    path = c.CACHE / f"p1_static_{model}_{kind}_{ind.RADIUS * 1e3:.1f}mm_{dx:.6f}.npy"
    return np.load(path) if path.exists() else None


def one_step(sel):
    """The run drawn at one cell size: h = 1 ms, or for a scheme whose steps are fractions of its
    stability limit, the middle one."""
    hs = sorted({r["h"] for r in sel})
    h = H if any(abs(x - H) < 1e-9 for x in hs) else hs[len(hs) // 2]
    return next(r for r in sel if abs(r["h"] - h) < 1e-9)


# the 5 mm cells pressed on to 7 mm, the depth of Brookfield's test: run file, statics cache name
DEEP_RUN = {"gs_mpm": ("gs_deep_mpm_5mm.json", None), "gs_pbd": ("gs_deep_pbd_5mm.json", None),
            "gs_fem_explicit": ("gs_deep_explicit_5mm_rigid.json", "p1_static_neohookean_vertex_12.7mm_0.005000_deep.npy"),
            "gs_fem_implicit": ("gs_deep_implicit_5mm.json", "p1_static_corotated_winkler_12.7mm_0.005000_deep.npy"),
            "gs_ipc": ("ipc_indent_parts/gs_ipc_deep_5.000mm_1000us.json",
                       "p1_static_neohookean_surface_12.7mm_0.005000_deep.npy")}


def deep_points(arm, model):
    """(depths mm, run / F_ref, statics / F_ref or None) of the 5 mm cells past 2.5 mm, or None.
    An arm with statics is drawn only with them."""
    run_name, statics_name = DEEP_RUN[arm]
    path = c.RESULTS / run_name
    if not path.exists() or (statics_name is not None and not (c.CACHE / statics_name).exists()):
        return None
    run = json.loads(path.read_text())
    ref = ind.reference(model, tuple(ind.DEEP))
    D = np.asarray(run["depths"])
    keep = D > ind.DEPTHS.max() + 1e-9
    FP = None if statics_name is None else (np.load(c.CACHE / statics_name) / ref)[keep]
    return D[keep] * 1e3, (np.asarray(run["force"]) / ref)[keep], FP


def force_panel(ax, rows, contact, ref, title, model, kind, deep=None):
    """kind None: the arm has no statics, its runs are drawn against the continuum alone."""
    D = ind.DEPTHS * 1e3
    style(ax)
    ax.axhline(0, color="#e4e3df", lw=1, zorder=0)
    worst, drawn = 0.0, []
    for dx_mm, color in DX_COLOR.items():
        sel = [r for r in rows if r["contact"] == contact and abs(r["dx"] * 1e3 - dx_mm) < 1e-3]
        dx = 0.01 / 3 if dx_mm == 3.333 else dx_mm / 1e3
        FP = None if kind is None else static(dx, model, kind)
        if not sel or (kind is not None and FP is None):        # no run is drawn without its statics
            continue
        drawn.append(dx_mm)
        run = np.array(one_step(sel)["force"])
        ax.plot(D, 100 * (run / ref - 1), "o", ms=5.5, color=color, mec="white", mew=0.6, zorder=3)
        if FP is not None:
            ax.plot(D, 100 * (FP / ref - 1), "o", ms=11, mfc="none", mec=color, mew=1.3, zorder=2)
            worst = max(worst, np.abs(run / FP - 1).max())
    if deep is not None:
        Dd, run, FP = deep
        color = DX_COLOR[5.0]
        ax.plot(Dd, 100 * (run - 1), "o", ms=5.5, color=color, mec="white", mew=0.6, zorder=3)
        if FP is not None:
            ax.plot(Dd, 100 * (FP - 1), "o", ms=11, mfc="none", mec=color, mew=1.3, zorder=2)
            worst = max(worst, np.abs(run / FP - 1).max())
    ax.set_xlabel(r"depth $\delta$ (mm)")
    ax.set_xlim(0, 7.3 if deep is not None else 2.6)
    ax.set_ylabel(r"$F / F_\mathrm{ref} - 1$ (%)")
    ax.set_title(title + (f"\nruns within {100 * worst:.1f}% of the statics" if worst else ""), fontsize=10.5)
    return drawn


def legend_entries(ax, drawn=None, statics=True):
    for dx_mm, color in DX_COLOR.items():
        if drawn is not None and dx_mm not in drawn:
            continue
        ax.plot([], [], "o", ms=6, color=color, label=rf"$\Delta x$ = {dx_mm:g} mm")
    if statics:
        ax.plot([], [], "o", ms=11, mfc="none", mec=INK, mew=1.3, label="statics")


GS_ARMS = (("gs_fem_explicit", "legacy", "neohookean", "vertex", "Genesis 1.5.0, FEM explicit: legacy coupler, stable neo-Hookean",
            "Statics: the same mesh and material solved for equilibrium at each depth,\nthe sphere touching its vertices."),
           ("gs_fem_implicit", "sap", "corotated", "winkler", "Genesis 1.5.0, FEM implicit: SAP coupler, linear corotated",
            "Statics: the same mesh and material solved for equilibrium at each depth, its whole surface on\n"
            "a compliant layer, SAP's pressure field of the FEM, 5e8 Pa/m."),
           ("gs_ipc", "ipc", "neohookean", "surface",
            "Genesis 1.5.0, FEM through the IPC coupler: libuipc's stable neo-Hookean",
            "Statics: the same mesh and material solved for equilibrium at each depth, its whole top surface\n"
            "kept out of the sphere.  IPC's barrier width: 2 um."),
           ("gs_mpm", "legacy", "fixed_corotated", None, "Genesis 1.5.0, MPM: legacy coupler, fixed corotated",
            "MPM has no statics to compare with: the continuum is its only reference."),
           ("gs_pbd", "legacy", "neohookean", None, "Genesis 1.5.0, PBD: legacy coupler, edge springs",
            "PBD has no statics to compare with: the continuum is its only reference.\n"
            r"Each edge's compliance: $1 / (E \Delta x)$.  $h_{\max}$: the largest step at which some compliance"
            "\nstill gives the slab's lowest frequency."))
# the step each arm is drawn at, for the caption
STEP = {"gs_fem_explicit": "half its stability limit", "gs_mpm": r"0.02 $\Delta x$ (s/m)",
        "gs_pbd": r"0.5 $h_{\max}$"}
# the continuum's material, for the caption: the arm's own, but PBD has none
REF_MATERIAL = {"gs_pbd": "neo-Hookean"}


def ipc_rows():
    """The IPC coupler's presses (genesis_ipc.py, run_ipc.py) as rows of the shape the others have."""
    rows = []
    for path in sorted((c.RESULTS / "ipc_indent_parts").glob("gs_ipc_std_*.json")):
        r = json.loads(path.read_text())
        rows.append(dict(r, arm="gs_ipc", contact="ipc"))
    return rows


def genesis():
    """Genesis 1.5.0, one figure per solver: the FEM explicit through the legacy coupler (vertex
    contact, stable neo-Hookean), implicit through SAP (the sphere's triangles against the FEM's
    pressure field, a compliant layer, linear corotated) and through the IPC coupler, each against
    the statics with that contact and that material; MPM (fixed corotated) and PBD (edge springs)
    through the legacy coupler (run_indent_particles.py), against the continuum alone."""
    use_house_style()
    rows = [json.loads(p.read_text()) for p in sorted((c.RESULTS / "indent_parts").glob("gs_fem_*.json"))]
    rows += ipc_rows()
    for arm in ("gs_mpm", "gs_pbd"):
        rows += [r for r in (json.loads(p.read_text()) for p in sorted((c.RESULTS / "indent_parts").glob(f"{arm}_*.json")))
                 if not r.get("cut")]
    for arm, contact, model, kind, title, statics in GS_ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        if not arm_rows:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.2))
        deep = deep_points(arm, model)
        drawn = force_panel(ax, arm_rows, contact, ind.reference(model), title, model, kind, deep)
        legend_entries(ax, drawn, statics=kind is not None)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        step = STEP.get(arm, "1 ms")
        fig.text(0.5, 0.01,
                 r"$F$ (N): the force on a 12.7 mm sphere at 2 mm/s into a 10 x 10 x 4 cm slab, $E$ = 11.3 kPa." + "\n"
                 rf"$F_\mathrm{{ref}}$ (N): the continuum's, {REF_MATERIAL.get(arm, 'in the same material')}.  $h$: {step}."
                 + ("  Past 2.5 mm, 5 mm cells only." if deep is not None else "")
                 + "\n" + statics,
                 ha="center", fontsize=8.5, color=INK2)
        fig.tight_layout(rect=(0, 0.15, 1, 1))
        clear_legends(fig)
        for e in ("png", "pdf"):
            fig.savefig(c.RESULTS / f"fig_indent_{arm}.{e}", dpi=200)
        plt.close(fig)
        print("wrote", c.RESULTS / f"fig_indent_{arm}.png")


if __name__ == "__main__":
    genesis()
