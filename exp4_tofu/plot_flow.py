"""The flow of E* and zeta* with the two discretizations, the step h and the cell size dx.

fig_flow_{arm}:       one per scheme, its panels in a row.  Against h, the part of the flow that
                      the step makes: E* divided by its h -> 0 value at the same dx, which the
                      mesh's modes give with nothing fitted, and zeta*.  If the two
                      discretizations separate, every dx falls on one line.  The first scheme of
                      each mass lumping (MuJoCo's equal masses, Genesis's by volume) adds a panel
                      against dx, the part the cells make: E* divided by its theoretical time
                      factor, every run of that lumping.  If they separate, every h falls on one
                      point per dx.
fig_flow_mpm:         MPM, which has no damping parameter: E* and its own damping against dx,
                      with the mode of its linearized step (mpm_linear.py), nothing fitted.
fig_flow_pbd:         PBD likewise, with its linearized step (pbd_linear.py).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common as c  # noqa: E402
import run_flow as rf  # noqa: E402
from figures import FIT_COLOR, INK, INK2, clear_legends, flow_figure, num, style, use_house_style  # noqa: E402

MISFIT_MAX = 1e-6
DX_COLOR = {10.0: "#86b6ef", 5.0: "#5598e7", 3.333: "#2a78d6", 2.5: "#1c5cab", 2.0: "#104281"}
DX_SIZE = {10.0: 11.0, 5.0: 9.0, 3.333: 7.5, 2.5: 6.0, 2.0: 4.5}   # rings where they coincide
LABEL = {"mj_euler": "MuJoCo 4.0.0, Euler\n(symplectic, damping explicit)",
         "mj_discrete": "MuJoCo 4.0.0, discrete\n(backward Euler, linearized)",
         "gs_fem_explicit": "Genesis 1.5.0, FEM explicit\n(symplectic, damping by a factor)",
         "gs_fem_implicit": "Genesis 1.5.0, FEM implicit\n(backward Euler)"}
# What the captions say: the parameters that are renormalized, and their units
E_DEF = (r"$E^\star$ (Pa): the Young's modulus given to the simulator so that the block's lowest mode "
         r"has the continuum's frequency, 14.4 Hz at $E$ = 11.3 kPa.")
ZETA_DEF = (r"$\zeta^\star$ (dimensionless): the damping ratio given to the simulator so that the mode "
            r"decays as the continuum's, $\zeta$ = 0.10.")


def coefficients(scheme):
    """(c1, c2) per ms and ms^2 of the theory's E*/E - 1 and zeta*/zeta - 1: quadratic in h."""
    hs = np.array([0.0, 1e-3, 2e-3])
    vals = np.array([c.time_prediction(scheme, h) for h in hs])
    out = []
    for k, base in ((0, 1.0), (1, c.ZETA0)):
        a = np.polyfit(hs * 1e3, vals[:, k] / base - 1, 2)
        a[np.abs(a) < 1e-12] = 0.0      # rounding, where the theory has no term
        out.append((a[1], a[0]))
    return out


def pct(x):
    """A deviation already in percent, as text."""
    return f"{x:.2g}%" if x >= 0.01 else f"{x:.0e}%"


def key(dx):
    return round(dx * 1e3, 3)


def h_column(arm, rows):
    rows = sorted((r for r in rows if r["arm"] == arm), key=lambda r: (r["dx"], r["h"]))
    scheme = rf.ARMS[arm][1]
    (e1, e2), (z1, z2) = (tuple(100 * np.array(t)) for t in coefficients(scheme))   # in percent
    h = np.array([r["h"] * 1e3 for r in rows])
    e = 100 * np.array([r["E_ratio"] / r["S_theory"] - 1 for r in rows])
    z = 100 * np.array([r["zeta"] / c.ZETA0 - 1 for r in rows])
    col = {"title": LABEL[arm], "h": h, "groups": np.array([key(r["dx"]) for r in rows]),
           "rows": [(e, (e1, e2)), (z, (z1, z2))], "rmse": np.zeros(len(rows))}
    if not scheme.startswith("backward"):
        hm = h.max() * 1.1
        col["h_max"] = hm
        col["f_ticks"] = f_ticks(hm)
    return col


def f_ticks(h_max):
    """Frequencies for the top axis, no two closer than a tenth of the range in h."""
    out = []
    for f in (1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5):
        h = 1e3 / f
        if h < h_max and all(abs(h - 1e3 / g) > 0.1 * h_max for g, _ in out):
            out.append((f, f"{f / 1e3:g}k"))
    return out


def wrap(text, width=100):
    """Break a caption's lines at spaces outside $...$, so that no line runs wider than `width`."""
    out = []
    for line in text.split("\n"):
        words, cur = line.split(" "), ""
        merged, buf = [], ""
        for w in words:                       # keep a math span in one piece
            buf = w if not buf else buf + " " + w
            if buf.count("$") % 2 == 0:
                merged.append(buf)
                buf = ""
        if buf:
            merged.append(buf)
        for w in merged:
            if cur and len(cur) + 1 + len(w) > width:
                out.append(cur)
                cur = w
            else:
                cur = w if not cur else cur + " " + w
        out.append(cur)
    return "\n".join(out)


H_ORDER = ("mj_discrete", "mj_euler", "gs_fem_explicit", "gs_fem_implicit")   # the README's order
# the scheme whose figure also carries the cell-size panel: (mass lumping, the schemes it holds)
DX_PANEL = {"mj_discrete": ("equal", ("mj_discrete", "mj_euler")),
            "gs_fem_explicit": ("volume", ("gs_fem_explicit", "gs_fem_implicit"))}
T_DEF = r"$T(h)$ (dimensionless): the theory's factor $E^\star / E^\star_{h \to 0}$ for the run's step $h$."


def scheme_figures(rows, left_out):
    """fig_flow_{arm}: one figure per scheme, its panels in a row: E* and zeta* against h, and
    for the first scheme of each mass lumping E* against dx, every run of that lumping."""
    for arm in H_ORDER:
        if not any(r["arm"] == arm for r in rows):
            continue
        col = h_column(arm, rows)
        # An explicit scheme's damping counterterm is zero or of second order and tiny: its row
        # keeps the range of the E* row above it rather than magnifying the noise.
        ylim, no_fit = [None], []
        if not rf.ARMS[arm][1].startswith("backward"):
            e = col["rows"][0][0]
            pad = 0.05 * (e.max() - e.min())
            ylim = [(min(0.0, e.min()) - pad, max(0.0, e.max()) + pad)]
            if col["rows"][1][1] == (0.0, 0.0):
                no_fit.append((1, 0))
        lines = [E_DEF, r"$E^\star_{h \to 0}$ (Pa): $E^\star$ at the same $\Delta x$ as $h \to 0$, "
                        r"from the modal analysis: the lowest mode of the same mesh, an eigenproblem.", ZETA_DEF]
        extra = []
        if arm in DX_PANEL:
            kind, arms = DX_PANEL[arm]
            extra = [lambda ax, kind=kind, arms=arms: dx_panel(ax, rows, kind, arms)]
            lines.append(T_DEF)
        if any(r["arm"] == arm for r in left_out):
            lines.append(r"$h$ = 5 ms is not shown: above $2\beta$ = 4.4 ms no $E^\star$, $\zeta^\star$ gives the mode.")
        n_panels = 2 + len(extra)
        flow_figure([col], [r"$E^\star / E^\star_{h \to 0} - 1$ (%)", r"$\zeta^\star / \zeta - 1$ (%)"],
                    [c.RESULTS / f"fig_flow_{arm}.{e}" for e in ("png", "pdf")], 5.0,
                    wrap("\n".join(lines), 170 if n_panels == 3 else 135),
                    [(4000, "4k"), (1000, "1k"), (500, "500"), (200, "200")],
                    point_colors=DX_COLOR, point_sizes=DX_SIZE, quadratic=True, show_rmse=False,
                    no_fit=no_fit, row_ylim={1: ylim}, horizontal=True, extra_panels=extra,
                    legend_entries=[(rf"$\Delta x$ = {k:g} mm", col_, DX_SIZE[k]) for k, col_ in DX_COLOR.items()
                                    if k in set(col["groups"])])
        print("wrote", c.RESULTS / f"fig_flow_{arm}.png")


def fit_powers(x, y, powers):
    """Least squares y = sum_k a_k x^k over the given powers; returns (a, R^2)."""
    A = np.stack([x ** k for k in powers], 1)
    a = np.linalg.lstsq(A, y, rcond=None)[0]
    r2 = 1 - np.sum((y - A @ a) ** 2) / np.sum((y - y.mean()) ** 2)
    return a, r2


SCHEME_NAME = {"mj_discrete": "discrete", "mj_euler": "Euler", "gs_fem_explicit": "explicit",
               "gs_fem_implicit": "implicit"}


def dx_panel(ax, rows, kind, arms):
    """E* / T(h) against dx, every h of the schemes that share a mass lumping, with the mesh's
    modes' value at each dx.

    Equal vertex masses integrate the kinetic energy with an error of first order in dx
    (p1.first_order_equal_mass); masses lumped by volume integrate linear fields exactly, so
    their error, like the stiffness's, is of second order.  The fit uses those powers.
    """
    import p1
    table = rf.theory_table()
    ax.set_xlim(0, 10.5)
    ax.set_xlabel(r"$\Delta x$ (mm)", fontsize=9)
    ax.axhline(0, color="#e4e3df", lw=1, zorder=0)
    sel = [r for r in rows if r["arm"] in arms]
    x = np.array([r["dx"] * 1e3 for r in sel])
    e = 100 * np.array([r["E_ratio"] / c.time_prediction(rf.ARMS[r["arm"]][1], r["h"])[0] - 1 for r in sel])
    S = 100 * np.array([r["S_theory"] - 1 for r in sel])
    dxs = np.array(rf.DXS) * 1e3
    St = 100 * (np.array([table[f"{d:.6f}_{kind}"]["S"] for d in rf.DXS]) - 1)
    # the theory is computed on each mesh, so it exists only at the cell sizes that divide the
    # block: rings, where a theory given by a formula is a line
    ax.plot(dxs, St, "o", ms=12, mfc="none", mec=INK, mew=1.2, zorder=2, label="modal analysis")
    if kind == "equal":
        powers, first = (1, 2), 100 * p1.first_order_equal_mass() / 1e3
        a, r2 = fit_powers(x, e, powers)
        text = (f"fit: $c_1 \\Delta x + c_2 \\Delta x^2$, $R^2$ = {r2:.4f}\n"
                f"$c_1$ = {num(a[0])} (theory {num(first)}), $c_2$ = {num(a[1])}")
    else:
        powers = (2, 3)
        a, r2 = fit_powers(x, e, powers)
        text = (f"fit: $c_2 \\Delta x^2 + c_3 \\Delta x^3$, $R^2$ = {r2:.4f}\n"
                f"$c_2$ = {num(a[0])}, $c_3$ = {num(a[1])}   (theory: $c_1$ = 0)")
    xd = np.linspace(0, 10.5, 200)
    ax.plot(xd, sum(ak * xd ** k for ak, k in zip(a, powers)), color=FIT_COLOR, lw=2, ls="--",
            zorder=1, label=text)
    n_h = len({r["h"] for r in sel})
    names = " and ".join(SCHEME_NAME[a_] for a_ in arms)
    ax.plot(x, e, "o", ms=5, color="#2a78d6", mec="white", mew=0.6, zorder=3,
            label=f"{len(sel)} runs of {names}, {n_h} steps $h$:\nwithin {pct(np.abs(e - S).max())} of the modal analysis")
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    ax.set_ylabel(r"$E^\star / (E\,T(h)) - 1$ (%)")


MPM_COLOR = {0.25: "#86b6ef", 0.5: "#5598e7", 1.0: "#1c5cab", 2.0: "#104281"}


def mpm_figure(rows):
    """MPM: E* and its own numerical damping against dx, one colour per step (a fraction of
    Genesis's suggested substep, 0.02 dx with dx in m), with mpm_linear.py's mode of the
    linearized step at every point, nothing fitted."""
    sel = [r for r in rows if r["arm"] == "gs_mpm"]
    path = c.RESULTS / "mpm_theory.json"
    theory = json.loads(path.read_text()) if path.exists() else []
    if not sel:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax in axes:
        style(ax)
        ax.set_xlim(0, 10.5)
        ax.set_xlabel(r"$\Delta x$ (mm)", fontsize=9)

    def of(pts, frac):
        return sorted((r for r in pts if abs(r["h"] / (0.02 * r["dx"]) - frac) < 0.01 * frac), key=lambda r: r["dx"])

    worst = [0.0, 0.0]
    for frac, color in MPM_COLOR.items():
        th, run = of(theory, frac), of(sel, frac)
        for k, (ax, get) in enumerate(((axes[0], lambda r: 100 * (r["E_ratio"] - 1)), (axes[1], lambda r: r["zeta"]))):
            if th:
                ax.plot([r["dx"] * 1e3 for r in th], [get(r) for r in th], "o", ms=12, mfc="none", mec=color,
                        mew=1.4, zorder=2)
            if run:
                ax.plot([r["dx"] * 1e3 for r in run], [get(r) for r in run], "o", ms=5, color=color, mec="white",
                        mew=0.6, zorder=3, label=rf"$h$ = {frac:g} $\times$ 0.02 $\Delta x$" if k == 0 else None)
            for r in run:
                t = [q for q in th if abs(q["dx"] - r["dx"]) < 1e-9]
                if t:
                    worst[k] = max(worst[k], abs(get(r) - get(t[0])))
    axes[0].plot([], [], "o", ms=12, mfc="none", mec=INK, mew=1.4, label="linearized step")
    axes[0].axhline(0, color="#e4e3df", lw=1, zorder=0)
    axes[1].axhline(c.ZETA0, color=INK, lw=1, ls=":", zorder=0)
    axes[1].text(0.2, c.ZETA0, "tofu's own damping, 0.10", va="bottom", fontsize=8.5, color=INK2)
    axes[0].set_ylabel(r"$E^\star / E - 1$ (%)")
    axes[1].set_ylabel(r"$\zeta_\mathrm{MPM}$")
    axes[0].set_title(f"Genesis 1.5.0, MPM: the stiffness\nruns within {pct(worst[0])} of the linearized step", fontsize=11)
    axes[1].set_title(f"the damping MPM adds, which no parameter removes\nruns within {worst[1]:.0e} of the linearized step",
                      fontsize=11)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.text(0.5, 0.01, E_DEF + "\n" + r"$\zeta_\mathrm{MPM}$ (dimensionless): the damping ratio of the mode "
             r"in MPM, which has no damping parameter."
             "\n" + r"Linearized step: one MPM step linearized about rest, its mode read as the runs' are.",
             ha="center", fontsize=8.5, color=INK2)
    axes[1].set_ylim(bottom=0)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    clear_legends(fig)
    for e_ in ("png", "pdf"):
        fig.savefig(c.RESULTS / f"fig_flow_mpm.{e_}", dpi=200)
    plt.close(fig)
    print("wrote", c.RESULTS / "fig_flow_mpm.png")


PBD_COLOR = {0.1: "#86b6ef", 0.25: "#5598e7", 0.5: "#2a78d6", 0.9: "#104281"}


def pbd_figure():
    """PBD: E* and its own numerical damping against dx, one colour per step (a fraction of
    h_max, the largest step at which zero compliance still reaches the continuum's frequency),
    with pbd_linear.py's linearized step at every point, nothing fitted."""
    path = c.RESULTS / "pbd.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax in axes:
        style(ax)
        ax.set_xlim(0, 10.5)
        ax.set_xlabel(r"$\Delta x$ (mm)", fontsize=9)
    worst = [0.0, 0.0]
    for frac, color in PBD_COLOR.items():
        sel = sorted((r for r in rows if abs(r["fraction"] - frac) < 1e-9), key=lambda r: r["dx"])
        if not sel:
            continue
        x = [r["dx"] * 1e3 for r in sel]
        for k, (ax, run, th, off, scale) in enumerate(((axes[0], "E_ratio", "E_theory", 1.0, 100.0),
                                                       (axes[1], "zeta", "zeta_theory", 0.0, 1.0))):
            ax.plot(x, [scale * (r[th] - off) for r in sel], "o", ms=12, mfc="none", mec=color, mew=1.4, zorder=2)
            ax.plot(x, [scale * (r[run] - off) for r in sel], "o", ms=5, color=color, mec="white", mew=0.6, zorder=3,
                    label=rf"$h$ = {frac:g} $h_\max$" if k == 0 else None)
            for r in sel:
                worst[k] = max(worst[k], 100 * abs(r[run] / r[th] - 1))
    axes[0].plot([], [], "o", ms=12, mfc="none", mec=INK, mew=1.4, label="linearized step")
    axes[0].set_yscale("log")
    ticks = [150, 200, 300, 500, 1000]
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([f"{t:g}" for t in ticks])
    axes[0].minorticks_off()
    axes[1].axhline(c.ZETA0, color=INK, lw=1, ls=":", zorder=0)
    axes[1].text(0.2, c.ZETA0, "tofu's own damping, 0.10", va="bottom", fontsize=8.5, color=INK2)
    axes[0].set_ylabel(r"$E^\star / E - 1$ (%)")
    axes[1].set_ylabel(r"$\zeta_\mathrm{PBD}$")
    axes[0].set_title(f"Genesis 1.5.0, PBD: the stiffness\nruns within {pct(worst[0])} of the linearized step", fontsize=11)
    axes[1].set_title(f"the damping PBD adds, which no parameter removes\nruns within {pct(worst[1])} of the linearized step",
                      fontsize=11)
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].set_ylim(0, 0.11)
    fig.text(0.5, 0.01,
             r"$E^\star$ (Pa): the Young's modulus whose edge compliance $1/(E^\star \Delta x)$ (m/N) gives the "
             r"block's lowest mode the continuum's frequency,"
             "\n" + r"14.4 Hz at $E$ = 11.3 kPa.  $\zeta_\mathrm{PBD}$ (dimensionless): the damping ratio of the mode "
             r"in PBD, which has no damping parameter."
             "\n" + r"$h_\max$ (s): past it no compliance is stiff enough, 1.03 ms at 10 mm, 0.25 ms at 2 mm.  "
             r"Genesis's defaults: 4 sweeps of the edges' stretch"
             "\n" + r"constraints, relaxation 0.1; its volume constraint does nothing (its correction carries "
             r"$|\nabla C|^2$ once too often)."
             "\n" + r"Linearized step: one PBD step linearized about rest, its mode read as the runs' are.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    clear_legends(fig)
    for e_ in ("png", "pdf"):
        fig.savefig(c.RESULTS / f"fig_flow_pbd.{e_}", dpi=200)
    plt.close(fig)
    print("wrote", c.RESULTS / "fig_flow_pbd.png")


def main():
    use_house_style()
    rows = json.loads((c.RESULTS / "flow.json").read_text())
    # Past the limit of the counterterm (backward Euler: h > 2 beta, 4.4 ms) no E*, zeta* gives the
    # continuum's mode, or the run is unstable; those points are left out.
    left_out = [r for r in rows if not r["misfit"] < MISFIT_MAX]
    for r in left_out:
        print(f"left out: {r['arm']} dx {r['dx'] * 1e3:.2f} mm, h {r['h'] * 1e3:g} ms, misfit {r['misfit']:.1e}")
    rows = [r for r in rows if r["misfit"] < MISFIT_MAX]
    scheme_figures(rows, left_out)
    mpm_figure(rows)
    pbd_figure()


if __name__ == "__main__":
    main()
