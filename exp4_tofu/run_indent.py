"""The ball press's runs, one point per parts file: Genesis's force on the (h, dx) grid.

  python run_indent.py --count
  python run_indent.py --task i      # a Slurm array index
  python run_indent.py --merge

Each point records the force at every depth of indentation.DEPTHS with the nominal E.  Genesis's
explicit FEM couples through the legacy coupler, its implicit FEM through SAP.  MuJoCo's Euler
runs only on 5 mm cells, the only ones above the speed cutoff, with MuJoCo's default contact and
a hard one, a tenth of its time constant; the stored mj_discrete runs had both.
"""

import argparse
import json
import os
import time

import common as c
import indentation as ind

PARTS = c.RESULTS / "indent_parts"
DXS = (0.01, 0.005, 0.01 / 3, 0.0025, 0.002)
STEPS = (2.5e-4, 5e-4, 1e-3, 2e-3)
GS_FRACTIONS = (0.25, 0.5, 0.85)   # of the explicit FEM's stability limit at rest
CONTACTS = {"default": dict(solref=(0.02, 1.0), solimp=(0.9, 0.95, 0.001, 0.5, 2)),
            "hard": dict(solref=(0.002, 1.0), solimp=(0.99, 0.99, 0.001, 0.5, 2))}
MJ_EULER_H = 4.5e-5   # s, on 5 mm cells: 0.026x real time on one core


def tasks():
    import run_flow as rf
    table = rf.theory_table()
    out = [("gs_fem_implicit", dx, h, "sap") for dx in DXS for h in STEPS]
    out += [("gs_fem_explicit", dx, float(f"{f * rf.h_max('gs_fem_explicit', dx, table):.4g}"), "legacy")
            for dx in DXS for f in GS_FRACTIONS]
    out = sorted(out, key=lambda t: (t[1], t[2]))    # the finest cells and steps first
    return out + [("mj_euler", 0.005, MJ_EULER_H, k) for k in CONTACTS]


def run_point(task):
    arm, dx, h, contact = task
    out = PARTS / f"{arm}_{contact}_{dx * 1e3:.3f}mm_{h * 1e6:.0f}us.json"
    if out.exists():
        return out.name
    t0 = time.time()
    row = {"arm": arm, "dx": dx, "h": h, "contact": contact, "depths": ind.DEPTHS.tolist()}
    if arm.startswith("mj"):
        from mujoco_sim import MjIndent
        row["force"] = MjIndent(arm, dx, h, ind.RADIUS, **CONTACTS[contact]).forces().tolist()
    else:
        from genesis_sim import GsIndent
        row["force"] = GsIndent(arm, dx, h, ind.RADIUS).forces().tolist()
    row["seconds"] = time.time() - t0
    os.listdir(PARTS.parent)   # an NFS client may cache that the directory is missing
    PARTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=1))
    return out.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        rows = [json.loads(p.read_text()) for p in sorted(PARTS.glob("*.json"))]
        (c.RESULTS / "indent.json").write_text(json.dumps(rows, indent=1))
        print("merged", len(rows))
        return
    if args.count:
        print(len(tasks()))
        return
    print("done", run_point(tasks()[args.task]), flush=True)


if __name__ == "__main__":
    main()
