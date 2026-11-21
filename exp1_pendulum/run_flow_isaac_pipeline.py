"""Experiment 1's MuJoCo sweep, run through the Isaac pipeline (isaac/run_single.py) instead.

Everything is run_flow.py's except what run_single.py does differently: 1 s of command
instead of 4 s, the error sampled once per control step instead of once per physics step,
the grid starting at h = 0.5 ms, a command delay fitted in every arm, and the Jacobian taken
by the batched central difference.  If the Isaac flow is ragged and this one is not, the
pipeline is not what makes it ragged.

Usage:  uv run python run_flow_isaac_pipeline.py [arm ...]     (default: A)
Writes results/flow_mujoco_isaac_pipeline.json.
"""

import json
import multiprocessing
import sys
from pathlib import Path

import run_flow as rf
from common import Sim, ground_truth, ref_sequence

T_FIT = 1.0  # s, run_single.py's --t-fit default
OUT = Path(__file__).parent / "results" / "flow_mujoco_isaac_pipeline.json"

rf.WITH_DELAY = True
rf.H_TARGET = rf.H_TARGET[rf.H_TARGET >= 0.5e-3 - 1e-12]
rf.ref_sequence = lambda f_ctrl: ref_sequence(f_ctrl, t_end=T_FIT)
rf.ground_truth = lambda refs, f_ctrl, dec: ground_truth(refs, f_ctrl, dec)[dec - 1::dec]


class ControlRateSim(Sim):
    """Isaac reads the joint once per control step, so only those samples enter the fit."""

    def rollout(self, refs, f_ctrl, dec, *params, **kw):
        out = super().rollout(refs, f_ctrl, dec, *params, **kw)
        return None if out is None else out[dec - 1::dec]


rf.make_sim = ControlRateSim
rf.make_batch = lambda sim, refs, f_ctrl, dec: (
    lambda rows: [sim.rollout(refs, f_ctrl, dec, *row) for row in rows])

if __name__ == "__main__":
    arms = sys.argv[1:] or ["A"]
    jobs = [(arm, fc) for arm in arms for fc in rf.F_CTRL]
    with multiprocessing.get_context("fork").Pool(len(jobs)) as pool:
        rows = [r for rs in pool.map(rf.chain, jobs) for r in rs]
    OUT.write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {len(rows)} fits to {OUT}")
