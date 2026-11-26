#!/bin/bash
# The whole sweep: the 25 chains (three schemes in MuJoCo, two in Genesis, five control rates
# each) side by side, one CPU core each, then the merge.  About 30 min on 25 cores.
# Run it again to resume.
cd "$(dirname "$0")"
PY=../.venv/bin/python   # the project's venv, with the genesis extra (uv sync --extra genesis)
LOGS=${LOGS:-/tmp/exp3_logs}
mkdir -p "$LOGS"
run() {   # engine, scheme, control rate; Genesis warns about the rod's thin inertia every build
  nice -n 10 $PY run_flow.py --engine $1 --arm $2 --fctrl $3 2>&1 \
    | grep --line-buffered -v 'dubious inertia' > "$LOGS/$1_$2_$3.log" &
}
for fc in 10 20 50 100 200; do
  for arm in euler_explicit rk4_explicit implicitfast_native; do
    run mujoco $arm $fc
  done
  for arm in euler_explicit implicitfast_native; do
    run genesis $arm $fc
  done
done
wait
$PY run_flow.py --merge
