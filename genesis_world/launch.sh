#!/bin/bash
# The Genesis sweep, as shipped (FP32): the ten chains of Euler + explicit PD and implicitfast +
# native PD side by side, one CPU core each, then each scheme's cross-seeding once its five
# chains are done, and the merge.  approximate_implicitfast, Genesis's default, is not swept:
# on this pendulum, which has no constraint, it is implicitfast to round-off (sanity_check.py).
# Run it again to resume.
cd "$(dirname "$0")"
PY=../.venv/bin/python   # the project's venv, with the genesis extra (uv sync --extra genesis)
LOGS=${LOGS:-/data/youran/genesis_logs}
ARMS=${ARMS:-"euler_explicit implicitfast_native"}
mkdir -p "$LOGS"
run() {   # log name, then the arguments; Genesis warns about the rod's thin inertia every build
  local name=$1; shift
  nice -n 10 $PY run_single.py "$@" 2>&1 | grep --line-buffered -v 'dubious inertia' \
    > "$LOGS/$name.log" &
}
for arm in $ARMS; do
  for fc in 10 20 50 100 200; do
    run ${arm}_$fc --arm $arm --fctrl $fc
  done
done
wait
for arm in $ARMS; do
  run cross_$arm --arm $arm --cross-seed
done
wait
$PY run_single.py --merge
