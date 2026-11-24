#!/bin/bash
# The SuperDex sweep, as shipped (FP32, Mochi's own Newton tolerances): the rod, which is the
# pendulum as built, and a lever carrying J/100, which shows what is left once Mochi's
# maximal-coordinate link inertia is taken out.  The rod runs A and C only: B's inertia flows
# below the rod's own, and Mochi's armature cannot be negative.  Then the behaviour sweeps of
# the rod, each scheme's cross-seeding once its five chains are done, and the merge.
cd "$(dirname "$0")"
LOGS=${LOGS:-/data/youran/superdex_logs}
MAX=${MAX:-10}
mkdir -p "$LOGS"
PY=../.venv/bin/python   # the project's venv, with the superdex extra (uv sync --extra superdex)
running() { pgrep -u "$(id -u)" -cf "^\.\./\.venv/bin/python run_(single|gap_single)\.py"; }
launch() {   # log name, then the script and its arguments
  local name=$1; shift
  while [ "$(running)" -ge "$MAX" ]; do sleep 30; done
  # Mochi prints every Newton blow-up; the simulator already reports those rollouts as failed.
  nice -n 10 $PY "$@" 2>&1 | grep --line-buffered -v -e 'Solution explosion' \
    -e 'newton_solver.cpp' -e '^\s*$' > "$LOGS/$name.log" &
  sleep 2
}
pairs="rod:A rod:C hundredth:A hundredth:B hundredth:C"
for pair in $pairs; do
  link=${pair%:*}; arm=${pair#*:}
  for fc in 10 20 50 100 200; do
    launch ${link}_${arm}_$fc run_single.py --setting default --link $link --arm $arm --fctrl $fc
  done
done
launch gap_rod_A run_gap_single.py --link rod --arm A
launch gap_rod_C run_gap_single.py --link rod --arm C
while [ "$(pgrep -u "$(id -u)" -cf "run_single.py --setting")" -gt 0 ]; do sleep 60; done
for pair in $pairs; do
  link=${pair%:*}; arm=${pair#*:}
  launch cross_${link}_${arm} run_single.py --setting default --link $link --arm $arm --cross-seed
done
wait
$PY run_single.py --merge
