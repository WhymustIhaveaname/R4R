#!/bin/bash
# The whole Isaac run: one process per scheme (see run_single.py), each on whichever GPU has
# room for one more Isaac app, then the merge.  Run it again to resume.
cd "$(dirname "$0")"
PY=${ISAAC_VENV:-/data/isaac/venv}/bin/python
LOGS=${LOGS:-/data/isaac/logs}
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$LOGS"
free_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | awk -F', ' '$2 > 3000 {print $1; exit}'
}
for arm in A B C; do
  while [ -z "$(free_gpu)" ]; do sleep 30; done
  CUDA_VISIBLE_DEVICES=$(free_gpu) setsid nohup $PY run_single.py --arm $arm \
    > "$LOGS/$arm.log" 2>&1 < /dev/null &
  sleep 60   # let it take its memory before the next one looks
done
while pgrep -u "$(id -u)" -f "^$PY run_single.py --arm" > /dev/null; do sleep 60; done
$PY run_single.py --merge
