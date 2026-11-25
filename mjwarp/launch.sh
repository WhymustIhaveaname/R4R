#!/bin/bash
# The whole sweep: one process per arm (see run_single.py), each on whichever GPU has room, then
# the merge.  RK4 is fitted with three parameters, as it is in MuJoCo, so it takes no --delay.
# Run it again to resume.
cd "$(dirname "$0")"
PY=/data/mjwarp/venv/bin/python
LOGS=/data/mjwarp/logs
mkdir -p $LOGS
free_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | awk -F', ' '$2 > 2000 {print $1; exit}'
}
for arm in euler_explicit rk4_explicit implicitfast_native; do
  [ "$arm" = rk4_explicit ] && opts="" || opts="--delay"
  while [ -z "$(free_gpu)" ]; do sleep 20; done
  gpu=$(free_gpu)
  CUDA_VISIBLE_DEVICES=$gpu WARP_CACHE_PATH=/data/mjwarp/cache$gpu \
    setsid nohup $PY run_single.py --arm $arm $opts > $LOGS/$arm.log 2>&1 < /dev/null &
  sleep 30
done
while pgrep -u "$(id -u)" -f "^$PY run_single.py --arm" > /dev/null; do sleep 30; done
$PY run_single.py --merge --delay
$PY run_single.py --merge
