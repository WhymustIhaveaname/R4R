#!/bin/bash
# The whole sweep: three arms x five nominal control rates, a few chains at a time on whichever
# GPU has room, then each arm's cross-seeding, then the merge.  B is fitted with three
# parameters, as it is in MuJoCo, so it takes no --delay.
cd "$(dirname "$0")"
PY=/data/mjwarp/venv/bin/python
LOGS=/data/mjwarp/logs
MAX=${MAX:-4}
mkdir -p $LOGS
running() { pgrep -u "$(id -u)" -cf "^$PY run_single.py --arm"; }
free_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | awk -F', ' '$2 > 2000 {print $1; exit}'
}
launch() {   # log name, then the arguments
  local name=$1; shift
  while [ "$(running)" -ge "$MAX" ] || [ -z "$(free_gpu)" ]; do sleep 20; done
  gpu=$(free_gpu)
  CUDA_VISIBLE_DEVICES=$gpu WARP_CACHE_PATH=/data/mjwarp/cache$gpu \
    setsid nohup $PY run_single.py "$@" > $LOGS/$name.log 2>&1 < /dev/null &
  sleep 30
}
opts() { [ "$1" = B ] && echo "" || echo "--delay"; }
for arm in A B C; do
  for fc in 10 20 50 100 200; do launch ${arm}_${fc} --arm $arm --fctrl $fc $(opts $arm); done
done
while [ "$(running)" -gt 0 ]; do sleep 30; done
for arm in A B C; do launch cross_$arm --arm $arm --cross-seed $(opts $arm); done
while [ "$(running)" -gt 0 ]; do sleep 30; done
$PY run_single.py --merge --delay
$PY run_single.py --merge
