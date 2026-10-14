#!/usr/bin/env bash
# Start / stop the recognition, arm-state and simulator services on robot1.
#
#   ./run.sh start [vision|arm|sim|all]     (default: all)
#   ./run.sh stop  [vision|arm|sim|all]
#   ./run.sh status
#   ./run.sh log   vision|arm|sim
#
# vision (8090) serves the LAN 192.168.32.0/24 and the private 172.16.0.0/12 (allow-list in VISION_ARGS); arm (8091) and sim (8092)
# bind to 127.0.0.1 only and are reached through the vision page. Open http://192.168.32.20:8090/
# from a machine on that subnet, or use an SSH forward:
#   ssh -N -L 8090:127.0.0.1:8090 win10.hpcs-sb1124.sec.llhi.org   then open http://127.0.0.1:8090/
# (the page proxies the arm service on :8091 and the simulator on :8092, so one forward is enough)
#
# Options are passed through environment variables:
#   VISION_ARGS="--device 18443"   ARM_ARGS="--port /dev/ttyACM0"   SIM_ARGS=""
# Nothing here sends commands to the motors: arm_state.py only requests feedback frames.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN="$ROOT/run"
mkdir -p "$RUN"

declare -A DIR=([vision]="$ROOT/oak" [arm]="$ROOT/rebot" [sim]="$ROOT/sim")
declare -A PY=([vision]="$ROOT/oak/.venv/bin/python" [arm]="$ROOT/rebot/reBotArm_control_py/.venv/bin/python" [sim]="$ROOT/sim/.venv/bin/python")
declare -A SCRIPT=([vision]="vision.py" [arm]="arm_state.py" [sim]="sim_server.py")
declare -A ARGS=([vision]="${VISION_ARGS:---device 18443 --host 0.0.0.0 --allow-from 127.0.0.1/32,192.168.32.0/24,172.16.0.0/12}" [arm]="${ARM_ARGS:---port /dev/ttyACM0}" [sim]="${SIM_ARGS:-}")
declare -A PORT=([vision]=8090 [arm]=8091 [sim]=8092)
# numpy's OpenBLAS starts one busy thread per core (24) for every matrix call; the simulator's grasp
# planning / pose generation then took all cores and the arm service missed its 50 Hz feedback for
# 0.3-0.8 s (CAN-loss faults, 2026-10-14, each time sim or vision was busy). Few BLAS threads for all,
# and vision / sim at a lower priority than the arm service.
declare -A THREADS=([vision]=2 [arm]=1 [sim]=2)
declare -A NICE=([vision]=5 [arm]=0 [sim]=5)

alive() { [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }

start() {
  local s=$1
  if alive "$s"; then echo "$s: already running (pid $(cat "$RUN/$s.pid"))"; return; fi
  cd "${DIR[$s]}" || return 1
  # shellcheck disable=SC2086
  MUJOCO_GL=egl OMP_NUM_THREADS=${THREADS[$s]} OPENBLAS_NUM_THREADS=${THREADS[$s]} MKL_NUM_THREADS=${THREADS[$s]} \
    setsid nice -n "${NICE[$s]}" "${PY[$s]}" -u "${SCRIPT[$s]}" ${ARGS[$s]} >"$RUN/$s.log" 2>&1 &
  echo $! >"$RUN/$s.pid"
  echo "$s: started pid $! (log: $RUN/$s.log, http://127.0.0.1:${PORT[$s]}/)"
}

stop() {
  local s=$1
  if ! alive "$s"; then echo "$s: not running"; rm -f "$RUN/$s.pid"; return; fi
  local pid; pid=$(cat "$RUN/$s.pid")
  if [ "$s" = arm ] && [ "${FORCE:-0}" != 1 ]; then
    st=$(curl -s --max-time 2 http://127.0.0.1:8091/state.json | python3 -c "import json,sys; print(json.load(sys.stdin).get('control',{}).get('state','read_only'))" 2>/dev/null)
    if [ -n "$st" ] && [ "$st" != read_only ]; then
      echo "arm: motors are ENABLED ($st). Turn torque off from the page first (or FORCE=1 to leave them enabled)."; return 1
    fi
  fi
  kill -TERM "$pid"            # SIGTERM = the same clean shutdown as Ctrl-C (releases the OAK / serial port)
  for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
  if kill -0 "$pid" 2>/dev/null && [ "$s" = arm ] && [ "${FORCE:-0}" = 1 ]; then kill -TERM "$pid"; sleep 1; fi
  if kill -0 "$pid" 2>/dev/null; then echo "$s: did not exit, sending SIGKILL"; kill -KILL "$pid"; fi
  rm -f "$RUN/$s.pid"; echo "$s: stopped"
}

which=${2:-all}
case "$which" in all) set_=(arm sim vision) ;; vision|arm|sim) set_=("$which") ;; *) echo "unknown service $which"; exit 2 ;; esac
case "${1:-}" in
  start) for s in "${set_[@]}"; do start "$s"; done ;;
  stop) for s in "${set_[@]}"; do stop "$s"; done ;;
  restart) for s in "${set_[@]}"; do stop "$s"; start "$s"; done ;;
  status) for s in vision arm sim; do if alive "$s"; then echo "$s: running pid $(cat "$RUN/$s.pid") port ${PORT[$s]}"; else echo "$s: stopped"; fi; done ;;
  log) tail -n 40 "$RUN/$which.log" ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
