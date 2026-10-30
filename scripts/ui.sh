#!/usr/bin/env bash
# Background port-forwards for Prometheus (9090) and Grafana (3000).
#   scripts/ui.sh start | stop | status
# Each forward runs in a loop: if its pod restarts and the forward drops, it reconnects.
set -u
NS=quake
LOG=/tmp/quake-ui
mkdir -p "$LOG"

forward() {   # name local:remote
  while true; do
    kubectl -n "$NS" port-forward "svc/$1" "$2" >>"$LOG/$1.log" 2>&1
    sleep 2
  done
}

case "${1:-status}" in
  start)
    "$0" stop >/dev/null
    forward prometheus 9090:9090 & echo $! > "$LOG/prometheus.pid"
    forward grafana    3000:3000 & echo $! > "$LOG/grafana.pid"
    sleep 2; "$0" status ;;
  stop)
    for n in prometheus grafana; do
      [ -f "$LOG/$n.pid" ] && kill "$(cat "$LOG/$n.pid")" 2>/dev/null && rm -f "$LOG/$n.pid"
    done
    pkill -f "kubectl -n $NS port-forward" 2>/dev/null
    echo stopped ;;
  status)
    for p in 9090 3000; do
      curl -s -o /dev/null -w "localhost:$p -> HTTP %{http_code}\n" "http://localhost:$p/" || echo "localhost:$p -> down"
    done ;;
esac
