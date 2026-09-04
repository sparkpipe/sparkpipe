#!/usr/bin/env bash
# Run ON a spark host: report the 100G RDMA fabric state relevant to the
# dsv4-pro host-rdma deployment. usage: fabric_probe_pro.sh SPARK_ALIAS
alias_name="${1:?usage: fabric_probe_pro.sh SPARK_ALIAS}"
g100="$(getent hosts "${alias_name}" | awk '{print $1}' | head -1)"
r200="$(ip -br addr show 2>/dev/null | awk '$1 ~ /enp/ && $3 ~ /10\.10\.200/ {print $3}' | head -1)"
active_ports="$(ibv_devinfo 2>/dev/null | grep -c 'PORT_ACTIVE')"
gids="$(ibv_devinfo -v 2>/dev/null | grep -o '::ffff:[0-9.]*' | tr '\n' ' ')"
busy_ports="$(ss -tln 2>/dev/null | grep -cE ':20480 |:6462[0-9] |:6463[0-5] ' || echo 0)"
residentd="$(pgrep -c -f 'sparkpipe_model_residentd' 2>/dev/null || echo 0)"
echo "alias=${alias_name} resolved=${g100} rail200=${r200} active_verbs_ports=${active_ports} busy_ports=${busy_ports} residentd_procs=${residentd}"
echo "gids: ${gids}"
