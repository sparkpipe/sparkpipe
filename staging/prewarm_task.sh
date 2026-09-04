#!/bin/bash
# glm53 weightd prewarm — respawn fresh daemons 16/16 and load rank packs
# into arenas OUTSIDE any mesh, so waves attach warm in seconds.
# Rank is resolved in THIS local shell (no remote arithmetic) and passed literal.
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
NODES="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
REV=84c6a6aa9497188e15a635ba793b0f95a79b1033
LOG=/tmp/prewarm_task.log
: > "$LOG"

rank_of() {
    case "$1" in
        spark0) echo 0;;  spark1) echo 1;;  spark2) echo 2;;  spark3) echo 3;;
        spark4) echo 4;;  spark5) echo 5;;  spark6) echo 6;;  spark7) echo 7;;
        spark8) echo 8;;  spark9) echo 9;;  sparka) echo 10;; sparkb) echo 11;;
        sparkc) echo 12;; sparkd) echo 13;; sparke) echo 14;; sparkf) echo 15;;
    esac
}

for h in $NODES; do
    r=$(rank_of "$h")
    (
        timeout 500 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "
            rr=\$HOME/sparkdata/glm5_next.tp16
            for p in \$(pgrep -f 'sparkpipe_weight[d]'); do
                [ \"\$(readlink /proc/\$p/exe 2>/dev/null)\" = \"\$rr/bin/sparkpipe_weightd\" ] && kill -TERM \$p
            done
            sleep 2
            cd \"\$rr\" && nohup ./bin/sparkpipe_weightd --socket /tmp/spark_weightd.sock >/tmp/weightd.out 2>&1 &
            for i in \$(seq 1 30); do [ -S /tmp/spark_weightd.sock ] && break; sleep 0.5; done
            [ -S /tmp/spark_weightd.sock ] || { echo '$h NO-SOCKET'; cat /tmp/weightd.out; exit 1; }
            HASH=\$(awk '{print \$1}' \"\$rr/packs/glm5_next_stage.tp16.rank${r}.g5nsp.sha256\" 2>/dev/null)
            SPARK_WEIGHTD_SOCKET=/tmp/spark_weightd.sock SPARK_WEIGHTD_PACK_SHA256=\$HASH \
                timeout 420 \$HOME/weightdctl load \
                \"\$rr/packs/glm5_next_stage.tp16.rank${r}.g5nsp\" glm5_next $REV 2>&1 | tail -3
            echo \"$h rank${r} rc=\$?\"
        " >> "$LOG" 2>&1
    ) &
done
wait
echo "PREWARM-DONE sockets=$(grep -L . /dev/null 2>/dev/null; echo)" >> "$LOG"
tail -5 "$LOG"
