#!/usr/bin/env bash
. "$(dirname "$0")/lib.sh"
step=${1:-}
case "$step" in
    stage|receipts-adopt|receipts-verify|receipts-check|publish|announce|wait|verify|rollback) ;;
    *) echo "usage: weightd.sh stage|receipts-adopt|receipts-verify|receipts-check|publish|announce|wait|verify|rollback"; exit 2;;
esac
weightd_changes || { echo "OLD_WEIGHTD = NEW_WEIGHTD ($NEW_WEIGHTD): this release does not change weightd"; exit 2; }
need BUILD_HOST WEIGHTD_BUNDLE_BUILD NEW_RECEIPT NEW_WARM
start_log "weightd-$step"
STAGE_BIN="\$HOME/$RELEASE_WEIGHTD_STAGE/bin"
ROLLBACK_COPY="release-staging/weightd-rollback/sparkpipe_weightd.$OLD_WEIGHTD"

per_rank() {
    local h i=0 pack
    for h in $RELEASE_NODES; do
        pack="\$HOME/sparkdata/$RELEASE_ROOT_NAME/${RELEASE_RANK_PACK//@hex@/$(printf %x $i)}"
        $RELEASE_SSH "$h" "${1//@PACK@/$pack}" 2>&1 | sed "s/^/$h: /" || echo "$h: FAILED rc=$?"
        i=$((i + 1))
    done
}

held_gate() {
    nodes --expect hold=yes --expect eng_n=0 --expect others=none "$@" --log "$log.nodes-gate" > /dev/null \
        || { nodes_bad "$log.nodes-gate"; die "every node must be held with no residentd at all (hold.sh; stop other lanes; the agent swaps weightd only when no engine runs)"; }
}

core_state() {
    hub "echo \$(tr -d '[:space:]' < ~/release/core/WEIGHTSD_BIN) \$(sha256sum < ~/release/core/bin/sparkpipe_weightd | cut -c1-16)"
}

build_check() {
    local b
    b=$($RELEASE_SSH "$BUILD_HOST" "cd $WEIGHTD_BUNDLE_BUILD && echo \$(git rev-parse HEAD) \$(sha256sum < build/sparkpipe_weightd | cut -c1-16) \$(sha256sum < build/weightd_receipt | cut -c1-16) \$(sha256sum < build/weightd_warm | cut -c1-16)")
    [ "$b" = "$SHA $NEW_WEIGHTD $NEW_RECEIPT $NEW_WARM" ] || die "bundle build is not the verified one: '$b' want '$SHA $NEW_WEIGHTD $NEW_RECEIPT $NEW_WARM'"
    echo "OK   bundle build $BUILD_HOST:$WEIGHTD_BUNDLE_BUILD = $b"
}

case "$step" in
    stage)
        build_check
        for h in $RELEASE_NODES; do
            $RELEASE_SSH "$BUILD_HOST" "ssh -o BatchMode=yes $h 'mkdir -p $STAGE_BIN' && scp -q $WEIGHTD_BUNDLE_BUILD/build/sparkpipe_weightd $WEIGHTD_BUNDLE_BUILD/build/weightd_receipt $WEIGHTD_BUNDLE_BUILD/build/weightd_warm $h:$RELEASE_WEIGHTD_STAGE/bin/" \
                && $RELEASE_SSH "$h" "grep -q $RELEASE_WEIGHTD_STAGE ~/KEEP 2>/dev/null || printf '%s\t%s\t%s\t%s\n' \$HOME/$RELEASE_WEIGHTD_STAGE release-$S7 KEEP 'weightd bundle binaries (release kit weightd.sh stage)' >> ~/KEEP; echo staged \$(sha256sum < $STAGE_BIN/sparkpipe_weightd | cut -c1-16)" | sed "s/^/$h: /"
        done
        nodes --expect stage_wd="$NEW_WEIGHTD" --expect stage_rc="$NEW_RECEIPT" --expect stage_warm="$NEW_WARM" --log "$log.nodes" | tail -n 1
        [ "${PIPESTATUS[0]}" -eq 0 ] || die "bundle not staged on every node"
        ;;
    receipts-adopt)
        per_rank "$STAGE_BIN/weightd_receipt adopt @PACK@; echo rc=\$?" | tee "$log.out"
        grep -q "HASH MISMATCH" "$log.out" && die "HASH MISMATCH: stop and investigate that pack"
        grep -q "FAILED rc=" "$log.out" && die "ssh failure"
        ;;
    receipts-verify)
        per_rank "$STAGE_BIN/weightd_receipt check @PACK@ >/dev/null 2>&1 && echo receipt-valid || { nice -n 19 ionice -c3 $STAGE_BIN/weightd_receipt verify @PACK@; echo rc=\$?; }" | tee "$log.out"
        grep -q "HASH MISMATCH\|FAILED rc=" "$log.out" && die "verify failed"
        ;;
    receipts-check)
        nodes --expect receipt=rc0 --log "$log.nodes" | tail -n 1
        [ "${PIPESTATUS[0]}" -eq 0 ] || die "a rank pack has no valid receipt (receipts-adopt, receipts-verify)"
        ;;
    publish)
        build_check
        held_gate --expect wd="$OLD_WEIGHTD"
        read -r announced core <<< "$(core_state)"
        [ "$announced $core" = "$OLD_WEIGHTD $OLD_WEIGHTD" ] || die "hub core is WEIGHTSD_BIN=$announced bin=$core, want $OLD_WEIGHTD for both"
        hub "mkdir -p ~/release-staging/weightd-rollback && r=~/$ROLLBACK_COPY && { [ -f \$r ] || cp ~/release/core/bin/sparkpipe_weightd \$r; } && test \"\$(sha256sum < \$r | cut -c1-16)\" = $OLD_WEIGHTD && echo rollback copy \$r ok" || die "no verified $OLD_WEIGHTD rollback copy on $RELEASE_HUB"
        $RELEASE_SSH "$BUILD_HOST" "scp -q $WEIGHTD_BUNDLE_BUILD/build/sparkpipe_weightd $RELEASE_HUB:release/core/bin/sparkpipe_weightd.new && scp -q $WEIGHTD_BUNDLE_BUILD/build/sparkpipe_mesh_status $RELEASE_HUB:release/core/bin/sparkpipe_mesh_status.new" || die "copy to the hub failed"
        hub "cd ~/release/core && exec 9>\$HOME/release/.core.publish.lock && flock 9 && test \"\$(sha256sum < bin/sparkpipe_weightd.new | cut -c1-16)\" = $NEW_WEIGHTD && chmod 755 bin/sparkpipe_weightd.new bin/sparkpipe_mesh_status.new && mv bin/sparkpipe_mesh_status.new bin/sparkpipe_mesh_status && mv bin/sparkpipe_weightd.new bin/sparkpipe_weightd && find bin -type f | sort | xargs sha256sum > MANIFEST.tmp && mv MANIFEST.tmp MANIFEST && echo core bin \$(sha256sum < bin/sparkpipe_weightd | cut -c1-16) WEIGHTSD_BIN still \$(cat WEIGHTSD_BIN)" || die "core publish failed"
        echo "WEIGHTD PUBLISHED (not announced). Next: weightd.sh announce"
        ;;
    announce)
        held_gate
        read -r announced core <<< "$(core_state)"
        [ "$core" = "$NEW_WEIGHTD" ] || die "hub core bin is $core, want $NEW_WEIGHTD (weightd.sh publish first)"
        hub 'bash -s' < "$REPO/tools/weightsd_announce.sh" || die "announce failed"
        read -r announced core <<< "$(core_state)"
        [ "$announced" = "$NEW_WEIGHTD" ] || die "WEIGHTSD_BIN is $announced after announce"
        echo "WEIGHTD ANNOUNCED $NEW_WEIGHTD. Next: weightd.sh wait"
        ;;
    wait)
        nodes --expect hold=yes --expect eng_n=0 --expect wd_inst="$NEW_WEIGHTD" --expect wd="$NEW_WEIGHTD" --expect wd_n=1 --expect wd_other=0 --wait-min 5 --interval 5 --quiet-poll --log "$log.nodes" | tail -n 1
        [ "${PIPESTATUS[0]}" -eq 0 ] || die "new weightd not running on every node after 5 min ('journalctl --user -u fleet-agent', ~/weightd.log; ss -ltnp | grep 61900). Roll back with weightd.sh rollback while still held"
        echo "WEIGHTD RUNNING $NEW_WEIGHTD $NODE_COUNT/$NODE_COUNT. Next: publish.sh"
        ;;
    verify)
        per_rank "echo weightd=\$(sha256sum < /proc/\$(pgrep -o -f 'sparkdata/weightd/sparkpipe_[w]eightd')/exe | cut -c1-16) verify=\$(grep -o 'pack-verify path=[^ ]*/'\$(basename @PACK@)' mode=[a-z0-9]*' \$HOME/weightd.log | tail -1 | sed 's/.*mode=/mode=/') reclaim_pack=\$($STAGE_BIN/weightd_warm /tmp/spark_weightd.sock --reclaim-pack 0000000000000000000000000000000000000000000000000000000000000000 2>&1 | grep -o 'freed=[0-9]* arenas=[0-9]* busy=[0-9]*' || echo unsupported)" | tee "$log.out"
        n=$(grep -c "weightd=$NEW_WEIGHTD .*reclaim_pack=freed=0 arenas=0 busy=0" "$log.out")
        echo "verify: $n/$NODE_COUNT on $NEW_WEIGHTD with RECLAIM_PACK live; receipt mode on $(grep -c "verify=mode=receipt" "$log.out")/$NODE_COUNT"
        [ "$n" -eq "$NODE_COUNT" ] || die "weightd verify failed"
        ;;
    rollback)
        held_gate
        hub "cd ~/release/core && exec 9>\$HOME/release/.core.publish.lock && flock 9 && test \"\$(sha256sum < ~/$ROLLBACK_COPY | cut -c1-16)\" = $OLD_WEIGHTD && cp ~/$ROLLBACK_COPY bin/sparkpipe_weightd.new && chmod 755 bin/sparkpipe_weightd.new && mv bin/sparkpipe_weightd.new bin/sparkpipe_weightd && find bin -type f | sort | xargs sha256sum > MANIFEST.tmp && mv MANIFEST.tmp MANIFEST && test \"\$(sha256sum < bin/sparkpipe_weightd | cut -c1-16)\" = $OLD_WEIGHTD && printf '%s\n' $OLD_WEIGHTD > WEIGHTSD_BIN.tmp && mv WEIGHTSD_BIN.tmp WEIGHTSD_BIN && echo rolled back to $OLD_WEIGHTD" || die "core rollback failed"
        nodes --expect hold=yes --expect eng_n=0 --expect wd_inst="$OLD_WEIGHTD" --expect wd="$OLD_WEIGHTD" --expect wd_n=1 --wait-min 5 --interval 5 --quiet-poll --log "$log.nodes" | tail -n 1
        [ "${PIPESTATUS[0]}" -eq 0 ] || die "old weightd not running on every node after 5 min"
        ;;
esac
echo "WEIGHTD $step DONE $(date -u +%H:%M:%SZ). log $log"
