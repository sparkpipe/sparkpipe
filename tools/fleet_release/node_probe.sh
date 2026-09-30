rr=$HOME/sparkdata/$ROOT_NAME
h=$(hostname)
pack=$rr/$RANK_PACK
stage=$HOME/$WEIGHTD_STAGE/bin
eng=""
eng_n=0
others=""
for x in $(pgrep -f "bin/sparkpipe_model_[r]esidentd"); do
    e=$(readlink /proc/$x/exe 2>/dev/null) || continue
    e=${e% (deleted)}
    [ "${e##*/}" = sparkpipe_model_residentd ] || continue
    d=$(readlink /proc/$x/cwd) || continue
    if [ "$d" = "$rr" ]; then
        eng_n=$((eng_n + 1))
        [ -z "$eng" ] && eng=$x
    else
        others="$others${others:+,}$d"
    fi
done
if [ -n "$eng" ]; then
    exe=$(sha256sum < /proc/$eng/exe | cut -c1-16)
    up=$(ps -o etimes= -p $eng | tr -d ' ')
else
    exe=none
    up=0
fi
w=$(pgrep -o -f "sparkdata/weightd/sparkpipe_[w]eightd")
if [ -n "$w" ]; then
    wd=$(sha256sum < /proc/$w/exe | cut -c1-16)
    wd_up=$(ps -o etimes= -p $w | tr -d ' ')
else
    wd=none
    wd_up=0
fi
wd_n=$(pgrep -c -f "sparkdata/weightd/sparkpipe_[w]eightd")
wd_other=$(( $(pgrep -c -f "sparkpipe_[w]eightd") - wd_n ))
wd_inst=$(sha256sum < $HOME/sparkdata/weightd/sparkpipe_weightd 2>/dev/null | cut -c1-16)
drv=$(sha256sum < $rr/$DRIVER 2>/dev/null | cut -c1-16)
rbin=$(sha256sum < $rr/bin/sparkpipe_model_residentd 2>/dev/null | cut -c1-16)
ready=$(grep -c "model_residentd ready" $rr/residentd.log 2>/dev/null)
errsite=$(grep -c ERRSITE $rr/residentd.log 2>/dev/null)
verify=$(grep -o "pack-verify path=[^ ]*/${pack##*/} mode=[a-z0-9]*" $HOME/weightd.log 2>/dev/null | tail -1 | sed "s/.*mode=//")
if [ -x $stage/sparkpipe_weightd ]; then
    stage_wd=$(sha256sum < $stage/sparkpipe_weightd | cut -c1-16)
    stage_rc=$(sha256sum < $stage/weightd_receipt 2>/dev/null | cut -c1-16)
    stage_warm=$(sha256sum < $stage/weightd_warm 2>/dev/null | cut -c1-16)
    $stage/weightd_receipt check $pack > /dev/null 2>&1
    receipt=rc$?
else
    stage_wd=none
    stage_rc=none
    stage_warm=none
    receipt=nostage
fi
hold=no
[ -f $rr/agent.hold ] && hold=yes
layout=legacy
for f in $rr/agent.env $rr/config/rank_index_* $rr/config/env_*.env; do [ -e "$f" ] && layout=layout; done
applied=$(sha256sum < $rr/.applied_manifest 2>/dev/null | cut -c1-16)
rootok=no
(cd $rr && sha256sum -c --quiet .applied_manifest > /dev/null 2>&1) && rootok=yes
mem=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
agent=$(systemctl --user is-active fleet-agent)
roots=$(grep -v "^$" $HOME/.fleet_agent_roots 2>/dev/null | tr "\n" "," | sed "s/,$//")
extra=""
if type probe_extra > /dev/null 2>&1; then extra=$(probe_extra "$rr"); fi
echo "host=$h agent=$agent hold=$hold layout=$layout applied=${applied:-none} rootok=$rootok eng_n=$eng_n exe=$exe up=$up rbin=${rbin:-none} drv=${drv:-none} ready=${ready:-0} errsite=${errsite:-0} wd=$wd wd_up=$wd_up wd_n=$wd_n wd_other=$wd_other wd_inst=${wd_inst:-none} verify=${verify:-none} stage_wd=$stage_wd stage_rc=$stage_rc stage_warm=$stage_warm receipt=$receipt mem_gib=$mem others=${others:-none} roots=${roots:-none} $extra"
