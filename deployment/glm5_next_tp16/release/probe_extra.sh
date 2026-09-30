probe_extra() {
    local rr=$1 kvline kvs kvref l2 gmode
    kvline=$(grep -o "GLM-KV-BYTES rank=[0-9]* tp=[0-9]* shard=[0-9]* physical_pages=[0-9]* latent_bytes=[0-9]* index_bytes=[0-9]* replicated_latent_bytes=[0-9]* replicated_index_bytes=[0-9]*" $rr/residentd.log 2>/dev/null | tail -1)
    kvs=none
    if [ -n "$kvline" ]; then
        kvs=$(echo "$kvline" | awk '{for(i=1;i<=NF;i++){split($i,a,"=");v[a[1]]=a[2]} print (v["shard"]=="1" && v["latent_bytes"]*v["tp"]==v["replicated_latent_bytes"] && v["index_bytes"]*v["tp"]==v["replicated_index_bytes"]) ? "1of" v["tp"] : "shard" v["shard"]}')
    fi
    kvref=$(grep -c "GLM-KV-SHARD-REFUSED\|GLM-KV-SHARD-REQUIRED\|TP-ALL-TO-ALL-UNSUPPORTED" $rr/residentd.log 2>/dev/null)
    l2=$(grep -o "GLM l2 prefetch=[a-z]*" $rr/residentd.log 2>/dev/null | tail -1 | cut -d= -f2)
    gmode=$(grep -o "GLM execution mode=[a-z]*" $rr/residentd.log 2>/dev/null | tail -1 | cut -d= -f2)
    echo "kvs=$kvs kvref=${kvref:-0} l2=${l2:-none} gmode=${gmode:-none}"
}
