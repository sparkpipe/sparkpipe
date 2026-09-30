#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_tp_mesh_round_control.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define PROBE_TIMEOUT_NS UINT64_C(20000000000)
#define PROBE_RECORD_BYTES_PER_ROW (4112u * 2u)
#define PROBE_CASES_MAX 16u
#define PROBE_SCATTER_DEGREE 16u

typedef struct ProbeMesh
{
    uint8_t *region;
    uint8_t *staging;
    uint32_t band;
    uint32_t degree;
    uint32_t local;
    uint32_t peer_route;
    uint64_t epoch;
    uint64_t sequence;
} ProbeMesh;

static uint64_t probe_now_ns(void)
{
    struct timespec now;
    (void)clock_gettime(CLOCK_MONOTONIC,&now);
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static uint64_t probe_word(uint64_t tag,uint32_t source,uint32_t target,uint64_t word)
{
    return tag * UINT64_C(0x9E3779B97F4A7C15) ^ ((uint64_t)source << 56u) ^ ((uint64_t)target << 48u) ^ word * UINT64_C(0xBF58476D1CE4E5B9);
}

static void probe_fill(uint8_t *destination,uint64_t tag,uint32_t source,uint32_t target,uint64_t bytes)
{
    uint64_t word,value;
    for (word=0u; word<bytes / 8u; word++)
    {
        value = probe_word(tag,source,target,word);
        memcpy(destination + word * 8u,&value,8u);
    }
}

static uint64_t probe_check(const uint8_t *source_bytes,uint64_t tag,uint32_t source,uint32_t target,uint64_t bytes)
{
    uint64_t word,value,mismatches = 0u;
    for (word=0u; word<bytes / 8u; word++)
    {
        memcpy(&value,source_bytes + word * 8u,8u);
        mismatches += value != probe_word(tag,source,target,word) ? 1u : 0u;
    }
    return mismatches;
}

static volatile uint64_t *probe_slot_tail(const ProbeMesh *mesh,uint32_t rank,uint64_t sequence)
{
    uint64_t slot = (uint64_t)mesh->band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + (uint64_t)rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ((sequence - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
    return (volatile uint64_t *)(mesh->region + (slot + 1u) * SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u);
}

static uint8_t *probe_slot(const ProbeMesh *mesh,uint32_t rank,uint64_t sequence)
{
    uint64_t slot = (uint64_t)mesh->band * SPARK_WEIGHTD_MESH_SLOTS_PER_BAND + (uint64_t)rank * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ((sequence - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
    return mesh->region + slot * SPARK_WEIGHTD_MESH_SLOT_BYTES;
}

static int probe_wait(volatile uint64_t *cell,uint64_t expected,const char *what)
{
    uint64_t deadline = probe_now_ns() + PROBE_TIMEOUT_NS;
    while ( __atomic_load_n(cell,__ATOMIC_ACQUIRE) != expected )
        if ( probe_now_ns() > deadline )
        {
            fprintf(stderr,"PROBE-TIMEOUT %s expected=%llx seen=%llx\n",what,(unsigned long long)expected,(unsigned long long)*cell);
            return -1;
        }
    return 0;
}

static int probe_round(ProbeMesh *mesh,uint64_t chunk_bytes,uint64_t *mismatches)
{
    uint64_t sequence = ++mesh->sequence;
    uint64_t tag = (mesh->epoch << 32u) | sequence;
    uint64_t slice = SparkTpMeshAllToAllSliceElements(SPARK_WEIGHTD_MESH_SLOT_BYTES,PROBE_SCATTER_DEGREE) * 2u;
    uint8_t *own = probe_slot(mesh,mesh->local,sequence);
    volatile uint64_t *entry = (volatile uint64_t *)(mesh->region + SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(mesh->band,mesh->local));
    SparkWeightdMeshRoute route = {0};
    uint32_t peer;
    uint64_t bytes;
    route.fields.peer_mask = ((1u << mesh->degree) - 1u) & ~(1u << mesh->local);
    if ( mesh->peer_route != 0u )
    {
        route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
        route.fields.slice_bytes = chunk_bytes;
        for (peer=0u; peer<mesh->degree; peer++)
            probe_fill(mesh->staging + SPARK_WEIGHTD_MESH_STAGING_OFFSET(mesh->band,peer),tag,mesh->local,peer,chunk_bytes);
        bytes = chunk_bytes;
    }
    else
    {
        route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_SCATTER;
        route.fields.slice_bytes = slice;
        for (peer=0u; peer<mesh->degree; peer++)
            probe_fill(own + peer * slice,tag,mesh->local,peer,chunk_bytes);
        bytes = slice * mesh->degree;
    }
    entry[1] = bytes;
    entry[2] = (uint64_t)mesh->local * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ((sequence - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
    entry[3] = route.word;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    *probe_slot_tail(mesh,mesh->local,sequence) = tag;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __atomic_store_n(&entry[0],tag,__ATOMIC_RELEASE);
    for (peer=0u; peer<mesh->degree; peer++)
    {
        uint8_t *slot;
        if ( peer == mesh->local ) continue;
        if ( probe_wait(probe_slot_tail(mesh,peer,sequence),tag,"peer-tail") != 0 ) return -1;
        slot = probe_slot(mesh,peer,sequence) + (mesh->peer_route != 0u ? 0u : (uint64_t)mesh->local * slice);
        *mismatches += probe_check(slot,tag,peer,mesh->local,chunk_bytes);
    }
    return probe_wait((volatile uint64_t *)(mesh->region + SPARK_WEIGHTD_MESH_SHIPPED_ENTRY(mesh->band,mesh->local)),tag,"shipped");
}

static int probe_parse_u32(const char *text,uint32_t maximum,uint32_t *value)
{
    char *end;
    unsigned long parsed;
    errno = 0;
    parsed = strtoul(text,&end,0);
    if ( errno != 0 || end == text || *end != '\0' || parsed > maximum ) return -1;
    *value = (uint32_t)parsed;
    return 0;
}

int main(int argc,char **argv)
{
    SparkWeightdClient *client = 0;
    SparkWeightdMeshTopology topology;
    ProbeMesh mesh;
    uint32_t lane,band_index,rows[PROBE_CASES_MAX],case_count,index,lane_out,epoch,rounds;
    uint64_t chunk_capacity;
    SparkStatus status;
    char *list,*token,*save = 0;
    memset(&mesh,0,sizeof(mesh));
    memset(&topology,0,sizeof(topology));
    if ( argc < 9 || argc - 9 > (int)PROBE_CASES_MAX || probe_parse_u32(argv[2],SPARK_WEIGHTD_MESH_MAX_LANES - 1u,&lane) != 0 ||
         probe_parse_u32(argv[3],1u,&band_index) != 0 || probe_parse_u32(argv[4],1u,&mesh.local) != 0 ||
         (strcmp(argv[6],"peer") != 0 && strcmp(argv[6],"scatter") != 0) || probe_parse_u32(argv[7],UINT32_MAX,&epoch) != 0 || epoch == 0u ||
         probe_parse_u32(argv[8],1000000u,&rounds) != 0 || rounds == 0u )
    {
        fprintf(stderr,"usage: weightd_peer_route_probe SOCKET LANE BAND_INDEX LOCAL PHYS_A,PHYS_B peer|scatter EPOCH ROUNDS ROWS...\n");
        return 2;
    }
    list = argv[5];
    for (token = strtok_r(list,",",&save); token != 0; token = strtok_r(0,",",&save))
    {
        if ( topology.rank_count >= 16u || probe_parse_u32(token,15u,&topology.physical_ranks[topology.rank_count]) != 0 ) return 2;
        topology.rank_count++;
    }
    case_count = (uint32_t)(argc - 9);
    for (index=0u; index<case_count; index++)
        if ( probe_parse_u32(argv[9 + index],SPARK_WEIGHTD_MESH_MAX_BATCH_ROWS,&rows[index]) != 0 || rows[index] == 0u ) return 2;
    if ( topology.rank_count == 0u || mesh.local >= topology.rank_count || (topology.rank_count == 1u && case_count != 0u) ) return 2;
    topology.local_rank = mesh.local;
    mesh.degree = topology.rank_count;
    mesh.peer_route = strcmp(argv[6],"peer") == 0 ? 1u : 0u;
    mesh.epoch = epoch;
    status = SparkWeightdClientConnect(argv[1],&client,0);
    if ( status == SPARK_STATUS_OK ) status = SparkWeightdClientLaneAcquire(client,lane,&topology,&lane_out,PROBE_TIMEOUT_NS);
    if ( status == SPARK_STATUS_OK ) status = SparkWeightdClientMeshMap(client,(void **)&mesh.region,PROBE_TIMEOUT_NS);
    if ( status == SPARK_STATUS_OK && mesh.peer_route != 0u ) status = SparkWeightdClientMeshStagingMap(client,(void **)&mesh.staging,PROBE_TIMEOUT_NS);
    if ( status == SPARK_STATUS_OK ) status = SparkWeightdClientMeshActivity(client,1u,1u,PROBE_TIMEOUT_NS);
    if ( status != SPARK_STATUS_OK )
    {
        fprintf(stderr,"PROBE-SETUP-FAIL status=%d\n",(int)status);
        SparkWeightdClientClose(client);
        return 1;
    }
    mesh.band = 2u * lane_out + band_index;
    printf("PROBE lane=%u band=%u local=%u degree=%u mode=%s capabilities=%llu staging_bytes=%llu abi=%u\n",lane_out,mesh.band,mesh.local,mesh.degree,argv[6],
        (unsigned long long)((const SparkWeightdMeshWaitRequest *)(mesh.region + SPARK_WEIGHTD_MESH_WAIT_ENTRY(mesh.band,mesh.local)))->capabilities,
        (unsigned long long)(mesh.peer_route != 0u ? SPARK_WEIGHTD_MESH_STAGING_BYTES : 0u),(unsigned)SPARK_WEIGHTD_IPC_ABI_VERSION);
    chunk_capacity = mesh.peer_route != 0u ? SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES : SparkTpMeshAllToAllSliceElements(SPARK_WEIGHTD_MESH_SLOT_BYTES,PROBE_SCATTER_DEGREE) * 2u;
    for (index=0u; index<case_count && status == SPARK_STATUS_OK; index++)
    {
        uint64_t total = (uint64_t)rows[index] * PROBE_RECORD_BYTES_PER_ROW,done,chunk,mismatches = 0u,began,elapsed,chunks = 0u;
        uint32_t repeat;
        began = probe_now_ns();
        for (repeat=0u; repeat<rounds && status == SPARK_STATUS_OK; repeat++)
            for (done=0u; done<total; done+=chunk)
            {
                chunk = total - done < chunk_capacity ? total - done : chunk_capacity;
                if ( probe_round(&mesh,chunk,&mismatches) != 0 ) status = SPARK_STATUS_IO_ERROR;
                chunks++;
            }
        elapsed = probe_now_ns() - began;
        printf("PROBE-CASE mode=%s rows=%u per_peer_bytes=%llu rounds_per_exchange=%llu exchanges=%u us_per_exchange=%.1f us_per_round=%.1f mismatches=%llu %s\n",
            argv[6],rows[index],(unsigned long long)total,(unsigned long long)(chunks / rounds),rounds,
            (double)elapsed / 1000.0 / rounds,(double)elapsed / 1000.0 / (double)(chunks != 0u ? chunks : 1u),
            (unsigned long long)mismatches,status == SPARK_STATUS_OK && mismatches == 0u ? "PASS" : "FAIL");
        if ( mismatches != 0u ) status = SPARK_STATUS_VALIDATION_FAILED;
    }
    (void)SparkWeightdClientMeshActivity(client,1u,0u,PROBE_TIMEOUT_NS);
    SparkWeightdClientClose(client);
    return status == SPARK_STATUS_OK ? 0 : 1;
}
