#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MESH_STATUS_EXIT_MET 0
#define MESH_STATUS_EXIT_FAULT 1
#define MESH_STATUS_EXIT_USAGE 2
#define MESH_STATUS_EXIT_TIMEOUT 3
#define MESH_STATUS_EXIT_MIXED 4
#define MESH_STATUS_EXIT_LANE_CONFLICT 5
#define MESH_STATUS_EXIT_ABSENT 6
#define MESH_STATUS_EXIT_DISABLED 7
#define MESH_STATUS_EXIT_UNRESPONSIVE 8
#define MESH_STATUS_POLL_NS UINT64_C(250000000)
#define MESH_STATUS_ATTEMPT_NS UINT64_C(2000000000)
#define MESH_STATUS_ATTEMPT_MIN_NS UINT64_C(100000000)
#define MESH_STATUS_NO_LANE UINT32_MAX

static const char *const mesh_status_counter_names[SPARK_WEIGHTD_MESH_STATUS_COUNTERS] = {
    "trywire","record_failures","record_invalid","wire_failures","rewires","unready",
    "repairs","cq_ok","cq_err","peer_resets","peer_reset_bits","lane_resets"};
static const char *const mesh_status_peer_names[] = {
    "absent","self","no_record","record_rejected","record_invalid","wire_failed","wired"};
static const char *const mesh_status_busy_names[] = {
    "none","activity","rpc","buffer","pending","wait","doorbell"};

static uint64_t MeshStatusNow(void)
{
    struct timespec now;
    if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
        return 0u;
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static int MeshStatusParse(const char *text,unsigned long low,unsigned long high,uint32_t *value)
{
    char *end;
    unsigned long parsed;
    if ( text == 0 || text[0] == '\0' || text[0] == '-' || text[0] == '+' )
        return 0;
    errno = 0;
    parsed = strtoul(text,&end,0);
    if ( errno != 0 || *end != '\0' || parsed < low || parsed > high )
        return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static void MeshStatusStateText(uint32_t state,char *text,size_t bytes)
{
    static const char *const names[] = {"disabled","wiring","ready"};
    if ( state < sizeof(names) / sizeof(names[0]) )
        (void)snprintf(text,bytes,"%s",names[state]);
    else
        (void)snprintf(text,bytes,"unknown(%u)",state);
}

static void MeshStatusPeerText(uint32_t state,char *text,size_t bytes)
{
    if ( state < sizeof(mesh_status_peer_names) / sizeof(mesh_status_peer_names[0]) )
        (void)snprintf(text,bytes,"%s",mesh_status_peer_names[state]);
    else
        (void)snprintf(text,bytes,"unknown(%u)",state);
}

static const char *MeshStatusBusyText(uint32_t reason)
{
    return reason < sizeof(mesh_status_busy_names) / sizeof(mesh_status_busy_names[0]) ? mesh_status_busy_names[reason] : "unknown";
}

static void MeshStatusPrintJson(const SparkWeightdIpcMeshStatusResult *result)
{
    char text[32];
    uint32_t index,rank;
    MeshStatusStateText(result->mesh_state,text,sizeof(text));
    printf("{\"status\":\"%s\",\"layout\":%u,\"layout_compat\":%u,\"mesh_state\":\"%s\",\"pid\":%u,\"rank\":%u,"
        "\"rank_mask\":\"0x%04x\",\"wired_mask\":\"0x%04x\",",
        SparkStatusToString((SparkStatus)result->status),result->layout,result->layout_compat,text,result->pid,
        result->local_rank,result->rank_mask,result->wired_mask);
    if ( result->pair_rank == UINT32_MAX )
        printf("\"pair_rank\":null,");
    else
        printf("\"pair_rank\":%u,",result->pair_rank);
    printf("\"daemon_generation\":%llu,\"boot_ns\":%llu,\"mesh_generation\":%llu,\"now_mono_ns\":%llu,\"ready_since_mono_ns\":%llu,\"counters\":{",
        (unsigned long long)result->daemon_generation,(unsigned long long)result->boot_ns,
        (unsigned long long)result->mesh_generation,(unsigned long long)result->now_mono_ns,
        (unsigned long long)result->ready_since_mono_ns);
    for (index=0u; index<SPARK_WEIGHTD_MESH_STATUS_COUNTERS; index++)
        printf("%s\"%s\":%llu",index != 0u ? "," : "",mesh_status_counter_names[index],(unsigned long long)result->counters[index]);
    printf("},\"peers\":[");
    for (index=0u; index<SPARK_WEIGHTD_MESH_RANKS; index++)
    {
        const SparkWeightdMeshPeerStatus *peer = &result->peers[index];
        MeshStatusPeerText(peer->state,text,sizeof(text));
        printf("%s{\"rank\":%u,\"state\":\"%s\",\"record_status\":\"%s\",\"qp_flags\":%u,\"wired_boot_ns\":%llu,\"record_boot_ns\":%llu,"
            "\"since_mono_ns\":%llu,\"last_ok_mono_ns\":%llu,\"last_err_mono_ns\":%llu,\"send_pending\":%u,\"rpc_pending\":%u,\"cq_err_since_ok\":%u}",
            index != 0u ? "," : "",index,text,SparkStatusToString((SparkStatus)peer->record_status),peer->qp_flags,
            (unsigned long long)peer->wired_boot_ns,(unsigned long long)peer->record_boot_ns,(unsigned long long)peer->since_mono_ns,
            (unsigned long long)peer->last_ok_mono_ns,(unsigned long long)peer->last_err_mono_ns,peer->send_pending,peer->rpc_pending,
            peer->cq_err_since_ok);
    }
    printf("],\"lanes\":[");
    for (index=0u; index<SPARK_WEIGHTD_MESH_MAX_LANES; index++)
    {
        const SparkWeightdMeshLaneStatus *lane = &result->lanes[index];
        printf("%s{\"lane\":%u,\"configured\":%s,\"owned\":%s,\"quarantined\":%s,\"ranks\":[",index != 0u ? "," : "",index,
            (lane->flags & SPARK_WEIGHTD_MESH_LANE_CONFIGURED) != 0u ? "true" : "false",
            (lane->flags & SPARK_WEIGHTD_MESH_LANE_OWNED) != 0u ? "true" : "false",
            (lane->flags & SPARK_WEIGHTD_MESH_LANE_QUARANTINED) != 0u ? "true" : "false");
        for (rank=0u; rank<lane->rank_count && rank<SPARK_WEIGHTD_MESH_RANKS; rank++)
            printf("%s%u",rank != 0u ? "," : "",(uint32_t)((lane->packed_ranks >> (4u * rank)) & 0xfu));
        printf("],\"local_rank\":%u,\"physical_mask\":\"0x%04x\",\"activity\":%u,\"failed_cells\":%u,\"pending_cells\":%u,"
            "\"configure_status\":\"%s\",\"busy_reason\":\"%s\",\"busy_index\":%u}",
            lane->local_rank,lane->physical_mask,lane->activity,lane->failed_cells,lane->pending_cells,
            SparkStatusToString((SparkStatus)lane->configure_status),MeshStatusBusyText(lane->busy_reason),lane->busy_index);
    }
    printf("]}\n");
}

static void MeshStatusPrintUnmet(const SparkWeightdIpcMeshStatusResult *result,uint32_t mask,uint32_t lane)
{
    char text[32];
    uint32_t rank,missing = (mask | result->rank_mask) & ~result->wired_mask;
    MeshStatusStateText(result->mesh_state,text,sizeof(text));
    fprintf(stderr,"MESH-STATUS-TIMEOUT state=%s missing=0x%04x\n",text,missing);
    for (rank=0u; rank<SPARK_WEIGHTD_MESH_RANKS; rank++)
    {
        const SparkWeightdMeshPeerStatus *peer = &result->peers[rank];
        if ( (missing & (1u << rank)) == 0u )
            continue;
        MeshStatusPeerText(peer->state,text,sizeof(text));
        fprintf(stderr,"MESH-STATUS-PEER rank=%u state=%s record_status=%s wired_boot_ns=%llu record_boot_ns=%llu age_ms=%llu cq_err_since_ok=%u\n",
            rank,text,SparkStatusToString((SparkStatus)peer->record_status),(unsigned long long)peer->wired_boot_ns,
            (unsigned long long)peer->record_boot_ns,
            (unsigned long long)(result->now_mono_ns > peer->since_mono_ns && peer->since_mono_ns != 0u ? (result->now_mono_ns - peer->since_mono_ns) / UINT64_C(1000000) : 0u),
            peer->cq_err_since_ok);
    }
    if ( lane != MESH_STATUS_NO_LANE )
        fprintf(stderr,"MESH-STATUS-LANE lane=%u flags=%u physical_mask=0x%04x configure_status=%s busy_reason=%s busy_index=%u\n",
            lane,result->lanes[lane].flags,result->lanes[lane].physical_mask,
            SparkStatusToString((SparkStatus)result->lanes[lane].configure_status),
            MeshStatusBusyText(result->lanes[lane].busy_reason),result->lanes[lane].busy_index);
}

static int MeshStatusOutcomeExit(uint32_t outcome,const char *socket_path)
{
    switch ( outcome )
    {
        case SPARK_WEIGHTD_MESH_QUERY_ABSENT:
            fprintf(stderr,"MESH-STATUS-ABSENT socket=%s\n",socket_path);
            return MESH_STATUS_EXIT_ABSENT;
        case SPARK_WEIGHTD_MESH_QUERY_UNRESPONSIVE:
            fprintf(stderr,"MESH-STATUS-UNRESPONSIVE socket=%s\n",socket_path);
            return MESH_STATUS_EXIT_UNRESPONSIVE;
        case SPARK_WEIGHTD_MESH_QUERY_UNSERVED:
        case SPARK_WEIGHTD_MESH_QUERY_INCOMPATIBLE:
            fprintf(stderr,"MESH-STATUS-MIXED-VERSION socket=%s outcome=%u\n",socket_path,outcome);
            return MESH_STATUS_EXIT_MIXED;
        default:
            fprintf(stderr,"MESH-STATUS-FAULT socket=%s\n",socket_path);
            return MESH_STATUS_EXIT_FAULT;
    }
}

static int MeshStatusMet(const SparkWeightdIpcMeshStatusResult *result,uint32_t mask,uint32_t lane)
{
    if ( result->status != SPARK_STATUS_OK || result->mesh_state != SPARK_WEIGHTD_MESH_STATE_READY ||
         (mask & ~result->wired_mask) != 0u )
        return 0;
    if ( lane == MESH_STATUS_NO_LANE )
        return 1;
    return (result->lanes[lane].flags & (SPARK_WEIGHTD_MESH_LANE_OWNED | SPARK_WEIGHTD_MESH_LANE_QUARANTINED)) == 0u &&
        result->lanes[lane].configure_status == SPARK_STATUS_OK;
}

static int MeshStatusWait(const char *socket_path,uint64_t timeout_ns,uint32_t mask,uint32_t lane)
{
    SparkWeightdIpcMeshStatusResult result;
    struct timespec pause;
    uint64_t started = MeshStatusNow(),deadline = started + timeout_ns,now,attempt,generation = 0u;
    uint32_t outcome,self,answered = 0u;
    for (;;)
    {
        now = MeshStatusNow();
        attempt = deadline > now ? deadline - now : 0u;
        attempt = attempt < MESH_STATUS_ATTEMPT_NS ? attempt : MESH_STATUS_ATTEMPT_NS;
        attempt = attempt > MESH_STATUS_ATTEMPT_MIN_NS ? attempt : MESH_STATUS_ATTEMPT_MIN_NS;
        (void)SparkWeightdMeshStatusQuery(socket_path,attempt,&result,&outcome);
        if ( outcome == SPARK_WEIGHTD_MESH_QUERY_UNSERVED || outcome == SPARK_WEIGHTD_MESH_QUERY_INCOMPATIBLE ||
             outcome == SPARK_WEIGHTD_MESH_QUERY_FAULT )
            return MeshStatusOutcomeExit(outcome,socket_path);
        answered = outcome == SPARK_WEIGHTD_MESH_QUERY_ANSWERED ? 1u : 0u;
        if ( answered != 0u )
        {
            if ( result.status == SPARK_STATUS_INVALID_ARGUMENT )
                return MeshStatusOutcomeExit(SPARK_WEIGHTD_MESH_QUERY_FAULT,socket_path);
            if ( result.status == SPARK_STATUS_UNSUPPORTED || result.mesh_state == SPARK_WEIGHTD_MESH_STATE_DISABLED )
            {
                fprintf(stderr,"MESH-STATUS-DISABLED socket=%s status=%s\n",socket_path,SparkStatusToString((SparkStatus)result.status));
                return MESH_STATUS_EXIT_DISABLED;
            }
            if ( generation != 0u && result.daemon_generation != generation )
                fprintf(stderr,"MESH-STATUS-DAEMON-RESTART previous=%llu current=%llu\n",
                    (unsigned long long)generation,(unsigned long long)result.daemon_generation);
            generation = result.daemon_generation;
            if ( (mask & ~result.rank_mask) != 0u )
            {
                fprintf(stderr,"MESH-STATUS-USAGE mask=0x%04x is outside the daemon rank_mask=0x%04x\n",mask,result.rank_mask);
                return MESH_STATUS_EXIT_USAGE;
            }
            self = 1u << result.local_rank;
            if ( lane != MESH_STATUS_NO_LANE && (result.lanes[lane].flags & SPARK_WEIGHTD_MESH_LANE_CONFIGURED) != 0u &&
                 (result.lanes[lane].physical_mask | self) != (mask | self) )
            {
                fprintf(stderr,"MESH-STATUS-LANE-CONFLICT lane=%u physical_mask=0x%04x requested=0x%04x\n",
                    lane,result.lanes[lane].physical_mask,mask);
                return MESH_STATUS_EXIT_LANE_CONFLICT;
            }
            if ( MeshStatusMet(&result,mask,lane) != 0 )
            {
                printf("MESH-STATUS-READY rank=%u mask=0x%04x wired=0x%04x daemon_generation=%llu boot_ns=%llu mesh_generation=%llu waited_ms=%llu\n",
                    result.local_rank,mask,result.wired_mask,(unsigned long long)result.daemon_generation,
                    (unsigned long long)result.boot_ns,(unsigned long long)result.mesh_generation,
                    (unsigned long long)((MeshStatusNow() - started) / UINT64_C(1000000)));
                return MESH_STATUS_EXIT_MET;
            }
        }
        now = MeshStatusNow();
        if ( now < deadline )
        {
            attempt = deadline - now < MESH_STATUS_POLL_NS ? deadline - now : MESH_STATUS_POLL_NS;
            pause.tv_sec = (time_t)(attempt / UINT64_C(1000000000));
            pause.tv_nsec = (long)(attempt % UINT64_C(1000000000));
            (void)nanosleep(&pause,0);
        }
        if ( MeshStatusNow() >= deadline )
        {
            if ( answered == 0u )
                return MeshStatusOutcomeExit(outcome,socket_path);
            MeshStatusPrintUnmet(&result,mask,lane);
            return MESH_STATUS_EXIT_TIMEOUT;
        }
    }
}

static int MeshStatusUsage(const char *program)
{
    fprintf(stderr,"usage: %s --socket PATH --timeout SECONDS [--wait-lane-peers MASK [--lane N]]\n",program);
    return MESH_STATUS_EXIT_USAGE;
}

int main(int argc,char **argv)
{
    SparkWeightdIpcMeshStatusResult result;
    const char *socket_path = 0;
    uint32_t timeout = 0u,mask = 0u,lane = MESH_STATUS_NO_LANE,outcome;
    int index;
    for (index=1; index<argc; index += 2)
    {
        const char *value = index + 1 < argc ? argv[index + 1] : 0;
        if ( strcmp(argv[index],"--socket") == 0 && value != 0 && value[0] != '\0' )
            socket_path = value;
        else if ( strcmp(argv[index],"--timeout") == 0 && MeshStatusParse(value,1ul,86400ul,&timeout) != 0 )
            continue;
        else if ( strcmp(argv[index],"--wait-lane-peers") == 0 && MeshStatusParse(value,1ul,0xfffful,&mask) != 0 )
            continue;
        else if ( strcmp(argv[index],"--lane") == 0 && MeshStatusParse(value,0ul,SPARK_WEIGHTD_MESH_MAX_LANES - 1u,&lane) != 0 )
            continue;
        else
            return MeshStatusUsage(argv[0]);
    }
    if ( socket_path == 0 || timeout == 0u || (lane != MESH_STATUS_NO_LANE && mask == 0u) )
        return MeshStatusUsage(argv[0]);
    if ( mask != 0u )
        return MeshStatusWait(socket_path,(uint64_t)timeout * UINT64_C(1000000000),mask,lane);
    (void)SparkWeightdMeshStatusQuery(socket_path,(uint64_t)timeout * UINT64_C(1000000000),&result,&outcome);
    if ( outcome != SPARK_WEIGHTD_MESH_QUERY_ANSWERED )
        return MeshStatusOutcomeExit(outcome,socket_path);
    MeshStatusPrintJson(&result);
    return result.status == SPARK_STATUS_OK || result.status == SPARK_STATUS_UNSUPPORTED ? MESH_STATUS_EXIT_MET : MESH_STATUS_EXIT_FAULT;
}
