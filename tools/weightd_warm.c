#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include "sparkpipe/spark_dsv4_parallel_shape.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

/*
 * Family identities (multidev): the GLM path above is the historical
 * default. --family dsv4_pro derives the EXACT identity the DSV4 Pro
 * module sends through SparkWeightdAttachMappedPack
 * (SparkDsv4ModuleWeightdAttach): model "dsv4", empty revision,
 * topology = the (u32-truncated) SparkDsv4TpNodeConfig configuration_hash
 * for this world rank's TP4xPP4 shape, geometry_fingerprint = the same
 * FNV-1a chain over the same pack-header fields in the same order. The
 * derivation shares the family's shape source, so warm and resident
 * identities match by construction; a source-contract test pins the hash
 * order against the module.
 */

static int parse_positive(const char *text,uint64_t maximum,uint64_t *value)
{
    char *end;
    if ( text == 0 || text[0] < '0' || text[0] > '9' )
        return 0;
    errno = 0;
    *value = strtoull(text,&end,10);
    return errno == 0 && *end == '\0' && *value != 0u && *value <= maximum;
}

typedef struct WsetEntry
{
    SparkWeightdExpertKey key;
    uint32_t index;
} WsetEntry;

static int compare_entries(const void *left,const void *right)
{
    const WsetEntry *a = (const WsetEntry *)left;
    const WsetEntry *b = (const WsetEntry *)right;
    if ( a->key.layer != b->key.layer )
        return a->key.layer < b->key.layer ? -1 : 1;
    if ( a->key.expert != b->key.expert )
        return a->key.expert < b->key.expert ? -1 : 1;
    return a->index < b->index ? -1 : a->index > b->index;
}

static uint32_t unique_in_order(SparkWeightdExpertKey *keys,uint32_t pairs)
{
    WsetEntry *entries = (WsetEntry *)calloc(pairs,sizeof(*entries));
    uint8_t *drop = (uint8_t *)calloc(pairs,1u);
    uint32_t index,count;
    count = 0u;
    if ( entries != 0 && drop != 0 )
    {
        for (index=0u; index<pairs; index++)
        {
            entries[index].key = keys[index];
            entries[index].index = index;
        }
        qsort(entries,pairs,sizeof(*entries),compare_entries);
        for (index=1u; index<pairs; index++)
            if ( entries[index].key.layer == entries[index - 1u].key.layer && entries[index].key.expert == entries[index - 1u].key.expert )
                drop[entries[index].index] = 1u;
        for (index=0u; index<pairs; index++)
            if ( drop[index] == 0u )
                keys[count++] = keys[index];
    }
    free(entries);
    free(drop);
    return count;
}

static SparkWeightdExpertKey *read_wset(const char *path,
    const SparkWeightdManifest *manifest,uint32_t *count)
{
    struct stat info;
    FILE *file = fopen(path,"rb");
    SparkWeightdExpertKey *keys;
    uint32_t pair[2],index,pairs;
    *count = 0u;
    if ( file == 0 )
        return 0;
    if ( fstat(fileno(file),&info) != 0 || !S_ISREG(info.st_mode) ||
         info.st_size <= 0 || (uint64_t)info.st_size % sizeof(pair) != 0u ||
         (uint64_t)info.st_size > UINT64_C(1048576) * sizeof(pair) )
    {
        fclose(file);
        return 0;
    }
    pairs = (uint32_t)((uint64_t)info.st_size / sizeof(pair));
    keys = (SparkWeightdExpertKey *)calloc(pairs,sizeof(*keys));
    if ( keys == 0 )
    {
        fclose(file);
        return 0;
    }
    for (index=0u; index<pairs; index++)
    {
        if ( fread(pair,1u,sizeof(pair),file) != sizeof(pair) ||
             SparkWeightdManifestFind(manifest,pair[0],pair[1]) == 0 )
        {
            free(keys);
            fclose(file);
            return 0;
        }
        keys[index].layer = pair[0];
        keys[index].expert = pair[1];
    }
    *count = unique_in_order(keys,pairs);
    if ( fgetc(file) != EOF || ferror(file) || *count == 0u )
    {
        free(keys);
        keys = 0;
    }
    fclose(file);
    return keys;
}

static int warm_keys(SparkWeightdClient *client,uint64_t generation,
    const SparkWeightdExpertKey *keys,uint32_t count,uint64_t timeout)
{
    SparkWeightdWorkingSetResult result = {0};
    SparkStatus status = SparkWeightdClientAcquire(client,generation,keys,count,&result,timeout);
    if ( status == SPARK_STATUS_OK && result.status == SPARK_STATUS_OK )
        status = SparkWeightdClientRelease(client,generation,result.lease_identifier,&result,timeout);
    if ( status == SPARK_STATUS_OK && result.status == SPARK_STATUS_OK )
        return 1;
    fprintf(stderr,"weightd_warm: FAILED layer=%u keys=%u status=%d daemon=%u\n",
        keys[0].layer,count,(int)status,result.status);
    return 0;
}

/* DSV4 Pro pack header (little-endian): 16 u32 + 2 u64. The geometry
 * chain hashes exactly the fields the module hashes, in its order:
 * format_version, codec_abi_version, linear/expert/kv codecs,
 * tensor_count, first_layer, layer_count, total_layer_count, hidden,
 * vocab, routed_expert_count, mtp_layer_count (u32 each), file_bytes
 * (u64). Mirrors SparkDsv4ModuleWeightdAttach. */
static int dsv4_pro_identity(const char *pack_path, uint32_t world_rank,
    SparkWeightdIdentity *identity)
{
    SparkDsv4TpShapeDescriptor shape;
    SparkDsv4TpNodeConfig config;
    uint32_t header[16];
    uint64_t wide[2], geometry;
    FILE *file;
    memset(&shape,0,sizeof(shape));
    memset(&config,0,sizeof(config));
    file = fopen(pack_path,"rb");
    if ( file == 0 || fread(header,4u,16u,file) != 16u ||
         fread(wide,8u,2u,file) != 2u )
    {
        if ( file != 0 ) fclose(file);
        return 0;
    }
    fclose(file);
    shape.abi_version = SPARK_DSV4_PARALLEL_SHAPE_ABI_VERSION;
    shape.tp_degree = 4u;
    shape.tp_rank = world_rank % 4u;
    shape.pp_stage_count = 4u;
    shape.pp_stage_index = world_rank / 4u;
    if ( SparkDsv4TpDeriveNodeConfig(&shape,&config) != SPARK_STATUS_OK )
        return 0;
    geometry = UINT64_C(1469598103934665603);
    geometry = SparkHashBytes(geometry,&header[1],sizeof(uint32_t));
    geometry = SparkHashBytes(geometry,&header[4],sizeof(uint32_t));
    geometry = SparkHashBytes(geometry,&header[5],sizeof(uint32_t));
    geometry = SparkHashBytes(geometry,&header[6],sizeof(uint32_t));
    geometry = SparkHashBytes(geometry,&header[7],sizeof(uint32_t));
    geometry = SparkHashBytes(geometry,&header[8],sizeof(uint32_t));
    geometry = SparkHashBytes(geometry,&header[9],sizeof(uint32_t));
    geometry = SparkHashBytes(geometry,&header[10],sizeof(uint32_t));
    geometry = SparkHashBytes(geometry,&header[11],sizeof(uint32_t));
    geometry = SparkHashBytes(geometry,&header[12],sizeof(uint32_t));
    geometry = SparkHashBytes(geometry,&header[13],sizeof(uint32_t));
    geometry = SparkHashBytes(geometry,&header[14],sizeof(uint32_t));
    geometry = SparkHashBytes(geometry,&header[15],sizeof(uint32_t));
    geometry = SparkHashBytes(geometry,&wide[1],sizeof(uint64_t));
    memset(identity,0,sizeof(*identity));
    identity->geometry_fingerprint = geometry;
    identity->topology = (uint32_t)config.configuration_hash;
    strcpy(identity->model,"dsv4");
    identity->revision[0] = '\0';
    return 1;
}

int main(int argument_count,char **arguments)
{
    SparkWeightdLazyAttachRequest request = {0};
    SparkWeightdLazyAttachResult attached = {0};
    SparkWeightdManifest manifest = {0};
    SparkWeightdClient *client = 0;
    SparkWeightdExpertKey *keys = 0;
    struct stat pack;
    SparkStatus status;
    char manifest_path[SPARK_WEIGHTD_PATH_BYTES + 16u];
    const char *wset_path = 0;
    const char *family = 0;
    uint64_t world_rank = 0;
    int world_rank_given = 0, identity_print = 0;
    char *filtered[24];
    int filtered_count = 0, index_argument;
    uint64_t layers = 45u,experts = 288u,seconds = 1800u,topology;
    uint32_t first,index,count = 0u;
    int exit_status = 1;
    for ( index_argument = 0; index_argument < argument_count; index_argument++ )
    {
        if ( strcmp(arguments[index_argument],"--family") == 0 &&
            index_argument + 1 < argument_count )
        {
            family = arguments[++index_argument];
            continue;
        }
        if ( strcmp(arguments[index_argument],"--world-rank") == 0 &&
            index_argument + 1 < argument_count )
        {
            char *end;
            errno = 0;
            world_rank = strtoull(arguments[++index_argument],&end,10);
            if ( errno != 0 || *end != '\0' || end == arguments[index_argument] ||
                world_rank > 15u )
                goto usage;
            world_rank_given = 1;
            continue;
        }
        if ( strcmp(arguments[index_argument],"--identity-print") == 0 )
        {
            identity_print = 1;
            continue;
        }
        if ( filtered_count == (int)(sizeof(filtered)/sizeof(filtered[0])) )
            goto usage;
        filtered[filtered_count++] = arguments[index_argument];
    }
    arguments = filtered;
    argument_count = filtered_count;
    if ( family != 0 && strcmp(family,"dsv4_pro") != 0 &&
         strcmp(family,"dsv41_flash") != 0 && strcmp(family,"k3") != 0 &&
         strcmp(family,"ling") != 0 )
    {
        fprintf(stderr,"weightd_warm: unknown family %s (dsv4_pro, dsv41_flash, k3, ling)\n",family);
        goto usage;
    }
    if ( family != 0 && strcmp(family,"dsv4_pro") == 0 && !world_rank_given )
    {
        fprintf(stderr,"weightd_warm: --family dsv4_pro requires --world-rank\n");
        goto usage;
    }
    if ( argument_count == 3 && strcmp(arguments[2],"--reclaim") == 0 )
    {
        /* Lane utility (additive): free every COLD arena (refcount 0, no
           leases). The pool size is fixed at arena creation, so a stale
           arena created with the wrong expert-pool budget blocks the
           correctly-sized one until reclaimed - measured on the lane-4
           rank3 cell (ACQUIRE-LOAD-STAGE stage=budget). */
        SparkWeightdReclaimResult reclaim = {0};
        SparkWeightdClient *reclaim_client = 0;
        status = SparkWeightdClientConnect(arguments[1],&reclaim_client,0);
        if ( status == SPARK_STATUS_OK )
            status = SparkWeightdClientReclaim(reclaim_client,&reclaim,
                30u * UINT64_C(1000000000));
        if ( status != SPARK_STATUS_OK || reclaim.status != SPARK_STATUS_OK )
        {
            fprintf(stderr,"weightd_warm: reclaim failed status=%d daemon=%u\n",
                (int)status,reclaim.status);
            SparkWeightdClientClose(reclaim_client);
            return 1;
        }
        fprintf(stderr,"weightd_warm: RECLAIM freed=%llu arenas=%u resident=%llu\n",
            (unsigned long long)reclaim.reclaimed_bytes,
            reclaim.reclaimed_arena_count,
            (unsigned long long)reclaim.resident_bytes);
        SparkWeightdClientClose(reclaim_client);
        return 0;
    }
    if ( argument_count > 6 && strcmp(arguments[6],"--wset") == 0 )
    {
        if ( argument_count < 8 )
            goto usage;
        wset_path = arguments[7];
        seconds = 300u;
    }
    if ( argument_count < 6 || argument_count > 9 ||
         !parse_positive(arguments[5],UINT32_MAX,&topology) ||
         (wset_path == 0 && argument_count > 6 && !parse_positive(arguments[6],UINT32_MAX,&layers)) ||
         (wset_path == 0 && argument_count > 7 && !parse_positive(arguments[7],SPARK_WEIGHTD_EXPERT_COUNT_MAX,&experts)) ||
         (argument_count > 8 && !parse_positive(arguments[8],UINT64_MAX / UINT64_C(1000000000),&seconds)) ||
         (getenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES") != 0 &&
          !parse_positive(getenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES"),
             SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT,&request.expert_pool_bytes)) ||
         (!identity_print &&
          getenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES") == 0) )
        goto usage;
    if ( strlen(arguments[2]) >= sizeof(request.pack_path) ||
         strlen(arguments[3]) != 64u ||
         strspn(arguments[3],"0123456789abcdefABCDEF") != 64u ||
         strlen(arguments[4]) >= sizeof(request.identity.revision) ||
         stat(arguments[2],&pack) != 0 || !S_ISREG(pack.st_mode) || pack.st_size <= 0 )
    {
        fprintf(stderr,"weightd_warm: invalid pack or identity\n");
        return 2;
    }
    memcpy(request.identity.pack_sha256,arguments[3],65u);
    strcpy(request.identity.model,"glm5_next_stage");
    strcpy(request.identity.revision,arguments[4]);
    request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    request.identity.arena_bytes = (uint64_t)pack.st_size;
    request.identity.topology = (uint32_t)topology;
    if ( family != 0 && strcmp(family,"dsv41_flash") == 0 )
    {
        /* Lane 4 (dsv41_flash): the module pins only the model tag
         * (SPARK_DSV41_FLASH_MODULE_TAG "dsv41_flash_stage") and sends
         * revision/topology from its node context, geometry unset (0).
         * REVISION/TOPOLOGY arguments stay authoritative here - they come
         * from the stage config - so the family hook pins the tag only. */
        strcpy(request.identity.model,"dsv41_flash_stage");
    }
    else if ( family != 0 && strcmp(family,"ling") == 0 )
    {
        /* Lane 9 (ling): the module pins only the model tag
         * (SPARK_LING_MODULE_TAG "ling_stage") and sends
         * revision/topology from its stage config (revision = the
         * stage model_revision, topology = tp_degree), geometry unset
         * (0). REVISION/TOPOLOGY arguments stay authoritative here -
         * they come from the stage config - so the family hook pins
         * the tag only, exactly the dsv41_flash shape. */
        strcpy(request.identity.model,"ling_stage");
    }
    else if ( family != 0 && strcmp(family,"k3") == 0 )
    {
        /* Lane 3 (k3): the runner pins model "kimi-k3" / revision
         * "mxfp4" / topology = tp_degree (4). arena_bytes keeps the
         * default pack-size fill - the daemon's size-mismatch contract
         * (WDATTACH) rejects any other value with INVALID_ARGUMENT,
         * measured against the release-shared weightd. */
        strcpy(request.identity.model,"kimi-k3");
        strcpy(request.identity.revision,"mxfp4");
        request.identity.topology = 4u;
        request.identity.geometry_fingerprint = 0u;
    }
    else if ( family != 0 )
    {
        /* Preserve the pack_sha256/abi/arena_bytes filled above; the
         * family owns model/revision/topology/geometry. */
        SparkWeightdIdentity derived;
        if ( !dsv4_pro_identity(arguments[2],(uint32_t)world_rank,&derived) )
        {
            fprintf(stderr,"weightd_warm: dsv4_pro identity derivation failed\n");
            return 2;
        }
        request.identity.geometry_fingerprint = derived.geometry_fingerprint;
        request.identity.topology = derived.topology;
        strcpy(request.identity.model,derived.model);
        strcpy(request.identity.revision,derived.revision);
    }
    if ( identity_print )
    {
        printf("identity model=%s revision=%s topology=%u geometry=%llu "
               "arena_bytes=%llu pack_sha256=%.64s\n",
            request.identity.model,request.identity.revision,
            request.identity.topology,
            (unsigned long long)request.identity.geometry_fingerprint,
            (unsigned long long)request.identity.arena_bytes,
            request.identity.pack_sha256);
        return 0;
    }
    strcpy(request.pack_path,arguments[2]);
    snprintf(manifest_path,sizeof(manifest_path),"%s.experts",arguments[2]);
    /* the manifest bound is the PACK size; identity.arena_bytes is the
     * daemon-side arena identity and for k3 is the pool budget, which
     * would reject every offset past 3 GiB of a ~98 GiB pack. */
    status = SparkWeightdManifestLoad(manifest_path,(uint64_t)pack.st_size,&manifest);
    if ( status != SPARK_STATUS_OK || manifest.group_count == 0u )
    {
        fprintf(stderr,"weightd_warm: expert manifest failed status=%d\n",(int)status);
        goto done;
    }
    if ( wset_path == 0 )
        for (index=0u; index<manifest.group_count; index++)
            if ( manifest.groups[index].layer >= layers || manifest.groups[index].expert >= experts )
            {
                fprintf(stderr,"weightd_warm: declared geometry excludes manifest expert %u/%u\n",
                    manifest.groups[index].layer,manifest.groups[index].expert);
                goto done;
            }
    if ( wset_path != 0 )
    {
        keys = read_wset(wset_path,&manifest,&count);
        if ( keys == 0 )
        {
            fprintf(stderr,"weightd_warm: invalid wset %s; require complete manifest key pairs\n",
                wset_path);
            goto done;
        }
    }
    else
    {
        keys = calloc(SPARK_WEIGHTD_LEASE_GROUPS_MAX,sizeof(*keys));
        if ( keys == 0 )
            goto done;
    }
    status = SparkWeightdClientConnect(arguments[1],&client,0);
    if ( status == SPARK_STATUS_OK )
        status = SparkWeightdClientAttachLazy(client,&request,&attached,seconds * UINT64_C(1000000000));
    if ( status != SPARK_STATUS_OK || attached.status != SPARK_STATUS_OK )
    {
        fprintf(stderr,"weightd_warm: attach failed status=%d daemon=%u\n",(int)status,attached.status);
        goto done;
    }
    if ( wset_path != 0 )
    {
        struct timespec began,ended;
        uint64_t elapsed;
        uint32_t chunk,warmed = 0u;
        (void)clock_gettime(CLOCK_MONOTONIC,&began);
        fprintf(stderr,"weightd_warm: WSET-ONE-SHOT file=%s keys=%u\n",wset_path,count);
        for (first=0u; first<count; first+=SPARK_WEIGHTD_LEASE_GROUPS_MAX)
        {
            chunk = count - first;
            if ( chunk > SPARK_WEIGHTD_LEASE_GROUPS_MAX )
                chunk = SPARK_WEIGHTD_LEASE_GROUPS_MAX;
            if ( !warm_keys(client,attached.arena_generation,&keys[first],chunk,
                seconds * UINT64_C(1000000000)) )
                goto done;
            warmed += chunk;
            fprintf(stderr,"weightd_warm: WSET-CHUNK warmed=%u/%u\n",warmed,count);
        }
        (void)clock_gettime(CLOCK_MONOTONIC,&ended);
        elapsed = (uint64_t)(ended.tv_sec - began.tv_sec) * UINT64_C(1000000000) +
            (uint64_t)ended.tv_nsec - (uint64_t)began.tv_nsec;
        fprintf(stderr,"weightd_warm: WSET-WARM keys=%u elapsed_ms=%llu\n",count,
            (unsigned long long)(elapsed / UINT64_C(1000000)));
    }
    else
        for (first=0u; first<manifest.group_count; first=index)
        {
            count = 0u;
            for (index=first; index<manifest.group_count &&
                 manifest.groups[index].layer == manifest.groups[first].layer; index++)
            {
                if ( count >= SPARK_WEIGHTD_LEASE_GROUPS_MAX )
                    goto done;
                keys[count].layer = manifest.groups[index].layer;
                keys[count++].expert = manifest.groups[index].expert;
            }
            if ( !warm_keys(client,attached.arena_generation,keys,count,seconds * UINT64_C(1000000000)) )
                goto done;
            fprintf(stderr,"weightd_warm: layer %u WARM experts=%u\n",keys[0].layer,count);
        }
    exit_status = 0;
done:
    SparkWeightdClientClose(client);
    SparkWeightdManifestDestroy(&manifest);
    free(keys);
    return exit_status;
usage:
    fprintf(stderr,"usage: weightd_warm SOCKET PACK SHA256 REVISION TOPOLOGY [LAYERS=45 [EXPERTS=288 [TIMEOUT_S=1800]]]\n"
        "       weightd_warm SOCKET PACK SHA256 REVISION TOPOLOGY --wset FILE [TIMEOUT_S=300]\n"
        "       weightd_warm SOCKET --reclaim\n"
        "       options (any position): --family dsv4_pro --world-rank R (derive the exact\n"
        "       DSV4 Pro module attach identity; REVISION/TOPOLOGY args are then ignored)\n"
        "                           --family dsv41_flash (pin the module tag; REVISION/TOPOLOGY stay authoritative)\n"
        "                           --family ling (pin the module tag; REVISION/TOPOLOGY stay authoritative)\n"
        "                           --family k3 (pin the k3 runner identity: kimi-k3/mxfp4,\n"
        "                              topology 4; arena bytes stay the pack size per the\n"
        "                              daemon's size-mismatch contract)\n"
        "                           --identity-print (print the derived identity and exit)\n"
        "       finite SPARK_WEIGHTD_EXPERT_POOL_BYTES is required\n");
    return 2;
}
