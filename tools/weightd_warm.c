#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static int parse_positive(const char *text,uint64_t maximum,uint64_t *value)
{
    char *end;
    if ( text == 0 || text[0] < '0' || text[0] > '9' )
        return 0;
    errno = 0;
    *value = strtoull(text,&end,10);
    return errno == 0 && *end == '\0' && *value != 0u && *value <= maximum;
}

static int read_wset(const char *path,const SparkWeightdManifest *manifest,
    SparkWeightdExpertKey *keys,uint32_t *count)
{
    struct stat info;
    FILE *file = fopen(path,"rb");
    uint32_t pair[2],index;
    int valid = 0;
    *count = 0u;
    if ( file == 0 )
        return 0;
    if ( fstat(fileno(file),&info) != 0 || !S_ISREG(info.st_mode) ||
         info.st_size <= 0 || (uint64_t)info.st_size % sizeof(pair) != 0u ||
         (uint64_t)info.st_size > SPARK_WEIGHTD_LEASE_GROUPS_MAX * sizeof(pair) )
        goto done;
    for (uint64_t offset=0u; offset<(uint64_t)info.st_size; offset+=sizeof(pair))
    {
        if ( fread(pair,1u,sizeof(pair),file) != sizeof(pair) ||
             SparkWeightdManifestFind(manifest,pair[0],pair[1]) == 0 )
            goto done;
        for (index=0u; index<*count; index++)
            if ( keys[index].layer == pair[0] && keys[index].expert == pair[1] )
                break;
        if ( index == *count )
        {
            keys[index].layer = pair[0];
            keys[index].expert = pair[1];
            (*count)++;
        }
    }
    valid = fgetc(file) == EOF && !ferror(file) && *count != 0u;
done:
    if ( fclose(file) != 0 )
        valid = 0;
    return valid;
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
    uint64_t layers = 45u,experts = 288u,seconds = 1800u,topology;
    uint32_t first,index,count = 0u;
    int exit_status = 1;
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
         !parse_positive(getenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES"),
             SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT,&request.expert_pool_bytes) )
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
    /* Family tag override (additive, lane 4): the attach identity model must
       equal the family module tag ("glm5_next_stage", "dsv41_flash_stage",
       ...) so the daemon reuses ONE arena per pack instead of mapping the
       same pack twice under two identities. Default keeps the GLM behavior
       byte-identical. */
    {
        const char *warm_model = getenv("SPARK_WEIGHTD_WARM_MODEL");
        if ( warm_model == 0 )
            warm_model = "glm5_next_stage";
        if ( strlen(warm_model) >= sizeof(request.identity.model) )
            goto usage;
        strcpy(request.identity.model,warm_model);
    }
    strcpy(request.identity.revision,arguments[4]);
    request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    request.identity.arena_bytes = (uint64_t)pack.st_size;
    request.identity.topology = (uint32_t)topology;
    strcpy(request.pack_path,arguments[2]);
    snprintf(manifest_path,sizeof(manifest_path),"%s.experts",arguments[2]);
    status = SparkWeightdManifestLoad(manifest_path,request.identity.arena_bytes,&manifest);
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
    keys = calloc(SPARK_WEIGHTD_LEASE_GROUPS_MAX,sizeof(*keys));
    if ( keys == 0 )
        goto done;
    if ( wset_path != 0 && !read_wset(wset_path,&manifest,keys,&count) )
    {
        fprintf(stderr,"weightd_warm: invalid wset %s; require 1..%u complete manifest key pairs\n",
            wset_path,SPARK_WEIGHTD_LEASE_GROUPS_MAX);
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
        (void)clock_gettime(CLOCK_MONOTONIC,&began);
        fprintf(stderr,"weightd_warm: WSET-ONE-SHOT file=%s keys=%u\n",wset_path,count);
        if ( !warm_keys(client,attached.arena_generation,keys,count,seconds * UINT64_C(1000000000)) )
            goto done;
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
        "       finite SPARK_WEIGHTD_EXPERT_POOL_BYTES is required\n"
        "       SPARK_WEIGHTD_WARM_MODEL overrides the attach identity tag (default glm5_next_stage)\n");
    return 2;
}
