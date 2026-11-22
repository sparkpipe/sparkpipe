from pathlib import Path
import argparse
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
static int test_close(int fd);
#define close test_close
#include "MAP_SOURCE"
#undef close
static int watched[2],close_count[2],replacement;
static unsigned imports,exports,mode;
static int test_close(int fd)
{
    unsigned index;
    for (index=0u; index<2u && watched[index]!=fd; index++);
    assert(index<2u);
    close_count[index]++;
    int result=close(fd);
    if (close_count[index]==1) assert(dup2(replacement,fd)==fd);
    return result;
}
SparkStatus SparkWeightdClientSlotExportBatch(SparkWeightdClient *client,uint64_t arena_generation,uint32_t slot_offset,SparkWeightdSlotBatch *batch,uint64_t timeout_nanoseconds)
{
    (void)client;(void)timeout_nanoseconds;
    assert(arena_generation==7u && slot_offset==0u && exports++==0u);
    memset(batch,0,sizeof(*batch));
    batch->status=SPARK_STATUS_OK;
    batch->chunk_bytes=mode==1u?8192u:4096u;
    batch->slot_count=2u;
    batch->batch_count=2u;
    batch->fds[0]=watched[0];
    batch->fds[1]=watched[1];
    return(SPARK_STATUS_OK);
}
SparkStatus SparkWeightdClientLeaseSlotsBatch(SparkWeightdClient *client,uint64_t arena_generation,uint64_t lease_identifier,uint32_t batch_offset,SparkWeightdLeaseSlotBatch *batch,uint64_t timeout_nanoseconds)
{
    (void)client;(void)arena_generation;(void)lease_identifier;(void)batch_offset;(void)batch;(void)timeout_nanoseconds;
    abort();
}
SparkStatus SparkDeviceMemoryImport(SparkDeviceMemoryHandle *memory,int fd,const char *owner)
{
    (void)owner;
    assert(fd==watched[imports]);
    imports++;
    if (mode==2u) return SPARK_STATUS_INVALID_ARGUMENT;
    *memory=imports;
    return SPARK_STATUS_OK;
}
SparkStatus SparkDeviceMemoryMap(SparkDeviceAddress address,uint64_t bytes,SparkDeviceMemoryHandle memory,uint64_t offset,uint32_t access)
{
    (void)address;(void)bytes;(void)memory;(void)offset;(void)access;
    abort();
}
static void check(unsigned check_mode)
{
    SparkWeightdMap map={0};
    SparkStatus expected;
    mode=check_mode;
    imports=exports=0u;
    replacement=open("/dev/zero",O_RDONLY);
    assert(replacement>=0);
    for (unsigned i=0u;i<2u;i++)
    {
        watched[i]=open("/dev/null",O_RDONLY);
        assert(watched[i]>=0);
        close_count[i]=0;
    }
    map.generation=7u;map.chunk_bytes=4096u;map.chunk_count=2u;
    expected=mode==1u?SPARK_STATUS_SCHEMA_ERROR:mode==2u?SPARK_STATUS_IO_ERROR:SPARK_STATUS_OK;
    assert(map_import_slots(&map)==expected);
    assert(exports==1u && imports==(mode==0u?2u:mode==2u?1u:0u));
    assert(mode!=0u || (map.slot_count==2u && map.slot_handles[0]==1u &&
        map.slot_handles[1]==2u && map.chunk_slots[0]==SPARK_WEIGHTD_SLOT_NONE && map.chunk_slots[1]==SPARK_WEIGHTD_SLOT_NONE));
    for (unsigned i=0u;i<2u;i++)
    {
        unsigned char value=255u;
        assert(fcntl(watched[i],F_GETFD)>=0);
        assert(read(watched[i],&value,1u)==1 && value==0u);
        assert(close_count[i]==1);
        assert(close(watched[i])==0);
    }
    free(map.slot_handles);
    free(map.chunk_slots);
    assert(close(replacement)==0);
}
int main(void)
{
    for (unsigned check_mode=0u;check_mode<3u;check_mode++) check(check_mode);
    puts("PASS weightd map slot descriptor ownership: imported, schema rejection, import failure; every received descriptor closed exactly once and six reused descriptors preserved");
    return 0;
}
'''


CREATE_HARNESS = r'''
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
static int test_close(int fd);
#define close test_close
#include "MAP_SOURCE"
#undef close
static unsigned init_fault,cleanup_faults,cleanup_hits,events,live_events[SPARK_WEIGHTD_LEASE_COUNT_MAX];
static unsigned handles[2],mappings[2],reserved,imports,maps;
static int fds[2],closes[2];
static const SparkDeviceAddress base=65536u;
static int cleanup(unsigned phase)
{
    unsigned bit=1u<<phase;
    if ((cleanup_faults & bit)==0u) return 0;
    cleanup_faults &= ~bit;
    cleanup_hits |= bit;
    return 1;
}
static int test_close(int fd)
{
    unsigned i=fd==fds[0]?0u:1u;
    assert(fd==fds[i] && closes[i]++==0);
    return close(fd);
}
cudaError_t cudaGetDevice(int *device) { *device=0;return cudaSuccess; }
SparkStatus SparkDeviceInitialize(const SparkDeviceConfiguration *configuration) { (void)configuration;return SPARK_STATUS_OK; }
SparkStatus SparkDeviceThreadInstance(SparkDeviceInstance *instance) { *instance=1u;return SPARK_STATUS_OK; }
SparkStatus SparkWeightdClientSlotExportBatch(SparkWeightdClient *client,uint64_t arena_generation,uint32_t slot_offset,SparkWeightdSlotBatch *batch,uint64_t timeout_nanoseconds)
{
    (void)client;(void)arena_generation;(void)slot_offset;(void)batch;(void)timeout_nanoseconds;
    abort();
}
SparkStatus SparkWeightdClientLeaseSlotsBatch(SparkWeightdClient *client,uint64_t arena_generation,uint64_t lease_identifier,uint32_t batch_offset,SparkWeightdLeaseSlotBatch *batch,uint64_t timeout_nanoseconds)
{
    (void)client;(void)arena_generation;(void)lease_identifier;(void)batch_offset;(void)batch;(void)timeout_nanoseconds;
    abort();
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,unsigned flags)
{
    assert(flags==cudaEventDisableTiming && events<SPARK_WEIGHTD_LEASE_COUNT_MAX);
    live_events[events]=1u;*event=(cudaEvent_t)(uintptr_t)(++events);
    return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    unsigned i=(unsigned)(uintptr_t)event-1u;
    assert(i<events && live_events[i]);
    if (cleanup(5u)) return 1;
    live_events[i]=0u;return cudaSuccess;
}
SparkStatus SparkDeviceMemoryGranularity(uint32_t granularity,uint64_t *bytes)
{
    assert(granularity==SPARK_DEVICE_GRANULARITY_MINIMUM);*bytes=4096u;return SPARK_STATUS_OK;
}
SparkStatus SparkDeviceAddressReserve(SparkDeviceAddress *address,uint64_t bytes,const char *owner)
{
    (void)owner;
    assert(bytes==12288u && reserved==0u);
    reserved=1u;*address=base;return SPARK_STATUS_OK;
}
SparkStatus SparkDeviceMemoryImport(SparkDeviceMemoryHandle *memory,int fd,const char *owner)
{
    unsigned i=fd==fds[0]?0u:1u;
    (void)owner;
    assert(fd==fds[i]);
    imports++;
    if (init_fault==1u+2u*i) return SPARK_STATUS_INVALID_ARGUMENT;
    assert(handles[i]==0u);handles[i]=1u;*memory=i+1u;
    return SPARK_STATUS_OK;
}
SparkStatus SparkDeviceMemoryMap(SparkDeviceAddress address,uint64_t bytes,SparkDeviceMemoryHandle memory,uint64_t offset,uint32_t access)
{
    unsigned i=(unsigned)memory-1u;
    assert(i<2u && handles[i] && !mappings[i] && reserved);
    assert(address==base+(i==0u?8192u:0u));
    assert(bytes==(i==0u?4096u:8192u) && offset==0u);
    assert(access==(i==0u?SPARK_DEVICE_ACCESS_DEVICE_READ:(SPARK_DEVICE_ACCESS_DEVICE_READ|SPARK_DEVICE_ACCESS_DEVICE_WRITE)));
    maps++;
    if (init_fault==2u+2u*i) return SPARK_STATUS_INVALID_ARGUMENT;
    mappings[i]=1u;return SPARK_STATUS_OK;
}
SparkStatus SparkDeviceMemoryUnmap(SparkDeviceAddress address,uint64_t bytes)
{
    unsigned i=address==base?1u:0u;
    assert(address==base+(i==0u?8192u:0u));
    assert(bytes==(i==0u?4096u:8192u) && mappings[i] && reserved);
    if (cleanup(i)) return SPARK_STATUS_INVALID_ARGUMENT;
    mappings[i]=0u;return SPARK_STATUS_OK;
}
SparkStatus SparkDeviceMemoryRelease(SparkDeviceMemoryHandle memory)
{
    unsigned i=(unsigned)memory-1u;
    assert(i<2u && handles[i] && !mappings[i]);
    if (cleanup(2u+i)) return SPARK_STATUS_INVALID_ARGUMENT;
    handles[i]=0u;return SPARK_STATUS_OK;
}
SparkStatus SparkDeviceAddressRelease(SparkDeviceAddress address,uint64_t bytes)
{
    assert(address==base && bytes==12288u && reserved);
    assert(!mappings[0] && !mappings[1] && !handles[0] && !handles[1]);
    if (cleanup(4u)) return SPARK_STATUS_INVALID_ARGUMENT;
    reserved=0u;return SPARK_STATUS_OK;
}
static void check(unsigned fault,unsigned failures)
{
    SparkWeightdLazyAttachResult attached={0};
    SparkWeightdMap *map=0;
    unsigned retries=0u;
    init_fault=fault;cleanup_faults=failures;cleanup_hits=0u;
    events=0u;imports=maps=0u;
    for (unsigned i=0u;i<2u;i++)
    {
        assert(!handles[i] && !mappings[i]);
        fds[i]=open("/dev/null",O_RDONLY);assert(fds[i]>=0);closes[i]=0;
    }
    assert(!reserved);
    attached.status=SPARK_STATUS_OK;attached.arena_generation=1u;
    attached.arena_bytes=8192u;attached.chunk_bytes=4096u;attached.chunk_count=2u;
    SparkStatus status=SparkWeightdMapCreate((SparkWeightdClient *)1,&attached,
        fds[0],fds[1],&map);
    assert(status==(fault?SPARK_STATUS_IO_ERROR:SPARK_STATUS_OK));
    for (unsigned i=0u;i<2u;i++) assert(closes[i]==1 && fcntl(fds[i],F_GETFD)==-1);
    assert(imports==(fault && fault<3u?1u:2u));
    assert(maps==(fault==1u?0u:fault && fault<4u?1u:2u));
    assert((map!=0)==(fault==0u || failures!=0u));
    if (fault && map) assert(map->failure==SPARK_STATUS_IO_ERROR);
    while (map)
    {
        status=SparkWeightdMapDestroy(map);
        if (status==SPARK_STATUS_OK) map=0;
        else
        {
            assert(status==SPARK_STATUS_IO_ERROR && map->failure!=SPARK_STATUS_OK);
            assert(++retries<=6u);
        }
    }
    assert(cleanup_hits==failures && cleanup_faults==0u && !reserved);
    assert(!handles[0] && !handles[1] && !mappings[0] && !mappings[1]);
    for (unsigned i=0u;i<events;i++) assert(live_events[i]==0u);
}
static void check_retirement(void)
{
    SparkWeightdMap map={0};
    uint64_t owners[1]={3u};
    uint8_t mapped[1]={1u};
    SparkDeviceMemoryHandle imported[1]={2u};
    map.base=base;map.chunk_bytes=8192u;map.chunk_count=1u;
    map.owners=owners;map.mapped=mapped;map.handles=imported;
    reserved=1u;handles[1]=mappings[1]=1u;
    cleanup_faults=(1u<<1u)|(1u<<3u);cleanup_hits=0u;
    assert(map_drop_slot(&map,0u)==SPARK_STATUS_OK && owners[0]==2u);
    assert(map_drop_slot(&map,1u)==SPARK_STATUS_OK && owners[0]==0u);
    assert(mapped[0] && imported[0] && handles[1] && mappings[1] && cleanup_hits==0u);
    map.pool_mapped=1u;owners[0]=1u;
    assert(map_drop_slot(&map,0u)==SPARK_STATUS_OK && !owners[0]);
    assert(mapped[0] && imported[0] && handles[1] && mappings[1] && !cleanup_hits);
    cleanup_faults=0u;
    handles[1]=mappings[1]=reserved=0u;
}
int main(void)
{
    for (unsigned phase=0u;phase<=4u;phase++) check(phase,0u);
    for (unsigned phase=0u;phase<6u;phase++)
    {
        check(0u,1u<<phase);
        if (phase!=1u) check(4u,1u<<phase);
    }
    check(0u,63u);check(4u,63u&~2u);
    check_retirement();
    puts("PASS weightd map create: 18 actual-source cases; four init failures, six cleanup retry boundaries, combined retries, exact resource ownership; a released lease keeps every mapping in slot and pooled maps");
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'runtime/spark_weightd_map.c')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='weightd-map-fd-') as directory:
        temp = Path(directory)
        source = temp / 'test.c'
        command = [os.environ.get('CC', 'cc'), '-std=c11', '-D_POSIX_C_SOURCE=200809L',
                   '-ffunction-sections', '-fdata-sections', '-I' + str(ROOT), '-I' + str(ROOT / 'include'),
                   '-I' + str(ROOT / 'tests/cuda_stub'), str(source), '-pthread',
                   '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
                   '-o', str(temp / 'test')]
        if args.sanitize:
            command += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        for harness in (HARNESS, CREATE_HARNESS):
            source.write_text(harness.replace('MAP_SOURCE', str(args.source.resolve())))
            subprocess.run(command, check=True)
            subprocess.run([str(temp / 'test')], check=True, timeout=10)


if __name__ == '__main__':
    main()
