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
static unsigned imports;
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
CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle,
    void *fd,CUmemAllocationHandleType type)
{
    (void)handle;(void)fd;(void)type;imports++;
    return CUDA_ERROR_INVALID_VALUE;
}
CUresult cuMemMap(CUdeviceptr address,size_t bytes,size_t offset,
    CUmemGenericAllocationHandle handle,unsigned long long flags)
{
    (void)address;(void)bytes;(void)offset;(void)handle;(void)flags;
    abort();
}
CUresult cuMemSetAccess(CUdeviceptr address,size_t bytes,
    const CUmemAccessDesc *descriptors,size_t count)
{
    (void)address;(void)bytes;(void)descriptors;(void)count;
    abort();
}
static void check(unsigned mode)
{
    SparkWeightdMap map={0};
    SparkWeightdExportBatch batch={0};
    CUmemGenericAllocationHandle handles[2]={0};
    uint64_t owners[2]={0};
    uint8_t mapped[2]={1u,1u};
    uint32_t last=0u;
    imports=0u;
    replacement=open("/dev/zero",O_RDONLY);
    assert(replacement>=0);
    for (unsigned i=0u;i<2u;i++)
    {
        watched[i]=open("/dev/null",O_RDONLY);
        assert(watched[i]>=0);
        close_count[i]=0;
        batch.fds[i]=watched[i];
        batch.chunk_indices[i]=i;
    }
    map.chunk_bytes=4096u;map.chunk_count=2u;
    map.mapped=mapped;map.owners=owners;map.handles=handles;
    batch.chunk_bytes=mode==1u?8192u:4096u;
    batch.chunk_count=2u;batch.batch_count=2u;
    if (mode==2u) mapped[0]=0u;
    SparkStatus expected=mode==1u?SPARK_STATUS_SCHEMA_ERROR:
        mode==2u?SPARK_STATUS_IO_ERROR:SPARK_STATUS_OK;
    assert(map_import_batch(&map,0u,&batch,&last,0u,2u,
        map_now()+UINT64_C(1000000000))==expected);
    assert(imports==(mode==2u?1u:0u));
    for (unsigned i=0u;i<2u;i++)
    {
        unsigned char value=255u;
        assert(fcntl(watched[i],F_GETFD)>=0);
        assert(read(watched[i],&value,1u)==1 && value==0u);
        assert(close_count[i]==1);
        assert(close(watched[i])==0);
    }
    assert(close(replacement)==0);
}
int main(void)
{
    for (unsigned mode=0u;mode<3u;mode++) check(mode);
    puts("PASS weightd map descriptor ownership: cached, schema rejection, import failure; six reused descriptors preserved");
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
static unsigned handles[2],mappings[2],reserved,imports,maps,accesses;
static int fds[2],closes[2];
static const CUdeviceptr base=65536u;
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
CUresult cuCtxGetCurrent(CUcontext *context) { *context=(CUcontext)1;return CUDA_SUCCESS; }
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
CUresult cuMemGetAllocationGranularity(size_t *value,const CUmemAllocationProp *prop,
    CUmemAllocationGranularity_flags flags)
{
    (void)prop;(void)flags;*value=4096u;return CUDA_SUCCESS;
}
CUresult cuMemAddressReserve(CUdeviceptr *address,size_t bytes,size_t alignment,
    CUdeviceptr hint,unsigned long long flags)
{
    assert(bytes==12288u && alignment==0u && hint==0u && flags==0u && reserved==0u);
    reserved=1u;*address=base;return CUDA_SUCCESS;
}
CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle,
    void *fd,CUmemAllocationHandleType type)
{
    unsigned i=(intptr_t)fd==fds[0]?0u:1u;
    assert((intptr_t)fd==fds[i] && type==CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    imports++;
    if (init_fault==1u+3u*i) return CUDA_ERROR_INVALID_VALUE;
    assert(handles[i]==0u);handles[i]=1u;*handle=(CUmemGenericAllocationHandle)(uintptr_t)(i+1u);
    return CUDA_SUCCESS;
}
CUresult cuMemMap(CUdeviceptr address,size_t bytes,size_t offset,
    CUmemGenericAllocationHandle handle,unsigned long long flags)
{
    unsigned i=(unsigned)(uintptr_t)handle-1u;
    assert(i<2u && handles[i] && !mappings[i] && reserved);
    assert(address==base+(i==0u?8192u:0u));
    assert(bytes==(i==0u?4096u:8192u) && offset==0u && flags==0u);
    maps++;
    if (init_fault==2u+3u*i) return CUDA_ERROR_INVALID_VALUE;
    mappings[i]=1u;return CUDA_SUCCESS;
}
CUresult cuMemSetAccess(CUdeviceptr address,size_t bytes,
    const CUmemAccessDesc *descriptors,size_t count)
{
    unsigned i=address==base?1u:0u;
    assert(mappings[i] && count==1u && bytes==(i==0u?4096u:8192u));
    assert(descriptors->flags==(i==0u?CU_MEM_ACCESS_FLAGS_PROT_READ:CU_MEM_ACCESS_FLAGS_PROT_READWRITE));
    accesses++;
    if (init_fault==3u+3u*i) return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
}
CUresult cuMemUnmap(CUdeviceptr address,size_t bytes)
{
    unsigned i=address==base?1u:0u;
    assert(address==base+(i==0u?8192u:0u));
    assert(bytes==(i==0u?4096u:8192u) && mappings[i] && reserved);
    if (cleanup(i)) return CUDA_ERROR_INVALID_VALUE;
    mappings[i]=0u;return CUDA_SUCCESS;
}
CUresult cuMemRelease(CUmemGenericAllocationHandle handle)
{
    unsigned i=(unsigned)(uintptr_t)handle-1u;
    assert(i<2u && handles[i] && !mappings[i]);
    if (cleanup(2u+i)) return CUDA_ERROR_INVALID_VALUE;
    handles[i]=0u;return CUDA_SUCCESS;
}
CUresult cuMemAddressFree(CUdeviceptr address,size_t bytes)
{
    assert(address==base && bytes==12288u && reserved);
    assert(!mappings[0] && !mappings[1] && !handles[0] && !handles[1]);
    if (cleanup(4u)) return CUDA_ERROR_INVALID_VALUE;
    reserved=0u;return CUDA_SUCCESS;
}
static void check(unsigned fault,unsigned failures)
{
    SparkWeightdLazyAttachResult attached={0};
    SparkWeightdMap *map=0;
    unsigned retries=0u;
    init_fault=fault;cleanup_faults=failures;cleanup_hits=0u;
    events=0u;imports=maps=accesses=0u;
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
    assert(imports==(fault && fault<4u?1u:2u));
    assert(maps==(fault==1u?0u:fault && fault<5u?1u:2u));
    assert(accesses==(fault && fault<3u?0u:fault && fault<6u?1u:2u));
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
    CUmemGenericAllocationHandle imported[1]={(CUmemGenericAllocationHandle)(uintptr_t)2u};
    map.base=base;map.chunk_bytes=8192u;map.chunk_count=1u;
    map.owners=owners;map.mapped=mapped;map.handles=imported;
    reserved=1u;handles[1]=mappings[1]=1u;
    cleanup_faults=(1u<<1u)|(1u<<3u);cleanup_hits=0u;
    assert(map_drop_slot(&map,0u)==SPARK_STATUS_OK && owners[0]==2u);
    assert(mapped[0] && handles[1] && mappings[1] && cleanup_hits==0u);
    assert(map_drop_slot(&map,1u)==SPARK_STATUS_IO_ERROR);
    assert(owners[0]==2u && mapped[0] && handles[1] && mappings[1]);
    assert(map_drop_slot(&map,1u)==SPARK_STATUS_IO_ERROR);
    assert(owners[0]==2u && !mapped[0] && handles[1] && !mappings[1]);
    assert(map_drop_slot(&map,1u)==SPARK_STATUS_OK);
    assert(!owners[0] && !mapped[0] && !imported[0] && !handles[1] && !mappings[1]);
    assert(cleanup_faults==0u && cleanup_hits==((1u<<1u)|(1u<<3u)));
    map.pool_mapped=1u;owners[0]=1u;mapped[0]=1u;imported[0]=(CUmemGenericAllocationHandle)(uintptr_t)2u;
    handles[1]=mappings[1]=1u;cleanup_hits=0u;
    assert(map_drop_slot(&map,0u)==SPARK_STATUS_OK && !owners[0]);
    assert(mapped[0] && imported[0] && handles[1] && mappings[1] && !cleanup_hits);
    handles[1]=mappings[1]=reserved=0u;
}
int main(void)
{
    for (unsigned phase=0u;phase<=6u;phase++) check(phase,0u);
    for (unsigned phase=0u;phase<6u;phase++)
    {
        check(0u,1u<<phase);
        check(6u,1u<<phase);
    }
    check(0u,63u);check(6u,63u);
    check_retirement();
    puts("PASS weightd map create: 21 actual-source cases; six init failures, six cleanup retry boundaries, combined retries, exact resource ownership; partial retirement retries and pooled retention");
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
                   '-ffunction-sections', '-fdata-sections', '-I' + str(ROOT / 'include'),
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
