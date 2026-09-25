import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
source = (ROOT / "model-families/common/include/sparkpipe/spark_tp_mesh_kernels.cuh").read_text()
begin = source.index("#define SPARK_TP_MESH_WAIT_SLEEP_NS")
end = source.index("\n\nstatic __device__ __forceinline__ float2", begin)
body = source[begin:end]
prelude = r'''
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <sched.h>
#define __global__
#define SPARK_TP_MESH_ERROR_CANCELLED 0xFFFFFFFFFE000000ull
static struct { unsigned x; } threadIdx={0},blockIdx={0};
static volatile uint64_t cancel_cell;
static volatile unsigned gate,parked;
static unsigned cancel_reads;
static unsigned long long SparkGlm5NextGlobalTimerNs(void)
{
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC,&time);
    return (unsigned long long)time.tv_sec * 1000000000ull + time.tv_nsec;
}
static unsigned long long SparkGlm5NextLdcvU64(const volatile void *address)
{
    if ( address == &cancel_cell && ++cancel_reads >= 2u &&
         __atomic_load_n(&gate,__ATOMIC_ACQUIRE) != 0u )
    {
        __atomic_store_n(&parked,1u,__ATOMIC_RELEASE);
        while ( __atomic_load_n(&gate,__ATOMIC_ACQUIRE) != 0u ) sched_yield();
    }
    return *(const volatile uint64_t *)address;
}
static void __nanosleep(unsigned nanoseconds) { (void)nanoseconds; sched_yield(); }
static unsigned long long atomicExch(unsigned long long *address,unsigned long long value)
{ return __atomic_exchange_n(address,value,__ATOMIC_SEQ_CST); }
'''
probe = r'''
static uint64_t band[32];
static unsigned long long tag=0x100000001ull,error_word,diag_word,expected;
static void *WaitMain(void *)
{
    SparkGlm5NextMeshWaitKernel(band,64u,&tag,2u,0u,2u,&error_word,
        100000000ull,&diag_word,&cancel_cell,&expected,0);
    return 0;
}
int main(void)
{
    pthread_t thread;
    gate = 1u;
    if ( pthread_create(&thread,0,WaitMain,0) != 0 ) return 2;
    uint64_t stop = SparkGlm5NextGlobalTimerNs() + 1000000000ull;
    while ( __atomic_load_n(&parked,__ATOMIC_ACQUIRE) == 0u )
    {
        if ( SparkGlm5NextGlobalTimerNs() >= stop ) return 3;
        sched_yield();
    }
    cancel_cell = 1u;
    expected = 1u;
    __atomic_store_n(&gate,0u,__ATOMIC_RELEASE);
    pthread_join(thread,0);
    if ( error_word != (SPARK_TP_MESH_ERROR_CANCELLED | tag) )
    {
        fprintf(stderr,"FAIL old wait lost cancellation after rearm error=%llx\n",error_word);
        return 1;
    }
    error_word = 0u;
    tag = 0x200000001ull;
    band[23] = tag;
    WaitMain(0);
    if ( error_word != 0u ) return 4;
    puts("PASS captured CUDA wait body: immutable cancel generation and drained rearm");
    return 0;
}
'''
with tempfile.TemporaryDirectory(prefix="tp-mesh-cancel-") as directory:
    fixture = pathlib.Path(directory) / "cancel.cpp"
    binary = pathlib.Path(directory) / "cancel"
    fixture.write_text(prelude + body + probe)
    subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
                    str(fixture), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=5)
    mutation = body.replace("     expected_cancel )", "     SparkGlm5NextLdcvU64(cancel_expected) )")
    assert mutation != body
    fixture.write_text(prelude + mutation + probe)
    subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
                    str(fixture), "-o", str(binary)], check=True)
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 1 and "lost cancellation after rearm" in result.stderr, result
    print("PASS negative control: mutable generation times out instead of cancelling")
