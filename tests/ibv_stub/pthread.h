#pragma once

#include_next <pthread.h>

#if defined(__APPLE__)
#include <stddef.h>
#include <stdint.h>
#define CPU_SETSIZE 1024
typedef struct
{
    uint64_t bits[16];
} cpu_set_t;
static inline void CPU_ZERO(cpu_set_t *set)
{
    for ( size_t i = 0u; i < 16u; i++ )
        set->bits[i] = 0u;
}
static inline void CPU_SET(uint32_t cpu,cpu_set_t *set)
{
    if ( cpu < CPU_SETSIZE )
        set->bits[cpu / 64u] |= UINT64_C(1) << (cpu % 64u);
}
static inline int pthread_setaffinity_np(pthread_t thread,size_t size,const cpu_set_t *set)
{
    (void)thread;
    (void)size;
    (void)set;
    return 0;
}
#endif
