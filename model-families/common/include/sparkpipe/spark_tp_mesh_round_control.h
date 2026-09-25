#pragma once
#include <stddef.h>
#include <stdint.h>

#define SPARK_TP_MESH_ROUND_CONTROL_WORDS 18u
#define SPARK_TP_MESH_ROUND_CONTROL_BYTES \
    (SPARK_TP_MESH_ROUND_CONTROL_WORDS * sizeof(uint64_t))

#define SPARK_TP_MESH_ROUND_CONTROL_WORD_SLOT_CURSOR 0u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_ROUNDS_TOTAL 1u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_ROUNDS_DONE 2u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_ROUND_SEQ 3u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_CANCEL_EXPECTED 4u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_DEADLINE_NS 5u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_SEQ 6u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_EPOCH 7u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_ERROR 8u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_DIAG 9u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_SOURCE_WAIT_NS 10u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_PEER_WAIT_NS 11u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_COPY_NS 12u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_COMBINE_NS 13u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_WAIT_STARTED_NS 14u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_MATH_STARTED_NS 15u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_MATH_FINISHED_NS 16u
#define SPARK_TP_MESH_ROUND_CONTROL_WORD_MATH_BLOCKS_DONE 17u

typedef struct SparkTpMeshRoundControl
{
    uint64_t slot_cursor;
    uint64_t rounds_total;
    uint64_t rounds_done;
    uint64_t round_seq;
    uint64_t cancel_expected;
    uint64_t deadline_ns;
    uint64_t seq;
    uint64_t epoch;
    uint64_t error_word;
    uint64_t diag_word;
    uint64_t source_wait_ns;
    uint64_t peer_wait_ns;
    uint64_t copy_ns;
    uint64_t combine_ns;
    uint64_t wait_started_ns;
    uint64_t math_started_ns;
    uint64_t math_finished_ns;
    uint64_t math_blocks_done;
} SparkTpMeshRoundControl;

#define SPARK_TP_MESH_ROUND_LOOP_DECISION_GO 0u
#define SPARK_TP_MESH_ROUND_LOOP_DECISION_CANCEL 1u
#define SPARK_TP_MESH_ROUND_LOOP_DECISION_TIMEOUT 2u

#if defined(__CUDACC__)
__host__ __device__
#endif
static inline uint32_t SparkTpMeshTreeLevels(uint32_t degree)
{
    uint32_t levels = 0u;
    while ( (1u << levels) < degree ) levels++;
    return levels;
}

#if defined(__CUDACC__)
__host__ __device__
#endif
static inline uint64_t SparkTpMeshDirectCapacity(uint64_t slot_bytes,uint32_t operation)
{
    return ((slot_bytes - 16u) / (operation == 2u ? 8u : 2u)) & ~UINT64_C(3);
}

#if defined(__CUDACC__)
__host__ __device__
#endif
static inline uint64_t SparkTpMeshDirectLocalElements(uint64_t elements,uint32_t degree,uint32_t operation)
{
    return operation == 0u ? elements / degree : elements;
}

#if defined(__CUDACC__)
__host__ __device__
#endif
static inline uint64_t SparkTpMeshDirectChunks(uint64_t elements,uint32_t degree,uint32_t operation,uint64_t slot_bytes)
{
    return (SparkTpMeshDirectLocalElements(elements,degree,operation) - 1u) / SparkTpMeshDirectCapacity(slot_bytes,operation) + 1u;
}

#if defined(__CUDACC__)
__host__ __device__
#endif
static inline uint32_t SparkTpMeshTreeRoute(uint32_t rank,uint32_t degree,
    uint32_t phase)
{
    uint32_t levels = SparkTpMeshTreeLevels(degree);
    uint32_t reduce = phase < levels;
    uint32_t step = 1u << (reduce != 0u ? phase : 2u * levels - phase - 1u);
    uint32_t residue = rank % (2u * step);
    uint32_t peer;
    if ( residue == 0u && rank + step < degree ) peer = rank + step;
    else if ( residue == step ) peer = rank - step;
    else return 0u;
    return (residue == step) == (reduce != 0u) ?
        (peer + 1u) << 16u : peer + 1u;
}
