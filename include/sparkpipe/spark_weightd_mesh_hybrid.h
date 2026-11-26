#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_WEIGHTD_MESH_CAPABILITY_HYBRID_SQ 256u
#define SPARK_WEIGHTD_MESH_HYBRID_MAGIC UINT64_C(0x5350485942524433)
#define SPARK_WEIGHTD_MESH_HYBRID_VERSION 1u
#define SPARK_WEIGHTD_MESH_HYBRID_OFFSET 32768u
#define SPARK_WEIGHTD_MESH_HYBRID_SHM_PATH_BYTES 128u
#define SPARK_WEIGHTD_MESH_HYBRID_SHM_BYTES (16u * 1024u * 1024u)
#define SPARK_WEIGHTD_MESH_HYBRID_DBR_ARENA_BYTES 65536u
#define SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS 16u
#define SPARK_WEIGHTD_MESH_HYBRID_OFFSET_NONE UINT32_MAX
#define SPARK_WEIGHTD_MESH_HYBRID_FLAG_DBREC_IN_SHM 1u
#define SPARK_WEIGHTD_MESH_HYBRID_FLAG_BF_UNIQUE 2u

#if defined(__CUDACC__)
#define SPARK_WEIGHTD_MESH_HYBRID_HD __host__ __device__ __forceinline__
#else
#define SPARK_WEIGHTD_MESH_HYBRID_HD inline
#endif

typedef struct SparkWeightdMeshHybridPeer
{
    uint32_t sq_shm_off;
    uint32_t dbrec_shm_off;
    uint32_t sq_depth;
    uint32_t sq_mask;
    uint32_t qpn_shifted;
    uint32_t rkey_be;
    uint32_t recv_lkey_be;
    uint32_t staging_lkey_be;
    uint64_t remote_base;
    uint64_t staging_peer_off;
    uint32_t max_inline;
    uint8_t peer;
    uint8_t target;
    uint16_t flags;
    uint32_t pad0;
    uint32_t pad1;
} SparkWeightdMeshHybridPeer;

typedef struct SparkWeightdMeshHybridTrigger
{
    uint64_t seq;
    uint64_t meta;
    uint64_t slot;
    uint64_t bytes;
    uint64_t pad[4];
} SparkWeightdMeshHybridTrigger;

typedef struct SparkWeightdMeshHybridCtrl
{
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    uint64_t shm_generation;
    uint64_t shm_bytes;
    uint64_t recv_iova_base;
    uint64_t staging_iova_base;
    uint32_t owner_lane;
    uint32_t owner_band;
    uint32_t owner_rank;
    uint32_t degree;
    uint32_t peer_count;
    uint32_t all_peers_mask;
    uint32_t min_max_inline;
    uint32_t dbrec_in_shm;
    uint64_t pad_hdr[6];
    char shm_path[SPARK_WEIGHTD_MESH_HYBRID_SHM_PATH_BYTES];
    uint64_t cpu_mode;
    uint64_t grant_gen;
    uint64_t pad_cpu_mode[6];
    uint64_t gpu_busy;
    uint64_t gpu_last_gen;
    uint64_t pad_gpu_busy[6];
    SparkWeightdMeshHybridTrigger trigger[2];
    uint32_t gpu_pi[SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS];
    uint32_t gpu_wrs[SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS];
    uint32_t cpu_sq_limit[SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS];
    uint32_t cpu_wr_limit[SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS];
    SparkWeightdMeshHybridPeer peers[SPARK_WEIGHTD_MESH_HYBRID_PEER_SLOTS];
} SparkWeightdMeshHybridCtrl;

static SPARK_WEIGHTD_MESH_HYBRID_HD uint64_t SparkWeightdMeshHybridMetaPack(
    uint32_t peer_mask,uint32_t ds,uint32_t num_bb,uint32_t grant_gen)
{
    return (uint64_t)(peer_mask & 0xffffu) |
        ((uint64_t)(ds & 0xffu) << 16u) |
        ((uint64_t)(num_bb & 0xffu) << 24u) |
        ((uint64_t)grant_gen << 32u);
}

static SPARK_WEIGHTD_MESH_HYBRID_HD uint32_t SparkWeightdMeshHybridMetaPeerMask(uint64_t meta)
{
    return (uint32_t)(meta & 0xffffu);
}

static SPARK_WEIGHTD_MESH_HYBRID_HD uint32_t SparkWeightdMeshHybridMetaDs(uint64_t meta)
{
    return (uint32_t)((meta >> 16u) & 0xffu);
}

static SPARK_WEIGHTD_MESH_HYBRID_HD uint32_t SparkWeightdMeshHybridMetaNumBb(uint64_t meta)
{
    return (uint32_t)((meta >> 24u) & 0xffu);
}

static SPARK_WEIGHTD_MESH_HYBRID_HD uint32_t SparkWeightdMeshHybridMetaGrantGen(uint64_t meta)
{
    return (uint32_t)(meta >> 32u);
}

#if !defined(__cplusplus)
_Static_assert(sizeof(SparkWeightdMeshHybridPeer) == 64u,
    "hybrid peer descriptor fits one cache line");
_Static_assert(sizeof(SparkWeightdMeshHybridTrigger) == 64u,
    "hybrid trigger occupies one cache line");
_Static_assert(offsetof(SparkWeightdMeshHybridCtrl,cpu_mode) == 256u &&
    offsetof(SparkWeightdMeshHybridCtrl,gpu_busy) == 320u &&
    offsetof(SparkWeightdMeshHybridCtrl,trigger) == 384u &&
    offsetof(SparkWeightdMeshHybridCtrl,gpu_pi) == 512u &&
    offsetof(SparkWeightdMeshHybridCtrl,cpu_sq_limit) == 640u &&
    offsetof(SparkWeightdMeshHybridCtrl,peers) == 768u &&
    sizeof(SparkWeightdMeshHybridCtrl) == 1792u,
    "hybrid control block keeps CPU, GPU, trigger, credit, and peer lines isolated");
#endif

#ifdef __cplusplus
}
#endif
