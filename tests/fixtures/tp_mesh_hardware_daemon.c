#include "tests/fixtures/tp_mesh_hardware_fixture.h"
#include "node/weightd_mesh.c"
#include <assert.h>

void SparkTestMeshWaitInitialize(void *region,uint32_t degree)
{
    uint32_t lane,rank;
    assert(degree >= 2u && degree <= SPARK_WEIGHTD_MESH_RANKS_PER_BAND && region != 0);
    memset(&weightd_mesh,0,sizeof(weightd_mesh));
    weightd_mesh.recv_buffer = region;
    weightd_mesh.rank_mask = (1u << degree) - 1u;
    weightd_mesh.activity_owners = degree;
    for (lane=0u; lane<SPARK_WEIGHTD_MESH_MAX_LANES; lane++)
    {
        weightd_mesh.lane_activity[lane] = 1u;
        weightd_mesh.lane_topology[lane].rank_count = degree;
        for (rank=0u; rank<degree; rank++)
            weightd_mesh.lane_topology[lane].physical_ranks[rank] = rank;
    }
    weightd_mesh.mesh_ready = 1u;
}

void SparkTestMeshWaitPoll(uint64_t now_ns)
{
    uint32_t rank;
    for (rank=0u; rank<weightd_mesh.lane_topology[0].rank_count; rank++)
    {
        uint32_t index = rank * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + rank;
        weightd_mesh.lane_topology[rank / 2u].local_rank = rank;
        SparkWeightdMeshWaitRequestsPollRange(now_ns,index,index + 1u);
    }
}
