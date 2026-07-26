#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "sparkpipe/spark_glm52_pp13_runtime.h"

static void SparkTestWriteFile(const char *path)
{
    FILE *file;

    file = fopen(path,"wb");
    assert(file != 0);
    assert(fputs("x\n",file) >= 0);
    assert(fclose(file) == 0);
}

static void SparkTestGlm52Pp13RuntimeRankPlan(void)
{
    SparkGlm52Pp13RuntimeRankPlan rank_plan;
    char error_buffer[256];
    char host_name[16];

    assert(SparkGlm52Pp13RuntimeRankHostName(
        0u,host_name,sizeof(host_name)) == SPARK_STATUS_OK);
    assert(strcmp(host_name,"10.10.100.10") == 0);
    assert(SparkGlm52Pp13RuntimeRankHostName(
        10u,host_name,sizeof(host_name)) == SPARK_STATUS_OK);
    assert(strcmp(host_name,"10.10.100.20") == 0);
    assert(SparkGlm52Pp13RuntimeRankHostName(
        12u,host_name,sizeof(host_name)) == SPARK_STATUS_OK);
    assert(strcmp(host_name,"10.10.100.22") == 0);
    assert(SparkGlm52Pp13RuntimeBuildRankPlan(
        0u,1024u,52100u,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,&rank_plan,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_OK);
    assert(rank_plan.first_layer_index == 0u);
    assert(rank_plan.layer_count == 6u);
    assert(rank_plan.listen_port == 52100u);
    assert(rank_plan.next_port == 52101u);
    assert(rank_plan.logical_lane_capacity == 1024u);
    assert(rank_plan.maximum_speculative_rows_per_lane == 8u);
    assert(rank_plan.execution_row_capacity == 1024u);
    assert(rank_plan.output_endpoint.max_active_sequence_count == 1024u);
    assert((rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_DENSE_PREFIX) != 0u);
    assert((rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) == 0u);
    assert(strcmp(rank_plan.next_host_name,"10.10.100.11") == 0);
    assert(strcmp(rank_plan.output_route_name,
        "10.10.100.10_to_10.10.100.11_hidden") == 0);
    assert(rank_plan.quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT);
    assert(SparkGlm52Pp13RuntimeExecutionRowCapacity(1u) == 8u);
    assert(SparkGlm52Pp13RuntimeExecutionRowCapacity(64u) == 512u);
    assert(SparkGlm52Pp13RuntimeExecutionRowCapacity(256u) == 1024u);
    assert(SparkGlm52Pp13RuntimeExecutionRowCapacity(1024u) == 1024u);
    assert(SparkGlm52Pp13RuntimeBuildRankPlan(
        12u,1024u,52100u,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,&rank_plan,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_OK);
    assert(rank_plan.first_layer_index == 72u);
    assert(rank_plan.layer_count == 6u);
    assert(rank_plan.listen_port == 52112u);
    assert(rank_plan.next_port == 0u);
    assert((rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u);
    assert((rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) == 0u);
    assert(strcmp(rank_plan.previous_host_name,"10.10.100.21") == 0);
    assert(strcmp(rank_plan.input_route_name,
        "10.10.100.21_to_10.10.100.22_hidden") == 0);
}

static void SparkTestGlm52Pp13RuntimeDsaCandidateBucket(void)
{
    assert(SparkGlm52Pp13RuntimeDsaCandidateBucket(0u) == 0u);
    assert(SparkGlm52Pp13RuntimeDsaCandidateBucket(1u) == 2048u);
    assert(SparkGlm52Pp13RuntimeDsaCandidateBucket(2048u) == 2048u);
    assert(SparkGlm52Pp13RuntimeDsaCandidateBucket(2049u) == 4096u);
    assert(SparkGlm52Pp13RuntimeDsaCandidateBucket(65536u) == 65536u);
    assert(SparkGlm52Pp13RuntimeDsaCandidateBucket(1048575u) == 1048576u);
    assert(SparkGlm52Pp13RuntimeDsaCandidateBucket(1048576u) == 1048576u);
    assert(SparkGlm52Pp13RuntimeDsaCandidateBucket(1048577u) == 0u);
}

static void SparkTestGlm52Pp13RuntimeFp8Packs(void)
{
    SparkGlm52Pp13RuntimeRankPlan rank_plan;
    char error_buffer[256];
    char pack_path[512];
    const char *pack_root;

    pack_root = "build/test_glm52_pp13_runtime_packs";
    (void)mkdir(pack_root,0775);
    (void)remove("build/test_glm52_pp13_runtime_packs/fp8_moe_pack_manifest.json");
    (void)remove("build/test_glm52_pp13_runtime_packs/resident_moe_pack_manifest.json");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0003_fp8_moe.spfp8");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0004_fp8_moe.spfp8");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0005_fp8_moe.spfp8");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0003_b12x_moe.spb12x");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0004_b12x_moe.spb12x");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0005_b12x_moe.spb12x");
    assert(SparkGlm52Pp13RuntimeBuildRankPlan(
        0u,1024u,52100u,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,&rank_plan,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_OK);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_NOT_FOUND);
    SparkTestWriteFile("build/test_glm52_pp13_runtime_packs/fp8_moe_pack_manifest.json");
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,3u,1u,0u,pack_path,sizeof(pack_path)) == SPARK_STATUS_OK);
    assert(strcmp(pack_path,
        "build/test_glm52_pp13_runtime_packs/glm52_layer_0003_fp8_moe.spfp8") == 0);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,4u,1u,0u,pack_path,sizeof(pack_path)) == SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_NOT_FOUND);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,5u,1u,0u,pack_path,sizeof(pack_path)) == SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_OK);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,SPARK_GLM52_MODEL_MTP_LAYER_INDEX,1u,0u,
        pack_path,sizeof(pack_path)) == SPARK_STATUS_OK);
    assert(strcmp(pack_path,
        "build/test_glm52_pp13_runtime_packs/glm52_layer_0078_fp8_moe.spfp8") == 0);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,SPARK_GLM52_MODEL_WEIGHT_LAYER_COUNT,1u,0u,
        pack_path,sizeof(pack_path)) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_MODULE_NOT_VALIDATED);
    assert(remove("build/test_glm52_pp13_runtime_packs/fp8_moe_pack_manifest.json") == 0);
    SparkTestWriteFile("build/test_glm52_pp13_runtime_packs/resident_moe_pack_manifest.json");
    assert(SparkGlm52Pp13RuntimeBuildRankPlan(
        0u,1024u,52100u,SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &rank_plan,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_NOT_FOUND);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,3u,1u,0u,pack_path,sizeof(pack_path)) ==
            SPARK_STATUS_OK);
    assert(strcmp(pack_path,
        "build/test_glm52_pp13_runtime_packs/glm52_layer_0003_b12x_moe.spb12x") == 0);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,4u,1u,0u,pack_path,sizeof(pack_path)) ==
            SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,5u,1u,0u,pack_path,sizeof(pack_path)) ==
            SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(remove("build/test_glm52_pp13_runtime_packs/resident_moe_pack_manifest.json") == 0);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_NOT_FOUND);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_NOT_FOUND);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,3u,1u,0u,pack_path,sizeof(pack_path)) ==
            SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,4u,1u,0u,pack_path,sizeof(pack_path)) ==
            SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,5u,1u,0u,pack_path,sizeof(pack_path)) ==
            SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,UINT32_MAX,3u,1u,0u,pack_path,sizeof(pack_path)) ==
            SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkGlm52Pp13RuntimeParseQuantizationMode(
        "fp8",&rank_plan.quantization_mode) == SPARK_STATUS_OK);
    assert(strcmp(SparkGlm52Pp13RuntimeQuantizationModeName(
        rank_plan.quantization_mode),"fp8") == 0);
    assert(SparkGlm52Pp13RuntimeParseQuantizationMode(
        "nvfp4",&rank_plan.quantization_mode) == SPARK_STATUS_OK);
    assert(strcmp(SparkGlm52Pp13RuntimeQuantizationModeName(
        rank_plan.quantization_mode),"nvfp4") == 0);
    assert(SparkGlm52Pp13RuntimeValidateFp8PlanCounts(
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,42u,42u) ==
        SPARK_STATUS_OK);
    assert(SparkGlm52Pp13RuntimeValidateFp8PlanCounts(
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,0u,0u) ==
        SPARK_STATUS_MODULE_NOT_VALIDATED);
    assert(SparkGlm52Pp13RuntimeValidateFp8PlanCounts(
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,0u,0u) ==
        SPARK_STATUS_OK);
    assert(SparkGlm52Pp13RuntimeValidateFp8PlanCounts(
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,1u,1u) ==
        SPARK_STATUS_MODULE_NOT_VALIDATED);
    assert(SparkGlm52Pp13RuntimeParseQuantizationMode(
        "auto",&rank_plan.quantization_mode) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkGlm52Pp13RuntimeExpectedMoeBackendKind(
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &rank_plan.quantization_mode) == SPARK_STATUS_OK);
    assert(rank_plan.quantization_mode ==
        SPARK_GLM52_PP13_RUNTIME_MOE_BACKEND_FP8_FLASHINFER_GROUPED);
    assert(SparkGlm52Pp13RuntimeExpectedMoeBackendKind(
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &rank_plan.quantization_mode) == SPARK_STATUS_OK);
    assert(rank_plan.quantization_mode ==
        SPARK_GLM52_PP13_RUNTIME_MOE_BACKEND_NVFP4_B12X);
}

static void SparkTestGlm52Pp13RuntimeFinalEventRoute(void)
{
    SparkGlm52Pp13RuntimeFinalEventRoute route;
    char error_buffer[256];

    assert(SparkGlm52Pp13RuntimeBuildFinalEventRoute(
        52100u,&route,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(route.source_rank_index == 12u);
    assert(route.sink_rank_index == 0u);
    assert(route.listen_port == 52300u);
    assert(route.connect_port == 52300u);
    assert(strcmp(route.source_host_name,"10.10.100.22") == 0);
    assert(strcmp(route.sink_host_name,"10.10.100.10") == 0);
    assert(strcmp(route.route_name,
        "10.10.100.22_to_10.10.100.10_final_events") == 0);
    assert(SparkGlm52Pp13RuntimeValidateFinalEventRoute(
        &route,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_OK);
    route.sink_rank_index = 1u;
    assert(SparkGlm52Pp13RuntimeValidateFinalEventRoute(
        &route,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_INVALID_ARGUMENT);
}

// Shape-driven rank plans: legacy builds carry degree-one shape defaults and
// a nonzero configuration hash; a TP4 x PP3 middle node derives the
// twenty-six layer slice, ring neighbors at the same TP rank one stage away,
// and same-stage collective peers; a tampered hash fails validation; and a
// shape whose linear node index exceeds the host table fails closed, which
// is exactly the TP16 state until the table grows with the hardware.
static void SparkTestPp13RuntimeShapePlans(void)
{
    SparkGlm52Pp13RuntimeRankPlan plan;
    SparkGlm52TpShapeDescriptor shape;
    char error_buffer[256];
    char pack_path[512];
    assert(SparkGlm52Pp13RuntimeBuildRankPlan(
        5u,16u,42000u,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &plan,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(plan.tp_degree == 1u && plan.tp_rank == 0u);
    assert(plan.pp_stage_count == 13u && plan.pp_stage_index == 5u);
    assert(plan.shape_configuration_hash != 0u);
    memset(&shape,0,sizeof(shape));
    shape.abi_version = SPARK_GLM52_TP_SHARD_ABI_VERSION;
    shape.tp_degree = 4u;
    shape.tp_rank = 2u;
    shape.pp_stage_count = 3u;
    shape.pp_stage_index = 1u;
    assert(SparkGlm52Pp13RuntimeBuildShapeRankPlan(
        &shape,16u,42000u,43000u,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &plan,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(plan.rank_index == 6u);
    assert(plan.first_layer_index == 26u && plan.layer_count == 26u);
    assert(plan.previous_rank_index == 2u && plan.next_rank_index == 10u);
    assert(plan.tp_collective_listen_port == 43006u);
    assert(plan.tp_peer_ports[0] == 43007u && plan.tp_peer_ports[1] == 43004u);
    assert(plan.tp_peer_host_names[0][0] != '\0');
    assert((plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_DENSE_PREFIX) == 0u);
    assert((plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) == 0u);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        "packs",plan.quantization_mode,7u,plan.tp_degree,plan.tp_rank,
        pack_path,sizeof(pack_path)) == SPARK_STATUS_OK);
    assert(strstr(pack_path,"_tp4r2.spfp8") != 0);
    plan.shape_configuration_hash += 1u;
    assert(SparkGlm52Pp13RuntimeValidateRankPlan(
        &plan,error_buffer,sizeof(error_buffer)) != SPARK_STATUS_OK);
    shape.tp_degree = 16u;
    shape.tp_rank = 15u;
    shape.pp_stage_count = 1u;
    shape.pp_stage_index = 0u;
    assert(SparkGlm52Pp13RuntimeBuildShapeRankPlan(
        &shape,16u,42000u,43000u,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
        &plan,error_buffer,sizeof(error_buffer)) != SPARK_STATUS_OK);
}

int main(void)
{
    SparkTestPp13RuntimeShapePlans();
    SparkTestGlm52Pp13RuntimeRankPlan();
    SparkTestGlm52Pp13RuntimeDsaCandidateBucket();
    SparkTestGlm52Pp13RuntimeFp8Packs();
    SparkTestGlm52Pp13RuntimeFinalEventRoute();
    return 0;
}
