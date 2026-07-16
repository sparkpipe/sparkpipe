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
    assert(rank_plan.execution_row_capacity == 8192u);
    assert(rank_plan.output_endpoint.max_active_sequence_count == 8192u);
    assert((rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_DENSE_PREFIX) != 0u);
    assert((rank_plan.flags &
        SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) == 0u);
    assert(strcmp(rank_plan.next_host_name,"10.10.100.11") == 0);
    assert(strcmp(rank_plan.output_route_name,
        "10.10.100.10_to_10.10.100.11_hidden") == 0);
    assert(rank_plan.quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT);
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
    (void)remove("build/test_glm52_pp13_runtime_packs/w8lut_moe_pack_manifest.json");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0003_fp8_moe.spfp8");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0004_fp8_moe.spfp8");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0005_fp8_moe.spfp8");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0003_b12x_moe.spb12x");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0004_b12x_moe.spb12x");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0005_b12x_moe.spb12x");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0003_w8lut_moe.spw8lut");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0004_w8lut_moe.spw8lut");
    (void)remove("build/test_glm52_pp13_runtime_packs/glm52_layer_0005_w8lut_moe.spw8lut");
    assert(SparkGlm52Pp13RuntimeBuildRankPlan(
        0u,1024u,52100u,
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,&rank_plan,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_OK);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_NOT_FOUND);
    SparkTestWriteFile("build/test_glm52_pp13_runtime_packs/fp8_moe_pack_manifest.json");
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,3u,pack_path,sizeof(pack_path)) == SPARK_STATUS_OK);
    assert(strcmp(pack_path,
        "build/test_glm52_pp13_runtime_packs/glm52_layer_0003_fp8_moe.spfp8") == 0);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,4u,pack_path,sizeof(pack_path)) == SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_NOT_FOUND);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,5u,pack_path,sizeof(pack_path)) == SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_OK);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,SPARK_GLM52_MODEL_MTP_LAYER_INDEX,
        pack_path,sizeof(pack_path)) == SPARK_STATUS_OK);
    assert(strcmp(pack_path,
        "build/test_glm52_pp13_runtime_packs/glm52_layer_0078_fp8_moe.spfp8") == 0);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,SPARK_GLM52_MODEL_WEIGHT_LAYER_COUNT,
        pack_path,sizeof(pack_path)) == SPARK_STATUS_INVALID_ARGUMENT);
    SparkTestWriteFile("build/test_glm52_pp13_runtime_packs/w8lut_moe_pack_manifest.json");
    assert(SparkGlm52Pp13RuntimeBuildRankPlan(
        0u,1024u,52100u,SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT,
        &rank_plan,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_MODULE_NOT_VALIDATED);
    assert(remove("build/test_glm52_pp13_runtime_packs/fp8_moe_pack_manifest.json") == 0);
    assert(remove("build/test_glm52_pp13_runtime_packs/w8lut_moe_pack_manifest.json") == 0);
    SparkTestWriteFile("build/test_glm52_pp13_runtime_packs/resident_moe_pack_manifest.json");
    assert(SparkGlm52Pp13RuntimeBuildRankPlan(
        0u,1024u,52100u,SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
        &rank_plan,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_NOT_FOUND);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,3u,pack_path,sizeof(pack_path)) ==
            SPARK_STATUS_OK);
    assert(strcmp(pack_path,
        "build/test_glm52_pp13_runtime_packs/glm52_layer_0003_b12x_moe.spb12x") == 0);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,4u,pack_path,sizeof(pack_path)) ==
            SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,5u,pack_path,sizeof(pack_path)) ==
            SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(remove("build/test_glm52_pp13_runtime_packs/resident_moe_pack_manifest.json") == 0);
    SparkTestWriteFile("build/test_glm52_pp13_runtime_packs/w8lut_moe_pack_manifest.json");
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_NOT_FOUND);
    assert(SparkGlm52Pp13RuntimeBuildRankPlan(
        0u,1024u,52100u,SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT,
        &rank_plan,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) ==
            SPARK_STATUS_NOT_FOUND);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,3u,pack_path,sizeof(pack_path)) ==
            SPARK_STATUS_OK);
    assert(strcmp(pack_path,
        "build/test_glm52_pp13_runtime_packs/glm52_layer_0003_w8lut_moe.spw8lut") == 0);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,4u,pack_path,sizeof(pack_path)) ==
            SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,rank_plan.quantization_mode,5u,pack_path,sizeof(pack_path)) ==
            SPARK_STATUS_OK);
    SparkTestWriteFile(pack_path);
    assert(SparkGlm52Pp13RuntimeValidateStageMoePackFiles(
        &rank_plan,pack_root,error_buffer,sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(SparkGlm52Pp13RuntimeBuildMoePackPath(
        pack_root,UINT32_MAX,3u,pack_path,sizeof(pack_path)) ==
            SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkGlm52Pp13RuntimeParseQuantizationMode(
        "fp8",&rank_plan.quantization_mode) == SPARK_STATUS_OK);
    assert(strcmp(SparkGlm52Pp13RuntimeQuantizationModeName(
        rank_plan.quantization_mode),"fp8") == 0);
    assert(SparkGlm52Pp13RuntimeParseQuantizationMode(
        "nvfp4",&rank_plan.quantization_mode) == SPARK_STATUS_OK);
    assert(strcmp(SparkGlm52Pp13RuntimeQuantizationModeName(
        rank_plan.quantization_mode),"nvfp4") == 0);
    assert(SparkGlm52Pp13RuntimeParseQuantizationMode(
        "w8lut",&rank_plan.quantization_mode) == SPARK_STATUS_OK);
    assert(strcmp(SparkGlm52Pp13RuntimeQuantizationModeName(
        rank_plan.quantization_mode),"w8lut") == 0);
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
    assert(SparkGlm52Pp13RuntimeValidateFp8PlanCounts(
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT,0u,0u) ==
        SPARK_STATUS_OK);
    assert(SparkGlm52Pp13RuntimeValidateFp8PlanCounts(
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT,1u,1u) ==
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
    assert(SparkGlm52Pp13RuntimeExpectedMoeBackendKind(
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT,
        &rank_plan.quantization_mode) == SPARK_STATUS_OK);
    assert(rank_plan.quantization_mode ==
        SPARK_GLM52_PP13_RUNTIME_MOE_BACKEND_W8LUT_BF16_WMMA);
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

int main(void)
{
    SparkTestGlm52Pp13RuntimeRankPlan();
    SparkTestGlm52Pp13RuntimeDsaCandidateBucket();
    SparkTestGlm52Pp13RuntimeFp8Packs();
    SparkTestGlm52Pp13RuntimeFinalEventRoute();
    return 0;
}
