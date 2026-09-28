#include <cstdio>
#include <cstring>

#include "spark_qwen38_max_resident_decode_stage_cuda.cu"

static int failures = 0;

static void Check(int condition, const char *name)
{
	if ( condition == 0 )
	{
		std::printf("FAIL %s\n",name);
		failures++;
	}
}

static LmGdnStageLinearView ExpertView(uint32_t weight_format, uint32_t input_dimension, uint32_t output_dimension, const void *storage)
{
	LmGdnStageLinearView view;
	std::memset(&view,0,sizeof(view));
	view.weight_format = weight_format;
	view.input_dimension = input_dimension;
	view.output_dimension = output_dimension;
	view.weight_payload = storage;
	view.weight_scale_e8m0 = (const uint8_t *)storage;
	return(view);
}

static cudaError_t LaunchGrouped(const LmGdnStageLinearView *view, uint32_t tp_degree, uint32_t tp_rank, uint32_t *storage)
{
	return(LmGdnStageLaunchGroupedExpertLinear(0,view,storage,storage,storage,storage,storage,1u,1u,tp_degree,tp_rank));
}

static void TestGroupedExpertGeometry(void)
{
	uint32_t storage[16] = {0u};
	LmGdnStageLinearView view;
	view = ExpertView(SPARK_LLM_WEIGHT_FORMAT_FP8_E4M3_F32B128,8192u,32u * 192u,storage);
	Check(LaunchGrouped(&view,16u,0u,storage) == cudaErrorInvalidValue,"fp8 grouped experts refuse rows per expert off the 128 block");
	view = ExpertView(SPARK_LLM_WEIGHT_FORMAT_FP8_E4M3_F32B128,192u,32u * 128u,storage);
	Check(LaunchGrouped(&view,16u,0u,storage) == cudaErrorInvalidValue,"fp8 grouped experts refuse an input width off the 128 block");
	view = ExpertView(SPARK_LLM_WEIGHT_FORMAT_NVFP4_PACKED,24u,32u * 128u,storage);
	Check(LaunchGrouped(&view,16u,0u,storage) == cudaErrorInvalidValue,"nvfp4 grouped experts refuse an input width off 16");
	view = ExpertView(SPARK_LLM_WEIGHT_FORMAT_FP8_E4M3_F32B128,8192u,32u * 128u,storage);
	Check(LaunchGrouped(&view,0u,0u,storage) == cudaErrorInvalidValue,"grouped experts refuse a zero tp degree");
	Check(LaunchGrouped(&view,2u * SPARK_LLM_ROUTED_EXPERT_COUNT,0u,storage) == cudaErrorInvalidValue,"grouped experts refuse more ranks than experts");
	Check(LaunchGrouped(&view,16u,16u,storage) == cudaErrorInvalidValue,"grouped experts refuse a rank past the degree");
	view = ExpertView(SPARK_LLM_WEIGHT_FORMAT_FP8_E4M3_F32B128,8192u,32u * 128u + 1u,storage);
	Check(LaunchGrouped(&view,16u,0u,storage) == cudaErrorInvalidValue,"grouped experts refuse rows not divisible by the rank's experts");
}

int main(void)
{
	TestGroupedExpertGeometry();
	if ( failures == 0 )
		std::printf("test_gdn_stage_launch_checks PASS\n");
	else
		std::printf("test_gdn_stage_launch_checks FAIL %d\n",failures);
	return(failures == 0 ? 0 : 1);
}
