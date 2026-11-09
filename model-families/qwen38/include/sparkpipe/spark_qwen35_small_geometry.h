#ifndef SPARKPIPE_SPARK_QWEN35_SMALL_GEOMETRY_H
#define SPARKPIPE_SPARK_QWEN35_SMALL_GEOMETRY_H

#include "sparkpipe/spark_qwen38_family.h"

struct Qwen35DenseGeometry : QwenHybridDefaults
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kVocab = 248320u;
	static constexpr uint32_t kPeriod = 4u;
	static constexpr uint32_t kPhase = 3u;
	static constexpr uint32_t kGdnKeyHeads = 16u;
	static constexpr uint32_t kHeadDim = 256u;
	static constexpr uint32_t kRopeDim = 64u;
	static constexpr float kRopeTheta = 10000000.0f;
	static constexpr bool kRankHeads = false;
	static constexpr bool kMoe = false;
	static constexpr uint32_t kExperts = 0u;
	static constexpr uint32_t kLocalExperts = 0u;
	static constexpr uint32_t kTopK = 0u;
	static constexpr uint32_t kMoeInter = 0u;
	static constexpr uint32_t kSharedRows = 0u;
	static constexpr bool kGdnSigmoidGate = false;
	static constexpr bool kTiedHead = true;
};

struct Qwen35_4bGeometry : Qwen35DenseGeometry
{
	static constexpr uint32_t kHidden = 2560u;
	static constexpr uint32_t kLayers = 32u;
	static constexpr uint32_t kGdnValueHeads = 32u;
	static constexpr uint32_t kHeads = 16u;
	static constexpr uint32_t kKvHeads = 4u;
};

struct Qwen35_2bGeometry : Qwen35DenseGeometry
{
	static constexpr uint32_t kHidden = 2048u;
	static constexpr uint32_t kLayers = 24u;
	static constexpr uint32_t kGdnValueHeads = 16u;
	static constexpr uint32_t kHeads = 8u;
	static constexpr uint32_t kKvHeads = 2u;
};

#endif
