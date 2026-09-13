// K-F2 bind-kind gate: SparkK3BindLayer must derive the per-layer kind from
// THE canonical rule (spark_k3_pool_sizing.h's SparkK3LayerIsMla - the 3:1
// period AND the trailing layer 92), not a private period-only form. The
// historical defect: layer 92 % 4 == 0, so the period-only rule bound it as
// GDN and the module resolved KDA names the pack does not carry for an MLA
// layer - a boot-time VALIDATION_FAILED on the real checkpoint.
//
// The PACK argument must cover layers 0..92 (93 tensors-deep: the synthetic
// full-backbone fixture tests/test_k3_bind_fixture.py builds one; a real
// full-model pack works too). A layers-0-3 slice pack can no longer exercise
// this gate.
#include "sparkpipe/spark_k3_bind.h"
#include "sparkpipe/spark_k3_pack_load.h"
#include "sparkpipe/spark_k3_pool_sizing.h"
#include "sparkpipe/spark_status.h"

#include <stdio.h>

static int expect(int condition, const char *what)
{
	printf("%s: %s\n", condition ? "PASS" : "FAIL", what);
	return(condition ? 0 : 1);
}

static int ExpectKind(SparkK3Pack *pack, uint32_t layer,
	uint32_t expect_gdn, uint32_t expect_dense, const char *label)
{
	SparkK3BoundLayer bound;
	int failures = 0;
	failures += expect(SparkK3BindLayer(pack, layer, &bound) ==
		SPARK_STATUS_OK, label);
	failures += expect(bound.layer_is_gdn == expect_gdn,
		expect_gdn ? "  kind is GDN" : "  kind is MLA");
	if ( expect_dense )
		failures += expect(bound.layer_is_dense == 1u, "  dense");
	return(failures);
}

int main(int argc, char **argv)
{
	SparkK3Pack pack;
	SparkK3BoundLayer bound;
	const SparkK3PackEntry *entry;
	int failures = 0;
	uint32_t layer, mla_count;
	if ( argc != 2 )
	{
		fprintf(stderr, "usage: test_k3_bind PACK   (must cover layers "
			"0..92 - see tests/test_k3_bind_fixture.py)\n");
		return(2);
	}
	failures += expect(SparkK3PackOpen(argv[1], &pack) == SPARK_STATUS_OK,
		"open");
	if ( pack.config.total_layers < 93u )
	{
		printf("FAIL pack carries %u layers, the gate needs the full "
			"93-layer backbone\n", pack.config.total_layers);
		return(1);
	}

	/* THE HEAD LAYERS {0,1,2,3}: three GDA (one dense) then the first MLA. */
	failures += ExpectKind(&pack, 0u, 1u, 1u, "bind layer 0 (dense KDA)");
	failures += expect(SparkK3BindLayer(&pack, 0u, &bound) == SPARK_STATUS_OK,
		"rebind layer 0");
	failures += expect(bound.tensor_count == 17u, "layer 0 binds 17 tensors");
	entry = SparkK3BoundEntry(&bound, "kda_gate_weight");
	failures += expect(entry != 0, "gate present");
	/* output_dim_heads x hidden, read from the pack's own config so the
	 * gate holds for the synthetic fixture AND the real checkpoint */
	if ( entry != 0 )
		failures += expect(entry->shape_count >= 2u &&
			entry->shape[0] ==
				pack.config.kda_heads * pack.config.kda_head &&
			entry->shape[1] == pack.config.hidden,
			"gate shape is output_dim_heads x hidden");
	failures += expect(SparkK3BoundPayload(&pack, &bound,
		"dense_gate_up_weight") != 0, "dense payload resolves");
	failures += expect(SparkK3BoundEntry(&bound, "expert_w1_weight") == 0,
		"no expert tensors on the dense layer");
	failures += ExpectKind(&pack, 1u, 1u, 0u, "bind layer 1 (routed KDA)");
	failures += expect(SparkK3BindLayer(&pack, 1u, &bound) ==
		SPARK_STATUS_OK, "rebind layer 1");
	failures += expect(SparkK3BoundEntry(&bound, "expert_w1_weight") != 0,
		"layer 1 expert w1 present");
	failures += expect(bound.tensor_count == 24u,
		"layer 1 binds 24 tensors (routed KDA)");
	failures += ExpectKind(&pack, 2u, 1u, 0u, "bind layer 2 (routed KDA)");
	failures += ExpectKind(&pack, 3u, 0u, 0u, "bind layer 3 (routed MLA)");
	failures += expect(SparkK3BindLayer(&pack, 3u, &bound) ==
		SPARK_STATUS_OK, "rebind layer 3");
	failures += expect(SparkK3BoundEntry(&bound, "mla_gate_weight") != 0,
		"layer 3 mla_gate present");
	failures += expect(SparkK3BoundEntry(&bound, "kda_qkv_beta_weight") == 0,
		"no KDA tensors on the MLA layer");

	/* THE TRAILING EXCEPTION {91,92}: the defect pair. 92 must be MLA -
	 * the period-only rule said GDN and failed the resolve. */
	failures += ExpectKind(&pack, 90u, 1u, 0u,
		"bind layer 90 (periodic GDN)");
	failures += ExpectKind(&pack, 91u, 0u, 0u,
		"bind layer 91 (periodic MLA: 91 % 4 == 3)");
	failures += ExpectKind(&pack, 92u, 0u, 0u,
		"bind layer 92 (trailing MLA - the F2 exception)");
	failures += expect(SparkK3BindLayer(&pack, 92u, &bound) ==
		SPARK_STATUS_OK, "rebind layer 92");
	failures += expect(SparkK3BoundEntry(&bound, "mla_kv_b_value_weight") != 0,
		"layer 92 mla_kv_b_value present");
	failures += expect(SparkK3BoundEntry(&bound, "kda_decay_bias") == 0,
		"no KDA decay tensors on layer 92");

	/* PP4 STAGE-3 SLICE (first 70, count 23): every layer of the slice
	 * binds, each kind matches the canonical predicate, and the counts
	 * agree with the pool-sizing arithmetic (7 MLA = six periodic + 92). */
	mla_count = 0u;
	for ( layer = 70u; layer < 93u; ++layer )
	{
		uint32_t is_mla = SparkK3LayerIsMla(layer);
		char what[128];
		snprintf(what, sizeof(what), "stage-3 layer %u binds (%s)", layer,
			is_mla ? "MLA" : "GDN");
		if ( SparkK3BindLayer(&pack, layer, &bound) != SPARK_STATUS_OK )
		{
			printf("FAIL: %s\n", what);
			failures++;
			continue;
		}
		if ( bound.layer_is_gdn == (is_mla != 0u) )
		{
			printf("FAIL: stage-3 layer %u kind flag disagrees with "
				"SparkK3LayerIsMla\n", layer);
			failures++;
		}
		if ( is_mla != 0u )
			mla_count++;
	}
	failures += expect(mla_count == 7u, "stage-3 holds exactly 7 MLA layers");
	failures += expect(mla_count ==
		SparkK3MlaLayersInSlice(70u, 23u),
		"bind kinds match the pool-sizing count for the slice");
	failures += expect(23u - mla_count == 16u,
		"stage-3 holds exactly 16 KDA layers");

	/* The exception cannot wander past the backbone end. */
	failures += expect(SparkK3BindLayer(&pack, 93u, &bound) ==
		SPARK_STATUS_VALIDATION_FAILED,
		"layer 93 is past the backbone -> VALIDATION_FAILED");

	SparkK3PackClose(&pack);
	printf("test_k3_bind: %d failures\n", failures);
	return(failures != 0 ? 1 : 0);
}
