import subprocess
pristine = 'modules/glm52_resident_decode_stage/validation/spark_glm52_resident_decode_stage_cuda_validation.cu'
src = open(pristine).read()

# 1. locals: owner lookup table
old_locals = '''	SparkGlm52ValMetrics metrics;
	uint32_t token,route,index;
	double maximum_weight_delta = 0.0;
	int status,local_result = 0;
	uint32_t packed_row_zero = 0u;
	uint16_t *expert_row_zero = 0u;'''
new_locals = '''	SparkGlm52ValMetrics metrics;
	uint32_t token,route,index;
	double maximum_weight_delta = 0.0;
	int status,local_result = 0;
	uint32_t packed_row_zero = 0u;
	uint32_t slot_owner[SPARK_GLM52_EXPERTS + 1u];
	uint32_t owner_expert = 0u;
	uint16_t *expert_row_zero = 0u;

	/* Route ORDER is not defined between the device top-k network and the
	 * host reference walk - only the selected SET and per-route weights are.
	 * A packed output row therefore cannot be paired with selected[r] by
	 * position. LmRouteBuild defines the binding that matters: packed row p
	 * belongs to the expert g whose half-open group range covers p. Read the
	 * row offsets once and invert them, so every comparison below pairs a
	 * captured row with its OWN expert whatever order each side emitted. */
	memset(slot_owner,0,sizeof(slot_owner));'''
assert src.count(old_locals) == 1
src = src.replace(old_locals,new_locals)

# 2. capture group offsets next to the packed-row capture
old_cap = '''			if ( cudaMemcpy(&packed_row_zero,fixture->route_packed_row,sizeof(uint32_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
				cudaMemcpy(expert_row_zero,fixture->expert_out + (uint64_t)packed_row_zero * SPARK_GLM52_VHIDDEN,
				SPARK_GLM52_VHIDDEN * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
			{
				free(expert_row_zero);
				return(SparkGlm52ValFail("routed_expert","readback"));
			}'''
new_cap = old_cap + '''
			if ( cudaMemcpy(slot_owner,fixture->group_row_offset,(uint64_t)(SPARK_GLM52_EXPERTS + 1u) * sizeof(uint32_t),
				cudaMemcpyDeviceToHost) != cudaSuccess )
			{
				free(expert_row_zero);
				return(SparkGlm52ValFail("routed_expert","readback"));
			}
			for (owner_expert = SPARK_GLM52_EXPERTS; owner_expert > 0u && slot_owner[owner_expert] > packed_row_zero; owner_expert--)
				;
			/* owner_expert is now the largest g with group_row_offset[g] <= p;
			 * empty leading groups collapse onto the owning one. */'''
assert src.count(old_cap) == 1
src = src.replace(old_cap,new_cap)

# 3. pair comparisons with owner_expert instead of selected[0]
old_pair1 = '''		SparkGlm52ValExpertGemmRow(fixture->experts.w1_payload,fixture->experts.w1_scales,selected[0],
			oracle.normed,oracle.gate_up,SPARK_GLM52_VW1_ROWS,SPARK_GLM52_VEXPERT_COLUMNS,SPARK_GLM52_VW1_ROWS);'''
new_pair1 = '''		SparkGlm52ValExpertGemmRow(fixture->experts.w1_payload,fixture->experts.w1_scales,owner_expert,
			oracle.normed,oracle.gate_up,SPARK_GLM52_VW1_ROWS,SPARK_GLM52_VEXPERT_COLUMNS,SPARK_GLM52_VW1_ROWS);'''
assert src.count(old_pair1) == 1
src = src.replace(old_pair1,new_pair1)

old_pair2 = '''		SparkGlm52ValExpertGemmRow(fixture->experts.w2_payload,fixture->experts.w2_scales,selected[0],
			oracle.intermediate,expert_reference,SPARK_GLM52_VW2_ROWS,SPARK_GLM52_VW2_COLUMNS,SPARK_GLM52_VHIDDEN);'''
new_pair2 = '''		printf("glm52_validation check=routed_expert_owner slot=%u owner=%u\n",
			packed_row_zero,owner_expert);
		SparkGlm52ValExpertGemmRow(fixture->experts.w2_payload,fixture->experts.w2_scales,owner_expert,
			oracle.intermediate,expert_reference,SPARK_GLM52_VW2_ROWS,SPARK_GLM52_VW2_COLUMNS,SPARK_GLM52_VHIDDEN);'''
assert src.count(old_pair2) == 1
src = src.replace(old_pair2,new_pair2)

# 4. tolerance: measured rounding envelope (const-slab probe: cosine exactly 1,
#    rel_l2 0.0226 from fp32-accumulation-order freedom against bf16 stores)
old_tol = '''		status = SparkGlm52ValReport("routed_expert_forward",&metrics,2e-2,0.999);'''
new_tol = '''		/* 3e-2, not 2e-2: with byte-identical slabs on both sides the
		 * forward measures rel_l2 0.0226 / cosine 1.0 - the fp32
		 * accumulation-order freedom against one bf16 store rounding. The
		 * threshold must sit above the codec's own noise floor so it keeps
		 * gating functional divergence (the mispairing this tier used to
		 * report measured 0.34). */
		status = SparkGlm52ValReport("routed_expert_forward",&metrics,3e-2,0.999);'''
assert src.count(old_tol) == 1
src = src.replace(old_tol,new_tol)

open('/tmp/validator_route_owner.cu','w').write(src)
print('edited ok')
