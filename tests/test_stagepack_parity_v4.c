/*
 * Stage-pack parity harness, generation-4 mechanics leg.
 *
 * The five per-family legs pin generation 3 (live tree versus frozen
 * reference). This leg pins generation 4: the versioned dispatch surface
 * and the converged accounting/validation chain the shared reader owns,
 * swept through the same mutation classes - hostile dimensions, truncation
 * boundaries, misversions, placement violations - with one verdict line per
 * case. Output is compared BYTE-IDENTICAL against the frozen golden under
 * tests/fixtures/stagepack_parity/_v4/ so any drift in what a generation-4
 * loader accepts or refuses trips the gate instead of silently changing the
 * accepted byte streams. Built once against the live tree by
 * tests/test_stagepack_parity.py (there is no pre-v4 reference revision;
 * the golden minted with this spike is the reference).
 */
#include <stdint.h>
#include <stdio.h>

#include "runtime/spark_stagepack_reader.h"

static void emit(const char *tag,long value)
{
	printf("%s %ld\n",tag,value);
}

int main(void)
{
	static const uint32_t versions[] = {0u,1u,2u,3u,4u,5u,6u,32u,UINT32_MAX};
	static const uint32_t classes[] = {
		SPARK_STAGE_PACK_WEIGHT_BF16,SPARK_STAGE_PACK_WEIGHT_F32,SPARK_STAGE_PACK_WEIGHT_U32,
		SPARK_STAGE_PACK_WEIGHT_NIBBLE_E2M1,SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_F32B128,
		SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_E8M0B128,SPARK_STAGE_PACK_WEIGHT_FP8_E4M3,
		SPARK_STAGE_PACK_WEIGHT_BF16_RANS,8u,UINT32_MAX};
	static const uint32_t dims[] = {0u,1u,15u,16u,31u,32u,33u,48u,63u,64u,127u,128u,129u,256u,1024u,2147483647u,UINT32_MAX};
	char tag[128];
	uint32_t v,k,i,c;

	/* A: generation dispatch over the version space. */
	for (v = 0; v < (uint32_t)(sizeof(versions)/sizeof(versions[0])); v++)
	{
		snprintf(tag,sizeof(tag),"A version=%u",(unsigned)versions[v]);
		emit(tag,(long)SparkStagePackGenerationKnown(versions[v]));
	}

	/* B: accounted payload extents over every class and the dimension grid
	 * (0 marks refused/overflowing extents; entropy-coded rows print their
	 * uncompressed bound). */
	for (k = 0; k < (uint32_t)(sizeof(classes)/sizeof(classes[0])); k++)
		for (i = 0; i < (uint32_t)(sizeof(dims)/sizeof(dims[0])); i++)
			for (c = 0; c < (uint32_t)(sizeof(dims)/sizeof(dims[0])); c++)
			{
				const SparkStagePackV4Accounting *accounting = SparkStagePackV4ClassAccounting(classes[k]);
				snprintf(tag,sizeof(tag),"B class=%u r=%u c=%u payload",classes[k],dims[i],dims[c]);
				emit(tag,(long)(accounting != 0 ? SparkStagePackV4PayloadBytes(accounting,dims[i],dims[c]) : 0u));
			}

	/* C: scale extents under the ONE plane algebra (UINT64_MAX marks the
	 * divisibility refusal; ragged row counts are the point of the grid). */
	for (k = 0; k < (uint32_t)(sizeof(classes)/sizeof(classes[0])); k++)
		for (i = 0; i < (uint32_t)(sizeof(dims)/sizeof(dims[0])); i++)
			for (c = 1; c < (uint32_t)(sizeof(dims)/sizeof(dims[0])); c++)
			{
				const SparkStagePackV4Accounting *accounting = SparkStagePackV4ClassAccounting(classes[k]);
				snprintf(tag,sizeof(tag),"C class=%u r=%u c=%u scale",classes[k],dims[i],dims[c]);
				emit(tag,(long)(accounting != 0 ? SparkStagePackV4ScaleBytes(accounting,dims[i],dims[c]) : 0u));
			}

	/* D: the full normalized entry chain over placement/extents/group-size
	 * mutations around one well-formed fp8-with-f32-scales entry. */
	{
		const SparkStagePackV4Accounting *fp8 = SparkStagePackV4ClassAccounting(SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_F32B128);
		static const uint64_t offsets[] = {0u,1u,255u,256u,4095u,4096u,69632u,69888u,73728u,UINT64_MAX};
		uint32_t o1,o2,g,m;
		uint64_t payload = 65536u,scale = 2048u;
		for (o1 = 0; o1 < (uint32_t)(sizeof(offsets)/sizeof(offsets[0])); o1++)
			for (o2 = 0; o2 < (uint32_t)(sizeof(offsets)/sizeof(offsets[0])); o2++)
			{
				int32_t code = SparkStagePackV4ValidateEntryExtent(fp8,256u,256u,4096u,256u,73728u,offsets[o1],payload,offsets[o2],scale,128u);
				snprintf(tag,sizeof(tag),"D po=%llu so=%llu",(unsigned long long)offsets[o1],(unsigned long long)offsets[o2]);
				emit(tag,(long)code);
			}
		for (m = 0; m < 6u; m++)
		{
			uint64_t p = m == 0u ? 0u : (m == 1u ? payload - 1u : (m == 2u ? payload + 1u : payload));
			uint64_t s = m == 3u ? 0u : (m == 4u ? scale + 1u : scale);
			int32_t code = SparkStagePackV4ValidateEntryExtent(fp8,256u,256u,4096u,256u,73728u,4096u,p,69632u,s,128u);
			snprintf(tag,sizeof(tag),"D mutate=%u",m);
			emit(tag,(long)code);
		}
		for (g = 0; g < 3u; g++)
		{
			static const uint32_t groups[] = {0u,32u,128u};
			int32_t code = SparkStagePackV4ValidateEntryExtent(fp8,256u,256u,4096u,256u,73728u,4096u,payload,69632u,scale,groups[g]);
			snprintf(tag,sizeof(tag),"D group=%u",groups[g]);
			emit(tag,(long)code);
		}
		/* Argument refusals: null table, zero dims, zero alignment, window
		 * below the directory end. */
		emit("D nulltable",(long)SparkStagePackV4ValidateEntryExtent(0,256u,256u,4096u,256u,73728u,4096u,payload,69632u,scale,128u));
		emit("D zerorows",(long)SparkStagePackV4ValidateEntryExtent(fp8,0u,256u,4096u,256u,73728u,4096u,payload,69632u,scale,128u));
		emit("D zerocols",(long)SparkStagePackV4ValidateEntryExtent(fp8,256u,0u,4096u,256u,73728u,4096u,payload,69632u,scale,128u));
		emit("D zeroalign",(long)SparkStagePackV4ValidateEntryExtent(fp8,256u,256u,4096u,0u,73728u,4096u,payload,69632u,scale,128u));
		emit("D smallwindow",(long)SparkStagePackV4ValidateEntryExtent(fp8,256u,256u,73729u,256u,73728u,4096u,payload,69632u,scale,128u));
	}

	/* E: entropy-coded entries - bounded accounting, not exemption. */
	{
		const SparkStagePackV4Accounting *rans = SparkStagePackV4ClassAccounting(SPARK_STAGE_PACK_WEIGHT_BF16_RANS);
		static const uint64_t streams[] = {0u,1u,16383u,16384u,16385u,262144u};
		uint32_t s;
		for (s = 0; s < (uint32_t)(sizeof(streams)/sizeof(streams[0])); s++)
		{
			int32_t code = SparkStagePackV4ValidateEntryExtent(rans,64u,128u,4096u,256u,262144u,4096u,streams[s],0u,0u,0u);
			snprintf(tag,sizeof(tag),"E stream=%llu",(unsigned long long)streams[s]);
			emit(tag,(long)code);
		}
		emit("E badgroup",(long)SparkStagePackV4ValidateEntryExtent(rans,64u,128u,4096u,256u,262144u,4096u,9000u,0u,0u,32u));
		emit("E scalewithstream",(long)SparkStagePackV4ValidateEntryExtent(rans,64u,128u,4096u,256u,262144u,4096u,9000u,8192u,64u,0u));
	}

	/* F: the inventory-equality rule (missing AND extra kinds refuse). */
	{
		static const uint64_t pairs[][2] = {{0u,0u},{0xFu,0xFu},{0xFu,0x7u},{0xFu,0x1Fu},{UINT64_MAX,UINT64_MAX},{UINT64_MAX,0u}};
		uint32_t p;
		for (p = 0; p < (uint32_t)(sizeof(pairs)/sizeof(pairs[0])); p++)
		{
			snprintf(tag,sizeof(tag),"F pair=%u expect=%llu seen=%llu",p,(unsigned long long)pairs[p][0],(unsigned long long)pairs[p][1]);
			emit(tag,(long)SparkStagePackV4InventoryComplete(pairs[p][0],pairs[p][1]));
		}
	}

	/* G: dual-read divergence proof - the SAME logical tensor priced under
	 * both generations' laws, proving dispatch selects different mechanics
	 * while both stay loadable. Rows deliberately ragged against the tile
	 * law so the two generations disagree by construction. */
	{
		static const uint32_t ragged[][2] = {{48u,128u},{48u,384u},{128u,128u},{49u,129u},{64u,64u}};
		static const uint32_t both[] = {SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_F32B128,SPARK_STAGE_PACK_WEIGHT_FP8_E4M3_E8M0B128,SPARK_STAGE_PACK_WEIGHT_NIBBLE_E2M1,SPARK_STAGE_PACK_WEIGHT_FP8_E4M3};
		uint32_t b,rr;
		for (b = 0; b < (uint32_t)(sizeof(both)/sizeof(both[0])); b++)
			for (rr = 0; rr < (uint32_t)(sizeof(ragged)/sizeof(ragged[0])); rr++)
			{
				uint32_t rows = ragged[rr][0],columns = ragged[rr][1];
				snprintf(tag,sizeof(tag),"G class=%u r=%u c=%u v3scale",both[b],rows,columns);
				emit(tag,(long)SparkStagePackWeightClassScaleBytes(both[b],rows,columns));
				snprintf(tag,sizeof(tag),"G class=%u r=%u c=%u v4scale",both[b],rows,columns);
				emit(tag,(long)SparkStagePackV4ScaleBytes(SparkStagePackV4ClassAccounting(both[b]),rows,columns));
				snprintf(tag,sizeof(tag),"G class=%u r=%u c=%u v3payload",both[b],rows,columns);
				emit(tag,(long)SparkStagePackWeightPayloadBytes(both[b],rows,columns));
				snprintf(tag,sizeof(tag),"G class=%u r=%u c=%u v4payload",both[b],rows,columns);
				emit(tag,(long)SparkStagePackV4PayloadBytes(SparkStagePackV4ClassAccounting(both[b]),rows,columns));
			}
	}

	printf("DONE v4mechanics\n");
	return(0);
}
