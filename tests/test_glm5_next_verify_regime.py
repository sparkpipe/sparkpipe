#!/usr/bin/env python3
"""The glm5_next single-sequence verify regime admits exactly the waves its captured table can replay, and every replayed row makes the attention choices of its own B1 step."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sparkpipe/spark_glm5_next_verify_regime.h"

#define REGIMES SPARK_GLM5_NEXT_GRAPH_REGIME_COUNT
#define TABLE SPARK_GLM5_NEXT_VERIFY_TABLE_COUNT

static void require(int condition,const char *what,uint32_t a,uint32_t b,uint32_t c)
{
	if ( condition )
		return;
	fprintf(stderr,"FAIL %s (%u %u %u)\n",what,a,b,c);
	exit(1);
}

static void check_parse(void)
{
	static const char *good[] = {"0","2","3","4","5","6","7","8"};
	static const char *bad[] = {"","1","9","10","02","2 ","-2","x","8x"};
	uint32_t index,rows;
	require(SparkGlm5NextVerifyRowsParse(0,&rows) == SPARK_STATUS_OK && rows == 0u,"absent setting is off",0u,0u,0u);
	for (index=0u; index<sizeof(good)/sizeof(good[0]); index++)
		require(SparkGlm5NextVerifyRowsParse(good[index],&rows) == SPARK_STATUS_OK && rows == (uint32_t)(good[index][0] - '0'),"valid row count parses",index,rows,0u);
	for (index=0u; index<sizeof(bad)/sizeof(bad[0]); index++)
		require(SparkGlm5NextVerifyRowsParse(bad[index],&rows) == SPARK_STATUS_INVALID_ARGUMENT && rows == 0u,"invalid row count is rejected",index,rows,0u);
	for (rows=SPARK_GLM5_NEXT_VERIFY_ROWS_MIN; rows<=SPARK_GLM5_NEXT_VERIFY_ROWS_MAX; rows++)
		require(SparkGlm5NextVerifyTableIndex(rows) == rows - SPARK_GLM5_NEXT_VERIFY_ROWS_MIN && SparkGlm5NextVerifyTableIndex(rows) < TABLE,"table index is dense and bounded",rows,0u,0u);
}

static void fill(uint32_t *slots,uint32_t *positions,uint32_t rows,uint32_t position)
{
	uint32_t row;
	for (row=0u; row<rows; row++)
	{
		slots[row] = 5u;
		positions[row] = position + row;
	}
}

static void check_shape(void)
{
	uint32_t slots[SPARK_GLM5_NEXT_VERIFY_ROWS_MAX],positions[SPARK_GLM5_NEXT_VERIFY_ROWS_MAX];
	fill(slots,positions,4u,100u);
	require(SparkGlm5NextVerifyWaveCheck(4u,8u,0u,1u,0u,slots,positions,64u,4096u) == SPARK_STATUS_OK,"consecutive single-sequence greedy wave is accepted",0u,0u,0u);
	require(SparkGlm5NextVerifyWaveCheck(4u,0u,0u,1u,0u,slots,positions,64u,4096u) == SPARK_STATUS_UNSUPPORTED,"regime off rejects every verify wave",0u,0u,0u);
	require(SparkGlm5NextVerifyWaveCheck(4u,8u,0u,1u,1u,slots,positions,64u,4096u) == SPARK_STATUS_UNSUPPORTED,"sampled rows are not in the greedy regime",0u,0u,0u);
	require(SparkGlm5NextVerifyWaveCheck(1u,8u,0u,1u,0u,slots,positions,64u,4096u) == SPARK_STATUS_INVALID_ARGUMENT,"one row is a decode step, not a verify",0u,0u,0u);
	require(SparkGlm5NextVerifyWaveCheck(4u,3u,0u,1u,0u,slots,positions,64u,4096u) == SPARK_STATUS_INVALID_ARGUMENT,"rows above the configured maximum",0u,0u,0u);
	require(SparkGlm5NextVerifyWaveCheck(4u,8u,1u,1u,0u,slots,positions,64u,4096u) == SPARK_STATUS_INVALID_ARGUMENT,"a partial wave is not a verify wave",0u,0u,0u);
	require(SparkGlm5NextVerifyWaveCheck(4u,8u,0u,2u,0u,slots,positions,64u,4096u) == SPARK_STATUS_INVALID_ARGUMENT,"two sequences are not a single-sequence verify",0u,0u,0u);
	slots[2] = 6u;
	require(SparkGlm5NextVerifyWaveCheck(4u,8u,0u,1u,0u,slots,positions,64u,4096u) == SPARK_STATUS_INVALID_ARGUMENT,"rows of another lane are rejected",0u,0u,0u);
	fill(slots,positions,4u,100u);
	positions[3] = 104u;
	require(SparkGlm5NextVerifyWaveCheck(4u,8u,0u,1u,0u,slots,positions,64u,4096u) == SPARK_STATUS_INVALID_ARGUMENT,"a position gap is rejected",0u,0u,0u);
	fill(slots,positions,4u,61u);
	require(SparkGlm5NextVerifyWaveCheck(4u,8u,0u,1u,0u,slots,positions,64u,4096u) == SPARK_STATUS_INVALID_ARGUMENT,"a wave straddling the split threshold is rejected",0u,0u,0u);
	fill(slots,positions,4u,4093u);
	require(SparkGlm5NextVerifyWaveCheck(4u,8u,0u,1u,0u,slots,positions,64u,4096u) == SPARK_STATUS_INVALID_ARGUMENT,"a wave past the sequence capacity is rejected",0u,0u,0u);
}

static void check_drafter(void)
{
	uint32_t kind;
	const char *path;
	static const char *bad[] = {"","Lookup","lookup ","oracle","oracle:","adversary:","recorded:","recorded","remote:rtx5090"};
	uint32_t index;
	require(SparkGlm5NextVerifyDrafterParse(0,0u,&kind,&path) == SPARK_STATUS_OK && kind == SPARK_GLM5_NEXT_VERIFY_DRAFTER_NONE && path == 0,"regime off with no drafter",0u,0u,0u);
	require(SparkGlm5NextVerifyDrafterParse("lookup",0u,&kind,&path) == SPARK_STATUS_INVALID_ARGUMENT && kind == SPARK_GLM5_NEXT_VERIFY_DRAFTER_NONE,"a drafter without the regime is rejected",0u,0u,0u);
	require(SparkGlm5NextVerifyDrafterParse(0,4u,&kind,&path) == SPARK_STATUS_INVALID_ARGUMENT,"the regime without a drafter is rejected",0u,0u,0u);
	require(SparkGlm5NextVerifyDrafterParse("lookup",4u,&kind,&path) == SPARK_STATUS_OK && kind == SPARK_GLM5_NEXT_VERIFY_DRAFTER_LOOKUP && path == 0,"lookup drafter",0u,0u,0u);
	require(SparkGlm5NextVerifyDrafterParse("oracle:/tmp/r.u32",4u,&kind,&path) == SPARK_STATUS_OK && kind == SPARK_GLM5_NEXT_VERIFY_DRAFTER_ORACLE && strcmp(path,"/tmp/r.u32") == 0,"oracle drafter names its recording",0u,0u,0u);
	require(SparkGlm5NextVerifyDrafterParse("adversary:r",8u,&kind,&path) == SPARK_STATUS_OK && kind == SPARK_GLM5_NEXT_VERIFY_DRAFTER_ADVERSARY && strcmp(path,"r") == 0,"adversary drafter names its recording",0u,0u,0u);
	require(SparkGlm5NextVerifyDrafterParse("recorded:/tmp/d.sprd",8u,&kind,&path) == SPARK_STATUS_OK && kind == SPARK_GLM5_NEXT_VERIFY_DRAFTER_RECORDED && strcmp(path,"/tmp/d.sprd") == 0,"recorded drafter names its draft table",0u,0u,0u);
	require(SparkGlm5NextVerifyDrafterUsesMtp(SPARK_GLM5_NEXT_VERIFY_DRAFTER_RECORDED) == 0u && SparkGlm5NextVerifyDrafterUsesLookup(SPARK_GLM5_NEXT_VERIFY_DRAFTER_RECORDED) == 0u,"a recorded table needs neither the MTP sidecar nor lookup",0u,0u,0u);
	for (index=0u; index<sizeof(bad)/sizeof(bad[0]); index++)
		require(SparkGlm5NextVerifyDrafterParse(bad[index],4u,&kind,&path) == SPARK_STATUS_INVALID_ARGUMENT && kind == SPARK_GLM5_NEXT_VERIFY_DRAFTER_NONE,"unknown drafter rejected",index,0u,0u);
}

static void check_depth(void)
{
	uint32_t budget,produced,rows_max,position,depth;
	for (budget=0u; budget<=8u; budget++)
		for (produced=0u; produced<=budget; produced++)
			for (rows_max=0u; rows_max<=SPARK_GLM5_NEXT_VERIFY_ROWS_MAX; rows_max++)
				for (position=40u; position<80u; position+=3u)
				{
					depth = SparkGlm5NextVerifyDepth(budget,produced,rows_max,position,64u,4096u);
					require(depth == 0u || (depth + 1u <= budget - produced && depth + 1u <= rows_max && rows_max >= SPARK_GLM5_NEXT_VERIFY_ROWS_MIN),"a round never exceeds the frame budget or the regime rows",budget,produced,rows_max);
					require(depth == 0u || SparkGlm5NextVerifyRowsFit(position,depth + 1u,64u,4096u) == depth + 1u,"a round stays inside one attention regime",position,depth,0u);
					require(depth != 0u || budget - produced < 2u || rows_max < SPARK_GLM5_NEXT_VERIFY_ROWS_MIN || SparkGlm5NextVerifyRowsFit(position,2u,64u,4096u) < 2u,"a round is planned whenever two rows fit",budget,produced,position);
				}
	require(SparkGlm5NextVerifyDepth(8u,0u,8u,100u,64u,4096u) == 7u,"a full frame drafts seven",0u,0u,0u);
	require(SparkGlm5NextVerifyDepth(8u,5u,8u,100u,64u,4096u) == 2u,"the rest of the frame bounds the depth",0u,0u,0u);
	require(SparkGlm5NextVerifyDepth(8u,0u,4u,100u,64u,4096u) == 3u,"the regime rows bound the depth",0u,0u,0u);
	require(SparkGlm5NextVerifyDepth(8u,0u,8u,60u,64u,4096u) + 1u == SparkGlm5NextVerifyRowsFit(60u,8u,64u,4096u) && SparkGlm5NextVerifyDepth(8u,0u,8u,60u,64u,4096u) < 7u,"the split threshold bounds the depth",0u,0u,0u);
	require(SparkGlm5NextVerifyDepth(8u,7u,8u,100u,64u,4096u) == 0u,"one token left is a plain step",0u,0u,0u);
}

static void capture_start(uint32_t bounds[REGIMES][TABLE],uint32_t rows_max,uint32_t threshold,uint32_t max_positions)
{
	uint32_t regime,rows,bound;
	for (regime=0u; regime<REGIMES; regime++)
		for (rows=SPARK_GLM5_NEXT_VERIFY_ROWS_MIN; rows<=SPARK_GLM5_NEXT_VERIFY_ROWS_MAX; rows++)
		{
			bound = regime != SPARK_GLM5_NEXT_GRAPH_REGIME_SELECTED && rows <= rows_max ? SparkGlm5NextVerifyCaptureBound(regime,rows,0u,threshold,max_positions) : 0u;
			require(bound == 0u || (SparkGlm5NextGraphRegime(bound,threshold) == regime && SparkGlm5NextVerifyRowsFit(bound - rows,rows,threshold,max_positions) == rows),"a captured verify graph replays a window inside its regime",regime,rows,threshold);
			bounds[regime][SparkGlm5NextVerifyTableIndex(rows)] = bound;
		}
}

static void replay(uint32_t bounds[REGIMES][TABLE],uint32_t rows,uint32_t position,uint32_t rows_max,uint32_t threshold,uint32_t max_positions)
{
	uint32_t context,regime,index,row,bound,other;
	context = position + rows;
	regime = SparkGlm5NextGraphRegime(context,threshold);
	index = SparkGlm5NextVerifyTableIndex(rows);
	if ( regime == SPARK_GLM5_NEXT_GRAPH_REGIME_SELECTED && (bounds[regime][index] == 0u || context > bounds[regime][index]) )
		for (other=SPARK_GLM5_NEXT_VERIFY_ROWS_MIN; other<=rows_max; other++)
			bounds[regime][SparkGlm5NextVerifyTableIndex(other)] = SparkGlm5NextVerifyCaptureBound(regime,other,context,threshold,max_positions);
	bound = bounds[regime][index];
	require(bound != 0u && context <= bound && bound <= max_positions,"an admitted verify wave has a captured graph that covers it",rows,position,threshold);
	for (row=0u; row<rows; row++)
		require(SparkGlm5NextAttentionSplit(bound,threshold) == SparkGlm5NextAttentionSplit(position + row + 1u,threshold) && (bound > SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K) == (position + row + 1u > SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K),"every verify row keeps its own B1 attention choice",rows,position + row,threshold);
}

static void check_table(uint32_t rows_max,uint32_t threshold,uint32_t max_positions)
{
	uint32_t bounds[REGIMES][TABLE],slots[SPARK_GLM5_NEXT_VERIFY_ROWS_MAX],positions[SPARK_GLM5_NEXT_VERIFY_ROWS_MAX],position,rows,admitted = 0u;
	SparkStatus status;
	capture_start(bounds,rows_max,threshold,max_positions);
	for (position=0u; position<max_positions; position++)
		for (rows=SPARK_GLM5_NEXT_VERIFY_ROWS_MIN; rows<=rows_max; rows++)
		{
			fill(slots,positions,rows,position);
			status = SparkGlm5NextVerifyWaveCheck(rows,rows_max,0u,1u,0u,slots,positions,threshold,max_positions);
			require(status == SPARK_STATUS_OK || status == SPARK_STATUS_INVALID_ARGUMENT,"shape check returns a named status",rows,position,threshold);
			require((status == SPARK_STATUS_OK) == (SparkGlm5NextVerifyRowsFit(position,rows,threshold,max_positions) == rows),"admission is exactly the regime fit",rows,position,threshold);
			if ( status != SPARK_STATUS_OK )
				continue;
			admitted++;
			replay(bounds,rows,position,rows_max,threshold,max_positions);
		}
	require(admitted != 0u,"some verify waves are admitted",rows_max,threshold,max_positions);
}

static void check_frame_class(void)
{
	uint32_t bits,shape,sampled,warm,graph,disabled,failed,captured,frame_class,rank_local;
	for (bits=0u; bits<128u; bits++)
	{
		shape = bits & 1u;
		sampled = (bits >> 1) & 1u;
		warm = (bits >> 2) & 1u;
		graph = (bits >> 3) & 1u;
		disabled = (bits >> 4) & 1u;
		failed = (bits >> 5) & 1u;
		captured = (bits >> 6) & 1u;
		frame_class = SparkGlm5NextVerifyFrameClass(shape,sampled,warm,graph,disabled,failed,captured);
		rank_local = graph == 0u || disabled != 0u || failed != 0u;
		require(frame_class < SPARK_GLM5_NEXT_VERIFY_FRAME_CLASS_COUNT && frame_class != SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_NO_DRAFT,"frame class is a pre-draft class",bits,frame_class,0u);
		require((frame_class == SPARK_GLM5_NEXT_VERIFY_FRAME_ELIGIBLE) == (shape != 0u && sampled == 0u && warm != 0u && !rank_local && captured != 0u),"only a warm captured greedy B1 decode frame on the graph path is eligible",bits,frame_class,0u);
		require((frame_class == SPARK_GLM5_NEXT_VERIFY_FRAME_RANK_LOCAL) == (shape != 0u && sampled == 0u && warm != 0u && rank_local),"rank-local graph state is its own class once the frame would otherwise verify",bits,frame_class,0u);
		if ( shape == 0u )
			require(frame_class == SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_SHAPE,"shape comes first",bits,frame_class,0u);
		else if ( sampled != 0u )
			require(frame_class == SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_SAMPLED,"sampling comes next",bits,frame_class,0u);
		else if ( warm == 0u || (!rank_local && captured == 0u) )
			require(frame_class == SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_COLD,"cold frames stay plain",bits,frame_class,0u);
	}
}

int main(void)
{
	static const uint32_t thresholds[] = {0u,1u,2u,9u,64u,1000u,2048u,2049u,4096u};
	static const uint32_t max_positions[] = {4096u,32768u};
	uint32_t t,m,rows_max;
	check_parse();
	check_drafter();
	check_depth();
	check_shape();
	check_frame_class();
	for (m=0u; m<2u; m++)
		for (t=0u; t<sizeof(thresholds)/sizeof(thresholds[0]); t++)
			for (rows_max=SPARK_GLM5_NEXT_VERIFY_ROWS_MIN; rows_max<=SPARK_GLM5_NEXT_VERIFY_ROWS_MAX; rows_max+=3u)
				if ( thresholds[t] <= max_positions[m] )
					check_table(rows_max,thresholds[t],max_positions[m]);
	return(0);
}
'''


def main():
    with tempfile.TemporaryDirectory() as directory:
        source, binary = Path(directory) / "verify.c", Path(directory) / "verify"
        source.write_text(HARNESS)
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I.", "-Iinclude",
                        "-Imodel-families/glm5_next/include", str(source), "-o", str(binary)], cwd=ROOT, check=True)
        subprocess.run([str(binary)], check=True, timeout=600)
    print("PASS verify waves are admitted exactly when a captured regime graph replays every row with its own B1 attention choice, drafters parse strictly and rounds fit the frame and the regime")


if __name__ == "__main__":
    main()
