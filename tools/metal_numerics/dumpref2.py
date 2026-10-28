from pathlib import Path
p = Path("tools/qwen38_27b_reference.c")
s = p.read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
rep("static const char *ref_dump_prefix = 0;\n", "static const char *ref_dump_prefix = 0;\nstatic float *ref_dump_out = 0;\n")
rep("""		fprintf(stderr,"dumped attention layer %d inputs: %u rows, %u keys\\n",ref_dump_index,rows,length);
	}
""", """		fprintf(stderr,"dumped attention layer %d inputs: %u rows, %u keys\\n",ref_dump_index,rows,length);
		ref_dump_out = (float *)malloc((size_t)rows * REF_Q * sizeof(float));
	}
""")
rep("""		for ( uint32_t row = 0u; row < rows; row++ )
			for ( uint32_t d = 0u; d < REF_HEAD_DIM; d++ )
			{
				const float g = gq[""", """		if ( ref_dump_out != 0 && (int)index == ref_dump_index )
			for ( uint32_t row = 0u; row < rows; row++ )
				memcpy(ref_dump_out + ((uint64_t)row * REF_HEADS + head) * REF_HEAD_DIM,out + (uint64_t)row * REF_HEAD_DIM,REF_HEAD_DIM * sizeof(float));
		for ( uint32_t row = 0u; row < rows; row++ )
			for ( uint32_t d = 0u; d < REF_HEAD_DIM; d++ )
			{
				const float g = gq[""")
rep("""	RefLinearRows(variant,gated,rows,REF_Q,layer->o,REF_HIDDEN,partial);
	free(gq);""", """	if ( ref_dump_out != 0 && (int)index == ref_dump_index )
	{
		char dump[4096];
		FILE *file;
		snprintf(dump,sizeof(dump),"%s.%s.out",ref_dump_prefix,variant == 0u ? "a" : variant == 1u ? "b" : variant == 2u ? "c" : "d");
		file = fopen(dump,"wb"); fwrite(ref_dump_out,sizeof(float),(size_t)rows * REF_Q,file); fclose(file);
		free(ref_dump_out);
		ref_dump_out = 0;
	}
	RefLinearRows(variant,gated,rows,REF_Q,layer->o,REF_HIDDEN,partial);
	free(gq);""")
rep("""			if ( strcmp(argv[5],"float-math-off") == 0 )
			{
				RefTeacher(&model,&state,prompt,argv[3],0u,"dump",0u);
				continue;
			}""", """			if ( strcmp(argv[5],"float-math-off") == 0 )
			{
				RefTeacher(&model,&state,prompt,argv[3],0u,"dump",0u);
				RefTeacher(&model,&state,prompt,argv[3],1u,"dumpb",0u);
				continue;
			}""")
p.write_text(s)
print("ok")
