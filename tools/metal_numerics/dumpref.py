from pathlib import Path
p = Path("tools/qwen38_27b_reference.c")
s = p.read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
rep("static uint16_t *ref_hidden_record = 0;\n", "static uint16_t *ref_hidden_record = 0;\nstatic int ref_dump_index = -1;\nstatic const char *ref_dump_prefix = 0;\n")
rep("""	memcpy(values + (uint64_t)first * REF_KV,vraw,(size_t)rows * REF_KV * sizeof(float));
""", """	memcpy(values + (uint64_t)first * REF_KV,vraw,(size_t)rows * REF_KV * sizeof(float));
	if ( (int)index == ref_dump_index && ref_dump_prefix != 0 )
	{
		char dump[4096];
		FILE *file;
		snprintf(dump,sizeof(dump),"%s.q",ref_dump_prefix);
		file = fopen(dump,"wb"); fwrite(query,sizeof(float),(size_t)rows * REF_Q,file); fclose(file);
		snprintf(dump,sizeof(dump),"%s.k",ref_dump_prefix);
		file = fopen(dump,"wb"); fwrite(keys,sizeof(float),(size_t)length * REF_KV,file); fclose(file);
		snprintf(dump,sizeof(dump),"%s.v",ref_dump_prefix);
		file = fopen(dump,"wb"); fwrite(values,sizeof(float),(size_t)length * REF_KV,file); fclose(file);
		fprintf(stderr,"dumped attention layer %d inputs: %u rows, %u keys\\n",ref_dump_index,rows,length);
	}
""")
rep("""	if ( argc != 5 && !(argc == 6 && (strcmp(argv[5],"fp32-activations") == 0""", """	if ( argc == 8 && strcmp(argv[5],"dump-attention") == 0 )
	{
		ref_dump_index = atoi(argv[6]);
		ref_dump_prefix = argv[7];
		argc = 6;
		argv[5] = (char *)"float-math-off";
	}
	if ( argc != 5 && !(argc == 6 && (strcmp(argv[5],"float-math-off") == 0 || strcmp(argv[5],"fp32-activations") == 0""")
rep("""			if ( strcmp(argv[5],"fp32-activations") == 0 )""", """			if ( strcmp(argv[5],"float-math-off") == 0 )
			{
				RefTeacher(&model,&state,prompt,argv[3],0u,"dump",0u);
				continue;
			}
			if ( strcmp(argv[5],"fp32-activations") == 0 )""")
p.write_text(s)
print("ok")
