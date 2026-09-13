#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_glm52_dspark_pack.h"
/* Container framing comes from the glm52 internal header - the draft-pack
 * container IS the glm52 stage-pack format (see the minter,
 * tools/glm52_dspark_stagepack.py --emit-pack). */

/* Argv-driven resolve driver for the MINTER ROUND-TRIP leg
 * (tests/test_glm52_dspark_stagepack.py): the python gate mints a dwarf
 * pack, wraps it with --emit-pack, then executes the REAL
 * SparkGlm52DsparkPackResolve here over that container and compares the
 * extracted artifacts' byte counts + sha256 against the mint receipt.
 * Machine-readable output: "status=<name>" always; one
 * "artifact <name> bytes=<n> sha256=<hex>" line per extracted artifact on
 * success. Exit code mirrors the resolver verdict so negative legs can
 * pin the exact refusal class: 0 OK, 10 HASH_MISMATCH, 11 SCHEMA_ERROR,
 * 1 anything else (including usage). */
int main(int argc,char **argv)
{
	char manifest_path[1024],config_path[1024],safetensors_path[1024];
	const char *paths[3];
	const char *names[3] = {
		"manifest.json","config.json","model.safetensors" };
	char hex[SPARK_SHA256_HEX_BYTES];
	SparkStatus status;
	int index;
	if ( argc != 3 )
	{
		fprintf(stderr,"usage: %s <pack-file> <scratch-dir>\n",
			argv[0]);
		return 1;
	}
	status = SparkGlm52DsparkPackResolve(argv[1],argv[2],
		manifest_path,sizeof(manifest_path),
		config_path,sizeof(config_path),
		safetensors_path,sizeof(safetensors_path));
	printf("status=%s\n",SparkStatusToString(status));
	if ( status != SPARK_STATUS_OK )
	{
		if ( status == SPARK_STATUS_HASH_MISMATCH )
			return 10;
		if ( status == SPARK_STATUS_SCHEMA_ERROR )
			return 11;
		return 1;
	}
	paths[0] = manifest_path;
	paths[1] = config_path;
	paths[2] = safetensors_path;
	for ( index = 0; index < 3; ++index )
	{
		FILE *file = fopen(paths[index],"rb");
		long bytes;
		if ( file == 0 )
			return 1;
		if ( fseek(file,0L,SEEK_END) != 0 || (bytes = ftell(file)) < 0L )
			{ fclose(file); return 1; }
		fclose(file);
		if ( SparkSha256File(paths[index],hex) != SPARK_STATUS_OK )
			return 1;
		printf("artifact %s bytes=%ld sha256=%s\n",
			names[index],bytes,hex);
	}
	return 0;
}
