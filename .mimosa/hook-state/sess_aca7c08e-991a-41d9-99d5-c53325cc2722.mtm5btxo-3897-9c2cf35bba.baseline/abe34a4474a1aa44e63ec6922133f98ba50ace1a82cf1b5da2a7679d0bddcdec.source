#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <dirent.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_model_resident_endpoint.h"
#include "sparkpipe/spark_weightd.h"

static const char *SparkWeightdSpawnResolvePackDigest(
	const char *runtime_root)
{
	static char digest[65];
	char dir_path[512];
	const char *candidate = 0;
	DIR *directory;
	struct dirent *entry;
	FILE *sidecar;
	size_t read_bytes;
	(void)snprintf(dir_path,sizeof(dir_path),"%s/packs",runtime_root);
	directory = opendir(dir_path);
	if ( directory == 0 )
		return 0;
	while ( (entry = readdir(directory)) != 0 )
	{
		size_t name_bytes = strlen(entry->d_name);
		if ( name_bytes > 7u && strcmp(entry->d_name + name_bytes - 7u,
			".sha256") == 0 )
		{
			if ( candidate != 0 )
			{
				(void)closedir(directory);
				return 0;
			}
			candidate = (const char *)entry->d_name;
		}
	}
	(void)closedir(directory);
	if ( candidate == 0 )
		return 0;
	{
		char sidecar_path[512];
		(void)snprintf(sidecar_path,sizeof(sidecar_path),"%s/packs/%s",
			runtime_root,candidate);
		sidecar = fopen(sidecar_path,"rb");
		if ( sidecar == 0 )
			return 0;
		read_bytes = fread(digest,sizeof(char),SPARK_WEIGHTD_SHA256_HEX_BYTES - 1u,sidecar);
		(void)fclose(sidecar);
		if ( read_bytes != 64u )
			return 0;
		digest[64] = '\0';
	}
	return digest;
}

void SparkModelResidentdEnsureWeightd(
	const char *runtime_root_argument,
	const char *socket_path_argument)
{
	const char *socket_path;
	const char *runtime_root;
	struct sockaddr_un address;
	struct stat binary_stat;
	struct timespec pause;
	int probe_fd;
	int attempts;
	char binary_path[512];
	pid_t child;

	if ( socket_path_argument == 0 || socket_path_argument[0] == '\0' )
		return;
	socket_path = socket_path_argument;
	runtime_root = runtime_root_argument;
	(void)setenv("SPARK_WEIGHTD_SOCKET",socket_path,1);
	if ( strlen(socket_path) >= sizeof(address.sun_path) )
		return;
	{
		const char *digest = SparkWeightdSpawnResolvePackDigest(runtime_root);
		if ( digest != 0 )
			(void)setenv("SPARK_WEIGHTD_PACK_SHA256",digest,1);
		else
			fprintf(stderr,"model_residentd weightd-no-digest-sidecar "
				"root=%s (seam falls back to direct load)\n",runtime_root);
	}
	probe_fd = socket(AF_UNIX,SOCK_STREAM,0);
	if ( probe_fd >= 0 )
	{
		memset(&address,0,sizeof(address));
		address.sun_family = AF_UNIX;
		(void)snprintf(address.sun_path,sizeof(address.sun_path),"%s",socket_path);
		if ( connect(probe_fd,(struct sockaddr *)&address,sizeof(address)) == 0 )
		{
			(void)close(probe_fd);
			return;
		}
		(void)close(probe_fd);
	}
	(void)snprintf(binary_path,sizeof(binary_path),"%s/bin/sparkpipe_weightd",runtime_root);
	if ( stat(binary_path,&binary_stat) != 0 )
	{
		fprintf(stderr,"model_residentd weightd-not-staged path=%s (seam falls back to direct load)\n",binary_path);
		return;
	}
	child = fork();
	if ( child < 0 )
		return;
	if ( child == 0 )
	{
		char *const argv[] = { (char *)"sparkpipe_weightd",
			(char *)"--socket",(char *)socket_path, 0 };
		(void)setsid();
		(void)execv(binary_path,argv);
		_exit(127);
	}
	for ( attempts = 0; attempts < 100; attempts++ )
	{
		probe_fd = socket(AF_UNIX,SOCK_STREAM,0);
		if ( probe_fd >= 0 )
		{
			memset(&address,0,sizeof(address));
			address.sun_family = AF_UNIX;
			(void)snprintf(address.sun_path,sizeof(address.sun_path),"%s",socket_path);
			if ( connect(probe_fd,(struct sockaddr *)&address,sizeof(address)) == 0 )
			{
				(void)close(probe_fd);
				fprintf(stderr,"model_residentd weightd-started pid=%ld socket=%s\n",(long)child,socket_path);
				return;
			}
			(void)close(probe_fd);
		}
		pause.tv_sec = 0;
		pause.tv_nsec = 100000000L;
		nanosleep(&pause,0);
	}
	(void)waitpid(child,0,WNOHANG);
	fprintf(stderr,"model_residentd weightd-start-timeout socket=%s (seam falls back to direct load)\n",socket_path);
}

