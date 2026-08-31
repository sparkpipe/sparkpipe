/* weightd spawn (docs/WEIGHTD_DESIGN.md): the residentd owns the daemon
 * lifecycle so every deployment inherits pack residency structurally.
 * This file deliberately carries the W2b env contract (kill-switch
 * parity with SPARK_STAGE_MODULE_LOAD_PIPELINE=0) so model_residentd.c
 * itself stays free of environment fallbacks (the production
 * fail-closed rule the selection-contract gate pins). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_model_resident_endpoint.h"

void SparkModelResidentdEnsureWeightd(const void *configuration_runtime_root);
#include <sys/wait.h>

void SparkModelResidentdEnsureWeightd(
	const void *configuration_runtime_root)
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

	socket_path = getenv("SPARK_WEIGHTD_SOCKET");
	if ( socket_path == 0 || socket_path[0] == '\0' )
		return;
	if ( getenv("SPARK_WEIGHTD_ATTACH") != 0 &&
		strcmp(getenv("SPARK_WEIGHTD_ATTACH"),"0") == 0 )
		return;
	if ( strlen(socket_path) >= sizeof(address.sun_path) )
		return;
	/* already up? the common case after the first launch */
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
	if ( configuration_runtime_root == 0 )
		return;
	runtime_root = (const char *)configuration_runtime_root;
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
	/* wait for the socket (the daemon binds before serving; cold loads
	 * happen on the first attach, inside the seam's own deadline) */
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

