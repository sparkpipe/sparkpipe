#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

#include "fixtures/test_child_guard.h"

static void *TestChildGuardThreadIdle(void *argument)
{
	(void)argument;
	for (;;)
		(void)pause();
	return(0);
}

int main(void)
{
	pthread_t helper;
	int pipe_fds[2];
	pid_t child;
	char byte;
	assert(pipe(pipe_fds) == 0);
	child = TestChildGuardFork();
	assert(child >= 0);
	if ( child == 0 )
	{
		(void)close(pipe_fds[0]);
		for (;;)
			(void)pause();
	}
	(void)close(pipe_fds[1]);
	(void)printf("fixture %ld\n",(long)child);
	(void)fflush(stdout);
	assert(pthread_create(&helper,0,TestChildGuardThreadIdle,0) == 0);
	assert(pthread_kill(helper,SIGTERM) == 0);
	for (;;)
	{
		ssize_t bytes = read(pipe_fds[0],&byte,1u);
		assert(bytes != 0);
	}
	return(0);
}
