#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "fixtures/test_child_guard.h"

#define TEST_GUARD_MAX_PIDS 8u
#define TEST_GUARD_GONE_WAIT_MS 5000u
#define TEST_GUARD_SURVIVE_MS 500u

typedef struct TestGuardMode
{
	const char *name;
	uint32_t guarded;
	uint32_t grandchild;
	int32_t signal_number;
	int32_t exit_code;
} TestGuardMode;

static const TestGuardMode TestGuardModes[] =
{
	{"assert",1u,0u,SIGABRT,-1},
	{"segv",1u,0u,SIGSEGV,-1},
	{"sigterm",1u,0u,SIGTERM,-1},
	{"sigint",1u,0u,SIGINT,-1},
	{"sighup",1u,0u,SIGHUP,-1},
	{"sigkill",1u,0u,SIGKILL,-1},
	{"exit",1u,0u,0,3},
	{"grandchild",1u,1u,SIGABRT,-1},
	{"control",0u,0u,SIGABRT,-1}
};

static pid_t TestGuardControlSurvivors[TEST_GUARD_MAX_PIDS];

static void TestGuardSleepMs(uint32_t milliseconds)
{
	struct timespec delay;
	delay.tv_sec = (time_t)(milliseconds / 1000u);
	delay.tv_nsec = (long)(milliseconds % 1000u) * 1000000L;
	(void)nanosleep(&delay,0);
}

static void TestGuardKillControlSurvivors(void)
{
	uint32_t index;
	for (index=0u; index<TEST_GUARD_MAX_PIDS; index++)
		if ( TestGuardControlSurvivors[index] > 0 )
			(void)kill(TestGuardControlSurvivors[index],SIGKILL);
}

static const TestGuardMode *TestGuardFindMode(const char *name)
{
	uint32_t index;
	for (index=0u; index<sizeof(TestGuardModes)/sizeof(TestGuardModes[0]); index++)
		if ( strcmp(TestGuardModes[index].name,name) == 0 )
			return(&TestGuardModes[index]);
	return(0);
}

static void TestGuardExecSleeper(void)
{
	execl("/bin/sleep","sleep","300",(char *)0);
	_exit(127);
}

static pid_t TestGuardVictimSpawn(const TestGuardMode *mode)
{
	pid_t child;
	child = mode->guarded != 0u ? TestChildGuardFork() : fork();
	assert(child >= 0);
	if ( child == 0 )
		TestGuardExecSleeper();
	return(child);
}

static pid_t TestGuardVictimSpawnWithGrandchild(pid_t *grandchild)
{
	int32_t relay[2];
	pid_t child,inner;
	assert(pipe(relay) == 0);
	child = TestChildGuardFork();
	assert(child >= 0);
	if ( child == 0 )
	{
		(void)close(relay[0]);
		inner = fork();
		if ( inner == 0 )
			TestGuardExecSleeper();
		if ( inner < 0 || write(relay[1],&inner,sizeof(inner)) != (ssize_t)sizeof(inner) )
			_exit(124);
		(void)close(relay[1]);
		for (;;)
			(void)pause();
	}
	(void)close(relay[1]);
	assert(read(relay[0],grandchild,sizeof(*grandchild)) == (ssize_t)sizeof(*grandchild));
	(void)close(relay[0]);
	return(child);
}

static int TestGuardVictim(const char *mode_name,int32_t report_fd,int32_t go_fd)
{
	const TestGuardMode *mode;
	pid_t pids[TEST_GUARD_MAX_PIDS];
	uint32_t count,index;
	char go;
	mode = TestGuardFindMode(mode_name);
	assert(mode != 0);
	assert(fcntl(report_fd,F_SETFD,FD_CLOEXEC) == 0);
	assert(fcntl(go_fd,F_SETFD,FD_CLOEXEC) == 0);
	count = 0u;
	pids[count++] = TestGuardVictimSpawn(mode);
	pids[count++] = TestGuardVictimSpawn(mode);
	if ( mode->grandchild != 0u )
	{
		pids[count] = TestGuardVictimSpawnWithGrandchild(&pids[count + 1u]);
		count += 2u;
	}
	if ( mode->guarded != 0u )
		assert(TestChildGuardTrackedCount() == (mode->grandchild != 0u ? 3u : 2u));
	for (index=0u; index<count; index++)
		dprintf(report_fd,"pid %ld\n",(long)pids[index]);
	dprintf(report_fd,"ready\n");
	assert(read(go_fd,&go,1u) == 1);
	if ( strcmp(mode->name,"exit") == 0 )
		exit(3);
	if ( strcmp(mode->name,"assert") == 0 || strcmp(mode->name,"grandchild") == 0 ||
		strcmp(mode->name,"control") == 0 )
		assert(0 && "deliberate failure with fixture children running");
	(void)raise(mode->signal_number);
	for (;;)
		(void)pause();
}

static uint32_t TestGuardProcessGone(pid_t pid)
{
	char path[64];
	char text[512];
	const char *state;
	size_t length;
	FILE *file;
	if ( kill(pid,0) != 0 && errno == ESRCH )
		return(1u);
	assert(snprintf(path,sizeof(path),"/proc/%ld/stat",(long)pid) > 0);
	file = fopen(path,"r");
	if ( file == 0 )
		return(0u);
	length = fread(text,1u,sizeof(text) - 1u,file);
	(void)fclose(file);
	text[length] = '\0';
	state = strrchr(text,')');
	return(state != 0 && state[1] == ' ' && (state[2] == 'Z' || state[2] == 'X') ? 1u : 0u);
}

static uint32_t TestGuardWaitGone(const pid_t *pids,uint32_t count,uint32_t wait_ms)
{
	uint32_t waited,index,gone;
	for (waited=0u; ; waited+=20u)
	{
		gone = 0u;
		for (index=0u; index<count; index++)
			gone += TestGuardProcessGone(pids[index]);
		if ( gone == count || waited >= wait_ms )
			return(gone);
		TestGuardSleepMs(20u);
	}
}

static uint32_t TestGuardHasMarker(pid_t pid,pid_t owner)
{
	char path[64];
	char environment[16384];
	char expected[96];
	size_t length,offset;
	uint32_t found;
	FILE *file;
	assert(snprintf(path,sizeof(path),"/proc/%ld/environ",(long)pid) > 0);
	assert(snprintf(expected,sizeof(expected),"%s=%ld.",TEST_CHILD_GUARD_OWNER_VARIABLE,(long)owner) > 0);
	file = fopen(path,"r");
	if ( file == 0 )
		return(0u);
	length = fread(environment,1u,sizeof(environment) - 1u,file);
	(void)fclose(file);
	environment[length] = '\0';
	found = 0u;
	for (offset=0u; offset<length; offset+=strlen(environment + offset) + 1u)
		if ( strncmp(environment + offset,expected,strlen(expected)) == 0 )
			found = 1u;
	return(found);
}

static void TestGuardCheckMarker(pid_t pid,pid_t owner)
{
	uint32_t waited;
	if ( access("/proc/self/environ",R_OK) != 0 )
		return;
	for (waited=0u; waited<TEST_GUARD_GONE_WAIT_MS && TestGuardHasMarker(pid,owner) == 0u; waited+=20u)
		TestGuardSleepMs(20u);
	assert(TestGuardHasMarker(pid,owner) == 1u);
}

static uint32_t TestGuardReadReport(FILE *report,pid_t *pids)
{
	char line[128];
	long value;
	uint32_t count;
	count = 0u;
	while ( fgets(line,sizeof(line),report) != 0 )
	{
		if ( strcmp(line,"ready\n") == 0 )
			return(count);
		assert(sscanf(line,"pid %ld",&value) == 1);
		assert(count < TEST_GUARD_MAX_PIDS);
		pids[count++] = (pid_t)value;
	}
	assert(0 && "victim closed its report before ready");
	return(0u);
}

static void TestGuardRunMode(const char *self,const TestGuardMode *mode)
{
	int32_t report_pipe[2],go_pipe[2];
	char report_text[16],go_text[16];
	pid_t pids[TEST_GUARD_MAX_PIDS];
	pid_t victim;
	uint32_t count,index,gone;
	int status;
	FILE *report;
	assert(pipe(report_pipe) == 0);
	assert(pipe(go_pipe) == 0);
	victim = TestChildGuardFork();
	assert(victim >= 0);
	if ( victim == 0 )
	{
		(void)close(report_pipe[0]);
		(void)close(go_pipe[1]);
		(void)snprintf(report_text,sizeof(report_text),"%d",report_pipe[1]);
		(void)snprintf(go_text,sizeof(go_text),"%d",go_pipe[0]);
		execl(self,self,"--victim",mode->name,report_text,go_text,(char *)0);
		_exit(127);
	}
	(void)close(report_pipe[1]);
	(void)close(go_pipe[0]);
	report = fdopen(report_pipe[0],"r");
	assert(report != 0);
	count = TestGuardReadReport(report,pids);
	assert(count == (mode->grandchild != 0u ? 4u : 2u));
	for (index=0u; index<count; index++)
	{
		assert(kill(pids[index],0) == 0);
		assert(TestGuardProcessGone(pids[index]) == 0u);
		if ( mode->guarded != 0u )
		{
			assert(getpgid(pids[index]) != getpgid(victim));
			if ( mode->grandchild == 0u || index != 2u )
				TestGuardCheckMarker(pids[index],victim);
		}
	}
	assert(write(go_pipe[1],"g",1u) == 1);
	(void)close(go_pipe[1]);
	assert(waitpid(victim,&status,0) == victim);
	if ( mode->exit_code >= 0 )
		assert(WIFEXITED(status) && WEXITSTATUS(status) == mode->exit_code);
	else
		assert(WIFSIGNALED(status) && WTERMSIG(status) == mode->signal_number);
	(void)fclose(report);
	if ( mode->guarded == 0u )
	{
		memcpy(TestGuardControlSurvivors,pids,sizeof(pids[0]) * count);
		TestGuardSleepMs(TEST_GUARD_SURVIVE_MS);
		assert(TestGuardWaitGone(pids,count,0u) == 0u);
		TestGuardKillControlSurvivors();
		assert(TestGuardWaitGone(pids,count,TEST_GUARD_GONE_WAIT_MS) == count);
		memset(TestGuardControlSurvivors,0,sizeof(TestGuardControlSurvivors));
		printf("PASS control: %u unguarded children outlive a failing test, so the check can see a leak\n",count);
		return;
	}
	gone = TestGuardWaitGone(pids,count,TEST_GUARD_GONE_WAIT_MS);
	if ( gone != count )
		for (index=0u; index<count; index++)
			if ( TestGuardProcessGone(pids[index]) == 0u )
			{
				fprintf(stderr,"FAIL %s: child %ld survived its test\n",mode->name,(long)pids[index]);
				(void)kill(pids[index],SIGKILL);
			}
	assert(gone == count);
	printf("PASS %s: %u fixture children gone after the test failed\n",mode->name,count);
}

int main(int argc,char **argv)
{
	uint32_t index;
	if ( argc == 5 && strcmp(argv[1],"--victim") == 0 )
		return(TestGuardVictim(argv[2],atoi(argv[3]),atoi(argv[4])));
	assert(argc == 1);
	assert(atexit(TestGuardKillControlSurvivors) == 0);
	assert(setvbuf(stdout,0,_IOLBF,0) == 0);
	for (index=0u; index<sizeof(TestGuardModes)/sizeof(TestGuardModes[0]); index++)
	{
#if !defined(__linux__)
		if ( TestGuardModes[index].signal_number == SIGKILL )
		{
			printf("SKIP sigkill: the parent-death signal needs Linux\n");
			continue;
		}
#endif
		TestGuardRunMode(argv[0],&TestGuardModes[index]);
	}
	printf("PASS test_child_guard\n");
	return(0);
}
