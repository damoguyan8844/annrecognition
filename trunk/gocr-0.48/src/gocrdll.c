// GocrDll.cpp : Defines the entry point for the DLL application.
//
// 
#include "stdafx.h"
#include "GocrDll.h"

#include "config.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#ifdef HAVE_GETTIMEOFDAY 
#include <sys/time.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "pnm.h"
#include "pgm2asc.h"
#include "pcx.h"
#include "ocr0.h"		/* only_numbers */
#include "progress.h"
#include "version.h"

BOOL APIENTRY DllMain( HANDLE hModule, 
                       DWORD  ul_reason_for_call, 
                       LPVOID lpReserved
					 )
{
    switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
			break;
		case DLL_THREAD_ATTACH:
		case DLL_THREAD_DETACH:
		case DLL_PROCESS_DETACH:
			break;
    }
    return TRUE;
}


static void mark_start(job_t *job) {
	assert(job);
	
// 		if (job->cfg.verbose) {
// 			out_version(0);
// 			/* insert some helpful info for support */
// 			fprintf(stderr, "# compiled: " __DATE__ );
// #if defined(__GNUC__)
// 			fprintf(stderr, " GNUC-%d", __GNUC__ );
// #endif
// #ifdef __GNUC_MINOR__
// 			fprintf(stderr, ".%d", __GNUC_MINOR__ );
// #endif
// #if defined(__linux)
// 			fprintf(stderr, " linux");
// #elif defined(__unix)
// 			fprintf(stderr, " unix");
// #endif
// #if defined(__WIN32) || defined(__WIN32__)
// 			fprintf(stderr, " WIN32");
// #endif
// #if defined(__WIN64) || defined(__WIN64__)
// 			fprintf(stderr, " WIN64");
// #endif
// #if defined(__VERSION__)
// 			fprintf(stderr, " version " __VERSION__ );
// #endif
// 			fprintf(stderr, "\n");
// 			fprintf(stderr,
// 				"# options are: -l %d -s %d -v %d -c %s -m %d -d %d -n %d -a %d -C \"%s\"\n",
// 				job->cfg.cs, job->cfg.spc, job->cfg.verbose, job->cfg.lc, job->cfg.mode, 
// 				job->cfg.dust_size, job->cfg.only_numbers, job->cfg.certainty,
// 				job->cfg.cfilter);
// 			fprintf(stderr, "# file: %s\n", job->src.fname);
// #ifdef USE_UNICODE
// 			fprintf(stderr,"# using unicode\n");
// #endif
// #ifdef HAVE_GETTIMEOFDAY
// 			gettimeofday(&job->tmp.init_time, NULL);
// #endif
// 		}
}

static void mark_end(job_t *job) {
	assert(job);
	
#ifdef HAVE_GETTIMEOFDAY
	/* show elapsed time */
	if (job->cfg.verbose) {
		struct timeval end, result;
		gettimeofday(&end, NULL);
		timeval_subtract(&result, &end, &job->tmp.init_time);
		fprintf(stderr,"Elapsed time: %d:%02d:%3.3f.\n", (int)result.tv_sec/60,
			(int)result.tv_sec%60, (float)result.tv_usec/1000);
	}
#endif
}

static int read_picture(job_t *job) {
	int rc=0;
	assert(job);
	
	if (strstr(job->src.fname, ".pcx"))
		readpcx(job->src.fname, &job->src.p, job->cfg.verbose);
	else
		rc=readpgm(job->src.fname, &job->src.p, job->cfg.verbose);
	return rc; /* 1 for multiple images, 0 else */
}

/* subject of change, we need more output for XML (ToDo) */
static void print_output(job_t *job) {
	int linecounter = 0;
	const char *line;
	
	assert(job);
    
	linecounter = 0;
	line = getTextLine(linecounter++);
	while (line) {
		/* notice: decode() is shiftet to getTextLine since 0.38 */
		fputs(line, stdout);
		if (job->cfg.out_format==HTML) fputs("<br />",stdout);
		if (job->cfg.out_format!=XML)  fputc('\n', stdout);
		line = getTextLine(linecounter++);
	}
	free_textlines();
}

static int print_output_ex(job_t *job,char * output,long length) {
	int linecounter = 0;
	const char *line;
	int index=0;
	int i=0;

	assert(job);
    
	linecounter = 0;
	line = getTextLine(linecounter++);
	while (line) {
		/* notice: decode() is shiftet to getTextLine since 0.38 */
		i=0;
		while(line[i]!='\0') output[index++]=line[i++]; 
		
		line = getTextLine(linecounter++);
	}
	output[index]='\0';
	free_textlines();
	return index;
}



/* FIXME jb: remove JOB; */
job_t *JOB;

GOCRDLL_API int __cdecl GocrDll_Eng( const char * imageFile,char * output,int length )
{
  int multipnm=1;
  job_t job;
  int ret=0;
  char f_name[256];
  strcpy(f_name,imageFile);

  JOB = &job;
  

  while (multipnm==1) {

    job_init(&job);
  
    //set parameter -i 
	job.src.fname = f_name;

    mark_start(&job);

    multipnm = read_picture(&job);
    /* separation of main and rest for using as lib
       this will be changed later => introduction of set_option()
       for better communication to the engine  */
    if (multipnm<0) break; /* read error */
  
    /* call main loop */
    pgm2asc(&job);

    mark_end(&job);
  
    ret=print_output_ex(&job,output,length);

    job_free(&job);
  
  }
  return ret;
}
