#ifndef COMMON_H
#define COMMON_H 1


/* Implementation constants */

#define FOR_UNIX  0
#define FOR_MAC   0
#define FOR_MSDOS 0
#if (defined(__unix__) || defined(unix) || defined(___AIX))
#undef FOR_UNIX
#define FOR_UNIX 1
#define OPSYS "Unix"
#endif
#ifdef THINKC
#undef FOR_MAC
#define FOR_MAC 1
#define OPSYS "Macintosh"
#endif
#ifdef __MSDOS__
#undef FOR_MSDOS
#define FOR_MSDOS 1
#define OPSYS "MSDOS"
#endif
#ifdef _MSDOS
#undef FOR_MSDOS
#define FOR_MSDOS 1
#define OPSYS "MSDOS"
#endif

#if defined(__NeXT__)
#define _NEXT_SOURCE
#endif

#ifdef SUN
#define strtoul(s,sp,base) ((unsigned long) strtol(s,sp,base))
#define CLK_TCK 1000000
#endif
#ifdef __vax
#define strtoul(s,sp,base) ((unsigned long) atol(s))
#define strtol(s,sp,base) (atol(s))
#define strtod(s,sp) (atof(s))
#define L_tmpnam 100
extern char *getenv(char *name);
#endif

#if ((defined(__TURBOC__)) || (defined(_MSC_VER)) || (defined(SUN)))
#define string_compare stricmp
#endif
#if (((defined(__ultrix))) || ((defined(NEXT))))
#define string_compare strcasecmp
#endif

#ifndef NETWORKING
#if (((defined(__ultrix))) || ((defined(__hpux))))
#define NETWORKING 1
#else
#define NETWORKING 0
#endif
#endif /* (ndef NETWORKING) */

#ifndef WINDOWS
typedef int Window_Type;
typedef int Image_Type;
#endif

#ifndef string_compare
#define string_compare framer_strcmp_ci
#endif



/* Miscellaneous definitions */

#ifndef AP
#ifdef _no_ansi_prototypes
#define AP(args)       ()
#else
#define AP(args)       args
#endif
#endif /* AP */

#undef True
#undef False

typedef enum BOOLEAN { True=1, False=0} boolean;

extern int default_display_width;

void *careful_allocate AP((size_t num,size_t size));
void *careful_reallocate AP((void *ptr,size_t new_size));
#define fra_allocate careful_allocate
#define fra_reallocate careful_reallocate
#define ALLOCATE(into,type,size) \
   into = (type *) fra_allocate(size,sizeof(type)); \
   if ((into == NULL) && (size != 0)) raise_crisis(Out_Of_Memory);

#define NOT(x) (!(x))

#define DO_TIMES(i,limit) \
    int i, _limit; \
    for (i=0,_limit=(limit);i<_limit;i++)

#define NULLP(x) (x == NULL)
#define EQ(x,y) (x == y)

#define IS_TERMINATOR_P(x) \
   (!(isgraph(x)) || (x == '/') || (x == '\"') || (x == '#') || \
    (x == '(') || (x == ')') || (x == '{') || (x == '}') || \
    (x == ' '))

#define WITH_TEMP_STRING(string,size) \
   char _tmp_string[size], *string; string=_tmp_string;
#define END_WITH_TEMP_STRING(string) \
   if (string != _tmp_string) free(string);

#ifndef DEFAULT_DISPLAY_WIDTH 
#define DEFAULT_DISPLAY_WIDTH 80
#endif


/* Handling exceptions */

/* A general-purpose exception-handling system for C
   by Jonathan Amsterdam, 1991 */
 
#include <setjmp.h>

#define EXCEPTION_DETAILS_SIZE 1000
typedef char *exception;
extern char exception_details[];
extern exception Out_Of_Memory;

typedef struct jbr {
  jmp_buf jb;
  struct jbr *next;
  struct jbr *self;
  }  jmp_buf_rec;
 
#define WITH_HANDLING {jmp_buf_rec _jbr; \
		       push_jbr(&_jbr);  \
		       if (setjmp(_jbr.jb) == 0) \
			 {theException = NULL;
#define ON_EXCEPTION pop_jbr();} else {
#define END_HANDLING }}
#define UNWIND_PROTECT WITH_HANDLING
#define ON_UNWIND    pop_jbr();}}
#define END_UNWIND   if (theException != NULL) reraise();
#define FLET(tp,pl,vl) {jmp_buf_rec _jbr; tp _tmp; \
                        _tmp=(pl); pl=(vl); \
			push_jbr(&_jbr);  \
                        if (setjmp(_jbr.jb) == 0) \
           		 {theException = NULL;

#define END_FLET(place) pop_jbr(); } place=_tmp; if (theException != NULL) reraise();}
 
extern exception theException;
#define CLEAR_EXCEPTION() theException=NULL; *exception_details='\0'
 
void push_jbr(jmp_buf_rec *jbr);
void pop_jbr(void);

int raise_event AP((exception ex));
int raise_crisis AP((exception ex));
int raise_crisis_with_details AP((exception ex,char *details));
int reraise AP((void));
void report_crisis AP((char *ex));

#endif /* ndef COMMON_H */
