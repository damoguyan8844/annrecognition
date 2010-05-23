/* C Mode */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* $Header: /mas/framer/headers/RCS/framer.h,v 1.3 1994/01/26 19:52:10 haase Exp $ */

#ifndef FRAMER_H
#define FRAMER_H

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
#define CLK_TCK 64
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
int strcasecmp(char *s1,char *s2);
#endif

#ifndef NETWORKING
#if (((defined(__ultrix))) || ((defined(__hpux))))
#define NETWORKING 1
#else
#define NETWORKING 0
#endif
#endif /* (ndef NETWORKING) */

#ifndef WINDOWS
typedef int *Window_Type;
typedef int *Image_Type;
typedef int *Sound_Type;
#endif

#ifndef string_compare
#define string_compare framer_strcmp_ci
int framer_strcmp_ci(char *s1,char *s2);
#endif

#define flip_word(wd) (((wd) << 24) | \
	     	       (((wd) << 8) & 0x00ff0000) | \
		       (((wd) >> 8) & 0x0000ff00) | \
                       ((unsigned long)(wd) >>24) )

#define strange_byte_order 0
#define normalize_word(x) (x)
#if (defined(__ultrix))
#undef strange_byte_order
#define strange_byte_order 1
#undef normalize_word
#define normalize_word(x) flip_word(x)
#endif


/* Handling exceptions */

/* A general-purpose exception-handling system for C
   by Jonathan Amsterdam, 1991 */
 
#include <setjmp.h>

#ifndef AP
#ifdef _no_ansi_prototypes
#define AP(args)       ()
#else
#define AP(args)       args
#endif
#endif /* AP */

#define EXCEPTION_DETAILS_SIZE 1000
typedef char *exception;
extern char exception_details[];

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


/* Miscellaneous definitions */

#undef True
#undef False

typedef enum BOOLEAN { True=1, False=0} boolean;

extern exception Out_Of_Memory;
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

extern boolean framer_terminators[];

#define IS_TERMINATOR_P(x) (framer_terminators[((int) x)+1])

#define WITH_TEMP_STRING(string,size) \
   char _tmp_string[size], *string; string=_tmp_string;
#define END_WITH_TEMP_STRING(string) \
   if (string != _tmp_string) free(string);

#ifndef DEFAULT_DISPLAY_WIDTH 
#define DEFAULT_DISPLAY_WIDTH 80
#endif

#endif /* ndef COMMON_H */

/* Macros for dealing with generic streams */

#define STREAMS_H 1

/* String streams */

typedef char *string_input_stream;
#define sgetc(s) ((*(*(s)) != '\0') ? (int) *((*(s))++) : EOF)
#define sungetc(c,s) ((c == EOF) ? EOF : ((*((*s)-1)) == c) ? (int) *(--(*s)) : EOF)
#define seof(s) ((*s) == '\0')

struct STRING_OUTPUT_STREAM { char *head, *point, *tail, *original;};
typedef struct STRING_OUTPUT_STREAM string_output_stream;
int string_putc(int c,string_output_stream *stream);
int string_puts(char *string,string_output_stream *stream);
int string_printf(string_output_stream *s, char *format, ...);
extern char sprintf_buffer[1024];
extern char *empty_string;

#define INITIALIZE_STRING_STREAM(stream,buffer,size) \
  char buffer[size];\
  struct STRING_OUTPUT_STREAM stream; \
  stream.point=stream.head=stream.original=buffer; \
  stream.tail = buffer+size-1; buffer[0]='\0';
#define CLOSE_STRING_STREAM(stream) \
  if (stream.head != stream.original) free(stream.head);

#define sputc(c,s) ((((s)->point) >= ((s)->tail)) ? \
                    (string_putc(c,s))              \
		    : (*((s)->point+1)='\0',*(((s)->point)++)=(c)))
#define sputs string_puts


/* Generic Streams */

typedef enum { string_input, string_output, file_io } stream_type_t;
struct GENERIC_STREAM { stream_type_t stream_type;
			union { string_input_stream *string_in;
				string_output_stream *string_out;
				FILE *file; } ptr;
		      };
typedef struct GENERIC_STREAM generic_stream;
int generic_printf AP((generic_stream *s,char *format, ...));

extern exception Writing_On_Input_String, Reading_from_Output_String,
		 Bad_UnGetc, No_String_So_Far;
extern generic_stream *standard_io, *standard_output;

#ifdef _MAIAJOHNG_
/* #if defined(__GNUC__) || defined(__cplusplus) */
/* changed from macros to inlines to add type checking
 * Wed Nov 24 21:30:59 1993 johng@media.mit.edu */

#ifndef __GNUC__
#define __inline__ inline
#endif

int gsgetint AP((generic_stream *s));

static __inline__ char* gsgets(char *string, int n, generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return (fgets(string,n,s->ptr.file));
   if (s->stream_type == string_input)
     return (string_gets(string,n,s->ptr.string_in));
  return (char*)(raise_crisis(Reading_from_Output_String),((char *) NULL));
}

static __inline__ int gsgetc(generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return getc(s->ptr.file);
   if ((s->stream_type == string_input))
     return sgetc((s->ptr.string_in));
  return (int)raise_crisis(Reading_from_Output_String);
}

static __inline__ int gsungetc(int c, generic_stream *s)
{
    if ((s->stream_type) == file_io) return ungetc(c,s->ptr.file);
    else if ((s->stream_type) == string_input) return sungetc(c,(s->ptr.string_in));
    else return (int)(raise_crisis(Reading_from_Output_String));
}

static __inline__ int gseof(generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return feof(s->ptr.file);
  if (s->stream_type == string_input)
    return seof((s->ptr.string_in));
  return (int)raise_crisis(Reading_from_Output_String);
}
     
static __inline__ int gsputc(int c, generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return (putc(c,s->ptr.file));
  if (s->stream_type == string_output)
    return (sputc(c,(s->ptr.string_out)));
  return (int)(raise_crisis(Writing_On_Input_String));
}

static __inline__ int gsputs(char *string, generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return fputs(string,s->ptr.file);
  if (s->stream_type == string_output)
    return sputs(string,(s->ptr.string_out));
  return (int)(raise_crisis(Writing_On_Input_String));
}

#define gsprintf(s,control,arg) \
       (((s->stream_type) == file_io) ? fprintf(s->ptr.file,control,arg) \
	: ((s->stream_type == string_output) ? \
	   (sprintf(sprintf_buffer,control,arg),sputs(sprintf_buffer,(s->ptr.string_out))) \
	   : raise_crisis(Writing_On_Input_String)))

static __inline__ int gserror(generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return (ferror(s->ptr.file));
  else return 0;  /* what else should I do here? */
}


#else  /* !_MAIAJOHNG_ */ /* !__GNUC__ || !__cplusplus */
int gsgetint AP((generic_stream *s));
int string_putc(int c,string_output_stream *stream);
int string_puts(char *string,string_output_stream *stream);
char *string_gets(char *string,int n,string_input_stream *stream);

char *gsgets(char *string, int n, generic_stream *s);

#if defined(FAST_STREAMS)
#pragma inline(gsgetc), inline(gsungetc), inline(gseof)
#pragma inline(gsputc), inline(gsputs), inline(gserror)

static int gsgetc(generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return getc(s->ptr.file);
   if ((s->stream_type == string_input))
     return sgetc((s->ptr.string_in));
  return (int)raise_crisis(Reading_from_Output_String);
}

static int gsungetc(int c, generic_stream *s)
{
    if ((s->stream_type) == file_io) return ungetc(c,s->ptr.file);
    else if ((s->stream_type) == string_input) return sungetc(c,(s->ptr.string_in));
    else return (int)(raise_crisis(Reading_from_Output_String));
}

static int gseof(generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return feof(s->ptr.file);
  if (s->stream_type == string_input)
    return seof((s->ptr.string_in));
  return (int)raise_crisis(Reading_from_Output_String);
}
     
static int gsputc(int c, generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return (putc(c,s->ptr.file));
  if (s->stream_type == string_output)
    return (sputc(c,(s->ptr.string_out)));
  return (int)(raise_crisis(Writing_On_Input_String));
}

static int gsputs(char *string, generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return fputs(string,s->ptr.file);
  if (s->stream_type == string_output)
    return sputs(string,(s->ptr.string_out));
  return (int)(raise_crisis(Writing_On_Input_String));
}

static int gserror(generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return (ferror(s->ptr.file));
  else return 0;  /* what else should I do here? */
}
#else
char *gsgets(char *string,int n,generic_stream *s);
int gsgetc(generic_stream *s);
int gsungetc(int c,generic_stream *s);
int gseof(generic_stream *s);
int gsputc(int c,generic_stream *s);
int gsputs(char *string,generic_stream *s);
int gserror(generic_stream *s);
#endif

#define gsprintf(s,control,arg) \
       (((s->stream_type) == file_io) ? fprintf(s->ptr.file,control,arg) \
	: ((s->stream_type == string_output) ? \
	   (sprintf(sprintf_buffer,control,arg),sputs(sprintf_buffer,(s->ptr.string_out))) \
	   : raise_crisis(Writing_On_Input_String)))

#endif  /* _MAIAJOHNG_ */ /* __GNUC__ || __cplusplus */
       

#define WITH_INPUT_FROM_STRING(gstream,string) \
    generic_stream _gstream, *gstream; \
    _gstream.stream_type=string_input; _gstream.ptr.string_in=(&string); \
    gstream=(&_gstream);
#define WITH_OUTPUT_TO_EXISTING_STRING(gstream,string,size) \
    generic_stream _gstream, *gstream; struct STRING_OUTPUT_STREAM _sstream; \
    _sstream.head=_sstream.original=_sstream.point=string; _sstream.tail=string+size-1; \
    _gstream.stream_type=string_output; _gstream.ptr.string_out=(&_sstream); \
    gstream=(&_gstream);
#define WITH_OUTPUT_TO_STRING(gstream,size) \
    char _buffer[size]; WITH_OUTPUT_TO_EXISTING_STRING(gstream,_buffer,size)
#define string_so_far(gstream) \
    ((gstream->stream_type== string_output) ? ((gstream->ptr.string_out)->head) : \
     (raise_crisis(No_String_So_Far),(char *) NULL))
#define string_size_so_far(gstream) \
    ((gstream->stream_type == string_output) ? \
     (((gstream->ptr.string_out)->point)-((gstream->ptr.string_out)->head)) : \
     (raise_crisis(No_String_So_Far),0))
#define END_WITH_OUTPUT_TO_STRING(gstream) \
    CLOSE_STRING_STREAM((*(gstream->ptr.string_out)));
#define OLD_WITH_OUTPUT_TO_STRING(gstream,sstream,buffer,size) \
    generic_stream _gstream, *gstream; INITIALIZE_STRING_STREAM(sstream,buffer,size); \
    _gstream.stream_type=string_output; _gstream.ptr.string_out=(&sstream); \
    gstream=(&_gstream)
   

/* The FRAME data structure */

/* 
  The core FRAMER data type is the frame which consists of a home, a
  prototype, a ground and a set of annotations; the annotations are
  each themselves frames whose home is the frame the annotate.

  The ground is a typed structure which refers to other values;
  the prototype points to another frame and the `prototype default rule'
  asserts that (by default) `my prototype is the like-named annotation
  of my home's prototype'.
  
  The frame structure is a lot like a directory structure with the
  addition of grounds and prototype information.
*/

typedef struct FRAME *Frame;
typedef struct FRAME *FrameP;
typedef struct FRAME_ARRAY Frame_Array;
typedef struct FRAME_APPENDIX Frame_Appendix;
typedef union TYPED_VALUE *Grounding;
typedef unsigned short search_word;

struct FRAME 
{
  unsigned char type; search_word bits;
  char *aname;
  Frame home;
  Frame prototype;
  Grounding ground;
  struct FRAME_APPENDIX *appendix;
};

struct ANNOTATION {char *key; Frame frame;};

struct FRAME_APPENDIX
{
  Frame_Array *spinoffs; int zip_code;
  struct ANNOTATION *annotations; short size, limit; 
};

struct FRAME_ARRAY
{
  Frame *elements;
  int size, limit;
};


struct FRAXL_STACK 
{
  Grounding expr; 
  unsigned char arity; unsigned char progress;
  Grounding *args; 
  struct FRAXL_STACK *previous; 
};

typedef struct FRAXL_STACK *FRAXL_STACK_PTR;
typedef unsigned char frame_bits;
extern frame_bits frame_creation_mask;

#define FRAME_P_MASK        (frame_bits) (1<<7)
#define FRAME_CURRENT_MASK  (frame_bits) (1<<6)
#define FRAME_MODIFIED_MASK (frame_bits) (1<<5)
#define FRAME_TYPE_MASK     (frame_bits) 31

enum FRAME_TYPE
{ local_frame=0, file_frame=1, ephemeral_frame=2, read_only_frame=3,
  deleted_frame=4, alias_frame=5, remote_frame=15, 
  network_alpha=16, network_beta=17, network_gamma=18, network_delta=19,
  network_epsilon=20 };
typedef enum FRAME_TYPE Frame_Type;

#define frame_type(frame)        ((frame->type) & FRAME_TYPE_MASK)

#define frame_current_p(frame)   \
   ((((frame)->type) & (FRAME_CURRENT_MASK | FRAME_P_MASK)) \
    == (FRAME_CURRENT_MASK | FRAME_P_MASK))
#define frame_modified_p(frame)  ((((frame)->type) & FRAME_MODIFIED_MASK) != 0)
#define frame_read_only_p(frame) ((((frame)->type) & FRAME_TYPE_MASK) == read_only_frame)
#define frame_ephemeral_p(frame) ((((frame)->type) & FRAME_TYPE_MASK) == ephemeral_frame)
#define frame_deleted_p(frame)   ((((frame)->type) & FRAME_TYPE_MASK) == deleted_frame)

#define set_frame_current_p(frame) \
	((frame)->type) |= FRAME_CURRENT_MASK
#define set_frame_modified_p(frame) \
	((frame)->type) |= FRAME_MODIFIED_MASK
#define set_frame_type(frame,tp) \
        ((frame->type)) = (((frame->type) & ~FRAME_TYPE_MASK) | tp)

#define clear_frame_current_p(frame) \
	((frame)->type) = (((frame)->type) & ~FRAME_CURRENT_MASK)
#define clear_frame_modified_p(frame) \
	((frame)->type) = (((frame)->type) & ~FRAME_MODIFIED_MASK)

#define set_frame_deleted_p(x) set_frame_type(x,deleted_frame)
#define clear_frame_deleted_p(x) ((frame_type(x) == deleted_frame) ? \
                                  (set_frame_type(x,frame_type(x->home)),x) : x)

#define FRAMER_SYSTEM_BIT     0  /* Used for system functions */
#define MNEMOSYNE_SEARCH_BIT  1  /* Used for Mnemosyne searches */
#define RESULT_REDUCTION_BIT 2   /* Used for result reduction */
#define clear_search_bit(frame,i) \
	((frame)->bits) = (((frame)->bits) & (~(1<<i)))
#define set_search_bit(frame,i) \
	((frame)->bits) |= (1<<i)
#define search_bit_p(frame,i) ((((frame)->bits) & (1<<i)) != 0)


/* Representing grounding */

/***********************************************************
  A grounding onsists of a type tag and a contents, which is
  dependent on the NATIVE_GROUND.  All implementations should
  identify the same type tags, but some programs may provide
  varieties of native grounds in the contents.

  FRAMER provides a set of core types: strings, integers, floats,
  vectors, and pairs; some programs may extend this but only if the
  provide functions to translate their own data types into the core
  language for preservation and restoration from disk. 
***********************************************************/

#define MAX_PRIMITIVE_ARITY 20

/* If your implementation has datatypes of its own which will be
   stored in grounds, reserve a type code here */

enum GROUND_TYPE 
{ /* FRAMER core types */
  any_ground=0, /* This should never be assigned */ 
  frame_ground_type=1, /* Neither should this, it's just a place holder */
  string_ground=2, pair_ground=3, 
  integer_ground=4, float_ground=5, bignum_ground=6, rational_ground=7,
  vector_ground=8, framer_function_ground=9, symbol_ground=10, 
  nd_ground=11, procedure_ground=12, comment_ground=13, delayed_parse=14,
  sound_ground=20,image_ground=21,video_ground=22,window_ground=23,
  frame_array_ground=31, integer_array_ground=32, float_array_ground=33,
  bitset_ground=34, short_point_ground=35
  };
typedef enum GROUND_TYPE Ground_Type;

extern Grounding (*interpreters[50])();
extern Grounding (*canonicalizers[50])();
extern void (*reclaimers[50])();

/* The PAIR type is native to FRAMER and has a left and a right.  Note
   that FRAMER pairs do not allow sharing of structure so as to avoid
   needing a garbage collector.  Future implementations may relax or
   remove this requirement (for instance, by addition of a reference
   counting garbage collector to allow sharing but not side-effects. */

struct PAIR 
{
  Grounding left;
  Grounding right;
};

typedef struct PAIR Rational;

struct SYMBOL
{
  char *name;
  Grounding value;
  Grounding (*dispatch)();
};

typedef struct PRIMITIVE_PROCEDURE { 
  char *name;
  int arity; 
  Grounding (*function)();
  Ground_Type types[MAX_PRIMITIVE_ARITY]; Ground_Type return_type;
} primitive_procedure;

struct DELAYED_PARSE { int size; char *string; };

struct VECTOR
{
  int size;
  Grounding *elements ;
};

#define BIGDIGIT_TYPE unsigned char /*This won't work w/ unsigned short int*/
/* must be unsigned and small enough to prevent overflow (at most half the
 size in memory of an unsigned long */

/*
        Explanation of BignumType  (Diagram courtesy of ed.)
bignum points here
|
|                and bignum->digits points here (to the "ones" digit)
|                |
v                v
[ : : ][ : : : ][ : : : ][ : : : ][ : : : ][ : : : ][ : : : ] ...... ;
[ This block of memory  ][ And this block of memory, which comes   ] ;
[ is the space for a    ][ right afterwards, is the (n-1) extra    ] ;
[ BignumType.           ][ digits which we allocate.               ] ;
  
can access the first digit (the "ones" digit) with bignum->digits[0],
  and any other digit with bignum->digits[n].                        ;
*/

typedef struct BIGNUM_STRUCT { 
  signed char sign;   /* {0,1, or -1} */
  unsigned long int numdigits;
  BIGDIGIT_TYPE digits[1];} *BignumType;

typedef struct INTEGER_MATRIX
{
  int dimensionality; int *dimensions;
  int *elements;
} integer_matrix;

typedef struct FLOAT_MATRIX
{
  int dimensionality; int *dimensions;
  int *elements;
} float_matrix;

struct BITSET
{
  Frame underneath; int which_bit;
};

/* NATIVE_GROUND is a union type indicating the various values which
   may be stored in a ground; the core types of FRAMER are frames (of
   course), strings, integers, flonums, pairs, and vectors; in
   addition, the `mystery' element of a union is used to refer to the
   canonical representation of a datatype from another implementation;
   its value is another grounding  For instance, the canonical form
   of LISP symbols consists of either strings (for symbols in the FRAMER
   package) or vectors of a symbol-name and package name; in a simple
   C environment a grounding f a lisp symbol would have a type of the
   LISP symbol type but store the corresponding string or vector in
   its `mystery' slot. */

typedef 
  union NATIVE_GROUND
      {
      char *string;
      long integer;
      BignumType bignum;
      double flonum;
      Rational rational;
      struct PAIR pair;
      struct VECTOR vector;
      struct BITSET bitset;
      struct DELAYED_PARSE delayed_parse;
      Grounding mystery;
      primitive_procedure *primitive;
      Grounding comment;
      struct SYMBOL *symbol;
      Frame_Array *frame_array;
      integer_matrix *imatrix;
      float_matrix *fmatrix;
      int short_point[2];
      /* This is for the `window type' */
      Window_Type window;
      Image_Type image;
      Sound_Type sound;
      /* This is for implementation specific grounds. */
      void *internal;
      /* This is for the free chain */
      struct GROUNDING *next_free;
      } 
  Native_Ground;

/* A grounding consists of a ground type, a reference count, and a contents */

struct GROUNDING {
  unsigned char type; 
  short reference_count;
  Native_Ground contents;
};

union TYPED_VALUE {struct GROUNDING ground; struct FRAME frame; unsigned char type; };

#define ground_type(x)        (x->type)
#define ground_contents(x)    (x->ground.contents)
#define the(aspect,x)         ((x)->ground.contents.aspect)
#define unground(x,aspect)    ((x)->ground.contents.aspect)
#define set_ground_contents(x,type,value)    \
        (x->ground.contents.type=value)
#define ref_count(x)          ((x)->ground).reference_count
#define set_ground_type(x,tp) x->type = (char) tp


/* Variables */

/* These are the declarations for functions and variables used by FRAMER */

/* The root of the frame tree and the default frame (defining declaration in frames.c) */
extern Frame root_frame;
extern Frame here;
extern char  *root_filename;
extern struct FRAXL_STACK *fraxl_stack;
extern boolean suppress_autoload;
extern boolean strict_syntax;
extern boolean announce_file_ops;
extern Frame read_root;
extern char *_interned;

extern exception File_Unwritable, File_Unreadable,
  Read_Error, Write_Error, Unexpected_EOF, Rootless_Reference, Nonexistent_File,
  Read_Only_Frame, Out_Of_Memory, Out_Of_Space, Dislocated_Frame,
  Eval_Failure, Type_Error, Arity_Error, Not_A_Function, Recursive_Prototype;

#if FOR_MSDOS
#define INITIAL_ZIP_CODES 200
#else
#define INITIAL_ZIP_CODES 100000
#endif

#ifndef ALPHA_RELEASE
#define ALPHA_RELEASE  0
#endif

#if FOR_MSDOS
#define FRAME_BLOCK_SIZE 50
#define GROUND_BLOCK_SIZE 50
#else
#define FRAME_BLOCK_SIZE 250
#define GROUND_BLOCK_SIZE 150
#endif

extern char *frame_names, *end_of_frame_names;
extern Frame frame_block, end_of_frame_block;
extern struct FRAME_APPENDIX *frame_appendix_block, *end_of_frame_appendix_block;
extern struct GROUNDING *ground_block, *end_of_ground_block;
extern struct PAIR *pair_block, *end_of_pair_block;

Frame frame_underflow AP((void));
struct FRAME_APPENDIX *frame_appendix_underflow AP((void));
char *allocate_permanent_string AP((char *name));
char *intern_frame_name AP((char *name));


/* Prototypes */

/* Constructors */

Frame make_annotation AP((Frame frame,char *name));
/* This returns an annotation of <frame> named <name> and --- if it
   must be newly consed --- arranges for it to obey the default
   prototype rule.  Calls to `use' may fail if <frame> is read_only or
   there is not enough memory to create the frame.  The call fails by
   printing an error message and returning NULL */
Frame make_ephemeral_annotation AP((Frame frame,char *name));
Frame make_unique_annotation AP((Frame frame,char *name));
Frame make_alias AP((Frame frame,char *name,Frame to_frame));
Frame use_annotation AP((Frame frame,char *name));

Frame probe_annotation AP((Frame frame,char *name));
/* This returns NULL if there is no annotation named <name> on <frame>
   or any of the prototypes of <frame>.  Otherwise, it returns an
   annotation named <name> on <frame> and creates a chain
   of prototype annotations so that the annotation it returns obeys
   the default prototype rule. */

Frame local_probe_annotation AP((Frame frame,char *name));
/* This returns NULL if there is no annotation named <name> on <frame>. 
   This does not inheritance. */

Frame inherits_frame AP((Frame frame,char *name));
/* This returns NULL if there is no annotation named <name> on <frame>
   or any of the prototypes of <frame>.  Otherwise, it returns an
   annotation named <name> on <frame> and creates a chain
   of prototype annotations so that the annotation it returns obeys
   the default prototype rule. */

Frame inherits_frame AP((Frame frame,char *name));
/* This returns the first annotation named <name> of <frame> or one of
   its prototypes given a search that proceeds from frame to
   prototype.  This conses no new structure and is usually used for
   telling whether CHECK or USE would create new structure */
Frame has AP((Frame frame,char *name)); /* Compatability */
   
Frame copy_frame AP((Frame source,Frame dest));
/* This copies all the contents of <source> into <dest>, including translating
   relative references within <source> into corresponding references within <dest>. */

Frame move_frame AP((Frame frame,Frame new_home,char *new_name));

boolean delete_annotation AP((Frame frame));
/* This removes the frame <frame> as an annotation, purging it from
   the tree of frames.  This is only done if no other frames refer
   to <frame> or any of its annoations as a prototype.  In purging it,
   all of its annotations are also purged.  Note that this may still
   leave dangling pointers in the grounds of frames, but that's life. */


/* Accessors and Mutators */

Frame frame_prototype_fn AP((Frame frame));
/* Gets the prototype (possibly NULL) of <frame>, loading <frame> if it
   has not been loaded yet. */

Frame set_prototype AP((Frame frame,Frame prototype));
/* This sets the prototype of <frame> to be <prototype> and records
   <frame> on the `spinoffs' of <prototype> if the relation does not
   obey the default prototype rule.  Note that this operation will
   fail (returning NULL) if <frame> is read only; otherwise, <frame>
   is returned. */
Frame set_prototypes AP((Frame frame,Frame prototype));
/* This recursively adjusts frame and its annotations to point to prototype
   and its annotations.  */

Grounding frame_ground_fn AP((Frame frame));
/* Gets the ground (possibly NULL) of <frame>, loading <frame> if it
   has not been loaded yet. */

Frame set_ground AP((Frame frame,Grounding ground));
/* This sets the ground of <frame> to <ground> calling gc_ground on
   the current ground and replacing it.  This fails (by returning
   NULL) if <frame> is read only and returns <frame> otherwise. */

Frame_Array *frame_spinoffs AP((Frame frame));
/* This returns a frame_Array of the spinoffs (either stored or
   implicit via the default prototype rule) for <frame>. */

Grounding get_inherited_ground AP((Frame frame));
/* Returns the first non-NULL ground stored on <frame> or its prototypes. */

boolean prototype_is_default AP((Frame frame,Frame prototype));
/* This returns true or false (1 or 0) if frame and prototype obey the
   default prototype rule; in particular, if the home of <prototype> is
   the prototype of the home of <frame>. */

Frame has_prototype(Frame f1,Frame f2);
/* Returns true if f1 has f2 as a prototype. */
boolean has_home(Frame f,Frame h);
/* Returns true if f is beneath h in the annotation hierarchy. */


/* Parsing and printing frames */

Frame parse_frame AP((generic_stream *stream));
/* This parses a frame pathname from <stream> given a root of <relative_to>.
   Additionally, it handles the `.' syntax to introduce paths relative to
   the read_root if it is defined. */
Frame parse_frame_path(generic_stream *stream,Frame root);
/* Parses a frame path relative to a root */
Frame fparse_frame AP((FILE *stream));
/* This parses a frame pathname from <stream> given a root of <relative_to>.
   Additionally, it handles the `.' syntax to introduce paths relative to
   the read_root if it is defined. */
Frame parse_frame_from_string AP((char *string));
/* This turns a pathname in <string> into a frame parsed relative to
   <relative_to>. */

void print_frame AP((generic_stream *stream,Frame frame));
/* This outputs a pathname for <frame> to <stream> */
void print_frame_name(char *string,generic_stream *stream);
/* This outputs a slashified string to <stream> */
void print_frame_under(generic_stream *stream,Frame frame,Frame under);
/* Prints a frame relative to another frame */
void fprint_frame AP((FILE *file_stream,Frame frame));
/* This outputs a pathname for <frame> to <stream>, given a root (for
   relative printing) of <relative_to>. This printing also generates
   the `.' syntax for shifts in the `read root'. */
char *print_frame_to_string AP((Frame frame));
/* This writes a pathname for <frame> into <string>, given a root (for
   relative printing) of <relative_to>. This printing also generates
   the `&' syntax for shifts in the `read root'. */
void sprint_frame AP((Frame frame, char *string));
/* This writes a pathname for <frame> into <string>, given a root (for
   relative printing) of <relative_to>. This printing also generates
   the `&' syntax for shifts in the `read root'. */

Frame frame_ref(char *path);
/* Returns a frame from parsing name.  This frame is created with
   autload suppressed, so it will not interfere with later or current
   loading. */


/* Functions for saving and restoring frames. */

void backup_frame AP((Frame frame));
/* This saves <frame> to disk by dumping whatever file frame it is
   embedded in. */
void backup_root_frame AP((char *filename));
/* This backs up the root frame to <filename>, backing up any modified
   sub-frames in the process. */
void write_framer_image AP((char *filename));
/* This writes the current framer image <filename>, including
   all current files in an image stored in .img and .idx files. */
void open_framer_image_file AP((char *filename));
/* This writes the current framer image <filename>, including
   all current files in an image stored in .img and .idx files. */
Frame load_frame_from_file AP((char *filename));
/* This loads and returns whatever frame is defined in <filename>. */

Frame get_current AP((Frame frame));
/* Restores the frame <frame> */

boolean backup_root_p(Frame f);
/* Returns true for frames which have their own filenames. */
char *backup_file_name(char *filename);
/* Translates a `logical pathname' into a physical one for
   backing up frames. */
char *frame_filename(Frame f);
/* If frame is a backup root, returns the filename to
   which frame is backed up (as a root). */
void set_frame_filename(Frame f,char *filename);
/* Sets the file to which a frame should be backed up */


/* Common ground declarations */

Grounding new_ground AP((void));
struct PAIR *new_pair AP((void));

Grounding touch AP((Frame frame));
Grounding string_to_ground AP((char *string));
Grounding integer_to_ground AP((long integer));
Grounding float_to_ground AP((float flonum));
void free_up_ground AP((struct GROUNDING *ground));

boolean equal_p AP((Grounding x,Grounding y));
boolean eqv_p AP((Grounding x,Grounding y));
boolean eq_p AP((Grounding x,Grounding y));
Grounding fraxl_equal_p AP((Grounding x,Grounding y));
Grounding fraxl_eqv_p AP((Grounding x,Grounding y));
Grounding fraxl_eq_p AP((Grounding x,Grounding y));

Grounding cons_pair AP((Grounding x,Grounding y));
Grounding make_procedure AP((Grounding args,Grounding body,Grounding env));
Grounding close_procedure AP((Grounding args_and_body,Grounding env));
Grounding make_comment AP((Grounding comment));
Grounding intern AP((char *string));
Grounding find_function AP((char *string));
Grounding merge_results AP((Grounding x,Grounding y));
Grounding find_result AP((Grounding x,Grounding y));
Grounding zap_result AP((Grounding x,Grounding y));
Grounding gather_results AP((Grounding combiner,Grounding ground,Grounding init));

int list_length(Grounding lst);

Grounding canonicalize_native_ground AP((Grounding ground));
Grounding interpret_native_ground AP((int code, Grounding ground));

extern Grounding t_symbol, quote_symbol, unquote_symbol, empty_list;
extern boolean strict_syntax;

Grounding parse_ground AP((generic_stream *stream));
/* This returns a grounding ased on parsing from <stream>. */
void print_ground AP((generic_stream *stream,Grounding ground));
/* This prints a grounding nto <stream> suitable for reconstitution
   by parse_ground. */
void pprinter(Grounding expr,generic_stream *stream,
	      int indent,int width,Grounding highlight);
/* This pretty prints a ground with a particular indentation, width, and
   highlighting a certain expression. */
Grounding fparse_ground AP((FILE *stream));
/* This returns a grounding ased on parsing from STREAM. */
void fprint_ground AP((FILE *stream,Grounding ground));
/* This prints a grounding nto <STREAM> suitable for reconstitution
   by parse_ground. */

Grounding parse_ground_from_string AP((char *string));
char *print_ground_to_string AP((Grounding grounding));

void pprint_ground AP((generic_stream *stream,Grounding gnd,int left,int right));
Grounding pretty_printer AP((Grounding gnd,Grounding width));

void ground_error(exception ex,char *herald,Grounding ground);
/* This signals an error with a ground */

/* These signal a frame error */
void frame_error(exception ex,char *herald,Frame frame);
void frame_error2(char *h1,Frame f1,char *h2,Frame f2);

/* This reclaims the storage taken by <ground>. */
void gc_ground AP((Grounding ground));

/* These are for ephemeral grounds which dump as {} but print with
   an integer id corresponding to their pointer. */
Grounding canonicalize_ephemeral_ground(Grounding ground);
Grounding interpret_ephemeral_ground(Ground_Type code,Grounding ground);


/* Macros for FRAMES */

/* These are used by various functions to ensure that frames are
   loaded and validly modifiable.  The definition of TOUCH assumes
   that it is used in a function which returns a pointer (since it
   returns NULL when its argument cannot be safely modified.) */
#define ensure_current(x) ((frame_current_p(x)) ? x : (get_current(x)))
#define ENSURE_CURRENT(x) x=ensure_current(x)
#define TOUCH(x) \
  if (frame_read_only_p(x) == True) \
    frame_error(Read_Only_Frame,"Can't modify ",x); \
  else set_frame_modified_p(x)
#define NOEOF(x,stream) if (x == EOF) if (gseof(stream)) raise_crisis(Unexpected_EOF);
#define remote_frame_p(x) (((int) frame_type(x)) >= 16)
#define default_prototype_p(frame,p) \
   (NOT(((p) == NULL) || ((frame)->home == NULL) || \
	((frame)->home->prototype != (p)->home)        || \
	(NOT(((frame)->aname == (p)->aname)            || \
	     (string_compare((frame)->aname,(p)->aname) == 0)))))

#define SKIP_WHITESPACE(input,stream)  \
   while (!(isgraph(input))) {input=getc(stream); NOEOF(input,stream);}

#define DO_FRAMES(x,frame_array) \
    Frame_Array *_tmp; Frame x; int _i=0; \
    _tmp =(frame_array); \
    if ((_tmp != NULL) && (_tmp->size > 0) && (_tmp->elements != NULL)) \
       while ((_i<_tmp->size) && ((x=_tmp->elements[_i++]) != NULL)) \
         if (NOT(frame_deleted_p(x)))
          
#define DO_SOFT_ANNOTATIONS(x,frm) \
  Frame x; struct FRAME_APPENDIX *app; int _i=0; app=((frm)->appendix); \
  if (app) while ((_i < app->size) ? (x=((app->annotations[_i++].frame)),1) : 0) \
     if (NOT(frame_deleted_p(x))) 
#define DO_ANNOTATIONS(x,frame) DO_SOFT_ANNOTATIONS(x,(ensure_current(frame)))
#define DO_FEATURES(f,frame) DO_ANNOTATIONS(f,frame) if (*(f->aname) != '+')

#define frame_prototype(x) \
    ((frame_current_p(x)) ? (((Frame) x)->prototype) : (frame_prototype_fn((Frame) x)))
#define frame_ground(x) \
    ((frame_current_p(x)) ? (((Frame) x)->ground) : (frame_ground_fn((Frame) x)))
#define frame_name(x) ((x)->aname)
#define frame_home(x) ((x)->home)

#define DO_PROTOTYPES(proto,frame)        \
      Frame proto; proto=frame;           \
      if (proto != NULL)                  \
        for (proto=ensure_current(proto); \
             (proto != NULL);             \
	     proto=((proto != NULL) ? frame_prototype(proto) : NULL))
#define DO_HOMES(hm,frame) \
      Frame hm; \
      for (hm=(frame);(hm != NULL);hm=hm->home)

#define TMP_BIND(x,value,tmp) tmp=x;x=value;
#define TMP_RESTORE(x,value,tmp) x=tmp;

#define WITH_SEARCH_BIT(bit) int bit; UNWIND_PROTECT bit=grab_search_bit();
#define END_WITH_SEARCH_BIT(bit) ON_UNWIND release_search_bit(bit); END_UNWIND


/* Macros for grounds */

extern exception Type_Error;
#define type_error(desc,gnd) ground_error(Type_Error,desc,gnd)
#define type_check(gnd,type,desc) \
  if (NOT(TYPEP(gnd,type))) type_error(desc,gnd)

#define TYPEP(x,typ) ((x != NULL) && \
		      ((typ == frame_ground_type) ? \
		       ((ground_type(x)) & FRAME_P_MASK) : \
		       (((Ground_Type) ground_type(x)) == typ)))
#define FRAMEP(x) (TYPEP((x),frame_ground_type))
#define STRINGP(x) (TYPEP((x),string_ground))
#define FIXNUMP(x) (TYPEP((x),integer_ground))
#define BIGNUMP(x) (TYPEP((x),bignum_ground))
#define RATIONALP(x) (TYPEP((x),rational_ground))
#define FLOATP(x) (TYPEP((x),float_ground))
#define SYMBOLP(x) (TYPEP((x),symbol_ground))
#define FUNCTIONP(x) (TYPEP((x),framer_function_ground))
#define CONSP(x) (TYPEP((x),pair_ground))
#define VECTORP(x) (TYPEP((x),vector_ground))
#define RESULT_SETP(x) (TYPEP((x),nd_ground))
#define ND_GROUND_P(x) (TYPEP((x),nd_ground))

#define INTEGERP(x) \
  ((((Ground_Type) ground_type(x)) == integer_ground) || \
   (((Ground_Type) ground_type(x)) == bignum_ground))
#define NUMBERP(x) \
  ((((Ground_Type) ground_type(x)) == integer_ground) || \
   (((Ground_Type) ground_type(x)) == bignum_ground) || \
   (((Ground_Type) ground_type(x)) == rational_ground) || \
   (((Ground_Type) ground_type(x)) == float_ground))

#define GFRAME(X) \
  ((FRAMEP(X)) ? ((Frame) X) : \
   (type_error("not a frame",X),(Frame) NULL))
#define GSTRING(X) \
  ((TYPEP(X,string_ground)) ? (unground(X,string)) : \
   (type_error("not a string",X),(char *)NULL))
#define GINTEGER(X) \
  ((TYPEP(X,integer_ground)) ? (unground(X,integer)) : \
   (type_error("not an integer",X),(int) NULL))
#define GFLOAT(X) \
  ((TYPEP(X,float_ground)) ? (unground(X,flonum)) : \
   (type_error("not a float",X),0.0))
#define GBIGNUM(X) unground(X,bignum)

#define frame_to_ground(X) ((Grounding) X)
#define f2g(X) frame_to_ground(X)  

#define SYMBOL_NAME(X) (the(symbol,X)->name)
#define SYMBOL_VALUE(X) (the(symbol,X)->value)
#define FUNCTION_NAME(x) (the(primitive,X)->name)

#define FREE_GROUND(ground) \
  if ((ground != NULL) && (NOT(FRAMEP(ground))) && \
      ((--ref_count(ground))  <= 0)) \
   gc_ground(ground)
#define USE_GROUND(ground) \
  if ((ground != NULL) && (NOT(FRAMEP(ground)))) \
    ref_count(ground)++
#define FREE_RESULT(ground) \
  if ((ground != NULL) && (NOT(FRAMEP(ground)))) ref_count(ground)--;
#define USED_GROUND(x) \
   ((x == NULL) ? x : \
    ((FRAMEP(x)) ? x : (x->ground.reference_count++,x)))
#define INITIALIZE_GROUND(ground,tp) \
  set_ground_type(ground,tp); \
  ref_count(ground)=0
#define INITIALIZE_NEW_GROUND(ground,tp) \
  ground=new_ground(); INITIALIZE_GROUND(ground,tp)

#define GCAR(X) ((X)->ground.contents.pair.left)
#define GCDR(X) ((X)->ground.contents.pair.right)
#define GCONS(X,Y,INTO) \
  {Grounding _cons, _x, _y; INITIALIZE_NEW_GROUND(_cons,pair_ground);\
   (GCAR(_cons))=_x=X; (GCDR(_cons))=_y=Y; USE_GROUND(_x); USE_GROUND(_y); \
   INTO=_cons;}
#define EMPTY_LIST_P(x) (x == empty_list)

#define GVELEMENTS(X) ((X)->ground.contents.vector.elements)
#define GVSIZE(X) ((X)->ground.contents.vector.size)
#define GVMAKE(INTO,I) \
  {Grounding *array; int _i=I;\
   INITIALIZE_NEW_GROUND(INTO,vector_ground); \
   array = (Grounding  *) fra_allocate(_i,sizeof(Grounding )); \
   (GVSIZE(INTO))=_i; (GVELEMENTS(INTO))=array; \
   while (_i > 0) {_i--;*array++=NULL;};}
#define GVREF(X,I) ((X)->ground.contents.vector.elements[I])
#define GVSET(X,I,G) {Grounding _new; int _i; _i=I; _new=G; USE_GROUND(_new); \
                      FREE_GROUND(GVREF(X,_i)); GVREF(X,_i)=_new;}

#define EQUAL_GROUND_P(x,y) \
  (((x) == (y)) || \
   (((x) != NULL) && ((y) != NULL) && (((ground_type(x))) == ((ground_type(y)))) \
    && (NOT(FRAMEP(x))) && (NOT(FRAMEP(y))) && equal_p(x,y))) 
#define EQUAL_P(x,y) \
   (((x) == (y)) || \
    (((x) != NULL) && ((y) != NULL) && (((ground_type(x))) == ((ground_type(y)))) \
     && (NOT(FRAMEP(x))) && (NOT(FRAMEP(y))) && equal_p(x,y))) 
#define EQL_P(x,y) \
  (((x) == (y)) || \
   (((x) != NULL) && ((y) != NULL) && \
    && (NOT(FRAMEP(x))) && (NOT(FRAMEP(y))) && eql_p(x,y))) 

#define DO_LIST(x,lst) \
    Grounding _tmp, x; \
    _tmp=(lst); if (_tmp != empty_list) x=GCAR(lst); \
    while ((_tmp != empty_list) ? (x=GCAR(_tmp),_tmp=GCDR(_tmp)) : 0)

#define ND_ELTS(X) ((X)->ground.contents.vector.elements)
#define ND_SIZE(X) ((X)->ground.contents.vector.size)
#define NDMAKE(INTO,I) \
  {Grounding *array; int _i=I;\
   INITIALIZE_NEW_GROUND(INTO,nd_ground); \
   array = (Grounding  *) fra_allocate(_i,sizeof(Grounding )); \
   (ND_SIZE(INTO))=_i; (ND_ELTS(INTO))=array; \
   while (_i > 0) {_i--;*array++=NULL;};}
#define STOP_DOING_RESULTS() _ptr=_last+1;
#define DO_RESULTS(x,set) \
    Grounding x, _set, *_ptr, *_last; _set=(set); \
    if (_set) \
     if (ground_type(_set) == nd_ground) \
       {_ptr=ND_ELTS(_set); _last=_ptr+ND_SIZE(_set)-1;} \
     else {_ptr=&_set; _last=&_set;} \
    if (_set) \
      while ((_ptr <= _last) ? (x=*_ptr++,1) : 0)

#define DO_VECTOR(x,vec) \
    Grounding x, _tmp, *_ptr, *_end; \
    _tmp=(vec); _ptr=GVELEMENTS(_tmp); _end=_ptr+GVSIZE(_tmp); \
    while (((_ptr < _end) ? ((x=(*(_ptr++))),1) : 0))

#define proc_env GCAR
#define proc_args(x) (GCAR(GCDR(x)))
#define proc_body(x) (GCDR(GCDR(x)))
#define proc_text(x) (GCDR(x))

#define bitset_bit(g) ((g)->ground.contents.bitset.which_bit)
#define bitset_root(g) ((g)->ground.contents.bitset.underneath)

#if TRACE_GC
/* For debugging */
#define FREE_GROUND(ground) \
  if ((ground != NULL) && (NOT(FRAMEP(ground)))) \
    {printf("\n@Line %d: %d-",__LINE__,ref_count(ground)); \
     print_ground(standard_output,ground); \
     if (--ref_count(ground) <= 0) gc_ground(ground); \
     printf(" [%d]",ground_memory_in_use); }
#define USE_GROUND(ground) \
  if ((ground != NULL) && (NOT(FRAMEP(ground)))) \
    {printf("\n@Line %d: %d+",__LINE__,ref_count(ground)); \
     print_ground(standard_output,ground); \
     ref_count(ground)++; printf(" [%d]",ground_memory_in_use); }
#define USED_GROUND(x) \
   ((x == NULL) ? x : \
    ((FRAMEP(x)) ? x : \
     ((((printf("\n@Line %d: %d+",__LINE__,ref_count(x))),(print_ground(standard_output,x))), \
       printf(" [%d]",ground_memory_in_use),x->ground.reference_count++),x)))
#endif


/* Macros for dealing with tables */

extern struct TABLE_STREAM
{ int tabs[20]; int position; char *background; generic_stream *stream;}
*current_table;

void string_at_tab AP((char *string,int tab,struct TABLE_STREAM *tablestream));
void ground_at_tab AP((Grounding ground,int tab,struct TABLE_STREAM *tablestream));
void labelled_ground_at_tab AP((char *label,Grounding ground,
				int tab,struct TABLE_STREAM *tablestream));
void frame_at_tab AP((Frame frame,Frame top,int tab,struct TABLE_STREAM *tablestream));

#define MAX_TABLE_WIDTH 30
#define MAX_TABLE_HEIGHT 100

extern Grounding table_being_made;
void add_cell AP((int column,char *string));
void print_table AP((Grounding table,generic_stream *stream));

#define WITH_TABLE() \
  Grounding _table; FLET(Grounding,table_being_made,empty_list)
   
#define END_WITH_TABLE() \
  _table=table_being_made; END_FLET(table_being_made); \
  if (table_being_made)                                \
     {table_being_made=cons_pair(reverse_list(_table),table_being_made); \
      FREE_GROUND(_table);} \
  else {Grounding rtable; rtable=reverse_list(_table); \
        print_table(rtable,standard_output); \
        FREE_GROUND(_table); FREE_GROUND(rtable); }


/* Macro for initializing FRAMER */

#ifndef CUSTOMIZE_FRAMER
#define CUSTOMIZE_FRAMER()   
#endif

#define INITIALIZE_JUST_FRAMER() \
  get_backup_paths_from_env(); \
  init_framer_memory();        \
  init_framer_grounds();       \
  init_network();              \
  CUSTOMIZE_FRAMER();          \
  if (root_filename) load_frame_from_file(root_filename); \
  else load_frame_from_file("radix")

#define INITIALIZE_FRAMER() INITIALIZE_JUST_FRAMER()

void get_backup_paths_from_env();
void init_framer_memory(void);
void init_framer_grounds(void);
void init_network(void);

#endif /* ndef FRAMER_H */
