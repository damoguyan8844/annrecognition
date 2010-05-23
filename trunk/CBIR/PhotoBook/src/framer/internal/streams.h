/* Macros for dealing with generic streams */

#ifndef STREAMS_H
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
char *string_gets(char *string,int n,string_input_stream *s);
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

#define gsprintf(s,control,arg) \
       (((s->stream_type) == file_io) ? fprintf(s->ptr.file,control,arg) \
	: ((s->stream_type == string_output) ? \
	   (sprintf(sprintf_buffer,control,arg),sputs(sprintf_buffer,(s->ptr.string_out))) \
	   : raise_crisis(Writing_On_Input_String)))


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
   
#endif /* STREAMS_H */


