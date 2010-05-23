/* -*- Mode: C -*- 
 
  Copyright (c) 1991, 1992 by the Massachusetts Institute of Technology.
  All rights reserved.

  This file is part of FRAMER, a representation language and semantic
  database developed by Kenneth B. Haase and his students at the Media
  Laboratory at the Massachusetts Institute of Technology in Cambridge,
  Massachusetts.  Research at the Media Lab is supported by funds and
  equipment from a variety of corporations and government sponsors whose
  contributions are gratefully acknowledged.

  Permission to use, copy, or modify these programs and their
  documentation for educational and research purposes only and without
  fee is hereby granted, provided that this copyright and permission
  notice appear on all copies and supporting documentation.  For any
  other uses of this software, in original or modified form, including
  but not limited to distribution in whole or in part, specific prior
  permission from M.I.T. must be obtained.  M.I.T. makes no
  representations about the suitability of this software for any
  purpose.  It is provided "as is" without express or implied warranty.

*************************************************************************/

/************************************************************************
  This file implements `generic streams' an abstraction for uniform output
  to files, strings, terminals, etc.

  Modification history:
    13 June 1992 - Began tracking modification history (Haase)
*************************************************************************/

/* Functions for using strings as streams */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "internal/common.h"
#include "internal/streams.h"

/* Standard I/O streams */
generic_stream standard_output_stream;
generic_stream *standard_output;
generic_stream standard_output_stream;
generic_stream *standard_output, *standard_io;
char *empty_string="";

char sprintf_buffer[1024];

exception
  Writing_On_Input_String = "Writing on an input string",
  Reading_from_Output_String = "Reading from output string",
  No_String_So_Far = "Can only get intermediate string from string output stream";

int gsgetint(generic_stream *stream)
{
  if (stream->stream_type == string_input)
    return strtoul(*(stream->ptr.string_in),stream->ptr.string_in,10);
  if (stream->stream_type == file_io)
    {int num; fscanf(stream->ptr.file,"%d",&num); return num;}
  else {raise_crisis(Reading_from_Output_String); return 0;}
}

int string_putc(int c,string_output_stream *stream)
{
  void *careful_allocate(size_t n_elts,size_t elt_size);
  if (stream->point < stream->tail) 
    {*(stream->point+1)='\0'; *(stream->point++)=c;}
  else {char *new_buffer; int new_size;
	new_size=(2*(stream->tail-stream->head))+5;
	new_buffer= (char *) careful_allocate(new_size,sizeof(char));
	strcpy(new_buffer,stream->head);
	if (new_buffer == NULL) return EOF;
	stream->point=new_buffer+(stream->point-stream->head);
	if (stream->head != stream->original) free(stream->head);
	stream->head=new_buffer; stream->tail=new_buffer+new_size-1;
	*(stream->point+1)='\0'; *(stream->point++)=c;}
  return (int) c;
}

int string_puts(char *string,string_output_stream *stream)
{
  while (*string != '\0') sputc(*(string++),stream);
  return (int) 0;
}

char *string_gets(char *string,int n,string_input_stream *stream)
{
  char *ptr, *limit; ptr=string; limit=ptr+n-1;
  while (ptr < limit)
    {char c; *ptr++=c=((*(*stream)));
     if (c == '\n') {*ptr='\0'; (*stream)++; return string;}
     else if (c == '\0') return string;
     else (*stream)++;}
  *ptr='\0'; return string;
}

char *gsgets(char *string, int n, generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return (fgets(string,n,s->ptr.file));
   if (s->stream_type == string_input)
     return (string_gets(string,n,s->ptr.string_in));
  return (char*)(raise_crisis(Reading_from_Output_String),((char *) NULL));
}

int gsgetc(generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return getc(s->ptr.file);
   if ((s->stream_type == string_input))
     return sgetc((s->ptr.string_in));
  return (int)raise_crisis(Reading_from_Output_String);
}

int gsungetc(int c, generic_stream *s)
{
    if ((s->stream_type) == file_io) return ungetc(c,s->ptr.file);
    else if ((s->stream_type) == string_input) return sungetc(c,(s->ptr.string_in));
    else return (int)(raise_crisis(Reading_from_Output_String));
}

int gseof(generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return feof(s->ptr.file);
  if (s->stream_type == string_input)
    return seof((s->ptr.string_in));
  return (int)raise_crisis(Reading_from_Output_String);
}
     
int gsputc(int c, generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return (putc(c,s->ptr.file));
  if (s->stream_type == string_output)
    return (sputc(c,(s->ptr.string_out)));
  return (int)(raise_crisis(Writing_On_Input_String));
}

int gsputs(char *string, generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return fputs(string,s->ptr.file);
  if (s->stream_type == string_output)
    return sputs(string,(s->ptr.string_out));
  return (int)(raise_crisis(Writing_On_Input_String));
}

int gserror(generic_stream *s)
{
  if ((s->stream_type) == file_io)
    return (ferror(s->ptr.file));
  else return 0;  /* what else should I do here? */
}

void setup_standard_io()
{
  standard_output_stream.stream_type=file_io; 
  standard_output_stream.ptr.file=stdout; 
  standard_output=(&standard_output_stream);
  standard_io=standard_output;
}
