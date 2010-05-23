/* -*- C -*- 
 
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
  This file defines the core functions for parsing, printing, and reclaiming
  FRAMER grounds.
*************************************************************************/
 
static char rcsid[] = 
  "$Header: /mas/framer/sources/RCS/printers.c,v 1.35 1994/01/26 18:46:07 haase Exp $";

#include <limits.h>
#include <errno.h>
#include <ctype.h>
#define FAST_STREAMS 1
#include "framer.h"

extern Grounding backquote_symbol, frame_ground_symbol;
int default_display_width=DEFAULT_DISPLAY_WIDTH;
boolean abbreviate_procedures=False;

/* A Panalopy of Printers */

void print_nd_ground(generic_stream *stream,Grounding ground)
{
  boolean first=True; gsputc('{',stream);
  {DO_RESULTS(r,ground) 
     {if (first) first=False; else {gsputc(' ',stream);}
      print_ground(stream,r);}}
  gsputc('}',stream);
  if (gserror(stream)) raise_crisis(Write_Error);
}

void print_vector_ground(generic_stream *stream,Grounding vector)
{
  int i, size;
  i=0; size=GVSIZE(vector);
  gsputc('#',stream);
  gsputc('(',stream);
  for (i=0;i<size;i++)
    {if (i>0) {gsputc(' ',stream);}
     print_ground(stream,GVREF(vector,i));}
  gsputc(')',stream);
}
 
#define CDRABLE(x) ((TYPEP(x,pair_ground)) && (GCDR(x) != NULL))

void print_pair_ground(generic_stream *stream,Grounding ground)
{
  Grounding point;
  if ((CDRABLE(ground)) && (CDRABLE(GCDR(ground))) && 
      ((GCDR(GCDR(ground))) == empty_list))
    {Grounding fcn, arg; fcn=GCAR(ground); arg=GCAR(GCDR(ground));
     if (fcn == quote_symbol)
       {gsputc('\'',stream); print_ground(stream,arg); return;}
     else if (fcn == backquote_symbol)
       {gsputc('`',stream); print_ground(stream,arg); return;}
     else if (fcn == unquote_symbol)
       {gsputc(',',stream); print_ground(stream,arg); return;}
     else if ((fcn == frame_ground_symbol) && (NOT(strict_syntax)) && (FRAMEP(arg)))
       {gsputs("#@",stream); print_frame(stream,GFRAME(arg)); return;}}
  gsputc('(',stream);
  print_ground(stream,GCAR(ground));
  point = GCDR(ground);
  while (NOT(EMPTY_LIST_P(point)) && TYPEP(point,pair_ground))
    {gsputc(' ',stream); 
     print_ground(stream,GCAR(point));
     point = GCDR(point);};
  if (point == empty_list) {gsputc(')',stream);}
  else {gsputc(' ',stream);gsputc('.',stream);gsputc(' ',stream);
	print_ground(stream,point);
	gsputc(')',stream);}
  if (gserror(stream)) raise_crisis(Write_Error);
}

static void print_symbol_name(generic_stream *stream,char *string)
{
  /* if (strict_syntax) gsputc(':',stream); */
  while (*string != '\0')
    {if (IS_TERMINATOR_P(*string)) gsputc('\\',stream);
     gsputc(*string++,stream);};
}
 
static void print_string_ground(generic_stream *stream,char *string)
{
  gsputc('"',stream);
  while (*string != '\0')
    {if ((*string == '"') || (*string == '\\')) gsputc('\\',stream);
     gsputc(*string++,stream);};
  gsputc('"',stream);
  if (gserror(stream)) raise_crisis(Write_Error);
}
 
static void print_integer_ground(generic_stream *stream,int integer)
{
  gsprintf(stream,"%d",integer);
}
 
static void print_bignum_ground(generic_stream *stream,Grounding bignum)
{
  char *bignum_to_string(Grounding big);
  char *string; string=bignum_to_string(bignum);
  gsputs(string,stream); free(string);
}
 
static void print_rational_ground(generic_stream *stream,Grounding rational)
{
  print_ground(stream,GCAR(rational)); 
  gsputc('/',stream);
  print_ground(stream,GCDR(rational)); 
}
 
static void print_flonum_ground(generic_stream *stream,float flonum)
{
  gsprintf(stream,"%f",flonum);
}


/* Printing procedures */

Grounding canonicalize_procedure_ground(Grounding thing)
{
  return cons_pair(proc_env(thing),proc_text(thing));
}


/* The top level printing function */

void print_ground(generic_stream *stream,Grounding ground)
{
  if (ground == NULL) gsputs("{}",stream);
  else if (FRAMEP(ground)) 
    if (frame_deleted_p(GFRAME(ground)))
      gsputs("{}",stream);
    else if ((strict_syntax) && (frame_ephemeral_p(GFRAME(ground))))
      gsputs("{}",stream);
    else if (stream->stream_type == file_io)
      {putc('#',stream->ptr.file); fprint_frame(stream->ptr.file,GFRAME(ground));}
    else {gsputc('#',stream); print_frame(stream,GFRAME(ground));}
  else switch (ground_type(ground))
    {
    case string_ground:
      print_string_ground(stream,GSTRING(ground)); break;
    case integer_ground:
      print_integer_ground(stream,GINTEGER(ground)); break;
    case float_ground:
      print_flonum_ground(stream,GFLOAT(ground)); break;
    case vector_ground:
      print_vector_ground(stream,ground); break;
    case pair_ground:
      if (ground == empty_list) gsputs("()",stream);
      else print_pair_ground(stream,ground); 
      break;
    case symbol_ground:
      print_symbol_name(stream,the(symbol,ground)->name); break;
    case bignum_ground:
      print_bignum_ground(stream,ground); break;
    case rational_ground:
      print_rational_ground(stream,ground); break;
    case nd_ground:
      print_nd_ground(stream,ground); break;
    case any_ground:
      gsputs("???",stream); break;
    case comment_ground:
      gsputs("#;",stream); print_ground(stream,the(comment,ground)); break;
    case procedure_ground:
      {Grounding canonical_form; 
       canonical_form=canonicalize_procedure_ground(ground);
       gsputs("#$12 ",stream); print_ground(stream,canonical_form);
       FREE_GROUND(canonical_form); break;}
    default:
      {Grounding canonical_form;
       gsprintf(stream,"#$%d ",(int) ground_type(ground));
       canonical_form=canonicalize_native_ground(ground);
       print_ground(stream,canonical_form);
       FREE_GROUND(canonical_form);
       break;}
    }
  if (stream->stream_type == file_io) fflush(stream->ptr.file);
}


/* Specialized printers for non-generic streams */

void fprint_ground(FILE *file_stream,Grounding ground)
{
  generic_stream stream;
  stream.stream_type = file_io; stream.ptr.file = file_stream;
  print_ground(&stream,ground);
}

void sprint_ground(string_output_stream *sstream,Grounding ground)
{
  generic_stream stream;
  stream.stream_type = string_output; stream.ptr.string_out = sstream;
  print_ground(&stream,ground);
}

char *print_ground_to_string(Grounding ground)
{
  generic_stream stream;
  INITIALIZE_STRING_STREAM(ss,buf,1);
  stream.stream_type = string_output; stream.ptr.string_out = &ss;
  print_ground(&stream,ground);
  return ss.head;
}


/* Pretty printing */

#define PPRINT_INDENT 2

Grounding highlighted_expression=NULL;
char *highlight_start, *highlight_end;

int flatsize(Grounding ground)
{
  int size;
  {WITH_OUTPUT_TO_STRING(gs,128)
     {print_ground(gs,ground); size=strlen(string_so_far(gs));}
   END_WITH_OUTPUT_TO_STRING(gs);}
  return size;
}
     
void print_hground(generic_stream *stream,Grounding ground)
{
  if ((ground) && (ground == highlighted_expression))
    {gsputs(highlight_start,stream); 
     print_ground(stream,ground);
     gsputs(highlight_end,stream);}
  else print_ground(stream,ground);
}

#define indent_line(stream,indent) \
    gsputc('\n',stream); {DO_TIMES(j,indent) gsputc(' ',stream);}


int pprint_elt(generic_stream *stream,Grounding elt,int left,int right,int xpos)
{
  int fsize; fsize=flatsize(elt);
  if (elt == highlighted_expression)
    fsize=fsize+strlen(highlight_start)+strlen(highlight_end);
  if (xpos == left)
    if ((xpos+fsize) > right)
      {pprint_ground(stream,elt,left,right);
       return right;}
    else {pprint_ground(stream,elt,left,right); return xpos+fsize;}
  else gsputc(' ',stream);
  if ((xpos+fsize+1) > right)
    {indent_line(stream,left); 
     return pprint_elt(stream,elt,left,right,left);}
  else
    {pprint_ground(stream,elt,left,right); return xpos+fsize+1;}
}

void pprint_vector_ground(generic_stream *stream,Grounding vector,int left,int right)
{
  int xpos;
  gsputc('#',stream); gsputc('(',stream); xpos=left=left+2;
  {DO_VECTOR(elt,vector)
     xpos=pprint_elt(stream,elt,left,right,xpos);}
  gsputc(')',stream);
}
 
void pprint_nd_ground(generic_stream *stream,Grounding set,int left,int right)
{
  int xpos;
  gsputc('{',stream); xpos=left=left+1;
  {DO_RESULTS(elt,set)
     xpos=pprint_elt(stream,elt,left,right,xpos);}
  gsputc('}',stream);
}
 
extern Grounding define_symbol, lambda_symbol;

void pprint_pair_ground(generic_stream *stream,Grounding ground,int left,int right)
{
  Grounding point; int xpos;
  if (((CDRABLE(ground)) && (CDRABLE(GCDR(ground))) && 
       ((GCDR(GCDR(ground))) == empty_list) &&
       (((GCAR(ground)) == quote_symbol) || 
	((GCAR(ground)) == backquote_symbol) ||
	((GCAR(ground)) == unquote_symbol))))
    {if ((GCAR(ground)) == quote_symbol) 
       gsputc('\'',stream); 
    else if ((GCAR(ground)) == backquote_symbol)
      gsputc('`',stream); 
    else gsputc(',',stream);
       pprint_ground(stream,(GCAR(GCDR(ground))),left+1,right);
       return;}
  gsputc('(',stream); point=ground; xpos=left=left+PPRINT_INDENT;
  while ((CONSP(point)) && (point != empty_list))
    {xpos=pprint_elt(stream,GCAR(point),left,right,xpos);
     point=GCDR(point);}
  if (NOT(EMPTY_LIST_P(point)))
    {if ((2+flatsize(point)+1) > right) 
       {indent_line(stream,left);} else gsputc(' ',stream);
     gsputc('.',stream); gsputc(' ',stream); 
     pprint_elt(stream,point,left+1,right,left+1);}
  gsputc(')',stream);
}

void pprint_ground(generic_stream *stream,Grounding ground,int left,int right)
{
  if ((ground) && (ground == highlighted_expression)) gsputs(highlight_start,stream);
  if (ground == NULL) gsputs("{}",stream);
  else if (FRAMEP(ground)) print_hground(stream,ground);
  else switch (ground_type(ground))
    {
    case pair_ground: 
      pprint_pair_ground(stream,ground,left,right); break;
    case nd_ground:
      pprint_nd_ground(stream,ground,left,right); break;
    case vector_ground:
      pprint_vector_ground(stream,ground,left,right); break;
    case procedure_ground:
      if ((NOT(strict_syntax)) && (abbreviate_procedures))
	{gsputs("#`",stream);	/* Make it unreadable */
	 pprint_ground(stream,proc_args(ground),left+2,right);
	 break;}
    default:
      if (NULLP(canonicalizers[((int) (ground_type(ground)))]))
	print_ground(stream,ground); 
      else {Grounding canonical_form;
	    gsprintf(stream,"#$%d ",(int) ground_type(ground));
	    canonical_form=canonicalize_native_ground(ground);
	    pprint_ground(stream,canonical_form,left+5,right);
	    FREE_GROUND(canonical_form);}
      break;
    }
  if ((ground) && (ground == highlighted_expression)) gsputs(highlight_end,stream);
  if (stream->stream_type == file_io) fflush(stream->ptr.file);
}

void pprinter(Grounding expr,generic_stream *stream,
	      int indent,int width,Grounding highlight)
{
  {FLET(Grounding,highlighted_expression,highlight)
     pprint_ground(stream,expr,indent,width);
   END_FLET(highlighted_expression);}
}

Grounding pretty_printer(Grounding input,Grounding width)
{
  pprinter(input,standard_output,0,GINTEGER(width),NULL);
  return input;
}

char *pprint_ground_to_string(Grounding ground,int width)
{
  generic_stream stream;
  INITIALIZE_STRING_STREAM(ss,buf,1);
  stream.stream_type = string_output; stream.ptr.string_out = &ss;
  pprint_ground(&stream,ground,0,width);
  return ss.head;
}

Grounding set_default_display_width(Grounding size)
{
  int old; old=default_display_width;
  default_display_width=GINTEGER(size);
  return integer_to_ground(old);
}

/*
  Local variables: ***
  compile-command: "../etc/compile-framer" ***
  tags-file-name: "../sources/TAGS" ***
  End: **
*/
