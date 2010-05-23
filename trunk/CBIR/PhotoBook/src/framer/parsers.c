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

  Modification history:
    13 June 1992 - Began tracking modification history (Haase)
    17 September 1992 - Added rationals
*************************************************************************/
 
/* This file implements parsers for various ground types. */

#include <limits.h>
#include <errno.h>
#include <ctype.h>
#define FAST_STREAMS 1
#include "framer.h"

static char rcsid[] = 
  "$Header: /mas/framer/sources/RCS/parsers.c,v 1.37 1994/01/26 18:41:41 haase Exp $";

#define toint(c) ((int) (c-'0'))

extern Grounding backquote_symbol, frame_ground_symbol, lisp_nil_symbol, period_symbol;


/* Parsing Vectors and ND-Sets */
 
Grounding parse_vector_ground(generic_stream *stream)
{
  int input; Grounding *elts, *ptr, *limit;
  input=gsgetc(stream);
  while ((isspace(input))) input=gsgetc(stream);
  {NOEOF(input,stream);}
  if (input == ')') return NULL; else gsungetc(input,stream);
  ptr=elts=(Grounding *) fra_allocate(sizeof(Grounding),100); limit=ptr+100;
  while (True)
    {if (ptr == limit)
       {Grounding *nelts, *nptr, *nlimit; 
	nelts=fra_reallocate(elts,10*(limit-elts)*sizeof(Grounding));
	nlimit=nelts+(limit-elts)*10; nptr=nelts+(ptr-elts);
	elts=nelts; ptr=nptr; limit=nlimit;}
     *ptr++=parse_ground(stream); 
     input=gsgetc(stream); while ((isspace(input))) input=gsgetc(stream);
     {NOEOF(input,stream);} 
     if (input == ')') 
       {Grounding temp, result, *rptr; GVMAKE(result,ptr-elts); 
	rptr=GVELEMENTS(result)+(ptr-elts); ptr--; rptr--;
	while (ptr >= elts)
	  {temp=*ptr--; USE_GROUND(temp); *rptr--=temp;}
	return result;}
     else gsungetc(input,stream);}
}

Grounding parse_nd_ground(generic_stream *stream)
{
  int input; Grounding *elts, *ptr, *limit;
  input=gsgetc(stream);
  while ((isspace(input))) input=gsgetc(stream);
  {NOEOF(input,stream);}
  if (input == '}') return NULL; else gsungetc(input,stream);
  ptr=elts=(Grounding *) fra_allocate(sizeof(Grounding),100); limit=ptr+100;
  while (True)
    {if (ptr == limit)
       {Grounding *nelts, *nptr, *nlimit; 
	nelts=fra_reallocate(elts,10*(limit-elts)*sizeof(Grounding));
	nlimit=nelts+(limit-elts)*10; nptr=nelts+(ptr-elts);
	elts=nelts; ptr=nptr; limit=nlimit;}
     *ptr++=parse_ground(stream); 
     input=gsgetc(stream); while ((isspace(input))) input=gsgetc(stream);
     {NOEOF(input,stream);} 
     if (input == '}') 
       {Grounding temp, result, *rptr; NDMAKE(result,ptr-elts); 
	rptr=ND_ELTS(result); ptr--;
	while (ptr >= elts)
	  {temp=*ptr--; USE_GROUND(temp); *rptr++=temp;}
	return result;}
     else gsungetc(input,stream);}
}

#if 0 /* From recursive version */
typedef struct VECTOR_PARSE
{int size; Grounding elt; struct VECTOR_PARSE *previous;}
vector_parse;
#endif 


/* Parsing lists */

boolean debugging_parser=False;
Grounding parse_pair_ground(generic_stream *stream)
{
  int input; Grounding left, right, result;
  input=gsgetc(stream); while (isspace(input)) input=gsgetc(stream);
  if (input == ')') return empty_list;
  else gsungetc(input,stream);
  left=parse_ground(stream);
  if (left == period_symbol)
    {left=parse_ground(stream);
     input=gsgetc(stream); while (isspace(input)) input=gsgetc(stream);
     NOEOF(input,stream);
     if (input != ')') raise_crisis("Missing close paren");
     return left;}
  else {right=parse_pair_ground(stream); GCONS(left,right,result);
	return result;}
}


/* Parsing procedures */

Grounding interpret_procedure_ground(int type_code,Grounding g)
{
  return close_procedure(GCDR(g),GCAR(g));
}


/* Parsing procedure grounds (old stuff) */

/* This is for an old printed format for procedures which handles recursive
   structures especially.  This version, however, does not.  */
Grounding parse_procedure_ground(generic_stream *stream)
{
  Grounding whole, temp, args, body, env, result;
  temp=whole=parse_ground(stream); 
  if ((NOT(((TYPEP(temp,pair_ground)) && ((list_length(temp)) == 3)))))
    raise_crisis(Read_Error);
  args=GCAR(temp); temp=GCDR(temp); body=GCAR(temp); temp=GCDR(temp); env=GCAR(temp); 
  result=make_procedure(args,body,env);
  FREE_GROUND(whole);
  return result;
}


/* Parsing strings */

Grounding parse_string_ground(generic_stream *stream)
{
  Grounding result; int input;
  INITIALIZE_STRING_STREAM(ss,buffer,1);
  input=gsgetc(stream); NOEOF(input,stream);
  for (;;)
    {if (input == '\\') input = (char) gsgetc(stream);
     else {if (input == '"') break;};
     NOEOF(input,stream);
     sputc(input,(&ss)); input=gsgetc(stream);}
  INITIALIZE_NEW_GROUND(result,string_ground);
  if (ss.head == ss.original)
    {char *temp; unground(result,string)=temp=fra_allocate(1,sizeof(char));
     *temp='\0';}
  else unground(result,string)=ss.head;
  return result;
}


/* Reading atoms */

Grounding make_rational(Grounding num,Grounding denom);
Grounding make_bignum(char *digits);

Grounding parse_atom(char *string)
{
  char first_char; Grounding make_bignum(char *digits);
  first_char=string[0];
  if (string[1] == '\0')
    if (isdigit(first_char)) 
      return integer_to_ground(toint(first_char));
    else return intern(string);
  else if ((isdigit(first_char)) ||  (first_char == '+') ||
	   (first_char == '-') || (first_char == '.'))
    {char *decimal_point, *divider, *scanner; scanner=string+1;
     decimal_point=strchr(string,'.'); divider=strchr(string,'/');
     if ((NOT(NULLP(decimal_point))) && (NOT(NULLP(divider))))
       return intern(string);
     else while ((*scanner != '\0') &&
		 ((isdigit(*scanner)) ||
		  (scanner == decimal_point) || (scanner == divider)))
       scanner++;
     if (NOT((*scanner) == '\0')) return intern(string);
     else if (decimal_point)
       {double dbl; dbl=strtod(string,NULL);
	return float_to_ground((float)dbl);}
     else if (divider)
       {Grounding result, num, denom; *(divider)='\0'; 
	num=parse_atom(string); denom=parse_atom(divider+1);
	USE_GROUND(num); USE_GROUND(denom);
	result=make_rational(num,denom); 
	FREE_GROUND(num); FREE_GROUND(denom);
	return result;}
     else {long fix; fix=strtol(string,NULL,10);
	   if ((errno == ERANGE) || (fix > INT_MAX) || (fix < INT_MIN))
	     {errno=0; return make_bignum(string);}
	   else return integer_to_ground(fix);}}
  else return intern(string);
}

Grounding read_atom(generic_stream *stream)
{
  char point; Grounding result;
  Grounding make_bignum(char *digits);
  INITIALIZE_STRING_STREAM(sstream,buffer,512);
  point=gsgetc(stream);
  while ((point == '/') || (NOT(IS_TERMINATOR_P(point))))
    if (point == '\\')
      {point=gsgetc(stream); sputc(point,&sstream); point=gsgetc(stream);}
    else {sputc(point,&sstream); point=gsgetc(stream);}
  gsungetc(point,stream);
  if (sstream.head == sstream.point)
    raise_crisis(Unexpected_EOF);
  else result=parse_atom(sstream.head);
  CLOSE_STRING_STREAM(sstream);
  /* This is a real kludge to handle the fact that LISP cannot be forced
     to print out NIL as the empty list. */
  if (result == lisp_nil_symbol) result=empty_list;
  return result;
}

Grounding parse_ground(generic_stream *stream)
{
  int prefix, int_type_code; Grounding inner_grounding;
  Grounding read_atom(generic_stream *stream);
  Frame parse_zip_coded_frame(generic_stream *stream);
  prefix=gsgetc(stream);
  while ((prefix == EOF) && (NOT(gseof(stream)))) prefix=gsgetc(stream);
  NOEOF(prefix,stream);
  while (isspace(prefix)) prefix=gsgetc(stream);
  switch ((char) prefix)
    {
    case ';':
      while ((prefix=gsgetc(stream)) != '\n');
      return parse_ground(stream);
    case '"':
      return parse_string_ground(stream);
    case ':':
      return read_atom(stream);
    case '(':
      return parse_pair_ground(stream);
    case '{':
      return parse_nd_ground(stream);
    case '#':
      {prefix=gsgetc(stream);
       if (isdigit(prefix))
	 {gsungetc(prefix,stream);
	  return frame_to_ground(parse_frame(stream));}
       switch ((char) prefix) 
	 { case '/': case '^': 
	     gsungetc(prefix,stream);
	     return frame_to_ground(parse_frame(stream));
	   case '(':
	     return parse_vector_ground(stream);
	   case '!': case '$':
	     if (stream->stream_type == file_io)
	       fscanf(stream->ptr.file,"%d",&int_type_code);
	     else {char buffer[200], digit, *ptr=buffer;
		   digit = gsgetc(stream);
		   while (isdigit(digit)) {*ptr++=digit;digit=gsgetc(stream);}
		   *ptr='\0'; sscanf(buffer,"%d",&int_type_code);};
	     if (gseof(stream)) {raise_crisis(Unexpected_EOF); return NULL;};
	     inner_grounding=parse_ground(stream);
	     return interpret_native_ground(int_type_code,inner_grounding);
	   case ';':
	     return make_comment(parse_ground(stream));
	   case '@':
	     {Frame f; f=parse_frame(stream);
	      return cons_pair(frame_ground_symbol,
			       cons_pair(frame_to_ground(f),empty_list));}
	   case '\'':
	     return parse_procedure_ground(stream);
	   default: 
	     {raise_crisis(Read_Error); return NULL;}}}
    case '`':     
      return cons_pair(backquote_symbol,cons_pair(parse_ground(stream),empty_list));
    case '\'':
      return cons_pair(quote_symbol,cons_pair(parse_ground(stream),empty_list));
    case ',':
      return cons_pair(unquote_symbol,cons_pair(parse_ground(stream),empty_list));
    case ')':
      raise_crisis("Too many close parens"); return NULL;
    default: 
      if ((prefix != '/') && (IS_TERMINATOR_P(prefix)))
	raise_crisis("Unexpected Terminator");
      gsungetc(prefix,stream);
      return read_atom(stream);
    }
}
 

/* Specialized parsing functions for non-generic streams. */

Grounding fparse_ground(FILE *file_stream)
{
  generic_stream stream;
  stream.stream_type = file_io; stream.ptr.file = file_stream;
  return parse_ground(&stream);
}

Grounding sparse_ground(char **sstream)
{
  generic_stream stream;
  stream.stream_type = string_input; stream.ptr.string_in = sstream;
  return parse_ground(&stream);
}

Grounding parse_ground_from_string(char *string)
{
  generic_stream stream;
  stream.stream_type = string_input; stream.ptr.string_in = &string;
  return parse_ground(&stream);
}

/*
  Local variables: ***
  compile-command: "../etc/compile-framer" ***
  tags-file-name: "../sources/TAGS" ***
  End: **
*/
