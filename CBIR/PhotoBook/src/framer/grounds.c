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
  This file defines the core functions for creating and reclaiming
  FRAMER grounds.
*************************************************************************/
 
static char rcsid[] = "$Header: /mas/framer/sources/RCS/grounds.c,v 1.46 1994/01/26 18:38:58 haase Exp $";

#include <limits.h>
#include <errno.h>
#include <ctype.h>
#include "framer.h"
#include "internal/private.h"


/* Declarations of interest */
 
Grounding interpret_frame_array AP((Ground_Type code,Grounding ground));
Grounding canonicalize_frame_array AP((Grounding ground));

Grounding interpret_function_ground AP((Ground_Type code,Grounding ground));
Grounding canonicalize_function_ground AP((Grounding ground));

Grounding interpret_procedure_ground AP((Ground_Type code,Grounding ground));
Grounding canonicalize_procedure_ground AP((Grounding ground));

/* These are used for translating special data types into local
   (implementation dependent) data types. */
Grounding (*interpreters[50])() =
   {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
Grounding (*canonicalizers[50])() =
   {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
void (*reclaimers[50])() =
   {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };

boolean framer_terminators[256];

Grounding t_symbol, quote_symbol, unquote_symbol, backquote_symbol,
  frame_ground_symbol, lisp_nil_symbol, period_symbol;
/* Used by archiving */
Grounding file_tag, read_only_tag, alias_tag; 


/* Composite grounds */

#if (!(CODE_RESOURCE))
struct GROUNDING an_empty_list;
Grounding empty_list= (Grounding) (&an_empty_list);
#else
struct GROUNDING an_empty_list; Grounding empty_list;
#endif

Grounding cons_pair(Grounding left,Grounding right)
/* Returns a cons of two elements. */
{
  Grounding result;
  GCONS(left,right,result);
  return result;
}
 
Grounding make_comment(Grounding from)
/* Makes a comment object from another object. */
{
  Grounding result;
  INITIALIZE_NEW_GROUND(result,comment_ground);
  set_ground_contents(result,comment,from);
  USE_GROUND(from);
  return result;
}

/* Frame arrays */

Grounding canonicalize_frame_array(Grounding ground)
/* Translates a frame array into a typed vector.
   (Not really neccessary anymore if we rewrote the
    FRAME imlementation to use grounded vectors...)
   [Though frame arrays can grow as vectors cannot.] */
{
  Grounding new_vec; int ix=0;
  GVMAKE(new_vec,the(frame_array,ground)->size);
  {DO_FRAMES(frame,(the(frame_array,ground)))
     {GVSET(new_vec,ix,frame_to_ground(frame));ix++;}}
  return new_vec;
}

Grounding interpret_frame_array(Ground_Type type_code,Grounding ground)
/* Takes a vector (representing a frame array) and returns the corresponding
   frame array. */
{
  Frame_Array *iarray; int size;
  Grounding *gframe, result; Frame *aframe;
  size=GVSIZE(ground);
  iarray=make_frame_array(size);
  aframe=iarray->elements; gframe=GVELEMENTS(ground);
  {DO_TIMES(i,size) *(aframe++) = (Frame ) *(gframe++);}
  iarray->size=size;
  INITIALIZE_NEW_GROUND(result,type_code);
  set_ground_contents(result,frame_array,iarray);
  return result;
}

void free_frame_array(Grounding ground)
/* Frees a frame array ground by freeing the elements of the array.
   It can't free the array itself because they're not maintained
   for reuse. */
{
  free(the(frame_array,ground)->elements);
}

/* Procedures */

Grounding make_procedure(Grounding args,Grounding body,Grounding env)
/* Makes a procedure object.  Procedures have funny garbage collection
   conventions but they're not handled here. */

{
  Grounding result, temp; INITIALIZE_NEW_GROUND(result,procedure_ground);
  GCAR(result)=env; USE_GROUND(env); 
  GCDR(result)=temp=cons_pair(args,body); USE_GROUND(temp);
  return result;
}

Grounding close_procedure(Grounding args_and_body,Grounding env)
/* Makes a procedure object.  Procedures have funny garbage collection
   conventions but they're not handled here. */

{
  Grounding result; INITIALIZE_NEW_GROUND(result,procedure_ground);
  GCAR(result)=env; USE_GROUND(env); 
  GCDR(result)=args_and_body; USE_GROUND(args_and_body);
  return result;
}

/* Symbols */

boolean strict_syntax=False;
exception IllegalSymbolName="Illegal symbol name";

Grounding *symbols=NULL;
int symbol_count=0;
int table_size=0;

Grounding find_symbol(char *string)
{
  int base,size; Grounding result; 
  size=symbol_count-1; base=0;
  BINARY_STRING_SEARCH(result,string,symbols,->ground.contents.symbol->name,base,size);
  return result;
}

Grounding intern(char *string)
{
  Grounding symbol; struct SYMBOL *sym; int base, size;
  if (*string == '\0') raise_crisis(IllegalSymbolName);
  size=symbol_count-1; base=0;
  BINARY_STRING_SEARCH(symbol,string,symbols,->ground.contents.symbol->name,base,size);
  if (symbol != NULL) return symbol;
  WITH_HANDLING
    {Grounding *space, *ptr; 
     INITIALIZE_NEW_GROUND(symbol,symbol_ground);
     ALLOCATE(sym,struct SYMBOL,1);
     set_ground_contents(symbol,symbol,sym);
     sym->name  = fra_allocate(strlen(string)+1,sizeof(char)); strcpy(sym->name,string);
     sym->value = NULL; sym->dispatch=NULL;
     if ((symbol_count+1) >= table_size)
       {Grounding *new_symbols; int new_table_size;
	new_table_size = (table_size+1+(table_size/5));
	new_symbols = 
	  (Grounding  *) 
	    fra_reallocate(symbols,sizeof(Grounding *)*new_table_size);
	if (new_symbols == NULL) raise_crisis(Out_Of_Memory);
	else {table_size=new_table_size; symbols=new_symbols;};};
     space=symbols+base; ptr=symbols+symbol_count;
     while (ptr > space) {(*ptr) = (*(ptr-1));ptr--;};
     symbols[base]=symbol;
     symbol_count++;}
  ON_EXCEPTION
    reraise(); return NULL;
  END_HANDLING
    return symbol;
}

/* FRAXL primitives */

Grounding *functions=NULL;
int function_count=0;
int function_table_size=0;

Grounding find_function(char *string)
{
  int base, size; Grounding result;
  size=function_count-1; base=0;
  BINARY_STRING_SEARCH(result,string,functions,->ground.contents.primitive->name,base,size);
  return result;
}

Grounding declare_function
  (Grounding (*func)(), char *string, int arity,
   Ground_Type type0, Ground_Type type1, 
   Ground_Type type2, Ground_Type type3)
{
  Grounding gfunc, gsym;
  int base, size;
  WITH_HANDLING
    {size=function_count-1; base=0;
     BINARY_STRING_SEARCH(gfunc,string,functions,->ground.contents.primitive->name,base,size);
     if (gfunc == NULL)
       {Grounding *space, *ptr; char *copied_string;
	struct PRIMITIVE_PROCEDURE *proc;
	INITIALIZE_NEW_GROUND(gfunc,framer_function_ground);
	ALLOCATE(proc,struct PRIMITIVE_PROCEDURE,1);
	gsym  = intern(string);
	copied_string = gsym->ground.contents.symbol->name;
	gfunc->ground.contents.primitive = proc;
	gfunc->ground.contents.primitive->name=copied_string;
	gsym->ground.contents.symbol->value=gfunc;
	if ((function_count+1) >= function_table_size)
	  {Grounding *new_functions; int new_table_size;
	   new_table_size = (function_table_size+1+(function_table_size/5));
	   new_functions = (Grounding  *) fra_reallocate(functions,sizeof(Grounding  *)*new_table_size);
	   if (new_functions == NULL) raise_crisis(Out_Of_Memory);
	   else {function_table_size=new_table_size; functions=new_functions;};};
	space=functions+base; ptr=functions+function_count;
	while (ptr > space) {(*ptr) = (*(ptr-1));ptr--;};
	functions[base]=gfunc;
	function_count++;};
     gfunc->ground.contents.primitive->arity=arity;
     gfunc->ground.contents.primitive->function=func;
     gfunc->ground.contents.primitive->types[0]=type0;
     gfunc->ground.contents.primitive->types[1]=type1;
     gfunc->ground.contents.primitive->types[2]=type2;
     gfunc->ground.contents.primitive->types[3]=type3;}
  ON_EXCEPTION
    reraise(); return NULL;
  END_HANDLING
    return gfunc;
}

Grounding declare_big_function
  (Grounding (*func)(), char *string, int arity,
   Ground_Type type0, Ground_Type type1, 
   Ground_Type type2, Ground_Type type3,
   Ground_Type type4, Ground_Type type5,
   Ground_Type type6, Ground_Type type7,
   Ground_Type type8, Ground_Type type9)
{
  Grounding gfunc, gsym;
  int base, size;
  WITH_HANDLING
    {size=function_count-1; base=0;
     BINARY_STRING_SEARCH(gfunc,string,functions,->ground.contents.primitive->name,
			  base,size);
     if (gfunc == NULL)
       {Grounding *space, *ptr; char *copied_string;
	struct PRIMITIVE_PROCEDURE *proc;
	INITIALIZE_NEW_GROUND(gfunc,framer_function_ground);
	ALLOCATE(proc,struct PRIMITIVE_PROCEDURE,1);
	gsym  = intern(string);
	copied_string = gsym->ground.contents.symbol->name;
	gfunc->ground.contents.primitive = proc;
	gfunc->ground.contents.primitive->name=copied_string;
	gsym->ground.contents.symbol->value=gfunc;
	if ((function_count+1) >= function_table_size)
	  {Grounding *new_functions; int new_table_size;
	   new_table_size = (function_table_size+1+(function_table_size/5));
	   new_functions = (Grounding  *) 
	     fra_reallocate(functions,sizeof(Grounding  *)*new_table_size);
	   if (new_functions == NULL) raise_crisis(Out_Of_Memory);
	   else {function_table_size=new_table_size; functions=new_functions;};};
	space=functions+base; ptr=functions+function_count;
	while (ptr > space) {(*ptr) = (*(ptr-1));ptr--;};
	functions[base]=gfunc;
	function_count++;};
     gfunc->ground.contents.primitive->arity=arity;
     gfunc->ground.contents.primitive->function=func;
     gfunc->ground.contents.primitive->types[0]=type0;
     gfunc->ground.contents.primitive->types[1]=type1;
     gfunc->ground.contents.primitive->types[2]=type2;
     gfunc->ground.contents.primitive->types[3]=type3;
     gfunc->ground.contents.primitive->types[4]=type4;
     gfunc->ground.contents.primitive->types[5]=type5;
     gfunc->ground.contents.primitive->types[6]=type6;
     gfunc->ground.contents.primitive->types[7]=type7;
     gfunc->ground.contents.primitive->types[8]=type8;
     gfunc->ground.contents.primitive->types[9]=type9;}
  ON_EXCEPTION
    reraise(); return NULL;
  END_HANDLING
    return gfunc;
}

Grounding declare_lexpr(Grounding (*func)(),char *string)
{
  Grounding gfunc, gsym;
  int base, size;
  WITH_HANDLING
    {size=function_count-1; base=0;
     BINARY_STRING_SEARCH(gfunc,string,functions,->ground.contents.primitive->name,
			  base,size);
     if (gfunc == NULL)
       {Grounding *space, *ptr; char *copied_string;
	struct PRIMITIVE_PROCEDURE *proc;
	INITIALIZE_NEW_GROUND(gfunc,framer_function_ground);
	ALLOCATE(proc,struct PRIMITIVE_PROCEDURE,1);
	gsym  = intern(string);
	copied_string = gsym->ground.contents.symbol->name;
	gfunc->ground.contents.primitive = proc;
	gfunc->ground.contents.primitive->name=copied_string;
	gsym->ground.contents.symbol->value=gfunc;
	if ((function_count+1) >= function_table_size)
	  {Grounding *new_functions; int new_table_size;
	   new_table_size = (function_table_size+1+(function_table_size/5));
	   new_functions = (Grounding  *) 
	     fra_reallocate(functions,sizeof(Grounding  *)*new_table_size);
	   if (new_functions == NULL) raise_crisis(Out_Of_Memory);
	   else {function_table_size=new_table_size; functions=new_functions;};};
	space=functions+base; ptr=functions+function_count;
	while (ptr > space) {(*ptr) = (*(ptr-1));ptr--;};
	functions[base]=gfunc;
	function_count++;};
     gfunc->ground.contents.primitive->arity=(-1);
     gfunc->ground.contents.primitive->function=func;}
  ON_EXCEPTION
    reraise(); return NULL;
  END_HANDLING
    return gfunc;
}

Grounding interpret_function_ground(code,ground)
     Ground_Type code;
     Grounding ground;
{
  extern exception Read_Error; Grounding result;
  if (code != framer_function_ground) raise_crisis(Read_Error);
  result=find_function(GSTRING(ground));
  if (result) return result;
  else return declare_function(NULL,GSTRING(ground),0,0,0,0,0);
}

Grounding canonicalize_function_ground(ground)
     Grounding ground;
{
  return string_to_ground(the(primitive,ground)->name);
}

/* Bitsets */

Grounding make_bitset(Grounding under)
{
  Grounding new; int grab_search_bit(void); 
  INITIALIZE_NEW_GROUND(new,bitset_ground);
  new->ground.contents.bitset.underneath=GFRAME(under);
  new->ground.contents.bitset.which_bit=grab_search_bit();
  return new;
}

Grounding clear_all_bits(Frame under,int bit)
{
  clear_search_bit(under,bit);
  {DO_SOFT_ANNOTATIONS(a,under) clear_all_bits(a,bit);}
  return frame_to_ground(under);
}

void free_bitset(Grounding bs)
{
  int bit;
  bit=bs->ground.contents.bitset.which_bit;
  clear_all_bits(bs->ground.contents.bitset.underneath,bit);
  release_search_bit(bit);
}


/* Canonicalizing ephemeral grounds */

Grounding canonicalize_ephemeral_ground(Grounding ground)
{
  if (strict_syntax) return NULL;
  else return integer_to_ground((int) ground);
}

Grounding interpret_ephemeral_ground(Ground_Type code,Grounding ground)
{
  if (ground == NULL) return NULL;
  else if (TYPEP(ground,integer_ground))
    return (Grounding) the(integer,ground);
  else raise_crisis("Can't interpret ephemeral ground");
  return NULL;
}


/* Canonicalizing and interpreting short point grounds */
Grounding canonicalize_point_ground(Grounding ground)
{
  return cons_pair(integer_to_ground(the(short_point,ground)[0]),
		   integer_to_ground(the(short_point,ground)[1]));
} 

Grounding interpret_point_ground(Ground_Type code,Grounding ground)
{
  Grounding new; INITIALIZE_NEW_GROUND(new,short_point_ground);
  the(short_point,new)[0]=GINTEGER(GCAR(ground));
  the(short_point,new)[1]=GINTEGER(GCDR(ground));
  return new;
}

Grounding make_point(short i,short j)
{
  Grounding new; INITIALIZE_NEW_GROUND(new,short_point_ground);
  the(short_point,new)[0]=i;
  the(short_point,new)[1]=j;
  return new;
}


/* Converting other types to grounds */

Grounding string_to_ground(char *string)
{
  Grounding result; char *copy;
  INITIALIZE_NEW_GROUND(result,string_ground);
  ALLOCATE(copy,char,(strlen(string)+1));
  strcpy(copy,string);
  the(string,result)=copy;
  return result;
}

Grounding integer_to_ground(long integer)
{
  Grounding result; 
  INITIALIZE_NEW_GROUND(result,integer_ground);
  the(integer,result)=integer;
  return result;
}

Grounding float_to_ground(float flt)
{
  Grounding result; 
  INITIALIZE_NEW_GROUND(result,float_ground);
  the(flonum,result)=(float) flt;
  return result;
}


/* Minimal numeric support ... plus more*/

Grounding (*string2big)();
char *(*big2string)();

extern Grounding generic_equal(Grounding x, Grounding y);
/* int (*generic_numeric_compare)()=NULL;  */
Grounding generic_numeric_compare(Grounding x, Grounding y)
{  return (generic_equal(x, y));  }

Grounding (*make_real_rational)()=NULL;

Grounding make_bignum(char *digits)
{
  if (string2big) return string2big(digits);
  else {Grounding result; char *copy; 
	ALLOCATE(copy,char,strlen(digits)+1); strcpy(copy,digits);
	INITIALIZE_NEW_GROUND(result,bignum_ground);
	the(string,result)=copy;
	return result;}
}

char *bignum_to_string(Grounding bignum)
{
  if (big2string) return big2string(bignum);
  else {char *copy; ALLOCATE(copy,char,strlen(the(string,bignum))+1);
	strcpy(copy,the(string,bignum));
	return copy;}
}

void free_fake_bignum(Grounding bignum)
{
  free(the(string,bignum));
}

Grounding make_rational(Grounding numerator, Grounding denominator)
{
  Grounding result; 
  if (make_real_rational) return make_real_rational(numerator,denominator);
  INITIALIZE_NEW_GROUND(result,rational_ground);
  GCAR(result)= numerator; GCDR(result) = denominator; 
  USE_GROUND(numerator); USE_GROUND(denominator);
  return result;
}

/* GCing Grounds */

/* Prototypes for allocation functions */

void free_up_ground(struct GROUNDING *ground);
void free_up_pair(struct PAIR *pair);

void free_ground(Grounding ground)
{
  FREE_GROUND(ground);
}

void gc_ground(Grounding ground)
{
  if (ground == NULL) return;
  switch (ground_type(ground))
    {
    case frame_ground_type: break;
    case string_ground: 
      free(the(string,ground)); break;
    case pair_ground: case rational_ground: case procedure_ground:
      if (ground == empty_list) return;
      FREE_GROUND(GCAR(ground));
      FREE_GROUND(GCDR(ground));
      break;
    case vector_ground: case nd_ground:
      {DO_VECTOR(elt,ground) FREE_GROUND(elt);}
      free(GVELEMENTS(ground));
      break;
    case integer_ground: case float_ground: case short_point_ground: break;
    case symbol_ground: case framer_function_ground:
      ground->ground.reference_count=1; return;
    default: 
      if (reclaimers[(int) ground_type(ground)] != NULL)
	(reclaimers[(int) ground_type(ground)])(ground);
      else {FREE_GROUND(the(mystery,ground));}
      break;
    };
  free_up_ground((struct GROUNDING *) ground);
}
 

/* Comparing grounds */

/* Two groundings are eq only if the are the same pointer */
boolean eq_p(Grounding x,Grounding y)
{
  return (x == y);
}

boolean eqv_p(Grounding x,Grounding y)
{
  if (x == y) return True;
  switch (ground_type(x)) 
    {
    case integer_ground:
    case float_ground:
    case bignum_ground:
    case rational_ground:
      switch (ground_type(y))
	{
	case integer_ground:
	case float_ground:
	case bignum_ground:
	case rational_ground:
	  return (generic_equal(x,y) != NULL);
	default:
	  return False;
	}
    default:
      return False;
    }
}

boolean equal_p(Grounding x,Grounding y)
{
  if (eqv_p(x,y)) return True;
  else if ((x == NULL) || (y == NULL)) return False;
  else if ((FRAMEP(x)) || (FRAMEP(y))) return False;
  if ((ground_type(x)) != (ground_type(y))) return False;
  else switch ((ground_type(x)))
    {
    case string_ground:
      return (strcmp(GSTRING(x),GSTRING(y))==0); 
    case vector_ground:
      {int i; i=GVSIZE(x);
       if (i != GVSIZE(y)) return False;
       for (i--;(i>=0);i--)
       	 if (!(EQUAL_P(GVREF(x,i),GVREF(y,i)))) return False;
       return True;}
    case pair_ground:
      if ((x == empty_list) || (y == empty_list)) return False;
      else if ((GCAR(x) == (GCAR(y))) || (EQUAL_P(GCAR(x),GCAR(y))))
      	return ((GCDR(x) == (GCDR(y))) || (EQUAL_P(GCDR(x),GCDR(y))));
      else return False;
    case symbol_ground:
      return (x==y);
    case nd_ground:
      { int size=0;
	{ DO_RESULTS(r,x)
       	  { boolean any=False;
	    size++;
	      { DO_RESULTS(s,y) if (EQUAL_P(r,s)) any=True; }
            if (NOT(any)) return False;  }}
        { DO_RESULTS(r,y) size--; }
        if (size == 0) return True; else return False; }
    default:
      return False;
    }
}

 
/* Dealing with `native grounds' */

Grounding interpret_native_ground(int type,Grounding source)
{
  Grounding result;
  if (interpreters[type] != NULL)
    {Grounding new; USE_GROUND(source);
     new=(interpreters[type])(type,source);
     USE_GROUND(new); FREE_GROUND(source);
     if ((new) && (NOT(FRAMEP(new)))) ref_count(new)--;
     return new;}
  WITH_HANDLING
    {INITIALIZE_NEW_GROUND(result,((Ground_Type)type));
     set_ground_contents(result,mystery,source);}
  ON_EXCEPTION
    reraise(); return NULL;
  END_HANDLING
    return result;
}
 
Grounding canonicalize_native_ground(ground)
     Grounding ground;
{
  if (canonicalizers[(int) (ground_type(ground))] != NULL)
    return (canonicalizers[(int) (ground_type(ground))])(ground);
  return the(mystery,ground);
}


extern char *highlight_start, *highlight_end;

void init_framer_grounds()
{
  Grounding root_frame_symbol;
  void setup_standard_io AP((void));
  setup_standard_io();
  /* Initialize terminators */
  {DO_TIMES(i,255)
     if (NOT(isgraph((char) i))) framer_terminators[i+1]=True;
     else framer_terminators[i+1]=False;}
  framer_terminators[(int) '/'+1]=True; framer_terminators[(int) '"'+1]=True;
  framer_terminators[(int) '#'+1]=True; framer_terminators[(int) '('+1]=True;
  framer_terminators[(int) ')'+1]=True; framer_terminators[(int) '{'+1]=True;
  framer_terminators[(int) '}'+1]=True; framer_terminators[(int) ' '+1]=True;
  framer_terminators[0]=True;
  /* Initialize table entries */
  canonicalizers[(int) frame_array_ground]= canonicalize_frame_array;
  interpreters[(int) frame_array_ground]= interpret_frame_array;
  reclaimers[(int) frame_array_ground]= free_frame_array;
  canonicalizers[(int) framer_function_ground]= canonicalize_function_ground;
  interpreters[(int) framer_function_ground]= interpret_function_ground;
  reclaimers[(int) framer_function_ground]= NULL;
  canonicalizers[(int) bitset_ground]= canonicalize_ephemeral_ground;
  interpreters[(int) bitset_ground]= interpret_ephemeral_ground;
  reclaimers[(int) bitset_ground]= free_bitset;
  canonicalizers[(int) procedure_ground]= canonicalize_procedure_ground;
  interpreters[(int) procedure_ground]= interpret_procedure_ground;
  canonicalizers[(int) short_point_ground]= canonicalize_point_ground;
  interpreters[(int) short_point_ground]= interpret_point_ground;
  reclaimers[(int) bignum_ground]=free_fake_bignum;
  /* Initialize variables */
  set_ground_type(empty_list,pair_ground);
  GCAR(empty_list)=NULL; GCDR(empty_list)=NULL; an_empty_list.reference_count=1;
  backquote_symbol = intern("BACKQUOTE");
  backquote_symbol->ground.contents.symbol->value = backquote_symbol;
  quote_symbol = intern("QUOTE");
  quote_symbol->ground.contents.symbol->value = quote_symbol;
  unquote_symbol = intern("UNQUOTE");
  unquote_symbol->ground.contents.symbol->value = unquote_symbol;
  t_symbol = intern("T");
  t_symbol->ground.contents.symbol->value = t_symbol;
  frame_ground_symbol = intern("FRAME-GROUND");
  root_frame_symbol = intern("ROOT-FRAME");
  the(symbol,root_frame_symbol)->value = frame_to_ground(root_frame);
  read_only_tag=intern("ROF"); alias_tag=intern("ALF"); file_tag=intern("FILE");
  lisp_nil_symbol=intern("LISP:NIL");
  period_symbol=intern(".");
  highlight_start="==>"; highlight_end="<==";
}

/*
  Local variables: ***
  compile-command: "../etc/compile-framer" ***
  tags-file-name: "../sources/TAGS" ***
  End: **
*/

