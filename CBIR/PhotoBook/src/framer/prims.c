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
  This file defines the core functions for parsing, printing, and reclaiming
  FRAMER grounds.
*************************************************************************/
 
/* This contains the C implementations of various FRAXL primitives. */

static char rcsid[] = "$Header: /mas/framer/sources/RCS/prims.c,v 1.81 1994/02/01 19:22:33 haase Exp $";

#include <limits.h>
#include <errno.h>
#include <ctype.h>
#include <time.h>
#include "framer.h"
#include "fraxl.h"
#include "internal/private.h"
#include "internal/eval.h"

extern Grounding
  if_symbol, amb_symbol, gather_symbol, lambda_symbol, define_symbol, 
  letrec_symbol, let_star_symbol,
  quote_symbol, backquote_symbol, unquote_symbol, send_symbol, delegate_symbol,
  t_symbol, root_frame_symbol, using_symbol, let_symbol, begin_symbol;
extern Grounding eval_function;
extern exception File_Unwritable, File_Unreadable;

Grounding make_bitset(Frame frame);

#define GCADR(x) (GCAR(GCDR(x)))


/* Generic operations */

Grounding fraxl_equal_p(Grounding x,Grounding y)
{
  if (equal_p(x,y)) return x; else return NULL;
}

Grounding fraxl_eqv_p(Grounding x,Grounding y)
{
  if (eqv_p(x,y)) return x; else return NULL;
}

Grounding fraxl_eq_p(Grounding x,Grounding y)
{
  if (eq_p(x,y)) return x; else return NULL;
}

Grounding fraxl_string_equal_p(Grounding x,Grounding y)
{
  if ((string_compare((GSTRING(x)),(GSTRING(y)))) == 0)
    return x;
  else return NULL;
}

#define FRAME_NAME(x)  (frame_name((Frame) x))

Grounding fraxl_alpha_less_p(Grounding x,Grounding y)
{
  Grounding generic_less_than(Grounding x,Grounding y);
  if (FRAMEP(x))
    if (NOT(FRAMEP(y))) return t_symbol;
    else if ((string_compare((FRAME_NAME(x)),FRAME_NAME(y))) < 0) 
      return t_symbol;
    else return NULL;
  else if (SYMBOLP(x))
    if (FRAMEP(y)) return NULL;
    else if (NOT(SYMBOLP(y))) return t_symbol;
    else if ((string_compare((SYMBOL_NAME(x)),SYMBOL_NAME(y))) < 0)
      return t_symbol;
    else return NULL;
  else if (STRINGP(x))
    if ((FRAMEP(y)) || (SYMBOLP(y))) return NULL;
    else if (NOT(STRINGP(y))) return t_symbol;
    else if ((string_compare((unground(x,string)),(unground(y,string)))) < 0)
      return t_symbol;
    else return NULL;
  else if (NUMBERP(x))
    if (NOT(NUMBERP(y))) return NULL;
    else if (generic_less_than(x,y)) return t_symbol;
    else return NULL;
  else return NULL;
}


/* ND-Set operations */

Grounding one_result(Grounding ground)
{
  if ((ground_type(ground))==nd_ground)
    return GVREF(ground,0);
  else return ground;
}

Grounding find_result(Grounding result,Grounding result_set)
{
  if (result_set == NULL) return NULL;
  else if ((ground_type(result_set)) != nd_ground)
    if (EQUAL_GROUND_P(result,result_set))
      return result_set;
    else return NULL;
  else {DO_RESULTS(r,result_set) if (EQUAL_GROUND_P(result,r)) return r;
	return NULL;}
}

/* This conses a new result (which should not be a result set) onto an
   existing (possibly empty) result set. */
Grounding cons_result(Grounding gnd,Grounding results)
{
  if (NULLP(results)) return gnd;
  else if (NULLP(gnd)) return results;
  else if (ground_type(gnd) == nd_ground) 
    {ground_error(Type_Error,"can't be nd ground",gnd); return NULL;}
  else if (ground_type(results) != nd_ground)
    if (EQUAL_GROUND_P(gnd,results)) return results;
    else {Grounding new_result_set; NDMAKE(new_result_set,2);
	  GVSET(new_result_set,0,gnd); GVSET(new_result_set,1,results);
	  return new_result_set;}
  else {Grounding new_result_set, *ptr;
	NDMAKE(new_result_set,GVSIZE(results)+1); GVSET(new_result_set,0,gnd); 
	ptr=ND_ELTS(new_result_set)+1; 
	{DO_RESULTS(r,results) {*ptr++=r; USE_GROUND(r);}}
	return new_result_set;}
}

Grounding merge_results(Grounding res1,Grounding res2)
{
  if (NULLP(res2)) return res1;
  else if (NULLP(res1)) return res2;
  else if (ground_type(res1) != nd_ground) return cons_result(res1,res2);
  else if (ground_type(res2) != nd_ground) return cons_result(res2,res1);
  else {WITH_RESULT_ACCUMULATOR(ra)
	  {{DO_RESULTS(r,res1) accumulate_result(r,ra);}
	   {DO_RESULTS(r,res2) accumulate_result(r,ra);}
	   return resolve_accumulator(ra);}}
}

Grounding zap_result(Grounding result,Grounding result_set)
{
  if (result_set == NULL) return NULL;
  else if (NOT(TYPEP(result_set,nd_ground)))
    if (EQUAL_GROUND_P(result,result_set)) return NULL;
    else return result_set;
  else {int i=0; DO_RESULTS(r,result_set)
	  if (EQUAL_GROUND_P(result,r)) {STOP_DOING_RESULTS();} else i++;
	if (i == GVSIZE(result_set)) return result_set;
	else {Grounding new; int j=0; NDMAKE(new,ND_SIZE(result_set)-1);
	      {DO_RESULTS(r,result_set) if (i-- != 0) {GVSET(new,j++,r);}}
	      return new;}}
}

Grounding resolve_accumulator(struct RESULT_ACCUMULATOR *ra)
{
  Grounding result, temp=empty_list, *ptr, *limit; 
  int how_many, duplicates=0; how_many=(ra->ptr-ra->accumulator);
  switch (how_many)
    {
    case 0: return NULL;
    case 1: {Grounding result; result=ra->accumulator[0]; 
	     if (NOT(FRAMEP(result))) ref_count(result)--;
	     return result;}
    case 2: {Grounding result; result=cons_result(ra->accumulator[0],ra->accumulator[1]);
	     USE_GROUND(result); 
	     FREE_GROUND(ra->accumulator[0]); FREE_GROUND(ra->accumulator[1]);
	     if (NOT(FRAMEP(result))) ref_count(result)--;
	     return result;}
    default: temp=empty_list;
    }
  ptr=ra->accumulator; limit=ra->ptr; while (ptr < limit) 
    {Grounding elt; elt=*ptr;
     if (FRAMEP(elt)) 
       if (search_bit_p((Frame)elt,RESULT_REDUCTION_BIT))
	 {duplicates++; *ptr=NULL;}
       else set_search_bit((Frame) elt,RESULT_REDUCTION_BIT);
     else {if (in_list(elt,temp)) duplicates++; else temp=cons_pair(elt,temp); 
	   FREE_GROUND(elt);}
     ptr++;}
  if (duplicates == (ra->ptr-ra->accumulator)) raise_crisis("Duplication error");
  else {NDMAKE(result,(ra->ptr-ra->accumulator)-duplicates);}
  {Grounding *nptr; ptr=ra->accumulator; nptr=GVELEMENTS(result);
   while (ptr < limit)
     {Grounding elt; elt=*ptr++;
      if (FRAMEP(elt))
	{if (search_bit_p((Frame)elt,RESULT_REDUCTION_BIT))
	   {*nptr++=elt; clear_search_bit((Frame) elt,RESULT_REDUCTION_BIT);}}}
   {DO_LIST(elt,temp) {USE_GROUND(elt); *nptr++=elt;}}}
  if (temp != empty_list) {FREE_GROUND(temp);}
  if (ra->size > 4) free(ra->accumulator);
  return result;
}

void accumulate_result(Grounding g,struct RESULT_ACCUMULATOR *r_acc)
{
  if (r_acc->ptr >= r_acc->limit)
    {Grounding *new_accumulator; int new_size; new_size=r_acc->size*2;
     ALLOCATE(new_accumulator,Grounding,new_size);
     memmove(new_accumulator,r_acc->accumulator,r_acc->size*sizeof(Grounding));
     r_acc->ptr=new_accumulator+(r_acc->ptr-r_acc->accumulator);
     if (r_acc->size > 4) free(r_acc->accumulator);
     r_acc->limit=new_accumulator+new_size; r_acc->size=new_size;
     r_acc->accumulator=new_accumulator;}
  *(r_acc->ptr++)=g; USE_GROUND(g);
}

Grounding list_to_result_set(Grounding list)
{
  WITH_RESULT_ACCUMULATOR(ra)
    {DO_LIST(elt,list) accumulate_result(elt,ra);
     return resolve_accumulator(ra);}
}

Grounding finalize_result_list(Grounding list)
{
  if (list == empty_list) return NULL;
  else if ((GCDR(list)) == empty_list)
    {Grounding result; result=GCAR(list);
     if (FRAMEP(result)) {FREE_GROUND(list); return result;}
     else {USE_GROUND(result); FREE_GROUND(list); ref_count(result)--; 
	   return result;}}
  else {WITH_RESULT_ACCUMULATOR(ra)
	  {Grounding result; DO_LIST(elt,list) accumulate_result(elt,ra);
	   result=resolve_accumulator(ra); FREE_GROUND(list);
	   return result;}}
}

Grounding gather_results(Grounding combiner,Grounding result_set,Grounding init)
{
  Grounding current, new; current=init; USE_GROUND(init);
  {DO_RESULTS(r,result_set)
     {if (!current) return NULL;  /* any combination with no results
				     terminates a gather */
     new=apply2(combiner,r,current); FREE_GROUND(current); current=new;}}
  return current;
}
 

/* Turtle Semantics */

extern Frame point;

Grounding fraxl_in_home(Grounding arg)
{
  read_root=point=GFRAME(arg);
  return NULL;
}

Grounding fraxl_proto_equals(Grounding args)
{
  if (args == empty_list) return (Grounding) set_prototype(point,NULL);
  else return (Grounding) set_prototype(point,GFRAME(GCAR(args)));
}

Grounding fraxl_ground_equals(Grounding args)
{
  Grounding list_elements(Grounding list);
  if (args == empty_list) return (Grounding) set_ground(point,NULL);
  else return (Grounding) set_ground(point,list_elements(args));
}

Grounding quotify(Grounding x)
{
  if ((TYPEP(x,pair_ground)) || (TYPEP(x,symbol_ground)))
    return cons_pair(quote_symbol,(cons_pair(x,empty_list)));
  else if ((TYPEP(x,procedure_ground)) && ((proc_env(x)) == empty_list))
    if (proc_args(x) == empty_list)
      return cons_pair(lambda_symbol,cons_pair(empty_list,proc_body(x)));
    else if (NULLP(GCAR(proc_args(x))))
      return cons_pair(lambda_symbol,cons_pair(GCDR(proc_args(x)),(proc_body(x))));
    else return cons_pair(define_symbol,cons_pair(proc_args(x),proc_body(x)));
  else return x;
}


/* Operations with search bits */

Grounding in_bitset(Grounding frame,Grounding bs)
{
  if (search_bit_p(GFRAME(frame),bitset_bit(bs)))
    return t_symbol;
  else return NULL;
}

Grounding add_to_bitset(Grounding frame,Grounding bs)
{
  set_search_bit((Frame) frame,bitset_bit(bs));  
  return NULL;
}

Grounding remove_from_bitset(Grounding frame,Grounding bs)
{
  clear_search_bit((Frame) frame,bitset_bit(bs));  
  return NULL;
}

void initialize_bitset(Frame f,int bit,Grounding proc)
{
  Grounding result; result=apply1(proc,frame_to_ground(f));
  if (result) set_search_bit(f,bit); FREE_GROUND(result);
  {DO_ANNOTATIONS(a,f) initialize_bitset(a,bit,proc);}
}

void for_bitset(Grounding fcn,Frame from,int bit)
{
  if (search_bit_p(from,bit))
    {Grounding r; r=apply1(fcn,frame_to_ground(from)); FREE_GROUND(r);}
  {DO_SOFT_ANNOTATIONS(a,from) for_bitset(fcn,a,bit);}
}

void gathering_bitset(struct RESULT_ACCUMULATOR *ra,Frame from,int bit)
{
  if (search_bit_p(from,bit))
    accumulate_result(frame_to_ground(from),ra);
  {DO_SOFT_ANNOTATIONS(a,from) gathering_bitset(ra,a,bit);}
}

Grounding gather_bitset(Frame from,int bit)
{
  WITH_RESULT_ACCUMULATOR(ra) gathering_bitset(ra,from,bit);
  return resolve_accumulator(ra);
}

Grounding fraxl_initialize_bitset(Grounding bs,Grounding fcn)
{
  initialize_bitset(bitset_root(bs),bitset_bit(bs),fcn);
  return bs;
}

Grounding fraxl_for_bitset(Grounding fcn,Grounding bs)
{
  for_bitset(fcn,bitset_root(bs),bitset_bit(bs));
  return NULL;
}

Grounding fraxl_gather_subset(Grounding bs)
{
  /* Fix for new result sets */
  return gather_bitset(bitset_root(bs),bitset_bit(bs));
}


/* Static primitives */
static int result_bitset, arg1_bitset, arg2_bitset;

static void bitset_op(Frame f,void (*op)())
{
  op(f);
  {DO_SOFT_ANNOTATIONS(a,f) bitset_op(a,op);}
}
static void bitset_and_op(Frame f)
{
  if ((search_bit_p(f,arg2_bitset)) && (search_bit_p(f,arg1_bitset)))
    set_search_bit(f,result_bitset);
}
static void bitset_or_op(Frame f)
{
  if ((search_bit_p(f,arg1_bitset)))
    set_search_bit(f,result_bitset);
}
static void bitset_not_op(Frame f)
{
  if ((search_bit_p(f,arg1_bitset)))
    clear_search_bit(f,result_bitset);
  else set_search_bit(f,result_bitset);
}


/* Boolean operators on bitsets */
Grounding bitset_and(Grounding bs1,Grounding bs2)
{
  Grounding new_bitset;
  /* Make sure bs1 is the `lowest' bitset */
  {Frame f1, f2; f1=bitset_root(bs1); f2=bitset_root(bs2);
   if (f1 == f2) {}
   else if (has_home(f1,f2)) {}
   else if (has_home(f2,f1)) {Grounding tmp; tmp=bs2; bs2=bs1; bs1=tmp;}
   else return NULL;}
  new_bitset=make_bitset(bitset_root(bs1));
  result_bitset=bitset_bit(new_bitset);
  arg1_bitset=bitset_bit(bs1);
  arg2_bitset=bitset_bit(bs2);
  bitset_op(bitset_root(bs1),bitset_and_op);
  return new_bitset;
}

Grounding bitset_or(Grounding bs1,Grounding bs2)
{
  Grounding new_bitset;
  /* Find the root above both bitsets */
  {Frame f1, f2, root; 
   f1=bitset_root(bs1); f2=bitset_root(bs2);
   if (has_home(f1,f2)) root=f2;
   else if (has_home(f2,f1)) root=f1;
   else {root=f1; while (NOT(has_home(f2,root))) root=frame_home(root);}
   new_bitset=make_bitset(root);}
  result_bitset=bitset_bit(new_bitset);
  arg1_bitset=bitset_bit(bs1);
  arg2_bitset=bitset_bit(bs2);
  bitset_op(bitset_root(bs1),bitset_or_op);
  bitset_op(bitset_root(bs2),bitset_or_op);
  return new_bitset;
}

Grounding bitset_not(Grounding bs)
{
  Grounding new_bitset;
  new_bitset=make_bitset(bitset_root(bs));
  result_bitset=bitset_bit(new_bitset);
  arg1_bitset=bitset_bit(bs);
  bitset_op(bitset_root(bs),bitset_not_op);
  return new_bitset;
}


/* Operations on points */

Grounding fraxl_make_point(Grounding x,Grounding y)
{
  Grounding make_point(short x,short y);
  return make_point(GINTEGER(x),GINTEGER(y));
}

Grounding point_x(Grounding pt)
{
  return integer_to_ground(the(short_point,pt)[0]);
}

Grounding point_y(Grounding pt)
{
  return integer_to_ground(the(short_point,pt)[1]);
}


/* List operations */

Grounding cons_p(Grounding x)
{
  if (x==empty_list) return NULL;
  if ((ground_type(x))==pair_ground) return x;
  else return NULL;
}

Grounding null_p(Grounding x)
{
  if (x==empty_list) return x;
  else return NULL;
}

Grounding in_list(Grounding elt,Grounding list)
{
  if (EMPTY_LIST_P(list) || (NOT(TYPEP(list,pair_ground)))) return NULL;
  else if (EQUAL_GROUND_P(elt,GCAR(list))) return GCAR(list);
  return in_list(elt,GCDR(list));
}

static int count(Grounding elt,Grounding list,int so_far)
{
  if (EMPTY_LIST_P(list)) return so_far;
  else if (TYPEP(list,pair_ground)) return so_far;
  else if (EQUAL_GROUND_P(elt,GCAR(list))) 
    return count(elt,GCDR(list),so_far+1);
  else return count(elt,GCDR(list),so_far);
}

Grounding remove_from_list(Grounding ground,Grounding list)
{
  if (EMPTY_LIST_P(list)) return list;
  else if (NOT(TYPEP(list,pair_ground))) return list;
  else if (EQUAL_GROUND_P(ground,GCAR(list)))
    return remove_from_list(ground,GCDR(list));
  else return cons_pair(GCAR(list),remove_from_list(ground,GCDR(list)));
}

Grounding remove_from_list_once(Grounding ground,Grounding list)
{
  if (EMPTY_LIST_P(list)) return list;
  else if (NOT(TYPEP(list,pair_ground))) return list;
  else if (EQUAL_GROUND_P(ground,GCAR(list))) return GCDR(list);
  else return cons_pair(GCAR(list),remove_from_list(ground,GCDR(list)));
}

static Grounding append(Grounding l1,Grounding l2)
{
  if (EMPTY_LIST_P(l1) || (NOT(TYPEP(l1,pair_ground)))) return l2;
  else return cons_pair(GCAR(l1),append(GCDR(l1),l2));
}

Grounding append_lexpr(Grounding lists)
{
  if (lists == empty_list) return lists;
  else {Grounding car; car=GCAR(lists);
	if (TYPEP(car,pair_ground))
	  return append(GCAR(lists),append_lexpr(GCDR(lists)));
	else {ground_error(Type_Error,"not a list to append: ",car);
	      return NULL;}}
}

Grounding list_lexpr(Grounding args)
{
  return args;
}

Grounding vector_lexpr(Grounding args)
{
  Grounding vec; int i=0; GVMAKE(vec,list_length(args)); 
  {DO_LIST(elt,args) {GVSET(vec,i,elt); i++;}}
  return vec;
}

Grounding vector_to_list(Grounding vector)
{
  Grounding list; int i, size; list=empty_list; size=GVSIZE(vector);
  for (i=size-1;i>=0;i--) list=cons_pair(GVREF(vector,i),list);
  return list;
}

Grounding list_to_vector(Grounding list)
{
  Grounding vector; int i, size;
  i=0; size=list_length(list); GVMAKE(vector,size);
  {DO_LIST(elt,list) {GVSET(vector,i++,elt);}}
  return vector;
}

Grounding maplist(Grounding (*fcn)(), Grounding list)
{
  Grounding result, *rptr; result=empty_list; rptr = (&result);
  USE_GROUND(result); /* Use the tail */
  {DO_LIST(elt,list) 
     {Grounding nelt; nelt=fcn(elt); USE_GROUND(nelt); TCONS(nelt,rptr);}} 
  ref_count(result)--; /* Strip off the extra use pointer from the head */
  return result;
}

Grounding reverse_list(Grounding list)
{
  Grounding new_list; new_list=empty_list;
  {DO_LIST(elt,list) {GCONS(elt,new_list,new_list);}}
  return new_list;
}

int list_length(Grounding list)
{
  int count=0;
  while (NOT(EMPTY_LIST_P(list))) {list=GCDR(list); count++;}
  return count;
}


/* Returns all the elements of a list as a result set. */
Grounding list_elements(Grounding list)
{
  Grounding results=NULL;
  {DO_LIST(x,list) results=merge_results(x,results);}
  return results;
}

Grounding pair_car(pair)
     Grounding pair;
{
  if (pair == empty_list) return NULL;
  else return GCAR(pair);
}
 
Grounding pair_cdr(Grounding pair)
{
  if (pair == empty_list) return NULL;
  else return GCDR(pair);
}
 
Grounding pair_cadr(Grounding pair)
{
  Grounding r; 
  if (pair == empty_list) return NULL; else r=GCDR(pair);
  if (r == empty_list) return NULL;
  else return (GCAR(r));
}

Grounding pair_cddr(Grounding pair)
{
  Grounding r; 
  if (pair == empty_list) return NULL; else r=GCDR(pair);
  if (r == empty_list) return NULL;
  else return (GCDR(r));
}

Grounding listref(Grounding lst,Grounding ix)
{
  int i; i=GINTEGER(ix);
  {DO_LIST(elt,lst) if (i-- == 0) return elt;}
  return NULL;
}

Grounding listcdrref(Grounding lst,Grounding ix)
{
  int i; i=GINTEGER(ix);
  while (lst != empty_list) 
    if (i-- == 0) return lst;
    else lst=GCDR(lst);
  return NULL;
}

Grounding fraxl_mapcar(Grounding fcn,Grounding lst)
{
  Grounding result, *rptr; result=empty_list; rptr = (&result);
  USE_GROUND(empty_list); /* Use the tail */
  {DO_LIST(elt,lst) {TCONS(apply1(fcn,elt),rptr);}} 
  ref_count(result)--; /* Strip off the extra use pointer from the head */
  return result;
}


/* Vector operations */
 
Grounding make_vector_ground(int size)
{
  Grounding result;
  GVMAKE(result,size);
  return result;
}
 
Grounding gvector_ref(Grounding gv,int i)
{
  if ((i < 0) || (i >= (GVSIZE(gv)))) return NULL;
  else return GVREF(gv,i);
}
 
int gvector_size(Grounding gv)
{
  return GVSIZE(gv);
}
 
void gvector_set(Grounding gv,int i,Grounding gs)
{
  GVSET(gv,i,gs);
}
 
Grounding gvref(Grounding vector,Grounding i)
{
  int index; index=the(integer,i);
  if ((index < 0) || (index >= GVSIZE(vector))) return NULL;
  else return GVREF(vector,index);
}
 
Grounding gvsize(Grounding vector)
{ return integer_to_ground(GVSIZE(vector)); }


/* Generic sequence operations */

Grounding generic_length(Grounding x)
{
  if (TYPEP(x,string_ground)) 
    return integer_to_ground(strlen(GSTRING(x)));
  else if (TYPEP(x,vector_ground)) 
    return integer_to_ground(GVSIZE(x));
  else if (TYPEP(x,pair_ground))
    return integer_to_ground(list_length(x));
  else {ground_error(Type_Error,"not a sequence: ",x); 
	return NULL;}
}

Grounding generic_elt(Grounding x,Grounding i)
{
  int index; index=GINTEGER(i);
  if (index < 0) return NULL;
  else if (TYPEP(x,string_ground)) 
    {char buf[2], *string; int size;
     string=GSTRING(x); size=strlen(string) ;buf[1]='\0'; 
     if (index >= size) return NULL;
     else {buf[0]=string[index]; return string_to_ground(buf);}}
  else if (TYPEP(x,vector_ground)) 
    if (index >= GVSIZE(x)) return NULL;
    else return GVREF(x,index);
  else if (TYPEP(x,pair_ground))
    {int count=0; DO_LIST(elt,x) if (count == index) return elt; else count++;
     return NULL;}
  else {ground_error(Type_Error,"not a sequence: ",x);
	return NULL;}
}

Grounding generic_reverse(Grounding x)
{
  if (TYPEP(x,pair_ground)) return reverse_list(x);
  else if (TYPEP(x,vector_ground)) 
    {int i, size; Grounding new; 
     size=GVSIZE(x); GVMAKE(new,size); size--; i=0;
     {DO_VECTOR(elt,x) {GVSET(new,size-i,elt); i++;}}
     return new;}
  else if (TYPEP(x,string_ground)) 
    {int size; char *iptr, *optr, *new; Grounding result;
     size=strlen((GSTRING(x))); ALLOCATE(new,char,size+1); 
     optr=new+size-1; iptr=GSTRING(x);
     while (*iptr != '\0') {*optr--=*iptr++;} 
     new[size+1]='\0'; result=string_to_ground(new); free(new);
     return result;}
  else {ground_error(Type_Error,"not a sequence: ",x);
	return NULL;}
}

Grounding generic_position(Grounding x,Grounding seq)
{
  int index=0;
  if (TYPEP(seq,string_ground)) 
    {char *key, *search, *find; key=GSTRING(x); search=GSTRING(seq);
     find=strchr(search,*key); 
     if (find) return integer_to_ground(find-search);
     else return NULL;}
  else if (TYPEP(seq,vector_ground)) 
    {DO_VECTOR(elt,seq) 
       if (EQUAL_GROUND_P(x,elt)) return integer_to_ground(index);
       else index++;
     return NULL;}
  else if (TYPEP(seq,pair_ground))
    {DO_LIST(elt,seq) 
       if (EQUAL_GROUND_P(x,elt)) return integer_to_ground(index);
       else index++;
     return NULL;}
  else {ground_error(Type_Error,"not a sequence: ",x);
	return NULL;}
}

Grounding generic_all_elts(Grounding x)
{
  WITH_RESULT_ACCUMULATOR(ra)
    if (TYPEP(x,string_ground)) 
      {char buf[2], *string; int size;
       string=GSTRING(x); size=strlen(string); buf[1]='\0'; 
       {DO_TIMES(i,size)
	  {buf[0]=string[i]; accumulate_result(string_to_ground(buf),ra);}}}
    else if (TYPEP(x,vector_ground)) 
      {DO_VECTOR(elt,x) accumulate_result(elt,ra);}
    else if (TYPEP(x,pair_ground))
      {DO_LIST(elt,x) accumulate_result(elt,ra);}
    else ground_error(Type_Error,"not a sequence: ",x);
  return resolve_accumulator(ra);
}


/* String and symbol ops */

Grounding fraxl_intern(Grounding string)
{
  return intern(GSTRING(string));
}

Grounding fraxl_symbol_name(Grounding symbol)
{
  return string_to_ground(SYMBOL_NAME(symbol));
}

Grounding fraxl_substring(Grounding string,Grounding gstart,Grounding gend)
{
  int len, start, end, size; char *result; 
  len=strlen(GSTRING(string)); start=GINTEGER(gstart); end=GINTEGER(gend);
  if (NOT((end > start) && (start > 0) && (end < len))) return NULL;
  else size=end-start+1;
  result=fra_allocate(size,sizeof(char)); 
  strncpy(result,GSTRING(string)+start,size); result[size]='\0';
  return string_to_ground(result);
}

Grounding fraxl_find_substring(Grounding string,Grounding in_string)
{
  char *inner_result, *outer_result; inner_result=strstr(GSTRING(in_string),GSTRING(string));
  if (inner_result == NULL) return NULL;
  else {outer_result=fra_allocate(strlen(inner_result)+1,sizeof(char));
	strcpy(outer_result,inner_result);
	return string_to_ground(outer_result);}
}

Grounding fraxl_capitalized_p(Grounding string)
{
  if (isupper(*(GSTRING(string)))) return string;
  else return NULL;
}


Grounding print_ground_to_stdout(Grounding ground)
{
  print_ground(standard_output,ground);
  if (standard_output->stream_type == file_io) fflush(standard_output->ptr.file);
  return ground;
}

Grounding display_ground_to_stdout(Grounding ground)
{
  if (TYPEP(ground,string_ground))
    gsputs(GSTRING(ground),standard_output);
  else print_ground(standard_output,ground);
  if (standard_output->stream_type == file_io) fflush(standard_output->ptr.file);
  return ground;
}

Grounding printout_lexpr(Grounding args)
{
  {DO_LIST(elt,args) 
     if (TYPEP(elt,string_ground)) gsputs(GSTRING(elt),standard_output);
     else print_ground(standard_output,elt);}
  gsputc('\n',standard_output); 
  if (standard_output->stream_type == file_io) fflush(standard_output->ptr.file);
  return NULL;
}

Grounding printoutp_lexpr(Grounding args)
{
  {DO_LIST(elt,args) 
     if (TYPEP(elt,string_ground)) gsputs(GSTRING(elt),standard_output);
     else print_ground(standard_output,elt);}
  if (standard_output->stream_type == file_io) fflush(standard_output->ptr.file);
  return NULL;
}

Grounding parse_ground_from_stdin()
{
  return parse_ground(standard_output);
}

Grounding newline_to_stdout()
{
  gsputc('\n',standard_output); 
  if (standard_output->stream_type == file_io) fflush(standard_output->ptr.file);
  return (Grounding) NULL;
}

Grounding print_result_to_stdout(Grounding result)
{
  gsputs("...> ",standard_output); 
  print_ground(standard_output,result);
  gsputc('\n',standard_output);
  return NULL;
}

Grounding print_string_ground_to_stdout(Grounding str)
{
  if ((ground_type(str)) != string_ground) return NULL;
  printf("%s",the(string,str)); fflush(stdout);
  return NULL;
}

Grounding file_as_string(Grounding filename)
{
  Grounding result; int c; FILE *fstream;
  fstream=fopen(GSTRING(filename),"r"); 
  if (NULLP(fstream)) raise_crisis_with_details(File_Unreadable,GSTRING(filename));
  {WITH_OUTPUT_TO_STRING(gs,1024)
     {while ((c=getc(fstream)) != EOF) gsputc(c,gs);
      result=string_to_ground(string_so_far(gs)); fclose(fstream);}
   END_WITH_OUTPUT_TO_STRING(gs);}
  return result;
}

Grounding fraxl_parse_frame_path(Grounding under_frame)
{
  return frame_to_ground(parse_frame_path(standard_output,GFRAME(under_frame)));
}

Grounding fraxl_readline()
{
  char buf[500]; int end;
  gsgets(buf,500,standard_output); end=strlen(buf);
  if ((buf[end-1]) == '\n') buf[end-1]='\0';
  return string_to_ground(buf);
}


/* Binding streams for ground printing */

Grounding binding_output_to_file(Grounding name,Grounding thunk)
{
  generic_stream gs; Grounding result;
  gs.stream_type=file_io; gs.ptr.file=fopen(GSTRING(name),"w");
  if (NULLP(gs.ptr.file)) raise_crisis_with_details(File_Unwritable,GSTRING(name));
  FLET(generic_stream *,standard_output,&gs)
    result=apply0(thunk);
  END_FLET(standard_output);
  fclose(gs.ptr.file);
  return result;
}

Grounding binding_input_to_file(Grounding name,Grounding thunk)
{
  generic_stream gs; Grounding result;
  gs.stream_type=file_io; gs.ptr.file=fopen(GSTRING(name),"r");
  if (NULLP(gs.ptr.file)) raise_crisis_with_details(File_Unreadable,GSTRING(name));
  FLET(generic_stream *,standard_output,&gs)
    result=apply0(thunk);
  END_FLET(standard_output);
  fclose(gs.ptr.file);
  return result;
}

Grounding binding_output_to_string(Grounding thunk)
{
  Grounding result;
  {INITIALIZE_STRING_STREAM(ss,buf,128);
   {generic_stream gs; 
    gs.stream_type=string_output; gs.ptr.string_out=(&ss);
    FLET(generic_stream *,standard_output,&gs)
      result=apply0(thunk);
    END_FLET(standard_output);
    FREE_GROUND(result);
    result=string_to_ground(ss.head);}
   CLOSE_STRING_STREAM(ss);}
  return result;
}

Grounding binding_input_to_string(Grounding string,Grounding thunk)
{
  Grounding result; generic_stream gs; 
  char *real_string; real_string=GSTRING(string);
  gs.stream_type=string_input; gs.ptr.string_in=(&real_string);
  FLET(generic_stream *,standard_output,&gs)
    result=apply0(thunk);
  END_FLET(standard_output);
  return result;
}


/* Declaring functions */
 
Grounding declare_unary_function(fcn,string,type0)
     Grounding (*fcn)();
     char *string;
     Ground_Type type0;
{
   return declare_function(fcn,string,1,type0,
			   any_ground,any_ground,any_ground);
}

Grounding declare_binary_function(fcn,string,type0,type1)
     Grounding (*fcn)();
     char *string;
     Ground_Type type0;
     Ground_Type type1;
{
   return declare_function(fcn,string,2,type0,type1,any_ground,any_ground);
}


/* Useful functions */

Grounding frame_p(Grounding x)
{
  if (x==NULL) return NULL;
  if (FRAMEP(x)) return x;
  else return NULL;
}

Grounding symbol_p(Grounding x)
{
  if (x==NULL) return NULL;
  if ((ground_type(x))==symbol_ground) return x;
  else return NULL;
}

/* Primitives on frames */

Grounding fraxl_frame_home(Frame x)
{
  return (Grounding) frame_home(x);
}

Grounding clear_frame_ground(Frame x)
{
  set_ground(x,NULL);
  return NULL;
}

Grounding clear_frame_prototype(Frame x)
{
  set_prototype(x,NULL);
  return NULL;
}

Grounding fraxl_frame_name(Frame x)
{
  return string_to_ground(x->aname);
}

Grounding fraxl_make_annotation(Frame x,Grounding name)
{
  return (Grounding) use_annotation(x,GSTRING(name));
}

Grounding fraxl_make_unique_annotation(Frame x,Grounding name)
{
  return (Grounding) make_unique_annotation(x,GSTRING(name));
}

Grounding fraxl_make_ephemeral_annotation(Frame x,Grounding name)
{
  return (Grounding) make_ephemeral_annotation(x,GSTRING(name));
}

Grounding fraxl_make_alias(Frame x,Grounding name,Frame to)
{
  return (Grounding) make_alias(x,GSTRING(name),to);
}

Grounding fraxl_probe_annotation(Frame x,Grounding name)
{
  return (Grounding) probe_annotation(x,GSTRING(name));
}

Grounding fraxl_local_probe_annotation(Frame x,Grounding name)
{
  return (Grounding) local_probe_annotation(x,GSTRING(name));
}

Grounding fraxl_soft_probe_annotation(Frame x,Grounding name)
{
  Frame raw_local_probe_annotation(Frame x,char *name);
  return (Grounding) raw_local_probe_annotation(x,GSTRING(name));
}

Grounding fraxl_inherits_frame(Frame x,Grounding name)
{
  return (Grounding) inherits_frame(x,GSTRING(name));
}

Grounding fraxl_delete_annotation(Frame x)
{
  Frame hm; hm = x->home;
  if (delete_annotation(x) == True)
    return (Grounding) hm; else return NULL;
}

Grounding fraxl_move_frame(Frame frame,Frame into,Grounding string)
{
  return (Grounding) move_frame(frame,into,GSTRING(string));
}

Grounding fraxl_frame_annotations(Frame x)
{
  ENSURE_CURRENT(x);
  if (NULLP(x->appendix)) return NULL;
  else {WITH_RESULT_ACCUMULATOR(results)
	  {DO_ANNOTATIONS(a,x) accumulate_result(frame_to_ground(a),results);
	   return resolve_accumulator(results);}}
}

Grounding fraxl_frame_spinoffs(Frame x)
{
  Frame_Array *spinoffs;
  spinoffs = frame_spinoffs(x);
  if (spinoffs == NULL) return (Grounding ) NULL;
  if (spinoffs->size == 0) return (Grounding ) NULL;
  else {int i; WITH_RESULT_ACCUMULATOR(results)
	  for (i=spinoffs->size-1;i>=0;i--)
	    accumulate_result(frame_to_ground((spinoffs->elements)[i]),
			      results);
	return resolve_accumulator(results);}
}

Grounding fraxl_default_prototype_p(Grounding x,Grounding y)
{
  if (prototype_is_default(GFRAME(x),GFRAME(y)))
    return t_symbol;
  else return NULL;
}

Grounding fraxl_has_default_prototype_p(Grounding x)
{
  if (prototype_is_default(GFRAME(x),frame_prototype(GFRAME(x))))
    return t_symbol;
  else return NULL;
}

Grounding fraxl_has_home(Frame frame,Frame home)
{
  while ((frame) && (frame != home)) frame=frame->home;
  return frame_to_ground(frame);
}

Grounding fraxl_backup_root_to_file(Grounding to)
{
  backup_root_frame(the(string,to));
  return frame_to_ground(root_frame);
}

Grounding fraxl_get_frame_filename(Grounding gframe)
{
  char *name; name=frame_filename(GFRAME(gframe));
  if (name) return string_to_ground(name);
  else return NULL;
}

Grounding fraxl_set_frame_filename(Grounding gframe,Grounding gstring)
{
  set_frame_filename(GFRAME(gframe),GSTRING(gstring));
  return NULL;
}

Grounding fraxl_set_backup_path(Grounding gstring)
{
  void set_backup_path(char *path);
  set_backup_path(GSTRING(gstring));
  return gstring;
}

Grounding load_frame_file(Grounding file)
{
  return frame_to_ground(load_frame_from_file(GSTRING(file)));
}

Grounding fraxl_touch(Grounding gframe)
{
  Frame frame; frame=GFRAME(gframe);
  ENSURE_CURRENT(frame);
  TOUCH(frame);
  return frame_to_ground(frame);
}

Grounding fraxl_current(Grounding gframe)
{
  Frame frame; frame=GFRAME(gframe);
  ENSURE_CURRENT(frame);
  return frame_to_ground(frame);
}

Grounding fraxl_current_p(Grounding gframe)
{
  if (frame_current_p(GFRAME(gframe))) return t_symbol;
  else return NULL;
}

Grounding add_to_ground(Frame frame,Grounding ground)
{
  Grounding old_ground, new_ground; 
  old_ground=frame_ground(frame); new_ground=merge_results(ground,old_ground);
  if (new_ground != old_ground) set_ground(frame,new_ground);
  return NULL;
}

Grounding remove_from_ground(Frame frame,Grounding ground)
{
  Grounding old_ground, new_ground; 
  old_ground=frame_ground(frame); new_ground=zap_result(ground,old_ground);
  if (new_ground != old_ground) set_ground(frame,new_ground);
  return NULL;
}

Grounding ground_contains_p(Frame frame,Grounding ground)
{
  return find_result(ground,frame_ground(frame));
}

/* Useful functions */

#define DECLARE_TYPE_PREDICATE(name,type) \
Grounding name(Grounding x)       \
{                                 \
   if (TYPEP(x,type)) return x;   \
   else return NULL;              \
}

DECLARE_TYPE_PREDICATE(fixnump,integer_ground)
DECLARE_TYPE_PREDICATE(floatp,float_ground)
DECLARE_TYPE_PREDICATE(stringp,string_ground)
DECLARE_TYPE_PREDICATE(listp,pair_ground)
DECLARE_TYPE_PREDICATE(vectorp,vector_ground)
DECLARE_TYPE_PREDICATE(framep,frame_ground_type)
DECLARE_TYPE_PREDICATE(symbolp,symbol_ground)
DECLARE_TYPE_PREDICATE(primitivep,framer_function_ground)
DECLARE_TYPE_PREDICATE(compound_procedurep,procedure_ground)

Grounding functionp(Grounding x)
{
  if (x == NULL) return x;
  else if ((ground_type(x)) == framer_function_ground)
    return x;
  else if (((ground_type(x)) == pair_ground) && (GCAR(x) == define_symbol))
    return x;
  else return NULL;
}

Grounding numberp(Grounding x)
{
  if (x==NULL) return NULL;
  if (((ground_type(x)) == integer_ground) || ((ground_type(x)) == float_ground) ||
      ((ground_type(x)) == bignum_ground) || ((ground_type(x)) == rational_ground))
    return x;
  else return NULL;
}

Grounding integerp(Grounding x)
{
  if (x==NULL) return NULL;
  if ((ground_type(x) == integer_ground) || (ground_type(x) == bignum_ground))
    return x;
  else return NULL;
}

Grounding rationalp(Grounding x)
{
  if (x==NULL) return NULL;
  if ((ground_type(x) == integer_ground) || (ground_type(x) == bignum_ground) ||
      (ground_type(x) == rational_ground))
    return x;
  else return NULL;
}

Grounding nullp(Grounding x) 
{
  if (x == empty_list) return x;
  else return NULL;
}


/* Special forms */

/* This special form evaluates its body while suppressing frame load notifications. */
static Grounding quietly_eval_handler(Grounding expr,Grounding env)
{
  Grounding result=NULL;
  {FLET(boolean,announce_file_ops,False)
     {DO_FORMS(form,GCDR(expr)) 
	{FREE_GROUND(result); result=eval(form,env);}}
   END_FLET(announce_file_ops);}
  return result;
}

Grounding stringify_eval_handler(Grounding expr,Grounding env)
{
  Grounding result=NULL;
  {WITH_OUTPUT_TO_STRING(gs,512)
     {{FLET(generic_stream *,standard_output,gs)
	 {DO_FORMS(form,GCDR(expr)) 
	    {result=eval(form,env); FREE_GROUND(result);}}
       END_FLET(standard_output);}
      result=string_to_ground(string_so_far(gs));}
   END_WITH_OUTPUT_TO_STRING(gs);}
  USE_GROUND(result);
  return result;
}


/* Environment inquiries */

Grounding datestring()
{
  char buf[50]; struct tm *now; time_t tnow; 
  tnow=time(NULL); now=localtime(&tnow);
  sprintf(buf,"%d/%d/%d",now->tm_mon+1,now->tm_mday,now->tm_year);
  return string_to_ground(buf);
}

Grounding timestring()
{
  char buf[50]; struct tm *now; time_t tnow; 
  tnow=time(NULL); now=localtime(&tnow);
  sprintf(buf,"%d:%02d:%02d",now->tm_hour,now->tm_min,now->tm_sec);
  return string_to_ground(buf);
}


/* Collecting statisitics */

static long n_frames, n_terminal, n_current, n_vacuous, n_empty, n_grounds;

static void lazy_collect_statistics(Frame f)
{
  n_frames++; 
  if (frame_current_p(f)) 
    {n_current++; if (NOT(dump_p(f))) n_vacuous++;
     if ((NULLP(f->appendix)) || (f->appendix->size == 0)) n_terminal++; 
     if (frame_ground(f)) {DO_RESULTS(g,frame_ground(f)) n_grounds++;}
     else n_empty++;}
  {DO_SOFT_ANNOTATIONS(a,f) lazy_collect_statistics(a);}
}

static void collect_statistics(Frame f)
{
  n_frames++; 
  ENSURE_CURRENT(f); if (NOT(dump_p(f))) n_vacuous++;
  if ((NULLP(f->appendix)) || (f->appendix->size == 0)) n_terminal++;
  if (frame_ground(f))
    {DO_RESULTS(g,frame_ground(f)) n_grounds++;}
  else n_empty++;
  {DO_ANNOTATIONS(a,f) collect_statistics(a);}
}

Grounding examine_tree(Grounding g)
{
  Grounding result;
  n_frames=n_terminal=n_current=n_vacuous=n_empty=n_grounds=0;
  collect_statistics(GFRAME(g));
  {WITH_OUTPUT_TO_STRING(gs,100)
     {gsprintf(gs,"%ld frames, ",n_frames);
      gsprintf(gs,"%ld current, ",n_frames);
      gsprintf(gs,"%ld terminal, ",n_terminal);
      gsprintf(gs,"%ld vacuous, ",n_vacuous);
      gsprintf(gs,"%ld empty, ",n_empty);
      gsprintf(gs,"%ld grounds",n_terminal);
      result=string_to_ground(string_so_far(gs));}
   END_WITH_OUTPUT_TO_STRING(gs);}
  return result;
}

Grounding lazy_examine_tree(Grounding g)
{
  Grounding result;
  n_frames=n_terminal=n_current=n_vacuous=n_empty=n_grounds=0;
  lazy_collect_statistics(GFRAME(g));
  {WITH_OUTPUT_TO_STRING(gs,100)
     {gsprintf(gs,"%ld frames, ",n_frames);
      gsprintf(gs,"%ld current, ",n_current);
      gsprintf(gs,"%ld terminal, ",n_terminal);
      gsprintf(gs,"%ld empty, ",n_empty);
      gsprintf(gs,"%ld grounds",n_terminal);
      result=string_to_ground(string_so_far(gs));}
   END_WITH_OUTPUT_TO_STRING(gs);}
  return result;
}


Grounding unary_pretty_printer(Grounding gnd)
{
  pprint_ground(standard_output,gnd,0,60);
  return gnd;
}

Grounding get_summary_from_image(Frame frame);
Grounding fraxl_get_image_index(Grounding gframe);
void open_framer_image_file(char *name);
extern boolean active_image_p;

Grounding fraxl_write_framer_image(Grounding name)
{
  write_framer_image(GSTRING(name));
  return NULL;
}

Grounding fraxl_open_framer_image(Grounding name)
{
  open_framer_image_file(GSTRING(name));
  return NULL;
}

Grounding fraxl_examine_framer_image(Grounding name)
{
  open_framer_image_file(GSTRING(name));
  active_image_p=False;
  return NULL;
}

Grounding fraxl_pretty_printer(Grounding input,Grounding width,Grounding highlight)
{
  pprinter(input,standard_output,0,GINTEGER(width),highlight);
  return input;
}

Grounding set_default_display_width(Grounding new_width);
Grounding fraxl_output_description(Grounding frame);

Grounding describe_grounded_frame(Grounding x);
Grounding make_comment(Grounding x);

/* Initializations */

Frame find_modified_annotation(Frame f);
Frame align_prototypes(Frame f,Frame p);

void init_framer_functions()
{
  void init_evaluator(void);
  void init_fraxl_numerics(void);
  init_evaluator();

  declare_keyword("QUIETLY",quietly_eval_handler);
  declare_keyword("STRINGIFY",stringify_eval_handler);
  /* FRAXL primitive declarations */
  declare_unary_function(numberp,"number?",any_ground);
  declare_unary_function(rationalp,"rational?",any_ground);
  declare_unary_function(integerp,"integer?",any_ground);
  declare_unary_function(fixnump,"fixnum?",any_ground);
  declare_unary_function(floatp,"float?",any_ground);

  declare_binary_function(fraxl_equal_p,"equal?",any_ground,any_ground);
  declare_binary_function(fraxl_equal_p,"equal",any_ground,any_ground);
  declare_binary_function(fraxl_eqv_p,"eqv?",any_ground,any_ground);
  declare_binary_function(fraxl_eqv_p,"eqv",any_ground,any_ground);
  declare_binary_function(fraxl_eqv_p,"eql?",any_ground,any_ground);
  declare_binary_function(fraxl_eqv_p,"eql",any_ground,any_ground);
  declare_binary_function(fraxl_eq_p,"eq?",any_ground,any_ground);
  declare_binary_function(fraxl_eq_p,"eq",any_ground,any_ground);
  declare_binary_function(fraxl_string_equal_p,"string=",any_ground,any_ground);
  declare_binary_function(fraxl_alpha_less_p,"alpha<",any_ground,any_ground);

  declare_unary_function(framep,"frame?",any_ground);
  declare_unary_function(stringp,"string?",any_ground);
  declare_unary_function(symbolp,"symbol?",any_ground);
  declare_unary_function(listp,"list?",any_ground);
  declare_unary_function(listp,"pair?",any_ground);
  declare_unary_function(nullp,"null?",any_ground);
  declare_unary_function(vectorp,"vector?",any_ground);
  declare_unary_function(primitivep,"primitive?",any_ground);
  declare_unary_function(compound_procedurep,"compound-procedure?",any_ground);

  /* Sequence functions */
  declare_unary_function(generic_all_elts,"elements",any_ground);
  declare_binary_function(generic_position,"position",any_ground,any_ground);
  declare_unary_function(generic_all_elts,"all-elements",any_ground);
  declare_unary_function(generic_length,"length",any_ground);
  declare_binary_function(generic_elt,"elt",any_ground,integer_ground);

  declare_binary_function(cons_pair,"cons",any_ground,any_ground);
  declare_unary_function(pair_car,"car",pair_ground);
  declare_unary_function(pair_cdr,"cdr",pair_ground);
  declare_unary_function(pair_cadr,"cadr",pair_ground);
  declare_unary_function(pair_cddr,"cddr",pair_ground);
  declare_lexpr(list_lexpr,"list");
  declare_lexpr(vector_lexpr,"vector");
  declare_lexpr(append_lexpr,"append");
  declare_unary_function(list_to_vector,"list->vector",pair_ground);
  declare_unary_function(vector_to_list,"vector->list",vector_ground);
  declare_binary_function(fraxl_mapcar,"mapcar",any_ground,pair_ground);
  declare_unary_function(list_elements,"list-elements",pair_ground);
  declare_binary_function(in_list,"member",any_ground,pair_ground);
  declare_unary_function(generic_reverse,"reverse",any_ground);
  declare_binary_function(remove_from_list,"remove",any_ground,pair_ground);
  declare_binary_function(listref,"list-ref",pair_ground,integer_ground);
  declare_binary_function(listcdrref,"list-tail",pair_ground,integer_ground);
  declare_unary_function(quotify,"quotify",any_ground);
  
  declare_binary_function(gvref,"vref",vector_ground,integer_ground);
  declare_binary_function(gvref,"vector-ref",vector_ground,integer_ground);
  declare_unary_function(gvsize,"vector-size",vector_ground);

  declare_unary_function(frame_fcn(frame_prototype_fn),"frame-prototype",frame_ground_type);
  declare_unary_function(frame_fcn(frame_prototype_fn),"prototype",frame_ground_type);
  declare_binary_function(frame_fcn(set_prototype),"set-frame-prototype",
			  frame_ground_type,frame_ground_type);
  declare_binary_function(fraxl_default_prototype_p,"default-prototype?",
			  frame_ground_type,frame_ground_type);
  declare_unary_function(fraxl_has_default_prototype_p,"has-default-prototype?",
			 frame_ground_type);
  declare_binary_function(frame_fcn(set_prototype),"set-prototype",
			  frame_ground_type,frame_ground_type);
  declare_binary_function(frame_fcn(set_prototypes),"set-prototypes",
			  frame_ground_type,frame_ground_type);
  declare_binary_function(frame_fcn(align_prototypes),"align-prototypes",
			  frame_ground_type,frame_ground_type);
  declare_unary_function(clear_frame_prototype,"clear-frame-prototype",frame_ground_type);
  declare_unary_function(clear_frame_prototype,"clear-prototype",frame_ground_type);

  declare_unary_function(frame_ground_fn,"frame-ground",frame_ground_type);
  declare_unary_function(frame_ground_fn,"ground",frame_ground_type);
  declare_binary_function(frame_fcn(set_ground),"set-frame-ground",
			  frame_ground_type,any_ground);
  declare_binary_function(frame_fcn(set_ground),"set-ground",
			  frame_ground_type,any_ground);
  declare_unary_function(clear_frame_ground,"clear-frame-ground",frame_ground_type);
  declare_unary_function(clear_frame_ground,"clear-ground",frame_ground_type);
  declare_binary_function(add_to_ground,"add-to-ground",
			  frame_ground_type,any_ground);
  declare_binary_function(remove_from_ground,"remove-from-ground",
			  frame_ground_type,any_ground);
  declare_binary_function(ground_contains_p,"ground-contains?",
			  frame_ground_type,any_ground);

  declare_unary_function(fraxl_in_home,"in-home",frame_ground_type);
  declare_lexpr(fraxl_proto_equals,"prototype=");
  declare_lexpr(fraxl_ground_equals,"ground=");
  declare_unary_function(fraxl_frame_home,"frame-home",frame_ground_type);
  declare_unary_function(fraxl_frame_name,"frame-name",frame_ground_type);
  declare_unary_function(fraxl_frame_annotations,"frame-annotations",
			 frame_ground_type);
  declare_unary_function(fraxl_frame_spinoffs,"frame-spinoffs",frame_ground_type);
  declare_binary_function(fraxl_make_annotation,"use-annotation",
			  frame_ground_type,string_ground);
  declare_binary_function(fraxl_make_annotation,"make-annotation",
			  frame_ground_type,string_ground);
  declare_binary_function(fraxl_make_unique_annotation,"make-unique-annotation",
			  frame_ground_type,string_ground);
  declare_binary_function(fraxl_make_ephemeral_annotation,
			  "make-ephemeral-annotation",frame_ground_type,string_ground);
  declare_function(fraxl_make_alias,"make-alias",3,
		   frame_ground_type,string_ground,frame_ground_type,any_ground);
  declare_binary_function(fraxl_probe_annotation,"probe-annotation",
			  frame_ground_type,string_ground);
  declare_binary_function(fraxl_local_probe_annotation,"local-probe-annotation",
			  frame_ground_type,string_ground);
  declare_binary_function(fraxl_soft_probe_annotation,"soft-probe-annotation",
			  frame_ground_type,string_ground);
  declare_unary_function(fraxl_delete_annotation,"delete-annotation",
			 frame_ground_type);
  declare_binary_function(((Grounding (*)()) copy_frame),"copy-frame",
			  frame_ground_type,frame_ground_type);
  declare_function(fraxl_move_frame,"move-frame",3,
		   frame_ground_type,frame_ground_type,string_ground,any_ground);
  declare_binary_function(fraxl_inherits_frame,"inherits-frame",
	                  frame_ground_type,frame_ground_type);
  declare_binary_function(frame_fcn(has_prototype),"has-prototype",
			  frame_ground_type,frame_ground_type);
  declare_binary_function(fraxl_has_home,"has-home",
			  frame_ground_type,frame_ground_type);
  declare_binary_function(fraxl_set_frame_filename,"set-filename",
			  frame_ground_type,string_ground);
  declare_unary_function(fraxl_get_frame_filename,"frame-filename",frame_ground_type);
  declare_unary_function(frame_fcn(backup_frame),"backup-frame",frame_ground_type);
  declare_unary_function(fraxl_touch,"touch",frame_ground_type);
  declare_unary_function(fraxl_current,"current",frame_ground_type);
  declare_unary_function(fraxl_set_backup_path,"set-backup-path",string_ground);
  declare_unary_function(fraxl_backup_root_to_file,"backup-root-frame",string_ground);
  declare_unary_function(fraxl_backup_root_to_file,"backup-all",string_ground);
  declare_unary_function(fraxl_write_framer_image,"write-image",string_ground);
  declare_unary_function(fraxl_open_framer_image,"open-image",string_ground);
  declare_unary_function(fraxl_examine_framer_image,"examine-image",string_ground);
  declare_unary_function(get_summary_from_image,"GET-SUMMARY",frame_ground_type);
  declare_unary_function(fraxl_get_image_index,"GET-IMAGE-INDEX",frame_ground_type);

  declare_unary_function(load_frame_file,"load-frame-from-file",string_ground);
  declare_unary_function(load_frame_file,"load-frame",string_ground);

  declare_unary_function(make_bitset,"make-bitset",frame_ground_type);
  declare_binary_function
    (fraxl_initialize_bitset,"initialize-bitset",bitset_ground,any_ground);
  declare_binary_function
    (fraxl_for_bitset,"for-bitset",procedure_ground,bitset_ground);
  declare_unary_function
    (fraxl_gather_subset,"gather-bitset",bitset_ground);

  declare_binary_function
    (in_bitset,"in-bitset",frame_ground_type,bitset_ground);
  declare_binary_function
    (add_to_bitset,"add-to-bitset",frame_ground_type,bitset_ground);
  declare_binary_function
    (remove_from_bitset,"remove-from-bitset",frame_ground_type,bitset_ground);
  declare_binary_function(bitset_and,"bitset-and",bitset_ground,bitset_ground);
  declare_binary_function(bitset_or,"bitset-or",bitset_ground,bitset_ground);
  declare_unary_function(bitset_not,"bitset-not",bitset_ground);

  declare_function(fraxl_substring,"substring",3,
		   string_ground,integer_ground,integer_ground,any_ground);
  declare_binary_function(fraxl_find_substring,"find-substring",
			  string_ground,string_ground);
  declare_unary_function(fraxl_capitalized_p,"capitalized?",string_ground);
  declare_unary_function(fraxl_intern,"string->symbol",string_ground);
  declare_unary_function(fraxl_symbol_name,"symbol->string",symbol_ground);
  declare_unary_function(fraxl_intern,"intern",string_ground);

  declare_unary_function(print_ground_to_stdout,"print-ground",any_ground);
  declare_unary_function(print_ground_to_stdout,"write",any_ground);
  declare_unary_function(display_ground_to_stdout,"display",any_ground);
  declare_binary_function(pretty_printer,"pprint+",any_ground,integer_ground);
  declare_unary_function(unary_pretty_printer,"pprint",any_ground);
  declare_fcn3(fraxl_pretty_printer,"prettily",any_ground,integer_ground,any_ground);
  declare_unary_function(print_result_to_stdout,"print-result",any_ground);
  declare_lexpr(printout_lexpr,"printout"); declare_lexpr(printoutp_lexpr,"printout+");
  declare_unary_function(print_string_ground_to_stdout,"print-string",
	                 string_ground);
  declare_function(newline_to_stdout,"newline",0,
	           any_ground,any_ground,any_ground,any_ground);
  declare_unary_function(set_default_display_width,"set-width",integer_ground);

  declare_unary_function(binding_output_to_string,"stringing-out",any_ground);
  declare_binary_function(binding_output_to_file,"filing-out",string_ground,any_ground);
  declare_binary_function(binding_input_to_string,"stringing-in",
			  string_ground,any_ground);
  declare_unary_function(file_as_string,"file-as-string",string_ground);
  declare_binary_function(binding_input_to_file,"filing-in",string_ground,any_ground);

  declare_function(timestring,"TIMESTRING",0,
	           any_ground,any_ground,any_ground,any_ground);
  declare_function(datestring,"DATESTRING",0,
	           any_ground,any_ground,any_ground,any_ground);

  declare_unary_function(make_comment,"MAKE-COMMENT",any_ground);
  declare_binary_function(point_x,"MAKE-POINT",integer_ground,integer_ground);
  declare_unary_function(point_x,"POINT-X",short_point_ground);
  declare_unary_function(point_y,"POINT-Y",short_point_ground);

  declare_function(parse_ground_from_stdin,"parse-ground",0,
	           any_ground,any_ground,any_ground,any_ground);
  declare_function(parse_ground_from_stdin,"read",0,
	           any_ground,any_ground,any_ground,any_ground);
  declare_function(fraxl_readline,"read-line",0,
	           any_ground,any_ground,any_ground,any_ground);
  declare_unary_function(fraxl_parse_frame_path,"parse-frame",frame_ground_type);
  init_fraxl_numerics();
  declare_unary_function(frame_fcn(find_modified_annotation),
			 "FIND-MODIFICATION",frame_ground_type);
  declare_unary_function(examine_tree,"STATS",frame_ground_type);
  declare_unary_function(lazy_examine_tree,"LAZY-STATS",frame_ground_type);
}

/*
  Local variables: ***
  compile-command: "../etc/compile-framer" ***
  tags-file-name: "../sources/TAGS" ***
  End: **
*/

