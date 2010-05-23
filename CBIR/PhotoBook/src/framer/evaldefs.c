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
  This file implements the FRAXL evaluator.
  *************************************************************************/

static char rcsid[] =
  "$Header: /mas/framer/sources/RCS/evaldefs.c,v 1.17 1994/01/28 23:41:30 rnewman Exp $";

#include <time.h>
#include "framer.h"
#include "fraxl.h"
#include "internal/private.h"
#include "internal/eval.h"

/* The array used for zip coding */
extern Frame_Array *zip_codes;
/* Various exceptions */
exception FRAXL_Error="Unrecoverable FRAXL error",
  FRAXL_Exception="Recoverable FRAXL exception";


/* Handlers for special forms */

/* Handler for IF */
static Grounding if_eval_handler(Grounding expr,Grounding env)
{
  Grounding result, ignore, test, body;
  POP_FORM(ignore,expr);		/* Ignore the IF */
  POP_FORM(test,expr);
  result = eval(test,env);
  if (result != NULL)
    {FREE_GROUND(result); POP_FORM(body,expr);
     return tail_eval(body,env);}
  else {POP_FORM(body,expr); 
	if (expr == empty_list) return NULL;
	else {POP_FORM(body,expr);}
	return tail_eval(body,env);}
}

static Grounding when_eval_handler(Grounding expr,Grounding env)
{
  Grounding ignore, test, test_form;
  POP_FORM(ignore,expr);		/* Ignore the WHEN */
  POP_FORM(test_form,expr);
  test = eval(test_form,env);
  if (test != NULL)
    {FREE_GROUND(test); 
     {Grounding final; DO_SIDE_EFFECT_FORMS(form,expr,final) eval(form,env);
      return tail_eval(final,env);}}
  else return NULL;
}

static Grounding unless_eval_handler(Grounding expr,Grounding env)
{
  Grounding ignore, test, test_form;
  POP_FORM(ignore,expr);		/* Ignore the UNLESS */
  POP_FORM(test_form,expr);
  test = eval(test_form,env);
  if (test == NULL)
    {Grounding final; DO_SIDE_EFFECT_FORMS(form,expr,final) eval(form,env);
     return tail_eval(final,env);}
  else {FREE_GROUND(test); return NULL;}
}

static Grounding cond_eval_handler(Grounding expr,Grounding env)
{
  Grounding ignore;
  POP_FORM(ignore,expr);		/* Ignore the COND */
  {DO_FORMS(cond_clause,expr)
     {Grounding test, test_expr; POP_FORM(test_expr,cond_clause);
      if ((test_expr == else_symbol) || (test_expr == t_symbol)) test=t_symbol;
      else test=eval(test_expr,env);
      if (test) 
	{Grounding final; DO_SIDE_EFFECT_FORMS(form,cond_clause,final)
	   side_eval(form,env);
	 FREE_GROUND(test);
	 return tail_eval(final,env);}}}
  return NULL;
}

/* Handler for NOT */
static Grounding not_eval_handler(Grounding expr,Grounding env)
{
  Grounding result, test;
  POP_FORM(test,expr);		/* Ignore the NOT */
  POP_FORM(test,expr);
  result = eval(test,env);
  if (result != NULL)
    {FREE_GROUND(result); return NULL;}
  else return USED_GROUND(t_symbol);
}

/* Handler for AND */
static Grounding and_eval_handler(Grounding expr,Grounding env)
{
  Grounding result=NULL, ignore, final;
  POP_FORM(ignore,expr);		/* Ignore the NOT */
  {DO_SIDE_EFFECT_FORMS(test,expr,final)
     {result=eval(test,env);
      if (NULLP(result)) return result;
      else {FREE_GROUND(result);}}
   return tail_eval(final,env);}
}

/* Handler for OR */
static Grounding or_eval_handler(Grounding expr,Grounding env)
{
  Grounding result=NULL, ignore, final;
  POP_FORM(ignore,expr);		/* Ignore the NOT */
  {DO_SIDE_EFFECT_FORMS(test,expr,final)
     {FREE_GROUND(result); result=eval(test,env);
      if (NOT(NULLP(result))) return result;}
   return tail_eval(final,env);}
}


Grounding maplist AP((Grounding (*fcn)(),Grounding list));

/* We don't handle ,@ (yet), but this is pretty straightforward */
static Grounding fraxl_eval_backquote(Grounding expr,Grounding env)
{
  if (expr == NULL) return NULL;
  else if (expr == empty_list) return USED_GROUND(empty_list);
  else if (ground_type(expr) != pair_ground) return USED_GROUND(expr);
  else if (GCAR(expr) == unquote_symbol)
    /* If you process an unquote, call the evaluator */
    {Grounding arg; POP_FORM(arg,expr); POP_FORM(arg,expr);
     return eval(arg,env);}
  /* Otherwise, construct the list by calling yourself recursively. */
  else {Grounding result, *rptr; result=empty_list; rptr = (&result);
	USE_GROUND(result);
	{DO_LIST(elt,expr) {TCONS(fraxl_eval_backquote(elt,env),rptr);}} 
	return result;}
}

/* Calls fraxl_eval_backquote */
static Grounding backquote_eval_handler(Grounding expr,Grounding env)
{
  Grounding arg;
  POP_FORM(arg,expr); POP_FORM(arg,expr);
  return fraxl_eval_backquote(arg,env);
}

/* Handling QUOTE: simplicity itself */
static Grounding quote_eval_handler(Grounding expr,Grounding env)
{
  Grounding arg;
  POP_FORM(arg,expr); POP_FORM(arg,expr);
  USE_GROUND(arg);
  return arg;
}

/* Handling DEFINE: make a procedure. */
static Grounding define_eval_handler(Grounding expr,Grounding env)
{
  Grounding args, temp, proc; POP_FORM(args,expr); /* Ignore `DEFINE' */
  proc=close_procedure(expr,env); USE_GROUND(proc);
  return proc;
}

/* Handling LAMBDA: make a procedure.
   (This could be done as a macro, but traces look better if we do this silently) */
static Grounding lambda_eval_handler(Grounding expr,Grounding env)
{
  Grounding args, proc; POP_FORM(args,expr); /* Ignore `LAMBDA' */
  POP_FORM(args,expr); args=cons_pair(NULL,args);
  proc=make_procedure(args,expr,env); USE_GROUND(proc);
  return proc;
}

/* Handling BEGIN: apply side_eval and do a tail call */
static Grounding begin_eval_handler(Grounding expr,Grounding env)
{
  Grounding final, result;
  Grounding body = GCDR(expr);
  if (body == empty_list) return NULL;
  {DO_SIDE_EFFECT_FORMS(form,body,final) side_eval(form,env);}
  result=tail_eval(final,env);
  return result;
}

/* Handling EVAL-HERE: Do a double eval */
static Grounding eval_here_handler(Grounding expr,Grounding env)
{
  Grounding form, form_to_eval, result;
  POP_FORM(form,expr); /* Ignored */ POP_FORM(form,expr);
  form_to_eval=eval(form,env); result=eval(form_to_eval,env);
  FREE_GROUND(form_to_eval);
  return result;
}

/* Handling INCLUDE: makes a bunch of other procedures callable */

static void do_inclusion(Grounding source,Grounding env)
{
  {DO_RESULTS(elt,source)
     if (FRAMEP(elt)) do_inclusion(get_inherited_ground(GFRAME(elt)),env);
     else if (TYPEP(elt,procedure_ground))
       {Grounding name; name=GCAR(proc_args(elt)); 
	add_binding(name,close_procedure(proc_text(elt),NULL),env);}
     else ground_error(Type_Error,"Don't know how to include ",elt);
   return;}
}

static Grounding include_eval_handler(Grounding expr,Grounding env)
{
  Grounding source; POP_FORM(source,expr); POP_FORM(source,expr);
  if (env == empty_list) raise_crisis("Can't include at top level");
  source=eval(source,env);
  do_inclusion(source,env);
  FREE_GROUND(source);
  return NULL;
}

/* Handling %XTRACE: Slightly hairy */
static Grounding xtrace_eval_handler(Grounding expr,Grounding env)
{
  Grounding form, result;
  POP_FORM(form,expr); POP_FORM(form,expr);
  if (NOT(TYPEP(form,pair_ground)))
    if (TYPEP(form,symbol_ground))
      {Grounding value; value=lookup_variable(form,env);
       output_trace2("Evaluation of",form,"yielded",value);
       return value;}
    else {output_trace("Self-evaluation of",form);
	  USE_GROUND(form); return form;}
  else if ((TYPEP(GCAR(form),symbol_ground)) && (KEYWORD_P(GCAR(form))))
    {output_trace("Evaluating",form);
     fraxl_debugger("Tracepoint",current_stack_frame);
     result=eval(form,env);
     output_trace2("Evaluation of",form,"yielded",result);
     return result;}
  else {USE_GROUND(form); FREE_GROUND(current_stack_frame->expr);
	current_stack_frame->expr=form; 
	output_trace("Evaluating",form);
	setup_combination(current_stack_frame);
	if (current_stack_frame->choices)
	  output_trace2("Evaluation of",form,"applies",current_stack_frame->choices);
	else if (current_stack_frame->choice)
	  output_trace2("Evaluation of",form,"applies",current_stack_frame->choice);
	else gsputs("has been snipped",standard_output);
	okay_to_tail_call=False; 
	fraxl_debugger("Tracepoint",current_stack_frame);
	if (current_stack_frame->choices)
	  {Grounding temp;
	   result=nd_applier(current_stack_frame->choices,empty_list);
	   temp=reverse_list(current_stack_frame->choices);
	   output_trace2("Returning",result,"from application of",temp);
	   FREE_GROUND(current_stack_frame->choices); FREE_GROUND(temp);
	   current_stack_frame->choices=NULL;}
	else if (current_stack_frame->choice)
	  {result=d_apply(current_stack_frame->choice);
	   output_trace2("Returning",result,"from application of",
			 current_stack_frame->choice);
	   FREE_GROUND(current_stack_frame->choice);
	   current_stack_frame->choice=NULL;}
	else result=NULL;
	return result;}
}

/* Handling %XTRACE: Slightly hairy */
static Grounding step_eval_handler(Grounding expr,Grounding env)
{
  Grounding form;
  POP_FORM(form,expr); POP_FORM(form,expr);
  current_stack_frame->stepping=True;
  return tail_eval(form,env);
}

static Grounding assert_eval_handler(Grounding expr,Grounding env)
{
  Grounding form, result; POP_FORM(form,expr); POP_FORM(form,expr);
  result=eval(form,env);
  if (result) return result;
  else {fraxl_debugger("Failed assertion",current_stack_frame);
	return result;}
}

Grounding error_lexpr(Grounding args)
{
  {DO_LIST(elt,args) 
     if (TYPEP(elt,string_ground)) gsputs(GSTRING(elt),standard_output);
     else print_ground(standard_output,elt);}
  if (standard_output->stream_type == file_io) fflush(standard_output->ptr.file);
  fraxl_debugger("Error",current_stack_frame);
  raise_crisis(FRAXL_Error);
  return NULL; /* Never reached */
}

Grounding exception_lexpr(Grounding args)
{
  {DO_LIST(elt,args) 
     if (TYPEP(elt,string_ground)) gsputs(GSTRING(elt),standard_output);
     else print_ground(standard_output,elt);}
  if (standard_output->stream_type == file_io) fflush(standard_output->ptr.file);
  fraxl_debugger("Exception",current_stack_frame);
  return NULL; /* Never reached */
}


/* Handling LETREC: This turns recursive environment pointers into NULLs */
static void fix_envs(Grounding form,Grounding env)
{
  {DO_RESULTS(r,form)
     if (TYPEP(r,procedure_ground))
       {if ((proc_env(r)) == env)
	  {proc_env(r)=NULL; FREE_GROUND(env);}}
     else if (TYPEP(r,pair_ground))
       {DO_LIST(e,r) fix_envs(e,env);}
     else if (TYPEP(r,vector_ground))
       {DO_VECTOR(e,r) fix_envs(e,env);}}
}

/* This actually handles LETREC */
static Grounding letrec_eval_handler(Grounding expr,Grounding env)
{
  Grounding bindings, new_env, final, result;
  POP_FORM(bindings,expr); POP_FORM(bindings,expr);
  new_env=cons_pair(cons_pair(empty_list,empty_list),env); USE_GROUND(new_env);
  {DO_FORMS(b,bindings)
     {Grounding var, val; POP_FORM(var,b); POP_FORM(val,b);
      val=eval(val,new_env); fix_envs(val,new_env);
      add_binding(var,val,new_env);}
   {UNWIND_PROTECT
      {DO_SIDE_EFFECT_FORMS(f,expr,final) side_eval(f,new_env);
       result=tail_eval(final,new_env);}
      ON_UNWIND
	{FREE_GROUND(new_env);}
      END_UNWIND;}
   return result;}
}


/* Non-deterministic special forms */

/* Handling AMB */
static Grounding amb_eval_handler(Grounding expr,Grounding env)
{
  Grounding result;
  {WITH_RESULT_ACCUMULATOR(ra)
     {DO_FORMS(expression,GCDR(expr))
	{Grounding results; results=eval(expression,env);
	 {DO_RESULTS(r,results) accumulate_result(r,ra);}
	 FREE_GROUND(results);}}
   result=resolve_accumulator(ra); USE_GROUND(result);}
  return result;
}

/* Handling GATHER */
static Grounding gather_eval_handler(Grounding expr,Grounding env)
{
  Grounding output, fcn, results, init;
  POP_FORM(fcn,expr); POP_FORM(fcn,expr); 
  POP_FORM(results,expr); POP_FORM(init,expr);
  fcn=eval(fcn,env); results=eval(results,env); init=eval(init,env);
  output=gather_results(fcn,results,init);
  FREE_GROUND(fcn); FREE_GROUND(results); FREE_GROUND(init);
  return output;
}

/* Handling PREFER */
static Grounding prefer_eval_handler(Grounding expr,Grounding env)
{
  Grounding result=NULL, ignore, final;
  POP_FORM(ignore,expr);		/* Ignore the NOT */
  if (expr == empty_list)  /* (prefer) with no arguments */
    return NULL;
  {DO_SIDE_EFFECT_FORMS(test,expr,final)
     {FREE_GROUND(result); result=eval(test,env);
      if (NOT(NULLP(result))) return result;}
   return tail_eval(final,env);}
}

Grounding intersect_result_sets(Grounding set1,Grounding set2)
{
  Grounding non_frames=NULL; int search_bit; search_bit=grab_search_bit();
  {DO_RESULTS(elt,set1)
     if (FRAMEP(elt)) set_search_bit(((Frame) elt),search_bit);
     else non_frames=cons_pair(elt,non_frames);}
  {WITH_RESULT_ACCUMULATOR(results)
     {DO_RESULTS(elt,set2)  
	if (FRAMEP(elt))
	  {if (search_bit_p(((Frame) elt),search_bit)) 
	     accumulate_result(elt,results);}
	else if (in_list(elt,non_frames)) accumulate_result(elt,results);}
   {DO_RESULTS(elt,set1)
      if (FRAMEP(elt)) clear_search_bit(((Frame) elt),search_bit);
    release_search_bit(search_bit);
    if (non_frames != empty_list) {FREE_GROUND(non_frames);}}
   return resolve_accumulator(results);}
}

Grounding difference_result_sets(Grounding set1,Grounding set2)
{
  Grounding non_frames=NULL; int search_bit; search_bit=grab_search_bit();
  {DO_RESULTS(elt,set2)
     if (FRAMEP(elt)) set_search_bit(((Frame) elt),search_bit);
     else non_frames=cons_pair(elt,non_frames);}
  {WITH_RESULT_ACCUMULATOR(results)
     {DO_RESULTS(elt,set1)  
	if (FRAMEP(elt))
	  {if (NOT(search_bit_p(((Frame) elt),search_bit)))
	     accumulate_result(elt,results);}
	else if (NOT(in_list(elt,non_frames)))
	  accumulate_result(elt,results);}
   {DO_RESULTS(elt,set2)
      if (FRAMEP(elt)) clear_search_bit(((Frame) elt),search_bit);
    release_search_bit(search_bit);
    if (non_frames != empty_list) {FREE_GROUND(non_frames);}}
   return resolve_accumulator(results);}
}

/* Handling INTERSECT */
static Grounding intersect_eval_handler(Grounding expr,Grounding env)
{
  Grounding initial_set, results=NULL;
  {Grounding ignore, first_form; 
   POP_FORM(ignore,expr); POP_FORM(first_form,expr); 
   results=initial_set=eval(first_form,env);}
  {DO_FORMS(form,expr)
     {Grounding set; set=eval(form,env);
      results=intersect_result_sets(set,results); FREE_GROUND(set);}}
  USE_GROUND(results); FREE_GROUND(initial_set);
  return results;
}

static Grounding difference_eval_handler(Grounding expr,Grounding env)
{
  Grounding initial_set, results=NULL;
  {Grounding ignore, first_form; 
   POP_FORM(ignore,expr); POP_FORM(first_form,expr); 
   results=initial_set=eval(first_form,env);}
  {DO_FORMS(form,expr)
     {Grounding new_results, set; set=eval(form,env);
      new_results=difference_result_sets(results,set);
      if (new_results != results) 
	{USE_GROUND(new_results); FREE_GROUND(results); results=new_results;}
      FREE_GROUND(set);}}
  return results;
}


/* Sorting result sets */

/* This global variable lets us use the ANSI qsort library, this
   is a dynamically bound sort predicate used by a C sort function. */
static Grounding sort_pred;

#if (__TURBOC__) /* TURBOC is picky about types and this placates it */
/* This refers to sort_pred to compute a comparison */
static int sort_compare(const void *v1,const void *v2)
{
  int result; Grounding comp; void **s1; void **s2;
  s1= (void * *) v1; s2= (void * *) v2;
  comp=apply2(sort_pred,(Grounding) *s1,(Grounding) *s2);
  if (comp) result=1; else result = -1;
  FREE_GROUND(comp); return result;
}
#else
/* This refers to sort_pred to compute a comparison */
static int sort_compare(Grounding *v1,Grounding *v2)
{
  int result; Grounding comp;
  comp=apply2(sort_pred,*v1,*v2);
  if (comp) result=1; else result = -1;
  FREE_GROUND(comp); return result;
}
#endif

/* Handling SORTED */
/* We basically put a set of results into a vector, bind the
   predicate to a global variable and then call qsort with a
   procedure that refers to the global variable to compare elements. */
static Grounding sorted_eval_handler(Grounding expr,Grounding env)
{
  Grounding form, pred, results, sorted, *v; int i=0, size=0; 
  POP_FORM(form,expr); POP_FORM(form,expr); POP_FORM(pred,expr);
  results=eval(form,env); pred=eval(pred,env); sorted=empty_list;
  {DO_RESULTS(r,results) size++;} ALLOCATE(v,Grounding,size);
  {DO_RESULTS(r,results) v[i++]=r;}
  {FLET(Grounding,sort_pred,pred) /* This will be used by sort_compare */
     qsort(v,size,sizeof(Grounding),sort_compare);
   END_FLET(sort_pred);}
  {DO_TIMES(i,size) sorted=cons_pair(v[i],sorted);}
  USE_GROUND(sorted); FREE_GROUND(results); FREE_GROUND(pred); free(v);
  return sorted;
}

/* Handler primitives */

static Frame global_handlers=NULL;
static char *commands_string;

/* Get the message name from a Grounding, whether the Grounding is
   a string or a symbol. If it's neither, raise an error. */
static char *grounding_to_message (Grounding g)
    {
    if (TYPEP(g, string_ground)) return GSTRING(g);
    else if (TYPEP(g, symbol_ground)) return SYMBOL_NAME(g);
    else ground_error(Type_Error,
          "Not a valid message identifier (string or symbol)", g);
    }

/* get_handlers gets the handlers frame for a frame, or the global table otherwise. */
Frame get_handlers(Frame frame)
{
  if (probe_annotation(frame,commands_string) != NULL)
    return probe_annotation(frame,commands_string);
  else return global_handlers;
}

/* This gets the handler function for a message to a frame. */
Grounding get_handler(Frame frame,char *message)
{
  Frame handler_frame; 
  handler_frame=probe_annotation(get_handlers(frame),message);
  if (handler_frame) return get_inherited_ground(handler_frame);
  else return NULL;
}

/* This gets the frame describing the handler function for a message to a frame. */
Frame get_handler_frame(Frame frame,char *message)
{
  return probe_annotation(get_handlers(frame),message);
}

/* This is a fraxl primitive for getting the frame describing the handler function
   for a message to a frame. */
Grounding fraxl_get_handler_frame(Grounding frame,Grounding message)
{
  return (Grounding) probe_annotation(get_handlers(GFRAME(frame)), grounding_to_message(message));
}

/* This is a fraxl primitive for getting the handler function for a message to a frame. */
Grounding fraxl_get_handler(Grounding frame,Grounding message)
{
  return get_handler(GFRAME(frame), grounding_to_message(message));
}


/* User level message sending */

Grounding send0(Frame to,char *message)
{
  Grounding method; method=get_handler(to,message);
  if (NULLP(method)) return NULL;
  else return apply1(method,frame_to_ground(to));
}

Grounding send1(Frame to,char *message,Grounding arg1)
{
  Grounding method; method=get_handler(to,message);
  if (NULLP(method)) return NULL;
  else return apply2(method,frame_to_ground(to),arg1);
}

Grounding send2(Frame to,char *message,Grounding arg1,Grounding arg2)
{
  Grounding method; method=get_handler(to,message);
  if (NULLP(method)) return NULL;
  else return apply3(method,frame_to_ground(to),arg1,arg2);
}


/* Message Passing */

Grounding dapply_handler(Grounding rail)
{
  if (okay_to_tail_call) return d_tail_apply(rail);
  else {Grounding result; USE_GROUND(rail); result=d_apply(rail);
	FREE_GROUND(rail); 
	if ((result) && (NOT(FRAMEP(result)))) ref_count(result)--; 
	return result;}
}

Grounding ndapply_handler(Grounding rail)
{
  if (okay_to_tail_call) return nd_tail_apply(rail);
  else {Grounding result; USE_GROUND(rail); result=nd_apply(rail);
	FREE_GROUND(rail); 
	if ((result) && (NOT(FRAMEP(result)))) ref_count(result)--; 
	return result;}
}

Grounding send_message(Grounding args)
{
  Grounding target, message, handler; Frame handler_frame; char *message_name; 
  POP_FORM(target,args); POP_FORM(message,args);
  if (NOT(FRAMEP(target)))
    ground_error(Type_Error,"sending message to a non-frame",target);
  message_name = grounding_to_message(message);
  handler_frame=get_handler_frame(GFRAME(target),message_name);
  if (handler_frame) handler=get_inherited_ground(handler_frame);
  else handler=NULL;
  if (NULLP(handler)) ground_error(Unhandled_Message,message_name,target);
  if (TYPEP(handler,nd_ground))
    return ndapply_handler(cons_pair(handler,cons_pair(target,args)));
  return dapply_handler(cons_pair(handler,cons_pair(target,args)));
}

Grounding request_message(Grounding args)
{
  Grounding target, message, handler; Frame handler_frame; char *message_name; 
  POP_FORM(target,args); POP_FORM(message,args);
  if (NOT(FRAMEP(target)))
    ground_error(Type_Error,"sending message to a non-frame",target);
  message_name = grounding_to_message(message);
  handler_frame=get_handler_frame(GFRAME(target),message_name);
  if (handler_frame) handler=get_inherited_ground(handler_frame);
  else handler=NULL;
  if (NULLP(handler)) return NULL;
  if (TYPEP(handler,nd_ground))
    return ndapply_handler(cons_pair(handler,cons_pair(target,args)));
  else return dapply_handler(cons_pair(handler,cons_pair(target,args)));
}

Grounding delegate_message(Grounding args)
{
  Grounding expert, target, message, handler; Frame handler_frame; char *message_name; 
  POP_FORM(expert,args); POP_FORM(target,args); POP_FORM(message,args);
  if (NOT(FRAMEP(target)))
    ground_error(Type_Error,"sending message to a non-frame",target);
  if (NOT(FRAMEP(expert)))
    ground_error(Type_Error,"delegating message to a non-frame",target);
  message_name = grounding_to_message(message);
  handler_frame=get_handler_frame(GFRAME(expert),message_name);
  handler=get_inherited_ground(handler_frame);
  if (TYPEP(handler,nd_ground))
    return ndapply_handler(cons_pair(handler,cons_pair(target,args)));
  else return dapply_handler(cons_pair(handler,cons_pair(target,args)));
}


/* Turtle semantics */

Frame point;

static Grounding within_frame_handler(Grounding expr,Grounding env)
{
  Frame new_point; Grounding frame_ref; 
  POP_FORM(frame_ref,expr); /* Ignored */
  POP_FORM(frame_ref,expr);
  /* If you don't have a body, don't do anything... */
  if (expr == empty_list) return NULL; 
  if (FRAMEP(frame_ref)) new_point=GFRAME(frame_ref);
  else if (TYPEP(frame_ref,string_ground)) 
    new_point=make_annotation(point,GSTRING(frame_ref));
  else if (TYPEP(frame_ref,symbol_ground))
    new_point=make_annotation(point,SYMBOL_NAME(frame_ref));
  else {Grounding eval_result; eval_result=eval(frame_ref,empty_list);
	new_point=(GFRAME(eval_result));}
  {FLET(Frame,point,new_point)
     {DO_FORMS(expression,expr) side_eval(expression,empty_list);}
   END_FLET(point);}
  return (Grounding) point;
}

Grounding defhandler_eval_handler(Grounding form,Grounding env)
{
  Frame handlers; handlers=probe_annotation(point,commands_string);
  if (NULLP(handlers)) 
    {handlers=use_annotation(point,commands_string);
     set_prototype(handlers,global_handlers);}
  {Grounding args; POP_FORM(args,form); POP_FORM(args,form);
   if (TYPEP(args,pair_ground))
     {Grounding name, definition, proc; Frame handler;
      name=GCAR(args); handler=use_annotation(handlers,SYMBOL_NAME(name));
      definition=cons_pair(define_symbol,cons_pair(args,form));
      USE_GROUND(definition); proc=eval(definition,env);
      set_ground(handler,proc); FREE_GROUND(definition);
      /* We don't FREE proc, since we're returning it to the evaluator. */
      return proc;}
   else if (TYPEP(args,symbol_ground))
     if (form == empty_list)
       {Frame method; method=use_annotation(handlers,SYMBOL_NAME(args));
	set_ground(method,get_inherited_ground(method));
	return frame_ground(method);}
     else {Frame method; method=use_annotation(handlers,SYMBOL_NAME(args));
	   set_ground(method,eval(GCAR(form),env));
	 return frame_ground(method);}
   else {raise_crisis("Bad DEFHANDLER form"); return NULL;}}
}


/* Syntactic sugar */

/* Gets the first element of an expression, ignoring comments. */
static Grounding form_car(Grounding expr)
{
  Grounding temp;
  POP_FORM(temp,expr); 
  return temp;
}

/* Gets the second element of an expression, ignoring comments. */
static Grounding form_cadr(Grounding expr)
{
  Grounding temp;
  POP_FORM(temp,expr); POP_FORM(temp,expr); 
  return temp;
}

/* Macro expander for turning lets into defines */
Grounding let_expander(Grounding expr)
{
  Grounding name, bindings;
  POP_FORM(name,expr); POP_FORM(name,expr);
  if (TYPEP(name,symbol_ground))
    {POP_FORM(bindings,expr);}
  else {bindings=name;name=NULL;}
  return cons_pair(cons_pair(define_symbol,
			     (cons_pair(cons_pair(name,
						  maplist(form_car,bindings)),expr))),
		   maplist(form_cadr,bindings));
}

/* Macro expander for turning let* expressions into nested defines */
Grounding let_star_expander(Grounding expr)
{
  Grounding bindings, body, binding, arg, value;
  POP_FORM(bindings,expr); POP_FORM(bindings,expr); POP_FORM(binding,bindings);
  arg=form_car(binding); value=form_cadr(binding);
  if ((bindings == NULL) || (bindings == empty_list)) body=expr; 
  else body=cons_pair(cons_pair(let_star_symbol,cons_pair(bindings,expr)),empty_list);
  return cons_pair(cons_pair(define_symbol,
			     cons_pair(cons_pair(NULL,cons_pair(arg,empty_list)),
				       body)),
		   cons_pair(value,empty_list));
}


/* Examining procedures */

Grounding procedure_arguments(Grounding procedure)
{
  return GCDR(proc_args(procedure));
}

Grounding procedure_name(Grounding procedure)
{
  return GCAR(proc_args(procedure));
}

Grounding procedure_body(Grounding procedure)
{
  return proc_body(procedure);
}

int procedure_arity(Grounding procedure)
{
  if (TYPEP(procedure,framer_function_ground))
    return the(primitive,procedure)->arity;
  else return list_length(procedure_arguments(procedure))-1;
}



/* Interface to evaluator */

extern void call_fraxl_debugger(exception ex);
extern Grounding fraxl_get_handler(Grounding frame,Grounding message);
extern Grounding fraxl_get_handler_frame(Grounding frame,Grounding message);
Grounding toggle_allocation_tracing(void);

static void do_top_level_define(Grounding proc)
{
  Grounding name, args, old_value;
  if (!TYPEP(proc,procedure_ground))
    raise_crisis ("Funny define");

  /* The "proc" should look like (define (f ...) ...), where
    "f" is a symbol.  Anything else is a syntax error. */
  args = proc_args(proc);
  if (!TYPEP(args,pair_ground) || (args == empty_list))
    ground_error(Syntax_Error, 
       "\"define\" must be followed by a non-empty argument list, not: ",
       args);
  name = GCAR(args);
  if (!TYPEP(name, symbol_ground))
    ground_error(Syntax_Error, "This can't be a function name: ", name);

  /* If the symbol was previously defined,
     decrement the old value's refcount. */
  old_value = SYMBOL_VALUE(name);
  FREE_GROUND(old_value);

  /* Install the new symbol definition */
  USE_GROUND(proc);
  SYMBOL_VALUE(name) = proc;
}

Grounding apply_primitive(Grounding procedure,Grounding arguments)
{
  Grounding nd_apply(Grounding rail);
  return nd_apply(cons_pair(procedure,arguments));
}

Grounding fraxl_eval(Grounding form)
{
  Grounding result;
  USE_GROUND(empty_list);
  result=eval(form,empty_list);
  FREE_GROUND(empty_list);
  /* This is a kludge to keep from counting this guy twice. */
  if (result && (NOT(FRAMEP(result))))
    ref_count(result)--;
  return result;
}


Grounding run_init(Grounding init)
{
  if (NULLP(init)) return NULL;
  else if (FRAMEP(init)) run_init(frame_ground(GFRAME(init)));
  else if (RESULT_SETP(init))
    {DO_RESULTS(each,init) run_init(each);}
  else {Grounding init_result = eval(init,empty_list);
	if (TYPEP(init,pair_ground) && (GCAR(init) == define_symbol))
	    do_top_level_define(init_result);
      }
  return NULL;
}

Grounding load_scheme_file(Grounding filename)
{
  Grounding form, value;
  generic_stream gstream;
  gstream.stream_type=file_io; gstream.ptr.file=fopen(GSTRING(filename),"r");
  WITH_HANDLING
    {form=parse_ground(&gstream);
     USE_GROUND(form);
     value=eval(form,empty_list);
     if (TYPEP(form,pair_ground) && (GCAR(form) == define_symbol))
       do_top_level_define(value);
     FREE_GROUND(form);
     }
  ON_EXCEPTION
    {fclose(gstream.ptr.file); 
     if (theException == Unexpected_EOF) {} reraise();}
  END_HANDLING;
  return NULL;
}


/* A read eval print loop */

extern boolean trap_unknown_frames, reject_unknown_frames;
extern long ground_memory_in_use;
double timing_threshold=1.0;
/* TPM: 4/13/94 */
#ifndef CLK_TCK
#define CLK_TCK 50
#endif
#define time_diff(start,end) (((double) (end-start))/CLK_TCK)
void setup_abort_handling(void); 

void report_statistics(FILE *stream,clock_t start,clock_t end,long old_mem,long new_mem)
{
  if (time_diff(start,end) > timing_threshold)
    {int days, hours, mins, secs; secs=((end-start)/CLK_TCK);
     mins=secs/60; hours=mins/60; days=hours/24;
     secs=secs-(mins*60); mins=mins-(hours*60); hours=hours-(days*24);
     fprintf(stream,";; Completed in");
     if (days) fprintf(stream," %d days",days);
     if ((days) || (hours)) fprintf(stream," %d hours",hours);
     if ((days) || (hours) || (mins))
       fprintf(stream," %d minutes and %f seconds\n",
	       mins,(time_diff(start,end)-mins*60-hours*3600-days*24*3600));
     else fprintf(stream, " %f seconds\n",((double) end-start)/CLK_TCK);}
  if (old_mem == new_mem) {}
  else if (old_mem < new_mem)
    fprintf(stream,";; Allocating %ld new bytes giving me %ld in use\n",
	    new_mem-old_mem,new_mem);
  else if (old_mem > new_mem)
    fprintf(stream,";; Freeing %ld bytes, leaving me with %ld in use\n",
	    old_mem-new_mem,new_mem);
}

void read_eval_print(FILE *in,FILE *out,void (*iter_fn)(),void (*error_fn)())
{
  Grounding input, result, quit_symbol; boolean running=True;
  long old_mem; clock_t start_time, end_time;
  quit_symbol=intern("%QUIT");
  while (running)
    {WITH_HANDLING
       {input=NULL; result=NULL; setup_abort_handling(); old_mem=ground_memory_in_use;
	fprintf(out,"Eval: "); fflush(out); 
	trap_unknown_frames=True; 
	input=fparse_ground(in); USE_GROUND(input);
	trap_unknown_frames=False;
	if (input == quit_symbol) running=False;
	else {start_time=clock(); 
	      result=eval(input,empty_list); 
	      end_time=clock();
	      if (TYPEP(input,pair_ground) && (GCAR(input) == define_symbol))
		do_top_level_define(result);
	      {DO_RESULTS(r,result)
		 {fprint_ground(out,r); putc('\n',out);}}
	      if (iter_fn) iter_fn();
	      FREE_GROUND(input); FREE_GROUND(result);
	      fflush(out); 
	      report_statistics(out,start_time,end_time,old_mem,ground_memory_in_use);}}
     ON_EXCEPTION
       {if (error_fn) error_fn();
	else {fflush(stdout); fflush(stderr);
	      fprintf(out,"#%%!& Error: %s: %s\n",theException,exception_details);
	      CLEAR_EXCEPTION();}
	if (input) {FREE_GROUND(input); input=NULL;}
	if (result) {FREE_GROUND(result); result=NULL;}}
     END_HANDLING;}
}


/* Eval from string */

Grounding eval_from_string(Grounding string,Grounding context)
{
  Grounding expr=NULL, result=NULL; 
  FLET(Frame,read_root,parse_frame_from_string(GSTRING(context)))
    {UNWIND_PROTECT
       {expr=parse_ground_from_string(GSTRING(string)); USE_GROUND(expr);
	result=eval(expr,empty_list);}
     ON_UNWIND
       {FREE_GROUND(expr); 
	if ((result) && (NOT(FRAMEP(result)))) ref_count(result)--;}
     END_UNWIND;}
  END_FLET(read_root);
  return result;
}

Grounding eval_from_string_without_interaction(Grounding string,Grounding context)
{
  Grounding result; boolean old_tuf, old_dbg;
  {WITH_HANDLING
     {old_tuf=trap_unknown_frames; trap_unknown_frames=False; 
      old_dbg=debugging; debugging=False;
      result=eval_from_string(string,context);}
   ON_EXCEPTION
     {fprintf(stdout,"#%%!& Error: %s: %s\n",theException,exception_details);
      result=NULL; CLEAR_EXCEPTION();}
   END_HANDLING;}
  trap_unknown_frames=old_tuf; debugging=old_dbg;
  return result;
}


/* Interface to tracing */

Grounding debugging_symbol, pruning_symbol, allocation_symbol, help_symbol,
  tail_recursion_symbol;

#define CONFIGURE_HELP \
  "You can activate or deactivate: DEBUGGING, PRUNING, TAIL-RECURSION, or ALLOCATION\n"

Grounding configure_debugging(Grounding args,boolean on)
{
  if (NOT(TYPEP(args,pair_ground))) args=cons_pair(args,empty_list);
  USE_GROUND(args);
  if (in_list(help_symbol,args))
    {gsputs(CONFIGURE_HELP,standard_output);}
  if (in_list(debugging_symbol,args)) debugging=on;
  if (in_list(pruning_symbol,args)) break_snips=on;
  if (in_list(allocation_symbol,args)) trace_allocation=on;
  if (in_list(tail_recursion_symbol,args)) use_tail_recursion=on;
  FREE_GROUND(args);
  return NULL;
}

Grounding debugger_activate(Grounding args)
{
  return configure_debugging(args,True);
}

Grounding debugger_deactivate(Grounding args)
{
  return configure_debugging(args,False);
}

Grounding set_max_stack_size(Grounding size)
{
  Grounding old; old=integer_to_ground(max_stack_limit);
  max_stack_limit=GINTEGER(size);
  return old;
}

void fraxl_crisis_function(exception ex)
{
  char buffer[256]; Frame_Array *ozc; ozc=zip_codes; zip_codes=NULL;
  if (ex == Non_Local_Exit) return;
  else if (NOT(debugging))
    {print_stack(current_stack_frame,standard_output,NULL,80,2);
     zip_codes=ozc; return;}
  sprintf(buffer,"(%s: %s)",ex,exception_details);
  if (current_stack_frame)
    {int debugging; debugging=1;
     while (debugging != 0)
       {UNWIND_PROTECT
	  {debugging=fraxl_debugger(buffer,current_stack_frame);
	   if (debugging) fprintf(stderr,"Bottom of stack!");}
	ON_UNWIND
	  zip_codes=ozc;
	END_UNWIND}}
}


/* Initializing the evaluator */

void init_special_forms()
{
  /* Conditional special forms */
  declare_keyword("if",if_eval_handler);
  declare_keyword("when",when_eval_handler);
  declare_keyword("unless",unless_eval_handler);
  declare_keyword("cond",cond_eval_handler);
  declare_keyword("and",and_eval_handler);
  declare_keyword("or",or_eval_handler);
  declare_keyword("not",not_eval_handler);

  /* Quote and unquote */
  declare_keyword("quote",quote_eval_handler);
  declare_keyword("backquote",backquote_eval_handler);

  /* Handling non determinism */
  declare_keyword("amb",amb_eval_handler);
  declare_keyword("prefer",prefer_eval_handler);
  declare_keyword("gather",gather_eval_handler);
  declare_keyword("intersect",intersect_eval_handler);
  declare_keyword("difference",difference_eval_handler);
  declare_keyword("sorted",sorted_eval_handler);

  /* Higher order function stuff */
  define_symbol   = declare_keyword("define",define_eval_handler);
  lambda_symbol   = declare_keyword("lambda",lambda_eval_handler);
  declare_keyword("eval-here",eval_here_handler);
  declare_keyword("include",include_eval_handler);
  declare_keyword("defhandler",defhandler_eval_handler);

  /* Binding constructs */
  declare_keyword("letrec",letrec_eval_handler);
  declare_keyword("begin",begin_eval_handler);
  declare_macro("let",declare_unary_function(let_expander,"expand-let",pair_ground));
  declare_macro("let*",declare_unary_function(let_star_expander,
					      "expand-let*",pair_ground));
  declare_keyword("within-frame",within_frame_handler);
  declare_keyword("within",within_frame_handler);

  /* Debugging helps */
  declare_keyword("<+>",xtrace_eval_handler);
  declare_keyword("step",step_eval_handler);
  declare_keyword("assert",assert_eval_handler);
}

void init_evaluator_functions()
{
  declare_lexpr(send_message,"send"); 
  declare_lexpr(delegate_message,"delegate");
  declare_lexpr(request_message,"request");
  declare_lexpr(error_lexpr,"error");
  declare_lexpr(exception_lexpr,"exception");
  declare_binary_function(eval_from_string,"string-eval",string_ground,string_ground);
  declare_binary_function
    (eval_from_string_without_interaction,"try-string-eval",string_ground,string_ground);
  point=root_frame;
  declare_unary_function(fraxl_eval,"eval",any_ground);
  declare_unary_function(load_scheme_file,"load",string_ground);
  declare_unary_function(run_init,"init",any_ground);
  declare_unary_function(set_max_stack_size,"max-stack=",integer_ground);
  declare_unary_function(debugger_activate,"configure",any_ground);
  declare_unary_function(debugger_activate,"activate",any_ground);
  declare_unary_function(debugger_deactivate,"deactivate",any_ground);
  declare_binary_function(apply_primitive,"apply",any_ground,pair_ground);
  declare_binary_function(apply_primitive,"prim-apply",any_ground,pair_ground);
  declare_function(fraxl_get_handler,"get-handler",2,
		   frame_ground_type,any_ground,any_ground,any_ground);
  declare_function(fraxl_get_handler_frame,"get-handler-frame",2,
		   frame_ground_type,any_ground,any_ground,any_ground);
  declare_unary_function
    (procedure_name,"procedure-name",procedure_ground);
  declare_unary_function
    (procedure_arguments,"procedure-arguments",procedure_ground);
  declare_unary_function
    (procedure_body,"procedure-body",procedure_ground);
}

void init_evaluator_symbols()
{
  debugging_symbol=intern("debugging");
  pruning_symbol=intern("snips");
  allocation_symbol=intern("alocation");
  help_symbol=intern("help");
  tail_recursion_symbol=intern("tail-recursion");
  else_symbol=intern("else");
  let_star_symbol=intern("let*");
}

void init_evaluator()
{
  init_special_forms();
  init_evaluator_symbols();
  init_evaluator_functions();
  global_handlers=frame_ref("/system/objects/default/+commands");  
  commands_string=intern_frame_name("+commands");
  /* This doesn't quite belong here, but I'm not sure where it goes. */
  intern_frame_name("+root");
  crisis_handler=fraxl_crisis_function;
}

/*
  Local variables: ***
  compile-command: "../etc/compile-framer" ***
  tags-file-name: "../sources/TAGS" ***
  End: **
*/

