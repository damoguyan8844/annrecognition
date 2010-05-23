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
  "$Header: /mas/framer/sources/RCS/eval.c,v 1.101 1994/01/26 18:36:01 haase Exp $";

/* #define TRACE_GC 1 */
#include "framer.h"
#include "fraxl.h"
#include "internal/private.h"
#include "internal/eval.h"

/* Declarations */

/* These are all special forms in the evaluator. */
Grounding lambda_symbol, define_symbol, let_star_symbol, else_symbol;
exception 
  Eval_Failure="Eval failure", 
  Type_Error="Bad Arg Type", 
  Arity_Error="Wrong number of arguments",
  Not_A_Function="Trying to apply a non-function",
  Unbound_Variable="Unbound variable",
  Non_Message="Can't be a message id",
  Unhandled_Message="Message has no handler",
  Syntax_Error="FRAXL syntax error";

boolean break_snips=False, /* Print trace statements */
        debugging=False,   /* Debug interactively */
        check_compound_arity=True; /* Signal error when a compound procedure
				      is called with the wrong number of arguments. */

/* Used in the read/eval/print loop */
extern long ground_memory_in_use;

/* Used to control tail recursion, okay_to_tail_call is set by the interpreter,
   use_tail_recursion set by the user. */
boolean okay_to_tail_call=False, use_tail_recursion=True;
/* This is used to do a tail call */
exception Non_Local_Exit="A non local exit (you shouldn't see this)";

/* Maintaining the stack */
struct STACK_FRAME *current_stack_frame=NULL, *throw_to=NULL;
Grounding throw_value=NULL;
int max_stack_limit=4000;


/* Trace statements */

extern boolean abbreviate_procedures;

int trace_indent=0;

/* Outputs a simple (one argument) trace statement */
void output_trace(char *header,Grounding expr)
{
  generic_stream stream; int xpos; 
  stream.stream_type = file_io; stream.ptr.file = stderr;
  {DO_TIMES(i,trace_indent) putc(' ',stderr);}
  fprintf(stderr,"%s",header); xpos=strlen(header)+trace_indent; 
  {FLET(boolean,abbreviate_procedures,True)
     pprint_elt((&stream),expr,trace_indent+2,default_display_width,xpos);
   END_FLET(abbreviate_procedures);}
  fprintf(stderr,"\n"); fflush(stderr);
}

/* Outputs a simple (one argument) trace statement */
void output_trace2(char *header,Grounding expr,char *header2,Grounding expr2)
{
  generic_stream stream; int xpos; 
  stream.stream_type = file_io; stream.ptr.file = stderr;
  {DO_TIMES(i,trace_indent) putc(' ',stderr);}
  {FLET(boolean,abbreviate_procedures,True)
     {fprintf(stderr,"%s",header); xpos=trace_indent+strlen(header);
      xpos=pprint_elt((&stream),expr,trace_indent+2,default_display_width,xpos);
      if ((xpos+strlen(header2)) > default_display_width)
	{fputc('\n',stderr); {DO_TIMES(i,trace_indent) putc(' ',stderr);}
	 fprintf(stderr,"  %s",header2); xpos=strlen(header2)+trace_indent;}
      else {fprintf(stderr," %s",header2); xpos=xpos+strlen(header2)+1;}
      pprint_elt((&stream),expr2,trace_indent+2,default_display_width,xpos);}
   END_FLET(abbreviate_procedures);}
  fprintf(stderr,"\n"); fflush(stderr);
}


/* Primitive operations */

/* Looks up a variable in an environment structure. 
   If this fails, it also looks in the global namespace. 
   It calls USE_GROUND on the value it returns. */
Grounding lookup_variable(Grounding symbol,Grounding env)
{
  while (env != empty_list)
    {Grounding call, args, values; 
     call=GCAR(env); args=GCAR(call); values=GCDR(call);
     while ((TYPEP(args,pair_ground)) && (args != empty_list) &&
	    (TYPEP(values,pair_ground)) && (values != empty_list))
       if ((GCAR(args)) == symbol) 
	 {Grounding val; val=(GCAR(values)); 
	  if ((TYPEP(val,procedure_ground)) && (NULLP(proc_env(val))))
	    val=close_procedure(proc_text(val),env); 
	  USE_GROUND(val); return val;}
       else {args=GCDR(args); values=GCDR(values);}
     if (args == symbol) {USE_GROUND(values); return values;}
     env=GCDR(env);}
  /* Default case */
  if (SYMBOL_VALUE(symbol))
    return USED_GROUND(SYMBOL_VALUE(symbol));
  else {ground_error(Unbound_Variable,"Can't evaluate the symbol ",symbol);
	return NULL;}
}

/* This adds a binding to the outermost rib of an environment */
void add_binding(Grounding var,Grounding val,Grounding env)
{
  if (env == empty_list) raise_crisis("Can't add binding to empty environment");
  else {Grounding rib, vars, vals, nvars, nvals;
	rib=GCAR(env); vars=GCAR(rib); vals=GCDR(rib);
	GCAR(rib)=nvars=cons_pair(var,vars); 
	GCDR(rib)=nvals=cons_pair(val,vals);
	USE_GROUND(nvars); USE_GROUND(nvals);
	FREE_GROUND(vars); FREE_GROUND(vals);}
}

/* This takes the expression in stack->expr, evaluates each
   of its elements and puts them (in backwards order) into
   stack->choices.  It also tracks whether any of the arguments
   are non-deterministic, returning the result of this. 
   It is called to set up a stack frame for evaluation. */
void setup_combination(struct STACK_FRAME *stack)
{
  boolean deterministic=True;
  stack->choices=empty_list; USE_GROUND(empty_list);
  {DO_FORMS(form,stack->expr)
     /* For each subexpression, evaluate it and push it on to stack->choices;
	if one form `fails', record it and set abort to True; if the
	result is non-deterministic, set deterministic to False.  
	Setting abort to True quits the DO_FORMS expression. */
     {Grounding arg; arg=eval(form,stack->env); 
      if (arg == NULL) 
	{abort=True; 
	 if (break_snips)
	   {output_trace2
	      ("The expression",stack->expr,"was snipped because of",form);
	    if (debugging) fraxl_debugger("Snipping",current_stack_frame);}
	 FREE_GROUND(stack->choices); stack->choices=NULL;}
      else {if (TYPEP(arg,nd_ground)) deterministic=False;
	    HCONS(arg,stack->choices,stack->choices);}}}
  /* When done, return whether the application is deterministic. */
  if (NULLP(stack->choices)) return;
  if (deterministic)
    {stack->choice=reverse_list(stack->choices);
     USE_GROUND(stack->choice); FREE_GROUND(stack->choices); 
     stack->choices=NULL;}
}

/* This is the main evaluate/apply loop, combined into one to handle
   tail recursion.  */ 
Grounding stack_call(struct STACK_FRAME *stack)
{
  Grounding nd_applier AP((Grounding nd_rail,Grounding d_rail));
  Grounding d_apply AP((Grounding d_rail));
  Grounding result=NULL;
  boolean old_okay_to_tail_call;
  old_okay_to_tail_call=okay_to_tail_call;
  /* This while loop does implicit iteration for tail recursion.  Each time
     around it, we evaluate a form; this evaluation may do a throw back
     to this loop which starts us again with the form thrown. */
  {WITH_HANDLING
     while (stack->looping)
       {Grounding head; 
	/* Start by setting stack and assuming that we're not in a tail call. */
	current_stack_frame=stack; stack->looping=False;
	/* It's okay to tail call if its possible to tail call (the user flag)
	   and the form is deterministic (i.e. a combination whose parameters
	   are all deterministic). */ 
	okay_to_tail_call=use_tail_recursion;
	if (stack->expr)	/* Regular eval */
	  {/* Shouldn't happen, but let's be safe anyway. */
	    head=GCAR(stack->expr); 
	    if (NULLP(head)) ground_error(Eval_Failure,"headless form",stack->expr);
	    if ((ground_type(head) == symbol_ground) && (KEYWORD_P(head)))
	      /* Here is where keywords and special forms are dispatched. */
	      result=(the(symbol,head))->dispatch(stack->expr,stack->env);
	    /* Otherwise, its a combination, so fill in stack->choices */
	    else setup_combination(stack);}
	else if (stack->choices)
	  {boolean deterministic=True;
	   DO_FORMS(elt,stack->choices) if (TYPEP(elt,nd_ground)) deterministic=False;
	   if (deterministic)
	     {stack->choice=reverse_list(stack->choices); USE_GROUND(stack->choice);
	      FREE_GROUND(stack->choices); stack->choices=NULL;}}
	if (stack->choices)
	  {/* Otherwise, if it is deterministic, it is okay to tail call. */
	   okay_to_tail_call=False;
	   /* Otherwise, call nd_applier. */
	   result=nd_applier(stack->choices,empty_list);}
	else if (stack->choice)
	  result=d_apply(stack->choice);}
   ON_EXCEPTION
     if ((theException == Non_Local_Exit) && (throw_to == stack))
       {result=throw_value; throw_to=NULL; throw_value=NULL;
	CLEAR_EXCEPTION();}
     else {FREE_GROUND(stack->expr); FREE_GROUND(stack->env); 
	   FREE_GROUND(stack->choices); FREE_GROUND(stack->choice);
	   okay_to_tail_call=old_okay_to_tail_call; 
	   current_stack_frame=stack->previous;
	   reraise();}
   END_HANDLING}
  /* Finally, clean up, and reset the bindings you saved initially. */
  FREE_GROUND(stack->expr); FREE_GROUND(stack->env); 
  FREE_GROUND(stack->choices); FREE_GROUND(stack->choice);
  current_stack_frame=stack->previous; okay_to_tail_call=old_okay_to_tail_call;
  return result;
}


/* Throws an expression and an environment to some other stack frame.
   The primary use for this is to do tail calls which throw to the current
   stack frame.  It does a non local exit (raising an event) which is
   caught by the eval loop replaces whatever it is evaluating with
   the expr and env passed to throw. */
Grounding tail_eval_pair(Grounding expr,Grounding env)
{
  /* Free the state of the current stack frame and replace them with your arguments. */
  USE_GROUND(expr); USE_GROUND(env); 
  FREE_GROUND(current_stack_frame->expr); FREE_GROUND(current_stack_frame->env); 
  FREE_GROUND(current_stack_frame->choices); FREE_GROUND(current_stack_frame->choice); 
  current_stack_frame->expr=expr; current_stack_frame->env=env;
  current_stack_frame->choices=NULL; current_stack_frame->choice=NULL; 
  current_stack_frame->looping=True;
  return NULL;
}

Grounding d_tail_apply(Grounding rail)
{
  /* Free the state of the current stack frame and replace them with your arguments. */
  USE_GROUND(rail); 
  FREE_GROUND(current_stack_frame->expr); FREE_GROUND(current_stack_frame->env); 
  FREE_GROUND(current_stack_frame->choices); FREE_GROUND(current_stack_frame->choice); 
  current_stack_frame->expr=NULL; current_stack_frame->env=NULL;
  current_stack_frame->choices=NULL; current_stack_frame->choice=rail; 
  current_stack_frame->looping=True;
  return NULL;
}

Grounding nd_tail_apply(Grounding rail)
{
  Grounding rrail;
  /* Free the state of the current stack frame and replace them with your arguments. */
  rrail=reverse_list(rail); USE_GROUND(rrail); FREE_GROUND(rail);
  USE_GROUND(rrail); 
  FREE_GROUND(current_stack_frame->expr); FREE_GROUND(current_stack_frame->env); 
  FREE_GROUND(current_stack_frame->choices); FREE_GROUND(current_stack_frame->choice); 
  current_stack_frame->expr=NULL; current_stack_frame->env=NULL;
  current_stack_frame->choices=rrail; current_stack_frame->choice=NULL; 
  current_stack_frame->looping=True;
  return NULL;
}

/* Evalutes a pair for side effect.  The special trick with this
   is the handling of `internal defines' by side-effecting the
   given environment which also requires some GC twiddling to avoid
   the problems with a reference counting garbage collector and 
   recursive structures. */
void side_eval_pair(Grounding expr,Grounding env)
{
  if ((GCAR(expr)) == define_symbol)
    {Grounding head; POP_FORM(head,expr); /* Ignore `DEFINE' */
     head=GCAR(expr);
     if ((TYPEP(head,pair_ground)) && (head != empty_list))
       add_binding(GCAR(head),close_procedure(expr,NULL),env);
     else ground_error(Syntax_Error,"Not a valid function spec: ",head);}
  else {Grounding result; result=eval_pair(expr,env); FREE_GROUND(result);}
}

/* This is one of the three workhorses of the evaluator. 
   (The others are nd_applier and d_apply). */
Grounding eval_pair(Grounding expr,Grounding env)
{
  /* The one special case */
  if (expr == empty_list) {USE_GROUND(expr); return expr;}
  else {struct STACK_FRAME stack; /* Initializing the new stack frame. */
	stack.previous=current_stack_frame; stack.looping=True;
	stack.expr=expr; stack.env=env; stack.choices=NULL; stack.choice=NULL;
	if (NULLP(current_stack_frame)) 
	  {stack.depth=1; stack.stepping=False;}
	else if (current_stack_frame->depth > max_stack_limit)
	  raise_crisis("Stack overflow");
	else {stack.depth=current_stack_frame->depth+1;
	      stack.stepping=current_stack_frame->stepping;}
	USE_GROUND(expr); USE_GROUND(env); USE_GROUND(stack.choices);
	return stack_call(&stack);}
}

/* This calls nd_applier after reversing the nd_rail to get it into the
   right order.  It also has to make a stack frame and we don't let it
   tail call (we could but we'd have to reproduce all the hair from eval_pair.
*/
Grounding nd_apply(Grounding nd_rail)
{
  Grounding rrail; struct STACK_FRAME stack; 
  rrail=reverse_list(nd_rail); USE_GROUND(rrail); FREE_GROUND(nd_rail);
  stack.previous=current_stack_frame; stack.looping=True;
  stack.choices=rrail; stack.choice=NULL; stack.expr=NULL; stack.env=NULL; 
  if (NULLP(current_stack_frame)) 
    {stack.depth=1; stack.stepping=False;}
  else if (current_stack_frame->depth > max_stack_limit)
    raise_crisis("Stack overflow");
  else {stack.depth=current_stack_frame->depth+1;
	stack.stepping=current_stack_frame->stepping;}
  return stack_call(&stack);
}


/* Determinstic apply */

/* This actually does an application.  A rail consists
   of a bunch of evaluated arguments; the first should be
   interpretable as a procedure and it is applied to the rest.
   There are two cases: primitive applications and lambda applications.
   The lambda applications check for arity and argument type; the procedure
   applications call out to apply_lambda. */
Grounding d_apply(Grounding rail)
{
  Grounding result, head; boolean step_this=False; head=GCAR(rail);
  /* Nothing without a ground type can be the head. */
  if ((head == NULL) || (FRAMEP(head)))
    ground_error(Not_A_Function,"I can't apply ",head);
  else {switch (ground_type(head))
	  {
	  case framer_function_ground: case procedure_ground: break;
	    /* We handle the special case where the head has to be
	       evaluated again.  This is a kludge but enables us to
	       use symbol names as primitives. */
	  case pair_ground: case symbol_ground:
	    {Ground_Type hd_type; head=eval(head,empty_list);
	     hd_type=ground_type(head);
	     if (NOT((hd_type == framer_function_ground) || 
		     (hd_type == procedure_ground)))
	       ground_error(Not_A_Function,"I can't apply ",head); break;
	     FREE_GROUND(GCAR(rail)); (GCAR(rail))=head; break;}
	  default: 
	    ground_error(Not_A_Function,"I can't apply ",head); break;
	  }}
  /* For stepping... */
  if (current_stack_frame->stepping)
    {step_this=True; okay_to_tail_call=False; trace_indent++;
     if (debugging) fraxl_debugger("Stepping",current_stack_frame);
     else output_trace("Applying",rail);}
  /* Now if we do the real dispatch. */
  switch (ground_type(head))
    { 
    case framer_function_ground:
      {Grounding args[MAX_PRIMITIVE_ARITY]; 
       struct PRIMITIVE_PROCEDURE *proc; int i=0;
       proc=the(primitive,head);
       if (proc->arity == -1)	/* -1 arity indicates a primitive lexpr */
	 {result=proc->function(GCDR(rail)); USE_GROUND(result); break;}
       {DO_FORMS(arg,GCDR(rail))
	  /* Otherwise, check the argument types; any_ground passes anything */
	  {if (((proc->types[i]) == any_ground) || (TYPEP(arg,(proc->types[i]))))
	     args[i++]=arg;
	  else ground_error(Type_Error,"argument has wrong type",arg);}}
       /* While checking argument types, we've also computed the number of
	  arguments; if that doesn't match the arity, we signal an error. */
       if (i != proc->arity)
	 ground_error(Arity_Error,"wrong number of arguments",rail);
       else switch (proc->arity) /* Intimidating but regular.... */
	 { 
	 case 0: result=proc->function(); break;
	 case 1: result=proc->function(args[0]); break;
	 case 2: result=proc->function(args[0],args[1]); break;
	 case 3: result=proc->function(args[0],args[1],args[2]); break;
	 case 4: result=proc->function(args[0],args[1],args[2],args[3]); break; 
	 case 5: result=proc->function(args[0],args[1],args[2],args[3],args[4]); break;
	 case 6: result=proc->function(args[0],args[1],args[2],args[3],args[4],args[5]); break;
	 case 7: result=proc->function(args[0],args[1],args[2],args[3],args[4],args[5],args[6]); break;
	 case 8: result=proc->function(args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7]); break;
	 case 9: result=proc->function(args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[8]); break;
	 case 10: result=proc->function(args[0],args[1],args[2],args[3],args[4],
					args[5],args[6],args[7],args[8],args[9]); 
	   break;
	 }
       USE_GROUND(result);	/* `Use' the result before passing it out. */
       break;
     }
    case procedure_ground: 
      /* To apply a form, we make a new environment and evaluate the forms
	 in it; most as side effects and the final one with throw to do a tail
	 call. */
      {Grounding new_env=NULL, final=NULL;
       if (check_compound_arity)
	 {Grounding arglist, paramlist; arglist=proc_args(head); paramlist=rail;
	  while ((TYPEP(arglist,pair_ground)) && (arglist != empty_list) &&
		 (TYPEP(paramlist,pair_ground)) && (paramlist != empty_list))
	    {arglist=GCDR(arglist); paramlist=GCDR(paramlist);}
	  if (NOT(arglist == paramlist))
	    if (NOT(TYPEP(arglist,symbol_ground)))
	      if (NOT(TYPEP(arglist,pair_ground)))
		ground_error(Syntax_Error,"Bad argument list: ",arglist);
	      else ground_error(Arity_Error,"for compound procedure",rail);}
       {UNWIND_PROTECT
	  {new_env=cons_pair(cons_pair(proc_args(head),rail),proc_env(head));
	   USE_GROUND(new_env);
	   {DO_SIDE_EFFECT_FORMS(expr,proc_body(head),final) side_eval(expr,new_env);}
	   result=tail_eval(final,new_env);}
	  ON_UNWIND
	    {FREE_GROUND(new_env);}
	  END_UNWIND}
       break;}
    }
  if (step_this)
    {output_trace2("Returning",result,"from application of",rail);
     trace_indent--;}
  return result;
}


/* Non-deterministic application */

void nd_applier_internal(Grounding nd_rail,Grounding d_rail,
			 struct RESULT_ACCUMULATOR *r_acc)
{
  if (nd_rail == empty_list)
    {Grounding results; results=d_apply(d_rail);
     {DO_RESULTS(r,results)
      {if (r_acc->ptr >= r_acc->limit)
	 {Grounding *new_accumulator; int new_size; new_size=r_acc->size*2;
	  ALLOCATE(new_accumulator,Grounding,new_size);
	  memmove(new_accumulator,r_acc->accumulator,
		  r_acc->size*sizeof(Grounding));
	  if (r_acc->size > 4) free(r_acc->accumulator);
	  r_acc->ptr=new_accumulator+(r_acc->ptr-r_acc->accumulator);
	  r_acc->limit=new_accumulator+new_size; r_acc->size=new_size;
	  r_acc->accumulator=new_accumulator;}
       *(r_acc->ptr++)=r; USE_GROUND(r);}}
     FREE_GROUND(results);}
  else {DO_RESULTS(elt,GCAR(nd_rail))
	  {Grounding new_d_rail;
	   UNWIND_PROTECT
	     {GCONS(elt,d_rail,new_d_rail); USE_GROUND(new_d_rail);
	      nd_applier_internal(GCDR(nd_rail),new_d_rail,r_acc);}
	   ON_UNWIND
	     {FREE_GROUND(new_d_rail);}
	   END_UNWIND}}
}

Grounding nd_applier(Grounding nd_rail,Grounding d_rail)
{
  Grounding cons_result(Grounding res1,Grounding res2);
  WITH_RESULT_ACCUMULATOR(ra)
    {USE_GROUND(d_rail); nd_applier_internal(nd_rail,d_rail,ra);
     {Grounding result, temp, *ptr, *limit; 
      int how_many, duplicates=0; how_many=(ra->ptr-ra->accumulator);
      switch (how_many)
	{
	case 0:	{FREE_GROUND(d_rail); return NULL;}
	case 1: {FREE_GROUND(d_rail); return ra->accumulator[0];}
	case 2: {FREE_GROUND(d_rail); 
		 result=cons_result(ra->accumulator[0],ra->accumulator[1]);
		 USE_GROUND(result); FREE_GROUND(ra->accumulator[0]);
		 FREE_GROUND(ra->accumulator[1]);
		 return result;}
	default: {temp=empty_list; FREE_GROUND(d_rail);} 
	  /* And continue onto the rest of the action.*/
	}
      ptr=ra->accumulator; limit=ra->ptr; while (ptr < limit) 
	{Grounding elt; elt=*ptr;
	 if (FRAMEP(elt)) 
	   if (search_bit_p((Frame)elt,RESULT_REDUCTION_BIT)) 
	     {duplicates++; *ptr=NULL;}
	   else set_search_bit((Frame)elt,RESULT_REDUCTION_BIT);
	 else {if (in_list(elt,temp)) duplicates++; else temp=cons_pair(elt,temp);
	       FREE_GROUND(elt);}
	 ptr++;}
      if (duplicates == (ra->ptr-ra->accumulator)) raise_crisis("Duplication error");
      else {NDMAKE(result,(ra->ptr-ra->accumulator)-duplicates);}
      {Grounding *nptr; ptr=ra->accumulator; nptr=ND_ELTS(result);
       while (ptr < limit)
	 {Grounding elt; elt=*ptr++;
	  if (FRAMEP(elt))
	    {if (search_bit_p((Frame)elt,RESULT_REDUCTION_BIT)) 
	       {*nptr++=elt; clear_search_bit((Frame)elt,RESULT_REDUCTION_BIT);}}}
       {DO_LIST(elt,temp) {*nptr++=elt; USE_GROUND(elt);}}}
      if (temp != empty_list) {FREE_GROUND(temp);}
      if (ra->size > 4) free(ra->accumulator);
      USE_GROUND(result);
      return result;}}
}


/* User versions of apply */

/* Utility function for thunk application. */
Grounding apply0(Grounding fcn)
{
  return nd_apply(cons_pair(fcn,empty_list));
}

/* Utility function for single argument apply. */
Grounding apply1(Grounding fcn,Grounding arg1)
{
  return nd_apply(cons_pair(fcn,cons_pair(arg1,empty_list)));
}

/* Utility function for two argument apply. */
Grounding apply2(Grounding fcn,Grounding arg1,Grounding arg2)
{
  return nd_apply(cons_pair(fcn,cons_pair(arg1,cons_pair(arg2,empty_list))));
}

/* Utility function for three argument apply. */
Grounding apply3(Grounding fcn,Grounding arg1,Grounding arg2,Grounding arg3)
{
  return nd_apply(cons_pair(fcn,cons_pair(arg1,cons_pair(arg2,cons_pair(arg3,empty_list)))));
}


/* Handling special forms */

Grounding expand_macro(Grounding expr,Grounding env)
{
  Grounding head, expansion, result; head=GCAR(expr);
  expansion=apply1(SYMBOL_VALUE(head),expr);
  result=eval(expansion,env); 
  FREE_GROUND(expansion);
  return result;
}

Grounding declare_keyword(char *string,Grounding (*handler)())
{
  Grounding sym; sym=intern(string);
  the(symbol,sym)->dispatch=handler;
  return sym;
}

Grounding declare_macro(char *string,Grounding expander)
{
  Grounding sym; sym=intern(string);
  the(symbol,sym)->value=expander;
  the(symbol,sym)->dispatch=expand_macro;
  return sym;
}

Grounding declare_lexpr(Grounding (*func)(),char *string);

/* Debugging */

void pprinter(Grounding expr,generic_stream *stream,
	      int indent,int width,Grounding highlight);


/* Prints the stack */

void print_stack(struct STACK_FRAME *stack,generic_stream *stream,
		 Grounding focus,int width,int stack_window)
{
  if (stack == NULL) return;
  else if (stack_window == 0) return;
  else {struct STACK_FRAME *cxt; cxt=stack;
	while ((cxt) && (NULLP(cxt->choice))) cxt=cxt->previous;
	if (cxt)
	  {Grounding temp, rail; rail=cxt->choice;
	   print_stack(cxt->previous,stream,cxt->expr,width,stack_window-1);
	   if (cxt->choices)
	     {temp=reverse_list(stack->choices);
	      gsprintf(stream,"; <<< [%d] From Space: ",stack->depth);
	      pprinter(temp,stream,23,width-23,NULL); 
	      gsputc('\n',stream); FREE_GROUND(temp);}
	   if (TYPEP(GCAR(rail),procedure_ground))
	     {Grounding proc, args, temp; 
	      proc=GCAR(rail); args=proc_args(proc);
	      if (NOT(TYPEP(args,pair_ground)))
		{gsputs("; <<< Applying ill-formed procedure\n",stream);}
	      else {temp=cons_pair(GCAR(args),GCDR(rail));
		    gsputs("; <<< Applying ",stream); 
		    pprinter(temp,stream,15,width-15,NULL); 
		    gsputc('\n',stream); FREE_GROUND(temp);
		    temp=cons_pair(define_symbol,proc_text(proc)); 
		    gsputs("   ",stream); pprinter(temp,stream,3,width-3,focus);
		    gsputc('\n',stream); FREE_GROUND(temp);}}
	   else {gsprintf(stream,"; <<< [%d] Applying ",cxt->depth);
		 pprinter(stack->choice,stream,15,width-15,NULL);
		 gsputc('\n',stream);}}
	else {print_stack(stack->previous,stream,NULL,width,stack_window-1);
	      gsprintf(stream,"; [%d] Evaluating ",stack->depth); 
	      pprinter(stack->expr,stream,18,width-18,focus);
	      gsputc('\n',stream);}}
}

#if 0
boolean print_stack(struct STACK_FRAME *stack,generic_stream *stream,
		    Grounding focus,int width,int stack_window)
{
  if (stack == NULL) return False;
  else if (stack_window == 0) return False;
  else if (stack->choice)
    {print_stack(stack->previous,stream,stack->expr,width,stack_window-1);
     if (stack->choices)
       {Grounding temp; temp=reverse_list(stack->choices);
	gsprintf(stream,"; <<< [%d] From Space: ",stack->depth);
	pprinter(temp,stream,23,width-23,NULL); 
	gsputc('\n',stream); FREE_GROUND(temp);}
     if (TYPEP(GCAR(stack->choice),procedure_ground))
       {Grounding proc_name, proc, temp; 
	proc=GCAR(stack->choice); proc_name=GCAR(proc_args(proc));
	temp=cons_pair(proc_name,GCDR(stack->choice));
	gsputs("; <<< Applying ",stream);
	pprinter(temp,stream,15,width-15,NULL); 
	gsputc('\n',stream); FREE_GROUND(temp);
	temp=cons_pair(define_symbol,proc_text(proc)); 
	gsputs("   ",stream); pprinter(temp,stream,3,width-3,focus);
	gsputc('\n',stream); FREE_GROUND(temp);
	return True;}
     else {gsprintf(stream,"; <<< [%d] Applying ",stack->depth);
	   pprinter(stack->choice,stream,15,width-15,NULL);
	   gsputc('\n',stream); return False;}}
  else if ((stack->choices) && (stack->expr))
    if (NOT(print_stack(stack->previous,stream,focus,width,stack_window)))
      {gsprintf(stream,"; [%d] Evaluating ",stack->depth); 
       pprinter(stack->expr,stream,18,width-18,focus);
       gsputc('\n',stream); 
       return False;}
    else return False;
  else return print_stack(stack->previous,stream,focus,width,stack_window);
}
#endif
/* Prints the current stack; this is designed to be bound to crisis_handler. */
void print_current_stack(exception ex)
{
  gsprintf(standard_output,"When %s occured:\n",ex);
  print_stack(current_stack_frame,standard_output,NULL,80,-1);
}


/* Calling an interactive debugger */

void clear_all_stepping(struct STACK_FRAME *to)
{
  struct STACK_FRAME *temp; temp=current_stack_frame;
  while (temp != to) {temp->stepping=False; temp=temp->previous;}
}

#define DEBUGGER_HELP \
"      b - stack backtrace (also w)\n\
      p - previous stack frame\n\
      n - next stack frame\n\
      c - clear stepping on this frame\n\
      s - step this frame (when you get back here)\n\
      r - return a value from this frame\n\
      e - evaluate an expression (in this environment)\n\
      q - quit debugger (continue or abort)\n\
      ? - print this message\n"

int fraxl_debugger(char *herald,struct STACK_FRAME *frame)
{
  char input[2000], *ptr=input; boolean suppress_redisplay=False;
  while (1)
    {if (NOT(suppress_redisplay))
       if (frame->choice) 
	 print_stack(frame,standard_output,NULL,80,1);
       else print_stack(frame,standard_output,frame->expr,80,1);
     suppress_redisplay=False;
     {DO_TIMES(i,trace_indent) putc(' ',stdout);}
     fprintf(stdout,">> %s [b,p,n,c,s,r,e,?,q] or Eval: ",herald); 
     fflush(stdout); gets(input);
     if ((*input == ' ') && (input[1] == '\0')) return 0;
     switch (*input)
       {
       case '?': fputs(DEBUGGER_HELP,stdout); suppress_redisplay=True; break;
       case 'b': case 'w':
	 print_stack(frame,standard_output,NULL,80,-1); 
	 suppress_redisplay=True; break;
       case 'q': case 'Q': return 0;
       case 'n': case 'N': return 1;
       case '\0': suppress_redisplay=True; break;
       case 'p': case 'P':
	 if (frame->previous) 
	   {if (NOT(fraxl_debugger(herald,frame->previous))) return 0;}
	 else fprintf(stdout,"At top of stack");
	 break;
       case 's': case 'S':
	 {struct STACK_FRAME *tmp; clear_all_stepping(NULL);
	  tmp=frame; while (tmp) {tmp->stepping=True; tmp=tmp->previous;}
	  return 0;}
       case 'c': case 'C':
	 clear_all_stepping(frame->previous); return 0;
       case 'r': case 'R':
	 {Grounding expr, result;
	  printf("Return value of: "); fflush(stdout);
	  if (input[1]=='\0') gets(input); else ptr++;
	  expr=parse_ground_from_string(ptr); USE_GROUND(expr);
	  result=eval(expr,frame->env); FREE_GROUND(expr);
	  throw_to=frame; throw_value=result; 
	  raise_crisis(Non_Local_Exit);}
       case 'e': case 'E':
	 {Grounding expr, result;
	  printf("Eval: "); fflush(stdout);
	  if (input[1]=='\0') gets(input); else ptr++;
	  expr=parse_ground_from_string(ptr); USE_GROUND(expr);
	  result=eval(expr,frame->env); 
	  {DO_RESULTS(r,result)
	     {gsputs("...> ",standard_output); 
	      print_ground(standard_output,r);
	      gsputs("\n",standard_output);}}
	  FREE_GROUND(expr); FREE_GROUND(result);
	  break;}
       case '(': 
	 {Grounding form, result; form=parse_ground_from_string(input);
	  result=eval(form,frame->env);
	  {DO_RESULTS(r,result)
	     {gsputs("...> ",standard_output); 
	      print_ground(standard_output,r);
	      gsputs("\n",standard_output);}}
	  FREE_GROUND(result); FREE_GROUND(form);
	  break;}
       }}
}

/*
  Local variables: ***
  compile-command: "../etc/compile-framer" ***
  tags-file-name: "../sources/TAGS" ***
  End: **
*/
