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

*/

/***************************************************************** 

  This file implements the ARLOtje `slot abstraction' on top of FRAMER.
  It consists of the basic operations on slots and a variety of utility
  functions.

  Modification history:
    13 June 1992 - Began tracking modification history (Haase)
     8 August 1992 - Added has_element to ARLOtje.
 
*****************************************************************/

#include "framer.h"
#include "fraxl.h"
#include "internal/eval.h"

Grounding home_symbol, set_symbol, value_symbol;
extern Grounding string_to_ground(char *string);
extern Grounding remove_from_list(Grounding elt, Grounding list);
extern Grounding remove_from_list_once(Grounding elt, Grounding list);
extern Grounding merge_results(Grounding list1, Grounding list2);
extern Grounding zap_result(Grounding victim, Grounding set);
extern Grounding find_result(Grounding goal, Grounding set);
extern Grounding fraxl_eval(Grounding expression);
extern Grounding in_list(Grounding elt,Grounding list);
extern Grounding pair_car(Grounding pair);
extern Grounding pair_cdr(Grounding pair);
extern Grounding apply(Grounding *grounds,int size);
extern Grounding eval_function;
Grounding eval_with_value_function;
Grounding add_to_ground(Frame f,Grounding g);
Grounding remove_from_ground(Frame f,Grounding g);

static char *get_method_name, *test_method_name, *add_method_name, *remove_method_name;

/* Using frames to store values */

struct RESOLUTION_STACK 
{Frame resolving; 
 struct RESOLUTION_STACK *previous;} *resolution_stack=NULL;

Grounding get_elements(Frame frame)
{
  Grounding result=NULL;
  struct RESOLUTION_STACK this_one, *temp=resolution_stack;
  this_one.resolving=frame; this_one.previous=resolution_stack;
  while (temp != NULL)
    if (temp->resolving == frame) return NULL;
    else temp=temp->previous;
  {FLET(struct RESOLUTION_STACK *,resolution_stack,(&this_one))
     {Grounding method; method=get_handler(frame,"get");
      if (NULLP(method)) result=frame_ground(frame);
      else {result=apply1(method,frame_to_ground(frame));
	    if ((result) && (NOT(FRAMEP(result)))) ref_count(result)--;}}
   END_FLET(resolution_stack);}
  return result;
}

Grounding has_element(Frame frame,Grounding value)
{
  Grounding method; method=get_handler(frame,"has");
  if (NULLP(method)) 
    if (find_result(value,frame_ground(frame))) return value;
    else return NULL;
  else {Grounding result; result=apply2(method,frame_to_ground(frame),value);
	if ((result) && (NOT(FRAMEP(result)))) ref_count(result)--;
	return result;}
}

Grounding add_element(Frame frame,Grounding value)
{
  Grounding method; method=get_handler(frame,"add");
  if (NULLP(method)) 
    {add_to_ground(frame,value); return value;}
  else {Grounding result; result=apply2(method,frame_to_ground(frame),value);
	if ((result) && (NOT(FRAMEP(result)))) ref_count(result)--;
	return result;}
}

Grounding remove_element(Frame frame,Grounding value)
{
  Grounding method; method=get_handler(frame,"remove");
  if (NULLP(method)) 
    {remove_from_ground(frame,value); return value;}
  else {Grounding result; result=apply2(method,frame_to_ground(frame),value);
	if ((result) && (NOT(FRAMEP(result)))) ref_count(result)--;
	return result;}
}


/* A Slot Abstraction */

Frame get_slot(Frame unit,Grounding slot)
{
  char *slot_name; Frame annotation;
  if (TYPEP(slot,string_ground)) slot_name=GSTRING(slot);
  else if (TYPEP(slot,frame_ground_type)) slot_name=frame_name(GFRAME(slot));
  else if (TYPEP(slot,symbol_ground)) slot_name=SYMBOL_NAME(slot);
  else ground_error(Type_Error,"not a slot",slot);
  annotation=probe_annotation(unit,slot_name);
  if (annotation) return annotation;
  else if (FRAMEP(slot))
    {annotation=use_annotation(unit,slot_name);
     set_prototype(annotation,GFRAME(slot));
     return annotation;}
  else return NULL;
}

Grounding get_value(Frame unit,char *slot)
{
  Frame set; ENSURE_CURRENT(unit);
  set=probe_annotation(unit,slot);
  if (set == NULL) return NULL;
  else return get_elements(set);
}

Grounding fraxl_get_value(Frame unit,Grounding slot)
{
  Frame set; set=get_slot(unit,slot);
  if (set == NULL) return NULL;
  else return get_elements(set);
}

Grounding has_value(Frame unit,char *slot,Grounding value)
{
  Frame set; ENSURE_CURRENT(unit);
  set=probe_annotation(unit,slot);
  if (set == NULL) return NULL;
  else return has_element(set,value);
}

Grounding fraxl_has_value(Frame unit,Grounding slot,Grounding value)
{
  Frame set; set=get_slot(unit,slot);
  if (set == NULL) return NULL;
  else return has_element(set,value);
}

Grounding put_value(Frame unit,char *slot,Grounding value)
{
  Frame set; ENSURE_CURRENT(unit);
  set=probe_annotation(unit,slot);
  if (set == NULL) return NULL;
  else return add_element(set,value);
}

Grounding fraxl_put_value(Frame unit,Grounding slot,Grounding value)
{
  Frame set; set=get_slot(unit,slot);
  if (set == NULL) return NULL;
  else return add_element(set,value);
}

Grounding retract_value(Frame unit,char *slot,Grounding value)
{
  Frame set; ENSURE_CURRENT(unit);
  set=probe_annotation(unit,slot);
  if (set == NULL) return NULL;
  else return remove_element(set,value);
}

Grounding fraxl_retract_value(Frame unit,Grounding slot,Grounding value)
{
  Frame set; set=get_slot(unit,slot);
  if (set == NULL) return NULL;
  else return remove_element(set,value);
}


/* Useful methods */

Grounding absorb_annotation_prototype(Frame frame,Frame annotation_prototype)
{
  Frame annotation;
  annotation = use_annotation(frame,annotation_prototype->aname);
  set_prototype(annotation,annotation_prototype);
  return NULL;
}

Grounding unabsorb_annotation_prototype(Frame frame,Frame annotation_prototype)
{
  Frame annotation;
  annotation = probe_annotation(frame,annotation_prototype->aname);
  if (annotation != NULL)
    if (frame_prototype(frame) != NULL)
      set_prototype(annotation,probe_annotation(frame->prototype,annotation->aname));
    else set_prototype(annotation,NULL);
  return NULL;
}

Grounding accumulate_grounding(Frame set)
{
  Grounding result; result = NULL; 
  {WITH_RESULT_ACCUMULATOR(results)
     {DO_PROTOTYPES(p,set)
	{DO_RESULTS(r,frame_ground(p)) accumulate_result(r,results);}}
   return resolve_accumulator(results);}
}


/* Demonic modifiers and accessors */

static Grounding unit_slot_value_list, unit_slot_list;

#define list3(x,y,z) \
  cons_pair((Grounding) x,cons_pair((Grounding) y,cons_pair((Grounding) z,empty_list)))
#define list2(x,y) \
  cons_pair((Grounding) x,cons_pair((Grounding) y,empty_list))

Grounding demonic_add(Frame frame,Grounding value)
{
  if ((find_result(value,frame_ground(frame))) == (Grounding ) NULL)
    {Grounding env; env=
       cons_pair(cons_pair(unit_slot_value_list,list3(frame_home(frame),frame,value)),
		 empty_list);
     add_to_ground(frame,value); USE_GROUND(env);
     {DO_PROTOTYPES(point,probe_annotation(frame,"put-demons"))
	{DO_RESULTS(demon,frame_ground(point))
	   {Grounding temp; temp=eval(demon,env); FREE_GROUND(temp);}}}
     FREE_GROUND(env);
     return NULL;}
  else return NULL;
}

Grounding demonic_remove(Frame frame,Grounding value)
{
  if ((find_result(value,frame_ground(frame))) != (Grounding ) NULL)
    {Grounding env; env=
       cons_pair(cons_pair(unit_slot_value_list,list3(frame_home(frame),frame,value)),
		 empty_list);
     remove_from_ground(frame,value); USE_GROUND(env);
     {DO_PROTOTYPES(point,probe_annotation(frame,"retract-demons"))
	{DO_RESULTS(demon,frame_ground(point))
	   {Grounding temp; temp=eval(demon,env); FREE_GROUND(temp);}}}
     FREE_GROUND(env);
     return NULL;}
  else return NULL;
}

Grounding demonic_get(Frame frame)
{
  Grounding env; env=
    cons_pair(cons_pair(unit_slot_list,list2(frame_home(frame),frame)),
	      empty_list);
  USE_GROUND(env);
  {WITH_RESULT_ACCUMULATOR(results)
     {DO_PROTOTYPES(point,probe_annotation(frame,"methods"))
	{DO_RESULTS(method,frame_ground(point))
	   {Grounding method_result; method_result=eval(method,env);
	    {DO_RESULTS(r,method_result) accumulate_result(r,results);
	     FREE_GROUND(method_result);}}}}
   FREE_GROUND(env);
   return resolve_accumulator(results);}
}

void do_demon_internal(Frame set,Grounding demon)
{
  /* Run the demon on this ones values */
  {DO_RESULTS(v,frame_ground(set))
     {Grounding temp, env; env=
	cons_pair(cons_pair(unit_slot_value_list,list3(frame_home(set),set,v)),
		  empty_list);
      temp=eval(demon,env); FREE_GROUND(temp); FREE_GROUND(env);}}
  {Frame_Array *spinoffs; spinoffs = frame_spinoffs(set);
   {DO_FRAMES(spinoff,spinoffs) do_demon_internal(spinoff,demon);}
   free(spinoffs->elements); free(spinoffs);}
}

Grounding do_demon(Frame frame,Grounding demon)
{
  do_demon_internal(frame,demon);
  return NULL;
}


/* Extract methods */

Grounding field_extractor(Grounding string,Grounding prefix,Grounding suffix)
{
  char *start, *end; generic_stream temp_stream; Grounding result;
  start=strstr(GSTRING(string),GSTRING(prefix));
  if (start == NULL) return NULL;
  else start=start+strlen(GSTRING(prefix));
  if (*(GSTRING(suffix)) == '\0')
    end=GSTRING(string)+strlen(GSTRING(string));
  else end=strstr(start,GSTRING(suffix));
  if (end == NULL) return NULL;
  else *end='\0';
  temp_stream.stream_type=string_input; temp_stream.ptr.string_in=(&start);
  WITH_HANDLING
    result=parse_ground(&temp_stream);
  ON_EXCEPTION
    CLEAR_EXCEPTION(); result=NULL;
  END_HANDLING;
  (*end) = (*(GSTRING(suffix)));
  return result;
}

char *get_string_till(char *start,char *till)
{
  char *result; ALLOCATE(result,char,till-start+1);
  strncpy(result,start,till-start); result[till-start]='\0';
  return result;
}

Grounding internal_match_string(Grounding pattern,char *string)
{
  if (pattern == empty_list) 
    if (*string == '\0') return empty_list;
    else return NULL;
  else if ((GCDR(pattern)) == empty_list)
    if (GCAR(pattern) == NULL) 
      return cons_pair(string_to_ground(string),empty_list);
    else {DO_RESULTS(r,GCAR(pattern)) 
	    if ((strcmp(GSTRING(r),string)) == 0) 
	      return cons_pair(r,empty_list);
	  return NULL;}
  else 
    {WITH_RESULT_ACCUMULATOR(results)
       if ((GCAR(pattern)) == NULL)
	 {DO_RESULTS(poss,GCAR(GCDR(pattern)))
	    {char *search_for, *found, *in; int size;
	     in=string; search_for=GSTRING(poss); size=strlen(search_for);
	     while ((found=strstr(in,search_for)) != NULL)
	       {DO_RESULTS(r,internal_match_string(GCDR(GCDR(pattern)),found+size))
		  {Grounding res; res=
		     cons_pair(string_to_ground(get_string_till(string,found)),
			       cons_pair(poss,r));
		   accumulate_result(res,results);}
		in=found+1;}}}
       else {DO_RESULTS(r,GCAR(pattern))
	       {char *pat; int size; pat=GSTRING(GCAR(pattern)); size=strlen(pat);
		if ((strlen(string) >= size) && (strncmp(pat,string,size) == 0))
		  {DO_RESULTS(r,internal_match_string(GCDR(pattern),string+size))
		     accumulate_result(cons_pair(GCAR(pattern),r),results);}}}
     return resolve_accumulator(results);}
}


Grounding match_string(Grounding pattern,Grounding string)
{
  return internal_match_string(pattern,GSTRING(string));
}


/* Local demonic record */

Grounding local_demonic_record(Frame set,Frame value)
{
  Frame local_frame; 
  Frame has_prototype(Frame x, Frame y);
  local_frame=probe_annotation(set,value->aname);
  if (local_frame == NULL)
    {set_prototype(set->home,value->home);
     local_frame=probe_annotation(set,value->aname);}
  else if (NOT(has_prototype(local_frame,value)))
    {set_prototype(set->home,value->home);
     set_prototype(local_frame,value);}
  demonic_add(set,frame_to_ground(local_frame));
  return NULL;
}


/* Describing frames */

Grounding describe_value_annotation(Frame annotation,Frame under)
{
  Frame prototype; Grounding values; 
  void labelled_frame_in_column(int column,char *label,Frame frame,Frame under);
  prototype  = frame_prototype(annotation);
  values     = get_elements(annotation); USE_GROUND(values);
  if (values == NULL) 
    add_cell(3,"No values");
  else {Grounding tmp; 
	tmp=apply1(find_function("VALUE-COLUMN"),values);
	FREE_GROUND(tmp);}
  if (!(prototype_is_default(annotation,prototype)))
    if (prototype == NULL) add_cell(3,"(no prototype)");
    else {labelled_frame_in_column(3,"is like ",prototype,under);}
  FREE_GROUND(values);
  return NULL;
}

/* A values setting command */

Grounding values_eq_prim(Grounding new_values)
{
  extern Frame point;
  Grounding current_values; current_values=get_elements(point);
  {DO_LIST(value,new_values)
     if (NOT(TYPEP(value,comment_ground)))
       if (NOT(find_result(value,current_values))) 
	 add_element(point,value);}
  {DO_RESULTS(value,current_values)
     if (NOT(in_list(value,new_values)))
       remove_element(point,value);}
  return (Grounding) point;
}


/* Text tools */

Frame lookup_char(Frame (*fcn)(),Frame bucket,char c)
{
  char buf[2]; buf[0]=c; buf[1]='\0';
  if (bucket) return fcn(bucket,buf);
  else return NULL;
}

#define hash_lookup(lookup_fn,bucket,string,index,size) \
   (((NULLP(bucket)) || (index >= size)) ? NULL \
    : (lookup_char(lookup_fn,bucket,string[index])))

Frame lexicon_lookup(Frame lexicon,char *word,Frame (*access_fn)())
{
  int size; Frame alpha, beta, gamma, delta;
  if (NULLP(probe_annotation(lexicon,"+indexed")))
    return access_fn(lexicon,word);
  size=strlen(word);
  alpha=lookup_char(access_fn,lexicon,word[0]);
  beta=lookup_char(access_fn,alpha,word[size-1]);
  gamma=lookup_char(access_fn,beta,word[size/2]);
  delta=lookup_char(access_fn,gamma,word[((size > 1) ? 1 :0)]);
  if (delta) return access_fn(delta,word);
  else return NULL;
}

Grounding lookup_word(Frame lexicon,Grounding string)
{
  return (Grounding) lexicon_lookup(lexicon,GSTRING(string),probe_annotation);
}

Grounding create_word(Frame lexicon,Grounding string)
{
  return (Grounding) lexicon_lookup(lexicon,GSTRING(string),make_annotation);
}

Grounding string_to_words(char *string)
{
  Grounding words; char *ptr; words=empty_list; ptr=string;
  while (*ptr != '\0')
    {char c; if (isalnum(*ptr)) while (isalnum(*ptr)) ptr++;
     else if (isspace(*ptr)) while (isspace(*ptr)) ptr++;
     else {c= *ptr; while (*ptr == c) ptr++;}
     if (NOT(isspace(*string)))
       {char c; c=(*ptr); (*ptr)='\0'; 
	words=cons_pair(string_to_ground(string),words);
	(*ptr)=c;}
     string=ptr;}
  if (words == empty_list) return words;
  else {Grounding in_order; in_order=reverse_list(words); FREE_GROUND(words);
	return in_order;}
}

Grounding fraxl_string_to_words(Grounding string)
{
  return string_to_words(GSTRING(string));
}


/* Initialization functions */

void init_arlotje()
{
  /* Initialize variables */
  home_symbol = intern("HOME");
  set_symbol = intern("SET");
  value_symbol = intern("VALUE");
  get_method_name=intern_frame_name("get-method"); 
  test_method_name=intern_frame_name("test-method");
  add_method_name=intern_frame_name("add-method");
  remove_method_name=intern_frame_name("remove-method");
  intern_frame_name("cache"); intern_frame_name("xcache");
  unit_slot_value_list=list3(home_symbol,set_symbol,value_symbol);
  unit_slot_list=list2(home_symbol,set_symbol);
  USE_GROUND(unit_slot_value_list); USE_GROUND(unit_slot_list);
  declare_fcn2(frame_fcn(get_slot),"get-slot",frame_ground_type,any_ground);
  declare_fcn2(fraxl_get_value,"get-value",frame_ground_type,any_ground);
  declare_fcn3(fraxl_put_value,"put-value",frame_ground_type,any_ground,any_ground);
  declare_fcn3(fraxl_has_value,"has-value",frame_ground_type,any_ground,any_ground);
  declare_fcn3(fraxl_retract_value,"retract-value",
	       frame_ground_type,any_ground,any_ground);
  declare_fcn1(get_elements,"get-elements",frame_ground_type);
  declare_fcn2(has_element,"has-element",frame_ground_type,any_ground);
  declare_fcn2(add_element,"add-element",frame_ground_type,any_ground);
  declare_fcn2(remove_element,"remove-element",frame_ground_type,any_ground);
  declare_lexpr(values_eq_prim,"values=");
  declare_fcn1(frame_ground_fn,"use-ground",frame_ground_type);
  declare_fcn2(add_to_ground,"add-to-set",frame_ground_type,any_ground);
  declare_fcn2(remove_from_ground,"remove-from-set",frame_ground_type,any_ground);
  declare_fcn2(frame_fcn(set_ground),"add-to-set-removing-current",
	       frame_ground_type,any_ground);
  declare_fcn1(get_inherited_ground,"inherit-grounding",frame_ground_type);
  declare_fcn1(get_inherited_ground,"get-inherited-ground",frame_ground_type);
  declare_fcn1(accumulate_grounding,"accumulate-grounding",frame_ground_type);
  declare_fcn1(demonic_get,"demonic-extraction",frame_ground_type);
  declare_fcn1(demonic_get,"demonic-get",frame_ground_type);
  declare_fcn2(demonic_add,"demonic-addition",frame_ground_type,any_ground);
  declare_fcn2(demonic_add,"demonic-add",frame_ground_type,any_ground);
  declare_fcn2(demonic_remove,"demonic-retraction",frame_ground_type,any_ground);
  declare_fcn2(demonic_remove,"demonic-remove",frame_ground_type,any_ground);
  declare_fcn2(local_demonic_record,"local-demonic-addition",
	       frame_ground_type,any_ground);
  declare_fcn2(local_demonic_record,"local-demonic-add",frame_ground_type,any_ground);
  declare_fcn3(field_extractor,"extract-field",string_ground,string_ground,string_ground);
  declare_fcn2(match_string,"match-string",pair_ground,string_ground);
  declare_fcn2(do_demon,"do-demon",frame_ground_type,pair_ground);
  declare_fcn2(absorb_annotation_prototype,"absorb-annotation-prototype",
	       frame_ground_type,frame_ground_type);
  declare_fcn2(unabsorb_annotation_prototype,"unabsorb-annotation-prototype",
	       frame_ground_type,frame_ground_type);
  declare_fcn2(describe_value_annotation,"describe-slot-value",
	       frame_ground_type,frame_ground_type);
  declare_fcn2(lookup_word,"lookup-word",frame_ground_type,string_ground);
  declare_fcn2(create_word,"create-word",frame_ground_type,string_ground);
  declare_fcn1(fraxl_string_to_words,"string->words",string_ground);
}

#if 0 /* Dead code */

/* Constrained values */

static Grounding changes=NULL;
boolean delay_changes=False;
Grounding accumulate_translated_ground(Frame f);
Grounding update_cache(Frame f,Frame cache);

#define invalidated_p(frame) search_bit_p(frame,CONSTRAINT_UPDATE_BIT)
#define set_invalidated_p(frame) set_search_bit(frame,CONSTRAINT_UPDATE_BIT)
#define clear_invalidated_p(frame) clear_search_bit(frame,CONSTRAINT_UPDATE_BIT)
#define get_analogous(x) frame_ground(x)

Grounding resolve_constraints(Grounding frame)
{
  Frame cache, constraints;
  Grounding results=NULL, cached_value; 
  if (results=frame_ground(GFRAME(frame))) return results;
  else cache=make_ephemeral_annotation(GFRAME(frame),"cache");
  if (cached_value=frame_ground(cache)) return cached_value;
  else if (cached_value=update_cache(GFRAME(frame),cache)) return cached_value;
  else return get_inherited_ground(GFRAME(frame));
}

Grounding assert_constrained_value(Frame frame,Grounding value)
{
  void update_dependents(Frame f);
  {Grounding old_value; if (old_value=frame_ground(frame)) 
     demonic_retract(frame,old_value);}
  demonic_record(frame,value); 
  update_dependents(frame);
  return value;
}

Grounding retract_constrained_value(Frame frame,Grounding value)
{
  Frame dependents; Grounding old_changes, old_value;
  void update_dependents(Frame f);
  demonic_retract(frame,value); 
  update_dependents(frame);
  return value;
}

void update_dependents(Frame frame)
{
  Frame dependents; Grounding old_changes;
  void invalidate_dependents(Frame f);
  void propogate_changes(Frame f);
  set_invalidated_p(frame);
  if (dependents=probe_annotation(frame,"dependents"))
    {DO_RESULTS(dependent,get_analogous(dependents))
       invalidate_dependents(GFRAME(dependent));}
  {UNWIND_PROTECT
     {Grounding new_changes;
      new_changes=get_analogous(probe_annotation(frame,"change-demons"));
      if (delay_changes) changes=merge_results(new_changes,changes);
      else {old_changes=changes; changes=new_changes;}
      USE_GROUND(changes); clear_invalidated_p(frame);
      if (dependents)
	{DO_RESULTS(dependent,get_analogous(dependents))
	   propogate_changes(GFRAME(dependent));}}
     ON_UNWIND
       if (NOT(delay_changes))
	 {DO_RESULTS(change,changes)
	    {Grounding temp; temp=eval(change,empty_list); FREE_GROUND(temp);}
	    FREE_GROUND(changes);
	    changes=old_changes;}
     END_UNWIND}
  
}

void invalidate_dependents(Frame f)
{
  Frame dependents, cache, xcache;
  if (invalidated_p(f)) return;
  else {set_invalidated_p(f); 
	cache=make_ephemeral_annotation(f,"cache"); 
	xcache=make_ephemeral_annotation(f,"xcache"); 
	xcache->ground=cache->ground; cache->ground=NULL;
	if (dependents=probe_annotation(f,"dependents"))
	  {DO_RESULTS(dependent,get_analogous(dependents))
	     invalidate_dependents(GFRAME(dependent));}}
}

Grounding update_cache(Frame f,Frame cache)
{
  Frame constraints; Grounding new_value=NULL;
  if (constraints=probe_annotation(f,"constraints")) /* Compute constraints */
    {DO_RESULTS(method,get_analogous(constraints))
       {Grounding add, newer_value; add=eval(method,empty_list);
	if ((add) && (new_value))
	  {newer_value=merge_results(add,new_value);
	   USE_GROUND(newer_value); FREE_GROUND(add); FREE_GROUND(new_value);
	   new_value=newer_value;}
	else if (new_value == NULL) {new_value=add; USE_GROUND(add);}}
     set_ground(cache,new_value); FREE_GROUND(new_value);}
  return new_value;
}

void propogate_changes(Frame f)
{
  Frame cache, xcache, constraints, dependents, controls; 
  Grounding ground, new_value=NULL;
  /* Remove marks, of if they're already removed, you've already done this node. */
  if (NOT(invalidated_p(f))) return; else clear_invalidated_p(f); 
  /* If there's a ground, clean up what you control first */
  if (ground=frame_ground(f))
    {if (controls=probe_annotation(f,"controls"))
       {DO_RESULTS(dependent,get_analogous(controls)) propogate_changes(f);}}
  /* Get the old value */
  cache=make_ephemeral_annotation(f,"cache");
  xcache=make_ephemeral_annotation(f,"xcache");
  /* If there was either an old value (which might have changed) or 
     an existing ground (which might be clobbered), recompute the
     current value. */
  if ((ground) || (xcache->ground))
    {new_value=update_cache(f,cache);
     /* If either they conflict with either the ground or the old value,
	add a change demon. */
     if (NOT((ground) ? (EQUAL_GROUND_P(ground,new_value)) :
	     (EQUAL_GROUND_P(xcache->ground,new_value))))
       {set_ground(f,NULL); set_ground(cache,NULL); update_cache(f,cache);
	changes=merge_results(get_analogous(probe_annotation(f,"change-demons")),
			      changes);}}
  set_ground(xcache,NULL);
  if (dependents=probe_annotation(f,"dependents"))
    {DO_RESULTS(dependent,get_analogous(dependents))
       if (invalidated_p(GFRAME(dependent)))
	 propogate_changes(GFRAME(dependent));}
}

#endif /* Dead Code */
