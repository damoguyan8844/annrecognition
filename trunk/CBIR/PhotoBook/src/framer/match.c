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
  This file implements the FRAMER matcher (underlying Mnemosyne).
*************************************************************************/

static char rcsid[] =
  "$Header: $";

#include <limits.h>
#include "framer.h"
#include "fraxl.h"
#include "internal/private.h"
#include "internal/eval.h"

/* Declarations for a short-sighted compiler... */
static void leave_trace(Frame frame);
static void mark_home(Frame frame);
void index_a_primitive(Frame frame);
Grounding get_indexed_spinoffs(Frame frame);
void reindex_after_shift(Frame frame);

#if 0 /* Simplify debugging for now... */
#pragma inline mapped_in_context_p
#pragma inline ground_matcher
#endif

#define prototype_marked_p(f)  search_bit_p(f,MNEMOSYNE_SEARCH_BIT)
#define mark_prototype(f)      set_search_bit(f,MNEMOSYNE_SEARCH_BIT)
#define unmark_prototype(f)    clear_search_bit(f,MNEMOSYNE_SEARCH_BIT)

#define MARK_PROTOTYPES(frame,bit) \
  DO_PROTOTYPES(p,(Frame) frame) set_search_bit(p,bit);
#define CLEAR_PROTOTYPES(frame,bit) \
  DO_PROTOTYPES(p,(Frame) frame) clear_search_bit(p,bit);


/* Computing mappings */

/* This returns the first prototype common to both <left> and <right>
   (if it exists) and NULL otherwise.  Note that here we count <left> and
   <right> as their own prototypes, so if <right> is a prototype of <left>,
   this would return <right>. */
Frame find_common_prototype(Frame left,Frame right)
{
  Frame common=NULL;
  {DO_PROTOTYPES(proto,left) mark_prototype(proto);}
  {DO_PROTOTYPES(proto,right)
    if (prototype_marked_p(proto))
     {common=proto;proto=NULL;}}
  {DO_PROTOTYPES(proto,left) unmark_prototype(proto);}
  return common;
}

/* Finds the cognate of <element> among the annotations of <in>.
   The cognate is the annotation of <in> which shares a prototype
   with <element> which is shared between no other pairs of annotations
   of <in> and <element>'s context (home). */
Frame find_cognate(Frame element,Frame in)
{
  Frame root=NULL, result=NULL;
  {DO_PROTOTYPES(proto,element) mark_prototype(proto);}
  {DO_FEATURES(f,frame_home(element))
     if (NOT(f == element))
       {DO_PROTOTYPES(proto,f) unmark_prototype(proto);}}
  {DO_FEATURES(b,in)
     {DO_PROTOTYPES(proto,b)
	if (prototype_marked_p(proto))
	  if (proto == root)
	    {unmark_prototype(proto); result=NULL;}
	  else {DO_PROTOTYPES(p,frame_prototype(proto)) unmark_prototype(p);
		result=b; root=proto;}}}
  {DO_PROTOTYPES(p,element) unmark_prototype(p);}
  return result;
}

/* Returns true (1) if target and base are cognates in their respective contexts. */
boolean mapped_p(Frame target,Frame base)
{
  return (boolean) (base == (find_cognate(target,frame_home(base))));
}

/* Returns true for elements in <context> which are cognates or elements
   outside of <context> which have a common prototype (could be cognates). */
boolean mapped_in_context_p(Frame target,Frame base,Frame context)
{
  Frame homer; homer=frame_home(target);
  while (NOT((homer == NULL) || (homer == context))) homer=homer->home;
  if (homer == NULL)
    {if ((find_common_prototype(target,base)) != NULL)
       return True; else return False;}
  else if ((find_cognate(target,frame_home(base))) == base)
    return True;
  else return False;
}

/* Returns true for elements in <context> which have common prototypes but
    no competing cognates or for elements outside of <context> which have
    a common prototype (could be cognates). */
boolean possibly_mapped_in_context_p(Frame target,Frame base,Frame context)
{
  Frame homer; homer=frame_home(target);
  while (NOT((homer == NULL) || (homer == context))) homer=homer->home;
  if (homer == NULL)
    {if ((find_common_prototype(target,base)) != NULL)
       return True; else return False;}
  else {Frame cognate; cognate=find_cognate(target,frame_home(base));
	if (cognate == base) return True;
	else if (NULLP(cognate))
	  {if ((NOT(find_cognate(base,frame_home(target)))) &&
	       (NOT(NULLP(find_common_prototype(target,base)))))
	     return True; else return False;}
	else return False;}
}


/* Where matching `grounds' out */

/* Returns true if TARGET and BASE match primitively, i.e.
   if <target> and <bases> handlers for the "match" message
   agree that they match. */
boolean match_primitives(Frame target,Frame base,Frame context)
{
  Grounding result; 
  result=send2(base,"match",frame_to_ground(target),frame_to_ground(context));
  if (result)
    {FREE_GROUND(result); return True;}
  else return False;
}

/* Returns true (1) if some element of <gx> is mapped to some
   element of <gy> in <context>. */
boolean ground_matcher(Grounding gx,Grounding gy,Frame context)
{
  if ((gx == gy) || (NULLP(gx))) return True;
  {DO_RESULTS(x,gx)
     {DO_RESULTS(y,gy)
	if ((NOT(FRAMEP(x))) || (NOT(FRAMEP(y)))) 
	  {if (EQUAL_GROUND_P(x,y)) return True;}
	else if (mapped_in_context_p(GFRAME(x),GFRAME(y),context))
	  return True;}}
  return False;
}

/* Returns true (1) if some pairing of elements from <gx> and <gy>
    is consistent with the known cognate mappings beneath <context>.
    E.G. the elements are other outside of <context> but share a common
    prototype, are inside of <context> and are mapped, or are inside of
    <context>, share a common prototype, and are mapped to no other elements.
*/
boolean weak_ground_matcher(Grounding gx,Grounding gy,Frame context)
{
  if ((gx == gy) || (NULLP(gx))) return True;
  {DO_RESULTS(x,gx)
     {DO_RESULTS(y,gy)
	if ((NOT(FRAMEP(x))) || (NOT(FRAMEP(y)))) 
	  {if (EQUAL_GROUND_P(x,y)) return True;}
	else if (possibly_mapped_in_context_p(GFRAME(x),GFRAME(y),context))
	  return True;}}
  return False;
}


/* Recursive structure matching */

/* The recursive matcher takes two frames and a context and
    matches them with respect to the context.  The context is
    used to determine which descriptive elements need to be
    strictly analogous and which need to only be *possibly*
    analogous.
*/
boolean matcher(Frame target,Frame base,Frame context)
{
  boolean some_features_exist=False;
  if (NOT(ground_matcher(frame_ground(target),frame_ground(base),context)))
    return False;
  else {DO_FEATURES(feature,target)
	  {Frame cognate; cognate=find_cognate(feature,base);
	   if ((cognate == NULL) || (NOT(matcher(feature,cognate,context))))
	     return False;
	   else some_features_exist=True;}
	  if (some_features_exist) return True;
	  /* This is a relative kludge that keeps empty descriptions from 
	     matching trivially. */
	  else if (NOT(NULLP(frame_ground(target)))) return True;
	  else if (NULLP(find_common_prototype(target,base))) return False;
	  else return True;}
}

boolean weak_matcher(Frame target,Frame base,Frame context)
{
  boolean some_features_exist=False;
  if (NOT(weak_ground_matcher(frame_ground(target),frame_ground(base),context)))
    return False;
  else {DO_FEATURES(feature,target)
	  {Frame cognate; cognate=find_cognate(feature,base);
	   if ((cognate == NULL) || (NOT(weak_matcher(feature,cognate,context))))
	     return False;
	   else some_features_exist=True;}
	  if (some_features_exist) return True;
	  /* This is a relative kludge that keeps empty descriptions from 
	     matching trivially. */
	  else if (NOT(NULLP(frame_ground(target)))) return True;
	  else if (NULLP(find_common_prototype(target,base))) return False;
	  else return True;}
}

/* This is a recursive matcher which sends messages at the primitive level.
   This makes it slower but more customizable than the common matchers. */
boolean object_matcher(Frame target,Frame base,Frame context)
{
  if ((NOT(NULLP((inherits_frame(base,"+commands"))))) &&
      (NOT(NULLP(get_handler(base,"match")))))
    /* Use primitive matcher if either target or base have matchers */
    return match_primitives(target,base,context);
  else {boolean some_features_exist=False;
	if (NOT(ground_matcher(frame_ground(target),frame_ground(base),context)))
	  return False;
        else {DO_FEATURES(feature,target)
		{Frame cognate; cognate=find_cognate(feature,base);
		 if ((cognate == NULL) || (NOT(matcher(feature,cognate,context))))
		   return False;
		 else some_features_exist=True;}
	      if (some_features_exist) return True;
	      /* This is a relative kludge that keeps empty descriptions from 
		 matching trivially. */
	      else if (NOT(NULLP(frame_ground(target)))) return True;
	      else if (NULLP(find_common_prototype(target,base))) return False;
	      else return True;}}
}


/* Shifting (and using the information) */

/* This changes the prototype of a frame to be another frame, propogating the
   change in prototype down to its annotations.  This propogate shifts every
   annotation to its cognate in the new prototype.  It also takes grounding
   relations to do further shifting so that if the grounds of one frame shifted
   to another have a common prototype, that relation is made into a cognate
   relation by shifting the corresponding frames.
*/

static int shift_bit; static Grounding shifted_frames;

static void i_shifter(Frame frame,Frame prototype,Frame context,boolean leave_traces)
{
  /* If you're leaving traces and there is a ground, leave them */
  if ((leave_traces) && (frame_ground(frame))) leave_trace(frame);
  /* Change the actual prototype relations unless it would create a cycle. */
  {Frame p; p=prototype; while ((p) && (p != frame)) p=p->prototype;
   if (NULLP(p)) set_prototype(frame,prototype);}
  /* Also, if you're leaving traces, reindex your primitives. */
  if ((leave_traces) && (frame_ground(frame))) index_a_primitive(frame);
  /* Shift all of your annotations to their cognates in the new prototype. */
  {DO_ANNOTATIONS(feature,frame)
     {Frame cognate; cognate=find_cognate(feature,prototype);
      if (cognate) i_shifter(feature,cognate,context,leave_traces);}}
  /* If the frame has a ground, try to use that ground to determine
     other cognate relations to create by further shifting. */
  if (frame_ground(frame))
    {Grounding targets, bases; targets=empty_list; bases=empty_list;
     /* This is basically a copy of find_cognate, operating over the result
	sets of targets and bases rather than all the features of a
	target and base. */
     {DO_RESULTS(g,frame_ground(frame))
	if ((FRAMEP(g)) && (NOT(search_bit_p((Frame) g,shift_bit))))
	  {Frame scan; scan=GFRAME(g);
	   while ((scan) && (scan != context)) scan=frame_home(scan);
	   if (scan) targets=cons_pair(g,targets);}}
     {DO_RESULTS(g,frame_ground(prototype))
	if (FRAMEP(g)) bases=cons_pair(g,bases);}
     {DO_LIST(element,targets)
	{Frame root=NULL, result=NULL;
	 {DO_PROTOTYPES(proto,GFRAME(element)) mark_prototype(proto);}
	 {DO_LIST(f,targets)
	    if (NOT(f == element))
	      {DO_PROTOTYPES(proto,(GFRAME(f))) unmark_prototype(proto);}}
	 {DO_LIST(b,bases)
	    {DO_PROTOTYPES(proto,(GFRAME(b)))
	       if (prototype_marked_p(proto))
		 if (proto == root)
		   {unmark_prototype(proto); result=NULL;}
		 else {DO_PROTOTYPES(p,frame_prototype(proto)) unmark_prototype(p);
		       result=GFRAME(b); root=proto;}}}
	 {DO_PROTOTYPES(p,GFRAME(element)) unmark_prototype(p);}
	 if (result)
	   {set_search_bit((Frame)element,shift_bit); 
	    shifted_frames=cons_pair(element,shifted_frames);
	    i_shifter(GFRAME(element),result,context,False);}}}
     FREE_GROUND(bases); FREE_GROUND(targets);}
}

static void just_leave_traces(Frame frame)
{
  if (frame_ground(frame)) leave_trace(frame);
  {DO_FEATURES(f,frame) just_leave_traces(f);}
}

static void top_shifter
  (Frame frame,Frame prototype,Frame context,boolean leave_traces)
{
  {UNWIND_PROTECT
     {shift_bit=grab_search_bit(); shifted_frames=empty_list;
      i_shifter(frame,prototype,context,leave_traces);}
   ON_UNWIND
     {{DO_LIST(elt,shifted_frames) 
	 clear_search_bit((Frame)elt,shift_bit);}
      if (shifted_frames != empty_list)
	{FREE_GROUND(shifted_frames);}
      release_search_bit(shift_bit);}
   END_UNWIND}
  if (leave_traces) reindex_after_shift(frame);
}

Frame shifter(Frame frame,Frame prototype,Frame context)
{
  if (NULLP(prototype))
    {just_leave_traces(frame); 
     set_prototype(frame,NULL); 
     reindex_after_shift(frame);}
  else if (find_common_prototype(frame,prototype))
    top_shifter(frame,prototype,context,False);
  else top_shifter(frame,prototype,context,True);
  return prototype;
}

Frame shift(Frame frame,Frame prototype)
{
  if ((NOT(NULLP(prototype))) &&
      (find_common_prototype(frame_home(frame),frame_home(prototype))))
    return shifter(frame,prototype,frame_home(frame));
  else return shifter(frame,prototype,frame);
}


/* Getting cognates */

Frame get_analog(Frame elt,Frame context)
{
  Frame result; result=find_cognate(elt,context);
  if (result) return result;
  shifter(frame_home(elt),context,frame_home(elt));
  return find_cognate(elt,context);
}

Grounding get_mappings(Frame target,Frame base)
{
  Grounding results=empty_list;
  i_shifter(target,base,target,False);
  {DO_FEATURES(f,target)
     {Frame cognate; cognate=find_cognate(f,base);
      if (cognate)
	{Grounding pair; pair=cons_pair(frame_to_ground(f),frame_to_ground(cognate));
	 results=cons_pair(pair,results);}}}
  return finalize_result_list(results);
}


/* Computing differences */

boolean unmapped_p(Frame target,Frame base)
{
  DO_ANNOTATIONS(f,target)
    if (find_cognate(f,base) == NULL) return True;
  return False;
}

Grounding unmapped(Frame target,Frame base)
{
  Grounding results=empty_list;
  {DO_FEATURES(f,target)
     {Frame cognate; cognate=find_cognate(f,base);
      if (cognate == NULL)
	results=cons_pair(frame_to_ground(f),results);}}
  return finalize_result_list(results);
}

/* What are these good for now? */
boolean difference_p(Frame target,Frame from)
{
  Frame cognate;
  cognate=find_cognate(target,from);
  if (cognate == NULL) return True;
  else if (matcher(target,cognate,frame_home(target)))
    return False;
  else return True;
}

Grounding differences(Frame target,Frame base)
{
  Grounding results=empty_list;
  {DO_FEATURES(f,target)
     {Frame cognate; cognate=find_cognate(f,base);
      if ((cognate == NULL) || (NOT(matcher(f,cognate,target))))
	results=cons_pair(frame_to_ground(f),results);}}
  return finalize_result_list(results);
}


/* Indexing frames */

/* This implements a bottom-up indexing and search mechanism for frames.
   The indexer applies to any FRAMER structure and indexes it and its
   annotations recursively.  The basic idea is that we decontextualize
   matching to make a weaker constraint which we can index on; thus for
   any descriptive element, we try to find the set of descriptive elements
   in memory which *might* match it and score each whole description by how
   many such elements they contain.

   A successful match between two descriptions is one where each element
   has a cognate with equal or cognate grounding.  The decontextualized
   version of the cognate relation is the common prototype relation;
   given this, the decontextualized version of the matching criteria is
   to find frames with common prototypes whose grounds are either equal
   or themselves common prototypes.  A contextual match between two
   descriptive elements always implies this weaker kind of match.

   The implementation of this indexing algorithm is constrained in two ways:
    + it must be efficient when dealing with large numbers of frames
    + it must be work by examining as few frames as possible, since
       each frame it examines means another load from memory

*/
static char *traces_string, *index_string, *mark_string, *score_string;
static int index_count=0;

/* This returns the frame which contains all frames with the same prototype
   as <frame> whose grounds contain <g> or some frame with a common prototype
   with <g>.  If <create> is true, the bucket is created if it doesn't exist. */
static Frame get_bucket(Frame frame,Grounding g,boolean create)
{
  Frame indices_frame; Grounding indices; _interned=index_string;
  if (create) indices_frame=make_annotation(frame,index_string);
  else indices_frame=probe_annotation(frame,index_string);
  if (indices_frame) indices=frame_ground(indices_frame); else indices=NULL;
  if (FRAMEP(g)) 
    {Frame p; p=GFRAME(g); 
     while (frame_prototype(p)) p=frame_prototype(p); g=frame_to_ground(p);
     {DO_RESULTS(bucket,indices)
	if ((GCAR(bucket)) == g) return (Frame) (GCDR(bucket));}}
  else {DO_RESULTS(bucket,indices)
	  if (EQUAL_P(GCAR(bucket),g)) return (Frame) (GCDR(bucket));}
  if (indices_frame)
    {Frame bucket; bucket=make_unique_annotation(indices_frame,"v");
     add_to_ground(indices_frame,cons_pair(g,frame_to_ground(bucket)));
     return bucket;}
  else return NULL;
}

/* Returns true if <f> or one of its home is marked as `no context',
   indicating that it should not be indexed. */
static boolean no_context_p(Frame f)
{
  Frame note;
  {DO_HOMES(home,f)
     if ((note=probe_annotation(home,"+nocontext")) &&
	 (NOT(NULLP(frame_ground(note)))))
       return True;}
  return False;
}

/* Leaving traces */
static void leave_trace(Frame frame)
{
  {DO_RESULTS(g,frame_ground(frame))
     {Frame p, bucket, traces; p=frame_prototype(frame);
      if ((NULLP(p)) || (frame_read_only_p(p)) || (no_context_p(p))) p=frame;
      bucket=get_bucket(p,g,True); traces=make_annotation(bucket,traces_string);
      add_to_ground(traces,frame_to_ground(frame));}}
}


/* Applies the function <search_fn> to all the frames co-indexed to <f>. */
void for_coindices(Frame frame,void (*search_fn)(Frame f))
{
  {DO_RESULTS(g,frame_ground(frame))
     {DO_PROTOTYPES(p,frame)
	{Frame bucket; bucket=get_bucket(p,g,False); 
	 if (bucket) 
	   {index_count++;
	    {DO_RESULTS(v,frame_ground(bucket)) 
	       search_fn((Frame) v);}}}}}
}

/* Indexes <frame> so that for_coindices will find it. */
void index_a_primitive(Frame frame)
{
  Frame p; p=frame_prototype(frame);
  if ((NULLP(p)) || (frame_read_only_p(p)) || (no_context_p(p))) p=frame;
  {DO_RESULTS(g,frame_ground(frame))
     add_to_ground(get_bucket(p,g,True),frame_to_ground(frame));}
}

/* Does a recursive descent of the FRAMER structure, 
   indexing primitives along the way. */
static void index_primitives_internal(Frame frame)
{
  {DO_FEATURES(f,frame) index_primitives_internal(f);}
  if (NOT(NULLP(frame_ground(frame)))) index_a_primitive(frame);
}

/* This is the user function for indexing primitives. */
Grounding index_primitives(Grounding frame)
{
  index_primitives_internal(GFRAME(frame));
  return frame;
}

/* This is the user function reindexing primitives */

static int reindex_bit; static Grounding reindexed_frames;

void reindex_internal(Frame frame)
{
  Grounding spinoffs=NULL;
  index_primitives_internal(frame_home(frame));
  set_search_bit(frame,reindex_bit);
  reindexed_frames=cons_pair(frame_to_ground(frame),reindexed_frames);
  spinoffs=get_indexed_spinoffs(frame);
  {DO_RESULTS(spinoff,spinoffs)
     if (NOT(search_bit_p((Frame)spinoff,reindex_bit)))
       reindex_internal((Frame) spinoff);}
  FREE_GROUND(spinoffs);
}

void reindex_after_shift(Frame frame)
{
  UNWIND_PROTECT
    {reindex_bit=grab_search_bit(); reindexed_frames=empty_list;
     reindex_internal(frame);}
  ON_UNWIND
    {{DO_LIST(elt,reindexed_frames) 
	clear_search_bit((Frame)elt,reindex_bit);}
     release_search_bit(reindex_bit);
     if (reindexed_frames != empty_list) {FREE_GROUND(reindexed_frames);}}
  END_UNWIND
}


/* Searching for similar frames */

/* The basic algorithm iterates down a frame's annotation structure, keeping
    track of the current depth.  At each annotation with a ground, it finds the
    co-indexed grounds and goes up based on the annotation's depth.  This gets
    the corresponding context which is then marked.

   The marking process uses an ephemeral +search annotation to store an id for
    the current search and a score which is incremented each time the frame is
    marked.  In addition, a +mark annotation makes sure that a home is only marked
    once for each original annotation in the key.

   The marking process adds a marked frame to search_result whenever its score
    passes the threshold of the current search.
*/

/* These identify the current search and the threshold past which
   results are added to search_result */
static short search_count=0, search_threshold=1;
/* This identifies the depth of the current search and
   the annotation currently being searched out from. */
static int search_depth=0;
/* This is where frames accumulate which have passed the threshold. */
static Grounding search_result;
/* This is the frame we are searching for analogs to */
static Frame search_home=NULL;

void search_annotations(Frame frame,int depth)
{
  {DO_FEATURES(f,frame) 
     search_annotations(f,depth+1);}
  if (frame_ground(frame))
    {search_depth=depth; for_coindices(frame,mark_home);}
}

static void mark_home(Frame primitive)
{
  Grounding make_point(short x,short y);
  /* Increment the local counter by one. */
  Frame top; Grounding value; int depth, icount; 
  /* Find the frame as far about <primitive> as the search
     is beneath its home. */
  {depth=search_depth; top=primitive; 
   while ((depth > 0) && (NOT(NULLP(top))))
     {top=frame_home(top); depth--;}}
  /* If there is no such frame or its your home, never mind... */
  if ((NULLP(top)) || (top == search_home)) return;
  /* This code makes sure that a frame only gets one point
     for each primitive annotation avoiding giving lots of points
     to frames with lots of potentially ambiguous matches. */ 
  {Frame mark; Grounding g; _interned=mark_string;
   mark=make_ephemeral_annotation(top,mark_string); g=frame_ground(mark);
   if ((g) && (TYPEP(g,integer_ground)) && ((GINTEGER(g)) == index_count)) return;
   else if (g) the(integer,g)=index_count; /* Kludge */
   else set_ground(mark,integer_to_ground(icount));}
  /* This code marks the home */
  {Frame score; _interned=score_string;
   score=make_ephemeral_annotation(top,score_string);
   if ((value=frame_ground(score)))
     if ((the(short_point,value)[0]) == search_count)
       {if ((++(the(short_point,value)[1])) == search_threshold)
	  search_result=cons_pair(frame_to_ground(top),search_result);}
     else {the(short_point,value)[0] = search_count;
	   the(short_point,value)[1] = 1;
	   if (search_threshold == 1) 
	     search_result=cons_pair(frame_to_ground(top),search_result);}
   else {set_ground(score,make_point(search_count,1));
	 if (search_threshold == 1) 
	   search_result=cons_pair(frame_to_ground(top),search_result);}}
}

/* This returns the indexing score for a particular frame. */
int index_score(Frame f)
{
  Frame score; Grounding pt;
  score=make_ephemeral_annotation(f,score_string);
  if (pt=frame_ground(score)) 
    if (TYPEP(pt,short_point_ground))
      return unground(pt,short_point)[1];
    else return 0;
  else return 0;
}


/* This is the top level function for doing the similarity search. 
   It takes a frame to start from and an integer threshold and returns
   the result set coming from the search.  This consists of all the frames
   marked by search_annotations. */
Grounding find_similar(Frame frame,int threshold)
{
  Grounding result;
  if (search_result != empty_list) 
    {FREE_GROUND(search_result); search_result=empty_list;}
  search_count++; search_threshold=threshold; search_home=frame;
  search_annotations(frame,0);
  result=list_to_result_set(search_result); 
  FREE_GROUND(search_result); search_result=empty_list; search_home=NULL;
  return result;
}


/* Incremental searches */

/* Incremental searches allow focussing on particular aspects of a frame.
   One starts an incremental search for a frame and then calls continue_search
   on those of its components which are found particularly valuable.  An incremental
   search is started by a call to start_search with a frame and a threshold; subsequent
   calls to continue_search on annotations beneath the initial frame will mark frames
   with common primitives at the same depth relative to the start frame.
   A repeated call to continue_search on a particular annotation increases the importance
   of that annotation in the overal scoring. The function get_search_result returns
   the current state of the search, while the function finish_search returns and
   clears the current state.
*/

/* This function starts a search for frames like <frame> whose similarity
   passes <threshold>. */
void start_search(Frame frame,int threshold)
{
  if (search_result != empty_list) 
    {FREE_GROUND(search_result); search_result=empty_list;}
  search_count++; search_threshold=threshold; 
  search_home=frame;
}

/* This function continues the current search focussing
   on the annotation <frame> */
void continue_search(Frame frame)
{
  Frame home; int depth=0; home=frame;
  while ((home) && (home != search_home)) {depth++; home=frame_home(home);}
  search_annotations(frame,depth);
}

/* This returns the state of the current search. */
Grounding get_search_result()
{
  return list_to_result_set(search_result);
}

/* This completes the current search, returning the final result. */
Grounding finish_search()
{
  Grounding result; 
  result=search_result; search_result=empty_list;
  return finalize_result_list(result);
}


/* Focussed sub searches */

/* A focussed sub search starts in an incremental search and does
   a search (from which it immediately returns) which is biased by
   the ongoing incremental search. */

static int subsearch_threshold; 
static int subsearch_bit;
static Grounding subsearch_result;

static void doing_subsearch(Frame primitive)
{
  Grounding make_point(short x,short y);
  /* Increment the local counter by one. */
  Frame top, score; Grounding value; int depth, icount; 
  /* Find the frame as far about <primitive> as the search
     is beneath its home. */
  {depth=search_depth; top=primitive; 
   while ((depth > 0) && (NOT(NULLP(top))))
     {top=frame_home(top); depth--;}}
  /* If there is no such frame, never mind, otherwise continue... */
  if (NULLP(top)) return; else _interned=score_string;
  if ((score=raw_local_probe_annotation(top,score_string)) &&
      ((value=frame_ground(score))) &&
      ((the(short_point,value)[0]) == search_count) &&
      (((the(short_point,value)[1])) > subsearch_threshold))
    if (NOT(search_bit_p(top,subsearch_bit)))
      {subsearch_result=cons_pair(frame_to_ground(top),subsearch_result);
       set_search_bit(top,subsearch_bit);}
}

static Grounding subsearch_annotations(Frame under,int depth)
{
  if (frame_ground(under))
    {search_depth=depth; for_coindices(under,doing_subsearch);}
  {DO_FEATURES(f,under) subsearch_annotations(f,depth+1);}
}


static int count_number_of_grounds(Frame frame)
{
  int count;
  if (frame_ground(frame)) count=1; else count=0;
  {DO_FEATURES(f,frame)
     count=count+count_number_of_grounds(f);}
  return count;
}

/* This function continues the current search focussing
   on the annotation <frame> */
Grounding subsearch_for(Frame frame,int threshold)
{
  Frame home; int depth=0; home=frame;
  while ((home) && (home != search_home)) {depth++; home=frame_home(home);}
  {UNWIND_PROTECT
     {subsearch_threshold=threshold; subsearch_result=empty_list; 
      subsearch_bit=grab_search_bit();
      subsearch_annotations(frame,depth);}
   ON_UNWIND
     {DO_LIST(elt,subsearch_result) clear_search_bit((Frame)elt,subsearch_bit);
      release_search_bit(subsearch_bit);}
   END_UNWIND}
  return finalize_result_list(subsearch_result);
}

/* This repeatedly calls subsearch_for with smaller and smaller
   (by a factor of 2) thresholds until it gets a result. */
Grounding subsearch(Frame frame)
{
  int threshold=0; threshold=count_number_of_grounds(search_home);
  while (threshold)
    {Grounding result; result=subsearch_for(frame,threshold);
     if (result) return result; else threshold=threshold/2;}
  return NULL;
}

/* This repeatedly calls find_similar with smaller and smaller
   (by a factor of 2) thresholds until it gets a result. */
Grounding find_most_similar(Frame frame)
{
  start_search(frame,100); continue_search(frame);
  return subsearch(frame);
}


/* Finding indexed spinoffs */

/* This code finds all the spinoffs of a frame which have been indexed. 
   It uses the indexing information on it and its annotations to find
    the corresponding frames and homes. */

static int find_spinoffs_depth;
static Grounding find_spinoffs_list;

static void collect_indexed_spinoffs(Frame frame)
{
  find_spinoffs_depth++;
  {DO_FEATURES(f,frame) collect_indexed_spinoffs(f);}
  find_spinoffs_depth--;
  if (NOT(NULLP(frame_ground(frame))))
    {Frame indices_frame; indices_frame=local_probe_annotation(frame,index_string);
     if (indices_frame)
       {DO_RESULTS(index,frame_ground(indices_frame))
	  {DO_RESULTS(r,frame_ground(GCDR(index)))
	     {Frame f; int depth; depth=find_spinoffs_depth; f=GFRAME(r); 
	      while ((f) && (depth > 0)) {f=frame_home(f); depth--;}
	      if ((NOT(NULLP(f))) && (NOT(search_bit_p(f,MNEMOSYNE_SEARCH_BIT))))
		{set_search_bit(f,MNEMOSYNE_SEARCH_BIT); 
		 find_spinoffs_list=
		   cons_pair(frame_to_ground(f),find_spinoffs_list);}}}}}
}

Grounding get_indexed_spinoffs(Frame frame)
{
  {UNWIND_PROTECT
     {find_spinoffs_list=empty_list; find_spinoffs_depth=0;
      collect_indexed_spinoffs(frame);}
   ON_UNWIND
     {DO_LIST(elt,find_spinoffs_list)
	clear_search_bit(GFRAME(elt),MNEMOSYNE_SEARCH_BIT);}
   END_UNWIND}
  return finalize_result_list(find_spinoffs_list);
}


/* Finding common primitives */

/* This is mostly for debugging.  It returns the primitives which are shared
   between two frames (and the reason for their similarity according
   to find_similar).
*/

static Grounding common_primitives;
static Frame primitive_home;

static void note_common_primitives(Frame primitive)
{
  Frame frame; int depth; depth=search_depth; frame=primitive;
  while ((depth > 0) && (NOT(NULLP(frame))))
    {frame=frame_home(frame); depth--;}
  if (NULLP(frame)) return;
  else if (frame != primitive_home) return;
  else {common_primitives=
	  cons_pair(frame_to_ground(primitive),common_primitives);}
}

static void note_coindexed_primitives(Frame frame,int depth)
{
  {DO_FEATURES(f,frame) note_coindexed_primitives(f,depth+1);}
  if (frame_ground(frame))
    {search_depth=depth; for_coindices(frame,note_common_primitives);}
}

Grounding find_common_primitives(Frame f1,Frame f2)
{
  primitive_home=f2; common_primitives=empty_list;
  note_coindexed_primitives(f1,0);
  return finalize_result_list(common_primitives);
}


/* Finding justified matches */

/* A justified match is a match with a precedent.  A match between
   x and y is justified if some frame similar to x was at some point
   shifted to y (and traces left). */

static boolean look_for_justified_matches(Frame frame,int depth)
{
  if (frame_ground(frame)) 
    {DO_RESULTS(g,frame_ground(frame))
       {DO_PROTOTYPES(p,frame)
	  {Frame bucket; bucket=get_bucket(p,g,False);
	   if (bucket) 
	     {Frame traces_frame; 
	      traces_frame=probe_annotation(bucket,traces_string);
	      if (traces_frame)
		{DO_RESULTS(trace,frame_ground(traces_frame))
		   {Frame home; int d; home=GFRAME(trace); d=depth;
		    while ((d) && (home)) {home=frame_home(home); d--;}
		    {DO_PROTOTYPES(p,home)
		       if (search_bit_p(p,MNEMOSYNE_SEARCH_BIT))
			 return True;}}}}}}}
  {DO_FEATURES(f,frame) 
     if (look_for_justified_matches(f,depth+1)) return True;}
  return False;
}

boolean justified_match_p(Frame left,Frame right)
{
  if (find_common_prototype(left,right)) return True;
  else {boolean result=False; 
	{UNWIND_PROTECT
	   {{DO_PROTOTYPES(p,left) set_search_bit(p,MNEMOSYNE_SEARCH_BIT);}
	    result=look_for_justified_matches(right,0);}
	 ON_UNWIND
	   {DO_PROTOTYPES(p,left) clear_search_bit(p,MNEMOSYNE_SEARCH_BIT);}
	 END_UNWIND}
	return result;}
}


/* Finding matches */

/* These are various functions for guessing at matches which are
   not currently cognates.  These can be either ambiguous cognates
   or elements which have no common prototypes at all.

   They all use the function return_match_set to handle competitive
   matching; this removes matches which conflict with one another. */

static Grounding remove_matches_to(Grounding target,Grounding match_list)
{
  if (match_list == empty_list) return match_list;
  else if ((GCDR(GCAR(match_list))) == target)
    return remove_matches_to(target,GCDR(match_list));
  else {Grounding new_tail; new_tail=remove_matches_to(target,GCDR(match_list));
	if (new_tail == (GCDR(match_list))) return match_list;
	else return cons_pair((GCAR(match_list)),new_tail);}
}

static Grounding minimize_match_list(Grounding list)
{
  if (list == empty_list) return list;
  else {Grounding head, tail; head=GCAR(list);
	tail=remove_matches_to(GCDR(head),GCDR(list));
	if (tail == GCDR(list)) 
	  {Grounding new_tail; new_tail=minimize_match_list(tail);
	   if (new_tail == tail) return list;
	   else return cons_pair(head,new_tail);}
	else {Grounding result; result=minimize_match_list(tail);
	      if (result != tail) {FREE_GROUND(tail);}
	      return result;}}
}

Grounding return_match_set(Grounding list)
{
  Grounding result, minimized; 
  if (list == empty_list) return NULL;
  else minimized=minimize_match_list(list);
  if (minimized != list) {FREE_GROUND(list);}
  result=finalize_result_list(minimized);
  return result;
}



/* find_disambiguated_matches finds matches among elements with a common
   prototype --- e.g. ambiguous cognate relations --- based on
   consistency with existing cognate relations */
Grounding find_disambiguated_matches(Frame description,Frame prototype)
{
  Grounding results=empty_list, differences, expectations;
  differences=unmapped(description,prototype);
  expectations=unmapped(prototype,description);
  {DO_RESULTS(difference,differences)
     {Grounding match=NULL;
      {DO_RESULTS(expectation,expectations)
	 if ((find_common_prototype(GFRAME(difference),GFRAME(expectation))) &&
	     (weak_matcher(GFRAME(difference),GFRAME(expectation),description)))
	   if (match) {match=NULL; STOP_DOING_RESULTS();}
	   else match=expectation;}
      if (match) results=cons_pair(cons_pair(difference,match),results);}}
  FREE_GROUND(differences); FREE_GROUND(expectations);
  return return_match_set(results);
}

/* find_consistent_matches returns justified matches which are consistent
   with existing cognate relations. */
Grounding find_consistent_matches(Frame description,Frame prototype)
{
  Grounding results=empty_list, differences, expectations;
  differences=unmapped(description,prototype);
  expectations=unmapped(prototype,description);
  {DO_RESULTS(difference,differences)
     {Grounding match=NULL;
      {DO_RESULTS(expectation,expectations)
	 if ((justified_match_p(GFRAME(difference),GFRAME(expectation))) &&
	     (weak_matcher(GFRAME(difference),GFRAME(expectation),description)))
	   if (match) {match=NULL; STOP_DOING_RESULTS();}
	   else match=expectation;}
      if (match) results=cons_pair(cons_pair(difference,match),results);}}
  FREE_GROUND(differences); FREE_GROUND(expectations);
  return return_match_set(results);
}

/* find_random_matches finds matches between arbitrary pairings of unmapped elements
    which are consistent with existing cognate relations. */
Grounding find_random_matches(Frame description,Frame prototype)
{
  Grounding results=empty_list, differences, expectations;
  differences=unmapped(description,prototype);
  expectations=unmapped(prototype,description);
  {DO_RESULTS(difference,differences)
     {Grounding match=NULL;
      {DO_RESULTS(expectation,expectations)
	 if ((weak_matcher(GFRAME(difference),GFRAME(expectation),description)))
	   if (match) {match=NULL; STOP_DOING_RESULTS();}
	   else match=expectation;}
      if (match) results=cons_pair(cons_pair(difference,match),results);}}
  FREE_GROUND(differences); FREE_GROUND(expectations);
  return return_match_set(results);
}


/* find_inconsistent_matches finds matches between arbitrary pairings of 
    elements which are either unmapped or incomplete matches, not caring
    about consistency with existing cognate relations. */
Grounding find_inconsistent_matches(Frame description,Frame prototype)
{
  Grounding results=empty_list, diffs, expectations;
  diffs=differences(description,prototype);
  expectations=differences(prototype,description);
  {DO_RESULTS(difference,diffs)
     {Grounding match=NULL;
      {DO_RESULTS(expectation,expectations)
	 if (matcher(GFRAME(difference),GFRAME(expectation),GFRAME(difference)))
	   if (match) {match=NULL; STOP_DOING_RESULTS();}
	   else match=expectation;}
      if (match) results=cons_pair(cons_pair(difference,match),results);}}
  FREE_GROUND(diffs); FREE_GROUND(expectations);
  return return_match_set(results);
}


/* Translating grounds */

Frame translate_frame(Frame ground,Frame from,Frame into)
{
  if (ground == from) return into;
  else if (has_home(from,ground))
    {while (ground != from) {from=frame_home(from); into=frame_home(into);}
     return into;}
  else if (has_home(ground,from))
    {Frame translated_home;
     translated_home=translate_frame(frame_home(ground),from,into);
     if (translated_home) return find_cognate(ground,translated_home);
     else return NULL;}
  else {while ((from) && (into) && (NOT(mapped_p(from,into))))
	  {from=frame_home(from); into=frame_home(into);}
	if ((from) && (into))
	  return translate_frame(ground,from,into);
	else return ground;}
}

Grounding translate_ground(Grounding ground,Frame from,Frame into)
{
  if (FRAMEP(ground)) 
    return frame_to_ground(translate_frame(GFRAME(ground),from,into));
  else if (ground == empty_list) return ground;
  else if (TYPEP(ground,pair_ground))
    return cons_pair(translate_ground(GCAR(ground),from,into),
		     translate_ground(GCDR(ground),from,into));
  else if (TYPEP(ground,nd_ground))
    {WITH_RESULT_ACCUMULATOR(results)
       {DO_RESULTS(r,ground) 
	  accumulate_result(translate_ground(r,from,into),results);}
       return resolve_accumulator(results);}
  else if (TYPEP(ground,vector_ground))
    {Grounding new_vec; GVMAKE(new_vec,GVSIZE(ground));
     {DO_TIMES(i,GVSIZE(ground))
	{GVSET(new_vec,i,translate_ground(GVREF(ground,i),from,into));}}
     return new_vec;}
  else if (TYPEP(ground,procedure_ground))
    return close_procedure(translate_ground(proc_text(ground),from,into),
			   translate_ground(proc_env(ground),from,into));
  else return ground;
}

Frame make_cognate(Frame of,Frame in)
{
  Frame result; result=find_cognate(of,in);
  if (NOT(NULLP(result))) return result;
  else return probe_annotation(in,frame_name(of));
}

Grounding get_translated_ground(Frame frame)
{
  Frame target; target=frame;
  while ((NOT(NULLP(target))) && (NULLP(frame_ground(target))))
    target=frame_prototype(target);
  if (target == NULL) return NULL;
  else if (target == frame) return frame_ground(target);
  else return translate_ground(frame_ground(target),target,frame);
}

Grounding accumulate_translated_ground(Frame frame)
{
  return frame_ground(frame);
}

Grounding cache_translated_ground(Frame frame)
{
  Frame target; target=frame;
  while ((NOT(NULLP(target))) && (NULLP(frame_ground(target))))
    target=frame_prototype(target);
  if (target == NULL) return NULL;
  else if (target == frame) return frame_ground(target);
  else {Grounding result;
	result=translate_ground(frame_ground(target),target,frame);
	set_ground(frame,result); 
	return result;}
}



/* FRAXL wrappers for matching functions */

Grounding fraxl_matcher(Grounding target,Grounding base,Grounding context)
{
  if (matcher(GFRAME(target),GFRAME(base),GFRAME(context)))
    return target;
  else return NULL; 
}

Grounding fraxl_match(Grounding target,Grounding base)
{
  if (matcher(GFRAME(target),GFRAME(base),GFRAME(target)))
    return target; else return NULL;
}

Grounding fraxl_weak_matcher(Grounding target,Grounding base,Grounding context)
{
  if (weak_matcher(GFRAME(target),GFRAME(base),GFRAME(context)))
    return target;
  else return NULL; 
}

Grounding fraxl_weak_match(Grounding target,Grounding base)
{
  if (weak_matcher(GFRAME(target),GFRAME(base),GFRAME(target)))
    return target;
  else return NULL; 
}

Grounding fraxl_unmapped_p(Grounding target,Grounding base)
{
  if (unmapped_p(GFRAME(target),GFRAME(base)))
    return target; else return NULL;
}

Grounding fraxl_difference_p(Grounding target,Grounding base)
{
  if (difference_p(GFRAME(target),GFRAME(base)))
    return target; else return NULL;
}

/* FRAXL wrappers for shifting functions. */

Grounding fraxl_shift(Grounding frame,Grounding prototype)
{
  shift(GFRAME(frame),GFRAME(prototype));
}

Grounding fraxl_shifter(Grounding frame,Grounding prototype,Grounding context)
{
  shifter(GFRAME(frame),GFRAME(prototype),GFRAME(context));
}

Grounding fraxl_snap(Grounding frame)
{
  shift(GFRAME(frame),NULL);
}


/* Fraxl wrappers for indexing functions */

/* This extracts the indexing score for a particular search result. */
Grounding fraxl_index_score(Frame f)
{
  return integer_to_ground(index_score(f));
}

/* This is a fraxl binding for find_similar */
Grounding fraxl_find_similar(Frame frame,Grounding threshold)
{
  return find_similar(frame,GINTEGER(threshold));
}

Grounding fraxl_subsearch_for(Grounding frame,Grounding threshold)
{
  return subsearch_for((Frame)frame,GINTEGER(threshold));
}

/* This is a fraxl binding for find_similar which assumes
   a threshold of 1. */
Grounding fraxl_find_all_similar(Frame frame)
{
  return find_similar(frame,1);
}

Grounding fraxl_start_search(Grounding frame,Grounding threshold)
{
  start_search(GFRAME(frame),GINTEGER(threshold));
  return frame;
}

Grounding fraxl_continue_search(Grounding frame)
{
  continue_search(GFRAME(frame));
  return frame;
}


/* Temporarily here. */

int get_ground_size(Grounding ground)
{
  if (ground == NULL) return 0;
  switch (ground_type(ground))
    {
    case frame_ground_type: return 0;
    case pair_ground: case rational_ground: case procedure_ground:
      if (ground == empty_list) return 0;
      return 1+get_ground_size(GCAR(ground))+get_ground_size(GCDR(ground));
    case vector_ground: case nd_ground:
      {int size=1;
       {DO_VECTOR(elt,ground) size=size+get_ground_size(elt)+1;}
       return size;}
    case integer_ground: case string_ground: case float_ground: case short_point_ground: return 1;
    case symbol_ground: case framer_function_ground: return 0;
    default: return 0;
    }
}

int get_frame_size(Frame f)
{
  int size=1;
  if (frame_ground(f)) size=size+get_ground_size(frame_ground(f));
  if (f->appendix) size++; /* Count annotations as an extra */
  {DO_ANNOTATIONS(a,f) size=size+get_frame_size(a);}
  return size;
}

Grounding fraxl_get_frame_size(Grounding frame)
{
  return integer_to_ground(get_frame_size(GFRAME(frame)));
}


/* Declaring the functions of interest. */
void init_mnemosyne()
{
  traces_string=intern_frame_name("traces");
  index_string=intern_frame_name("+index");
  mark_string=intern_frame_name("+mark");
  score_string=intern_frame_name("+score");
  common_primitives=empty_list; search_result=empty_list;
  declare_fcn2(frame_fcn(find_common_prototype),"find-common-prototype",
	       frame_ground_type,frame_ground_type);
  declare_fcn2(frame_fcn(find_cognate),"find-cognate",
	       frame_ground_type,frame_ground_type);
  declare_fcn1(fraxl_snap,"snap",frame_ground_type);
  declare_fcn2(fraxl_shift,"shift",frame_ground_type,frame_ground_type);
  declare_fcn3(fraxl_shifter,"shifter",
	       frame_ground_type,frame_ground_type,frame_ground_type);
  declare_fcn3(fraxl_matcher,"matcher",
	       frame_ground_type,frame_ground_type, frame_ground_type);
  declare_fcn2(fraxl_match,"match",frame_ground_type,frame_ground_type);
  declare_fcn3(fraxl_weak_matcher,"weak-matcher",
	       frame_ground_type,frame_ground_type, frame_ground_type);
  declare_fcn2(fraxl_weak_match,"weak-match",frame_ground_type,frame_ground_type);
  declare_fcn2(unmapped,"unmapped",frame_ground_type,frame_ground_type);
  declare_fcn2(differences,"differences",frame_ground_type,frame_ground_type);
  declare_fcn2(fraxl_unmapped_p,"unmapped-p",frame_ground_type,frame_ground_type);
  declare_fcn2(fraxl_difference_p,"difference-p",frame_ground_type,frame_ground_type);
  declare_fcn2(find_disambiguated_matches,"find-disambiguated-matches",
	       frame_ground_type,frame_ground_type);
  declare_fcn2(find_consistent_matches,"find-consistent-matches",
	       frame_ground_type,frame_ground_type);
  declare_fcn2(find_random_matches,"find-random-matches",
	       frame_ground_type,frame_ground_type);
  declare_fcn2(find_inconsistent_matches,"find-inconsistent-matches",
	       frame_ground_type,frame_ground_type);
  declare_fcn1(index_primitives,"index-primitives",frame_ground_type);
  declare_fcn1(fraxl_find_all_similar,"find-all-similar",frame_ground_type);
  declare_fcn1(find_most_similar,"find-most-similar",frame_ground_type);
  declare_fcn1(subsearch,"subsearch",frame_ground_type);
  declare_fcn1(fraxl_index_score,"index-score",frame_ground_type);
  declare_fcn2(fraxl_find_similar,"find-similar",frame_ground_type,integer_ground);
  declare_fcn2(fraxl_start_search,"start-search",frame_ground_type,integer_ground);
  declare_fcn2(fraxl_subsearch_for,"subsearch-for",frame_ground_type,integer_ground);
  declare_fcn1(fraxl_continue_search,"continue-search",frame_ground_type);
  declare_fcn0(get_search_result,"search-results");
  declare_fcn0(finish_search,"finish-search");
  declare_fcn2(find_common_primitives,"find-common-primitives",
	       frame_ground_type,frame_ground_type);
  declare_fcn3(translate_ground,"translate-ground",
	       any_ground,frame_ground_type,frame_ground_type);
  declare_fcn1(get_translated_ground,"get-translated-ground",frame_ground_type);
  declare_fcn1(cache_translated_ground,"cache-translated-ground",
	       frame_ground_type);
  declare_fcn1(accumulate_translated_ground,"accumulate-translated-ground",
	       frame_ground_type); 
  declare_fcn2(frame_fcn(get_analog),"get-analog",frame_ground_type,frame_ground_type);
  declare_fcn2(get_mappings,"get-mappings",frame_ground_type,frame_ground_type);
  declare_fcn1(get_indexed_spinoffs,"indexed-spinoffs",frame_ground_type);
  declare_fcn1(fraxl_get_frame_size,"frame-size",frame_ground_type);
}

#if 0

/* Temporary for cleanup */

static Grounding frames_to_delete;

static void prepare_for_cleanup(Frame f)
{
  char *name; name=intern_frame_name(frame_name(f));
  if ((name == occurrences_string) || 
      (name == variants_string) ||
      (name == traces_string) ||
      (name == occurrence_of_string))
    {set_prototype(f,NULL); set_ground(f,NULL);}
  else {DO_ANNOTATIONS(a,f) prepare_for_cleanup(a);}
}

static void final_cleanup(Frame f)
{
  char *name; name=intern_frame_name(frame_name(f));
  if ((name == occurrences_string) || 
      (name == variants_string) ||
      (name == traces_string) ||
      (name == occurrence_of_string))
    delete_annotation(f);
  else {DO_ANNOTATIONS(a,f) prepare_for_cleanup(a);}
}

static Grounding cleanup1(Grounding g)
{
  prepare_for_cleanup((Frame)g);
  return NULL;
}

static Grounding cleanup2(Grounding g)
{
  final_cleanup((Frame)g);
  return NULL;
}
#endif

/*
  Design notes:
   The tutoring process is one of giving the system correspondences
    between analogous elements of texts.  Suppose we are trying to teach
    it about political appointments.  We take two texts with different
    phrasal expressions of appointments and shift the corresponding elements
    to each other.  Then to check that we generalize appropriately, we find
    other similar phrasings and try and extract the analogous elements.

   Ultimately, the understanding process will not just involve indexing, but
    will involve the search for similar prototype stories and the drawing of
    analogies between them.  Perhaps a good starting place to look for matches
    would focus on verbs, but we could play with that as events demand.

   The common element is indexing on the word, sentence or phrase.  That is what
    the interface needs to support.

   So here's the routine:
    You want the system to `understand' stories about X.  You do a search
     for one phrasing and then see if get-analog does its thing.  If it doesn't,
     you're hosed.  If it does, search for another phrasing and give the system
     the analogies you're interested in for one of the alternative phrasings.  Then
     shift all of the other texts with that phrasing to the first one.  This should
     allow you to extract analogies through the other case.  Continue doing so.

    What do you do if get-analog fails?  Well, since you picked out a related
     phrasing, you've got words down for you.  The problems might be either
     initial mis-categorization (it doesn't seem X as a person, for instance)
     or amibiguity.  Let's see how much of a problem ambiguity is....
*/

/*
  Local variables: ***
  compile-command: "../etc/compile-framer" ***
  tags-file-name: "../sources/TAGS" ***
  End: **
*/

