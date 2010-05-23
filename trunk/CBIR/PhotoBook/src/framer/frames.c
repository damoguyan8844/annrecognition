/* $Header: /mas/framer/sources/RCS/frames.c,v 1.82 1994/01/14 00:55:11 rnewman Exp $ */

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
  This file implements the frame data structure and the basic operations on
  frames.  It is the kernel of FRAMER.

  Modification history:
    13 June 1992 - Began tracking modification history (Haase)
*************************************************************************/

/* This file defines the basic operations on frames. */

#define FAST_STREAMS 1
#include "framer.h"
#include "internal/private.h"

exception Read_Only_Frame="Cannot modify a read-only frame";
exception Out_Of_Memory="Out of memory (malloc failed)";
exception Rootless_Reference="Attempt to read root relative reference without a root";
exception Recursive_Prototype="a frame cannot be its own prototype!";
exception Illegal_Frame_Name="Illegal frame name";
exception Frame_Collision="Collision between frame names";
exception No_Zip_Codes="Reading a frame ZIP code out of context";
exception Unknown_Zip_Code="A Zip code is out of bounds";
extern exception Type_Error;
boolean trap_unknown_frames=False;
struct FRAME actual_root_frame =
   {(FRAME_P_MASK | FRAME_CURRENT_MASK), 0, "root",
      NULL,NULL,NULL,NULL};
Frame root_frame = &actual_root_frame;
Frame read_root=NULL;
int root_print_depth=3;
Frame_Array *zip_codes=NULL;
boolean suppress_autoload=False;
unsigned char frame_creation_mask=(FRAME_P_MASK | FRAME_TYPE_MASK);
char *_interned;

boolean restore_from_image(Frame frame);
extern boolean active_image_p;
static boolean caching_zip_codes=False;

Frame set_remote_ground AP((Frame f,Grounding g));
Frame set_remote_prototype AP((Frame f,Frame p));
Frame add_remote_annotation AP((Frame f,char *name));


/* Allocation Macrology */

/* Special stuff for allocating memory.  Combines with code in allocate.c */

/* This allocates a structure from a block of uninitialized structures
   bounded between <first> and <last>; if the block is empty, <overflow>
   is evaluated */
#define ALLOCATOR(first,last,overflow) \
   ((first == last) ? (overflow) : (first++))

#define allocate_frame() (ALLOCATOR(frame_block,end_of_frame_block,frame_underflow()))
#define allocate_frame_appendix() \
   ALLOCATOR(frame_appendix_block,end_of_frame_appendix_block,frame_appendix_underflow())

#define FRAME_ARRAY_MINIMUM_INCREMENT 5

/* This is for debugging memory */
#if ALPHA_RELEASE
#undef allocate_frame
Frame frame_checker AP((Frame frame));
#define allocate_frame() \
  frame_checker(ALLOCATOR(frame_block,end_of_frame_block,frame_underflow()))
#endif /* ALPHA_RELEASE */


/* The Annotation Hierarchy */

/* Interning frame names */

/* A table of frame names serves two purposes: in small-memory environments,
   it saves on string space; in other environments, it lets us use == comparison
   to find matching frame names.  This is especially helpful in using the annotation
   cache.
*/

static struct CHAR_BUCKET name_lookup_table = {'\0',{NULL,NULL}};

char **frame_name_table;
int number_of_frame_names=0, space_in_frame_name_table=0;

struct CHAR_BUCKET *find_bucket(char *string,struct CHAR_BUCKET *beneath)
{
  struct CHAR_BUCKET *bottom, *top; char kc; 
  while (1)
    {kc=tolower(*string); 
     bottom=beneath->data.branches.head; top=beneath->data.branches.tail; 
     if (top == NULL) /* When its an empty node */
       {struct CHAR_BUCKET *new; new=fra_allocate(sizeof(struct CHAR_BUCKET),1);
	new->key=kc; beneath->data.branches.head=new; beneath->data.branches.tail=new;
	if (kc == '\0') {new->data.string=NULL; return new;}
	else {new->data.branches.head=NULL; new->data.branches.tail=NULL;
	      beneath=new; string=string+1;}}
     else {struct CHAR_BUCKET *halfway=NULL;
	   while (top >= bottom)
	     {halfway=bottom+(top-bottom)/2;
	      if (halfway->key == kc) top=NULL;
	      else if (kc < halfway->key) top=halfway-1;
	      else bottom=halfway+1;}
	   if (NULLP(top))
	     if (kc == '\0') return halfway; 
	     else {string=string+1; beneath=halfway;}
	   else {struct CHAR_BUCKET *new; int index, size;
		 size=beneath->data.branches.tail-beneath->data.branches.head+1; 
		 if (top < halfway)
		   index=halfway-beneath->data.branches.head;
		 else index=(halfway-beneath->data.branches.head)+1;
		 new= (struct CHAR_BUCKET *) 
		   realloc(beneath->data.branches.head,
			   (size+1)*sizeof(struct CHAR_BUCKET));
		 beneath->data.branches.head=new; 
		 beneath->data.branches.tail=new+size+1-1;
		 memmove(new+index+1,new+index,(size-index)*sizeof(struct CHAR_BUCKET));
		 halfway=new+index; halfway->key=kc; 
		 if (kc == '\0') 
		   {halfway->data.string=NULL; return halfway;}
		 else {halfway->data.branches.head=NULL; halfway->data.branches.tail=NULL;
		       beneath=halfway; string=string+1;}}}
   }
}

char *intern_frame_name(char *string)
{
  struct CHAR_BUCKET *bucket; char *new_name;
  bucket=find_bucket(string,&name_lookup_table);
  if (bucket->data.string) return bucket->data.string;
  else new_name=allocate_permanent_string(string);
  if (number_of_frame_names == space_in_frame_name_table)
    {char **new_table; int new_table_size;
     if (space_in_frame_name_table)
       new_table_size=(space_in_frame_name_table*2);
     else new_table_size=300;
     if (space_in_frame_name_table)
       new_table = (char * *) 
	 fra_reallocate(frame_name_table,sizeof(char *)*new_table_size);
     else new_table = (char * *) fra_allocate(new_table_size,sizeof(char *));
     space_in_frame_name_table=new_table_size; frame_name_table=new_table;}
  frame_name_table[number_of_frame_names++]=new_name; bucket->data.string=new_name;
  return new_name;
}


/* Frame appendices */

struct FRAME_APPENDIX *add_appendix(Frame frame)
{
  if (frame->appendix) return frame->appendix;
  else {struct FRAME_APPENDIX *appendix;
	appendix=allocate_frame_appendix();
	appendix->limit=0; appendix->size=0; appendix->annotations=NULL;
	appendix->spinoffs=NULL; appendix->zip_code=-1;
	frame->appendix=appendix;
	return appendix;}
}


/* Operations on frame arrays */

/* Frame arrays are used for organizing the spinoffs and annotations of frames
   as well as for returning results of certain operations.  The definitions
   below ensure that frame arrays are grown when neccessary. */

/* Creates a frame array of <size> elements. */
Frame_Array *make_frame_array(int size)
{
  Frame_Array *array; ALLOCATE(array,struct FRAME_ARRAY,1);
  if (size == 0) array->elements=NULL;
  else array->elements = (Frame  *) fra_allocate(size,sizeof(Frame ));
  array->limit=size; array->size=0;
  return array;
}

void grow_frame_array(Frame_Array *array)
{
  Frame *elements; int increment, new_limit; increment=array->limit;
  if (increment < FRAME_ARRAY_MINIMUM_INCREMENT) 
    increment=FRAME_ARRAY_MINIMUM_INCREMENT;
  new_limit = (array->limit+increment);
  elements = fra_reallocate(array->elements,sizeof(Frame )*new_limit);
  /* Update the elements and limits of the frame array */
  array->elements = elements;
  array->limit = new_limit;
}

/* Adds a frame to array, growing it if neccessary. */
Frame_Array *add_frame_at_end(Frame frame,Frame_Array *array)
{
  /* If the array is empty (null), make an empty array. */
  if (array == NULL) array=make_frame_array(FRAME_ARRAY_MINIMUM_INCREMENT);
  else if (array->size == array->limit) grow_frame_array(array);
  array->elements[array->size++] = frame;
  return array;
}

/* Adds a frame to array, growing it if neccessary. */
Frame_Array *add_frame(Frame frame,Frame_Array *array)
{
  /* If the array is empty (null), make an empty array. */
  if (array == NULL) 
    {array=make_frame_array(FRAME_ARRAY_MINIMUM_INCREMENT);
     array->elements[array->size++]=frame;}
  else if (array->size == 0) 
    {array->elements = (Frame  *) 
       fra_allocate(FRAME_ARRAY_MINIMUM_INCREMENT,sizeof(Frame ));
     array->size=0; array->limit=FRAME_ARRAY_MINIMUM_INCREMENT;
     array->elements[array->size++]=frame;}
  else if (array->size == array->limit)
    {Frame *ptr, *tail; ptr=array->elements; tail=ptr+array->size-1;
     while ((ptr < tail) && (NOT(frame_deleted_p(*ptr)))) ptr++;
     if (ptr < tail) (*ptr)=frame;
     else {grow_frame_array(array); array->elements[array->size++]=frame;}}
  else array->elements[array->size++]=frame;
  return array;
}

/* Removes a frame from an array.  This just replaces it with NULL,
   and compaction happens when the array runs out of space.
*/
static void remove_frame(Frame frame,Frame_Array *array)
{
  Frame *elements, *max_element;
  elements=array->elements; max_element=elements+array->size;
  while (elements < max_element)
    if ((*elements) == frame) {*elements=NULL; return;}
    else elements++;
  return;
}

void add_spinoff(Frame spinoff,Frame frame)
{
  struct FRAME_APPENDIX *appendix; appendix=frame->appendix;
  if (NULLP(appendix)) appendix=add_appendix(frame);
  if (appendix->spinoffs) add_frame(spinoff,appendix->spinoffs);
  else appendix->spinoffs=add_frame(spinoff,make_frame_array(3)); 
}

void remove_spinoff(Frame spinoff,Frame frame)
{
  struct FRAME_APPENDIX *appendix;
  if ((appendix=(frame->appendix)))
    if (appendix->spinoffs) remove_frame(frame,appendix->spinoffs);
}


/* Finding and adding annotations to frames */

/* Annotations are stored in alphabetical name order and searched by
   a binary search.  The function binary_search_for_frame does
   this search, comparing names alphabetically; the function add_frame_ordered
   is like add_frame but maintains the alphabetically. */

static void grow_annotations(struct FRAME_APPENDIX *appendix)
{
  if (appendix->limit)
    {int new_limit=appendix->size*2;
     appendix->annotations=
       fra_reallocate(appendix->annotations,new_limit*sizeof(struct ANNOTATION));
     appendix->limit=new_limit;}
  else {appendix->annotations=fra_allocate(5,sizeof(struct ANNOTATION));
	appendix->limit=5; appendix->size=0;}
}

static void update_annotations(Frame home,Frame new,char *name)
{
  struct FRAME_APPENDIX *appendix; appendix=home->appendix;
  if (NULLP(appendix)) appendix=add_appendix(home);
  if (appendix->size == appendix->limit) grow_annotations(appendix);
  appendix->annotations[appendix->size].key=name;
  appendix->annotations[appendix->size++].frame=new;
}

/* Removes a frame from an array.  This moves all the succeeding
   elements of the array back to fill in the gap.
   (This is seldom called).
*/
static void remove_annotation(Frame frame)
{
  struct ANNOTATION *ptr, *limit; 
  if (NULLP(frame->appendix)) return;
  ptr=frame_home(frame)->appendix->annotations; 
  limit=ptr+frame_home(frame)->appendix->size;
  while (ptr < limit)
    if ((*ptr).frame == frame) {(*ptr).key=NULL; return;}
}

/* add_annotation makes a new annotation on a frame; it takes a frame,
   a name, and a prototype for the new frame.  The chief subtlety is the
   real_name of the frame which is either an interned name or a copy of
   the given argument (we copy it in case it is ephemeral).  The interned
   name is gotten either from the prototype (which already went through
   all this) or the name table.  Otherwise, it is just copied.
*/
static Frame add_annotation(Frame frame,char *name,Frame prototype)
{
  Frame made;
  /* Make a new frame data structure */
  made =  allocate_frame();
  /* Initialize the new frame structure */
  made->home = frame; 
  if (prototype) made->aname=prototype->aname;
  else made->aname=allocate_permanent_string(name);
  made->prototype = prototype; made->ground = NULL;
  made->appendix=NULL;
  made->type= frame->type & frame_creation_mask; made->bits= 0;
  if ((prototype) && (frame_type(prototype) == ephemeral_frame))
    set_frame_type(made,ephemeral_frame);
  /* Add it to the annotations of its home */
  INTERNED_FRAME_NAME(name);
  update_annotations(frame,made,name);
  return made;
}

/* The function local_probe_annotation sees if an annotation named <name>
   exists on <frame>.  It makes the frame up to date before operating on it. */
Frame local_probe_annotation(Frame frame,char *name)
/* Returns the annotation of `frame' named `name' or NULL otherwise.
  This does no inheritance of annotations through prototypes. */
{
  struct FRAME_APPENDIX *appendix;
  /* If we are autoloading, ensure that a frame is current before looking
     for an annotation on it.  This needs to go first to handle aliasing. */
  if (!(suppress_autoload)) ENSURE_CURRENT(frame);
  /* If there aren't any annotations, there can't be one named name. */
  appendix=frame->appendix; 
  if (NULLP(appendix)) return NULL;
  else if (appendix->size == 0) return NULL;
  /* Otherwise, just do a linear search. */
  else {Frame found; struct ANNOTATION *ptr, *head, *limit; 
	INTERNED_FRAME_NAME(name);
	ptr=head=appendix->annotations; limit=ptr+appendix->size;
	while (ptr < limit)
	  if (ptr->key == name) 
	    {found=ptr->frame; 
	     ptr->frame=head->frame; head->frame=found;
	     ptr->key=head->key; head->key=name; 
	     return found;}
	  else ptr++;
	return NULL;}
}

/* The function raw_local_probe_annotation sees if an annotation named <name>
   exists on <frame> but doesn't require that <frame> be loaded. */
Frame raw_local_probe_annotation(Frame frame,char *name)
{
  struct FRAME_APPENDIX *appendix;
  /* If there aren't any annotations, there can't be one named name. */
  appendix=frame->appendix; 
  if (NULLP(appendix)) return NULL;
  /* If all the annotations are deleted, we won't find name there */
  else if (appendix->size == 0) return NULL;
  /* Otherwise, just do a linear search. */
  else {Frame found; struct ANNOTATION *ptr, *head, *limit; 
	INTERNED_FRAME_NAME(name);
	ptr=head=appendix->annotations; limit=ptr+appendix->size;
	while (ptr < limit)
	  if (ptr->key == name) 
	    {found=ptr->frame; 
	     ptr->frame=head->frame; head->frame=found;
	     ptr->key=head->key; head->key=name; 
	     return found;}
	  else ptr++;
	return NULL;}
}


/* Prototypes and Grounds */

/* Prototypes are kept in two directions; every frame knows
   both its prototype and those current frames which use it
   as a prototype.  This second relation is stored sparsely as the
   spinoffs of the frame; if the relation is the default one, we do
   not store it since we can easily regenerate it by looking at the
   spinoffs of the frames home.  This is done by frame_spinoffs
   defined some pages below */

/* This determines whether a prototype relation would obey the
   default prototype rule.  This is used by the system all over the
   place and its fraxl version is DEFAULT-PROTOTYPE? */
boolean prototype_is_default(frame,prototype)
     Frame frame, prototype;
{
  if (default_prototype_p(frame,prototype))
    return True;
  else return False;
}

/* This returns true if some prototype (eventually) of <x> is <proto>.
   The FRAXL version of this is HAS-PROTOTYPE. */
Frame has_prototype(Frame x,Frame proto)
{
  Frame seek;
  if (proto == NULL) return x;
  else seek=x;
  while ((seek != NULL) && (seek != proto)) seek=frame_prototype(seek);
  if (seek) return x; else return NULL;
}

/* This returns true if <x> is somewhere beneath <home>
   in the annotation tree.  The FRAXL version of this
   is HAS-HOME. */
boolean has_home(Frame x,Frame home)
{
   x=x->home;
   while (NOT((x == NULL) || (x == home))) x=x->home;
   if (x == home) return True; else return False;
}

/* This changes the prototype of a frame.  This is slightly complicated
   because we need to maintain the minimal set of spinoffs between frames.
   The FRAXL analog of this function is SET-PROTOTYPE. */
Frame set_prototype(Frame frame,Frame prototype)
{
  /* Make sure its current and writable. */
  ENSURE_CURRENT(frame); 
  if (prototype == frame->prototype) return frame;
  else {TOUCH(frame);}
  if (remote_frame_p(frame))	/* Change this */
    {set_remote_prototype(frame,prototype); return frame;}
  if (prototype != NULL)
    if (has_prototype(prototype,frame)) /* Catch error */
      frame_error2(Recursive_Prototype,frame," loops with",prototype);
  if (prototype == NULL) {}
  else if (NOT(default_prototype_p(frame,prototype)))
    /* If prototype is not a default, add it to the prototype's spinoffs.
       This man entail making */
    add_spinoff(frame,prototype);
  {DO_ANNOTATIONS(a,frame)
     if (a->prototype == NULL) {}
     else if (default_prototype_p(a,a->prototype)) add_spinoff(a,a->prototype);
     else if ((a->prototype->home == prototype) &&
	      ((a->aname == a->prototype->aname) ||
	       (string_compare(a->aname,a->prototype->aname) == 0)))
       remove_spinoff(a,a->prototype);}
  /* Otherwise, remove the old spinoff relation (supposing it wasn't to NULL) */
  if (NOT((frame->prototype == NULL) || (default_prototype_p(frame,frame->prototype))))
    remove_spinoff(frame,frame->prototype);
  frame->prototype = prototype;
  return frame;
}

/* This changes the prototype of a frame, without forcing it to be loaded. */
Frame internal_modify_prototype(Frame frame,Frame prototype)
{
  if (prototype == frame->prototype) return frame;
  if (prototype != NULL)
    {Frame p; p=prototype; while ((p) && (p != frame)) p=p->prototype;
     if (p) /* Catch error */
       frame_error2(Recursive_Prototype,frame," loops with",prototype);}
  if (prototype == NULL) {}
  else if (NOT(default_prototype_p(frame,prototype)))
    /* If prototype is not a default, add it to the prototype's spinoffs.
       This man entail making */
    add_spinoff(frame,prototype);
  {DO_ANNOTATIONS(a,frame)
     if (a->prototype == NULL) {}
     else if (default_prototype_p(a,a->prototype))
       add_spinoff(a,a->prototype);
     else if ((a->prototype->home == prototype) &&
	      ((a->aname == a->prototype->aname) ||
	       (string_compare(a->aname,a->prototype->aname) == 0)))
       remove_spinoff(a,a->prototype);}
  /* Otherwise, remove the old spinoff relation (supposing it wasn't to NULL) */
  if (NOT((frame->prototype == NULL) || (default_prototype_p(frame,frame->prototype))))
    remove_spinoff(frame,frame->prototype);
  frame->prototype = prototype;
  return frame;
}

Frame set_prototypes(Frame frame,Frame new_prototype)
/* Sets the prototypes of a frame *and* its annotations. */
{
  set_prototype(frame,new_prototype);
  {DO_ANNOTATIONS(a,frame)
     {Frame proto; proto=probe_annotation(new_prototype,frame_name(a));
      if (proto) set_prototypes(a,proto);}}
  return frame;
}

Frame align_prototypes(Frame frame,Frame new_prototype)
/* Sets the prototypes of a frame *and* its annotations. */
{
  set_prototype(frame,new_prototype);
  {DO_ANNOTATIONS(a,frame)
     {Frame proto; 
      if (new_prototype) 
	proto=probe_annotation(new_prototype,frame_name(a));
      else proto=NULL;
      align_prototypes(a,proto);}}
  return frame;
}

/* This updates (if neccessary) and gets the prototype of a frame.
   The FRAXL analog of this function is FRAME-PROTOTYPE or just PROTOTYPE.
   This is referenced by the macro frame_prototype which calls it when
   a frame needs to be made current before having its prototype fetched. */
Frame frame_prototype_fn(Frame frame)
{
  ENSURE_CURRENT(frame);
  return frame->prototype;
}

/* This sets the ground of a frame, loading the frame if need be,
   checking that the frame is not read_only and possibly reclaiming the
   storage taken by the old ground. */
Frame set_ground(Frame frame,Grounding ground)
{
  Grounding old_ground;
  /* Make sure that the frame is current and writable. */
  ENSURE_CURRENT(frame); TOUCH(frame);
  if (remote_frame_p(frame))
    {set_remote_ground(frame,ground); return frame;}
  else {USE_GROUND(ground);
	old_ground = frame->ground; frame->ground = ground;
	FREE_GROUND(old_ground);
	return frame;}
}

/* This gets the ground of a frame, ensuring that it is first current.
   It is referenced by the macro frame_ground which calls it whenever
   the frame doesn't seem to be current. */
Grounding frame_ground_fn(Frame frame)
{
  ENSURE_CURRENT(frame);
  return frame->ground;
}

/* This returns the first non NULL ground on <frame> or its prototypes.
   It may be used by different programs which use the prototype hierarchy
   for inheritance. Its fraxl analog is GET-INHERITED-GROUND. */
Grounding get_inherited_ground(Frame frame)
{
  ENSURE_CURRENT(frame);
  while ((frame != NULL) && (frame->ground == NULL))
    {frame=frame->prototype; if (frame != NULL) {ENSURE_CURRENT(frame);}}
  if (frame == NULL) return NULL; else return frame->ground;
}


/* Search bits */

/* Every frame has a `search word' consisting of bits which may be
   quickly modified and checked for use in various search and marker-passing
   algorithms.  The functions grab_search_bit and release_search_bit manage
   indices into this search word to ensure that processes using search bits
   don't step on one another.  In addition, the lower three bits of the
   word are already reserved for system functions:
     + the first is used for various system procedures
     + the second is used for MNEMOSYNE's cognate searches
     + the third is used for constraint propogation
  The unsigned char search_bit_use_mask has a bit for every bit
  currently being used; the reserved bits above initialize this
  mask to 7.
*/
search_word search_bit_use_mask=7;

/* Reserves a new search bit by a linear search for a bit which works.
   (Actually, this could be sped up by starting at the third bit, but
    who cares) */
int grab_search_bit()
{
  {DO_TIMES(i,(sizeof(search_word)*8))
     if (((1<<i) & (search_bit_use_mask)) == 0)
       {search_bit_use_mask=((1<<i) | (search_bit_use_mask));
	return i;}}
  raise_crisis("No more search bits"); 
  return -1; /* Never should be reached */
}

/* Releases a search bit by clearing the corresponding bit in search_bit_use_mask. */
void release_search_bit(int i)
{
  search_bit_use_mask=((search_bit_use_mask) & ~(1<<i));
}


/* User functions */

/* `probe_annotation' returns the annotation named `name' of `frame' if either:
     a) it is already there
     b) a like-named annotation is on some prototype of `frame'
    In this latter case, the newly constructed annotation has
    a prototype of this like-named inherited annotation
*/
Frame probe_annotation (Frame frame,char *name)
{
  Frame local, prototype;
  /* Ensure that frame is current (and dereference aliases). */
  ENSURE_CURRENT(frame);
  /* Get the local version. */
  local = local_probe_annotation(frame,name);
  /* If its local and not marked as deleted, just return it. */
  if ((local != NULL) && (NOT(frame_deleted_p(local)))) return local;
  /* Otherwise, look for a potentialy prototype for the new annotatino by
      doing a recursive probe_annotation on the prototype of <frame>.
     If no such prototype exists, you can just return NULL. */
  else if (frame->prototype) prototype = probe_annotation(frame->prototype,name);
  else return NULL;
  /* If there isn't a good prototype, fail */
  if (prototype == NULL) return NULL;
  else if (local)
    /* If you're resurrecting a deleted frame, undelete it and set
       its prototype. */
    {local->prototype=prototype; clear_frame_deleted_p(local);}
  /* Otherwise, make a new annotation with the appropriate prototype. */
  else local = add_annotation(frame,name,prototype);
  return local;
}

/* `use_annotation' returns the annotation named `name' of `frame'.
   If such an annotation exists, it is returned
   If no such annotation exists, one is created obeying the default
    prototype rule (if it applies).
*/
Frame use_annotation (Frame frame,char *name)
{
  Frame local, prototype=NULL;
  local = local_probe_annotation(frame,name); /* Is it there? */
  /* If it's there and undeleted, return it. */
  if ((local) && (NOT(frame_deleted_p(local)))) return local;
  /* Otherwise, try and get its prototype. */
  else if (suppress_autoload) {}
  else if (frame->prototype)
    {prototype=probe_annotation(frame->prototype,name);
     /* If it would have no prototype, you are modifying its home. */
     if (NULLP(prototype)) TOUCH(frame);}
  if (local) {clear_frame_deleted_p(local); local->prototype=prototype;}
  else local=add_annotation(frame,name,prototype);
  /* If the prototype is NULL but you're not autoloading, it means you didn't check;
     thus, the new frame isn't really current. */
  if (NOT(suppress_autoload)) set_frame_current_p(local);
  if (prototype == NULL)
    {if (suppress_autoload) clear_frame_current_p(local);
     if (remote_frame_p(frame)) add_remote_annotation(frame,name);}
  return local;
}

Frame make_ephemeral_annotation (Frame frame,char *name)
/* `use_annotation' returns the annotation named `name' of `frame'.
   If no such annotation exists, one is created.
   If some prototype of `frame' has an annotation named `name'
   a chain of inheritance from this annotation is created.
*/
{
  Frame local, prototype;
  local = raw_local_probe_annotation(frame,name); /* Is it there? */
  /* If it's there and undeleted, return it. */
  if ((local) && (NOT(frame_deleted_p(local)))) return local;
  /* Otherwise, try and get its prototype. */
  else if ((frame->prototype) && (frame_current_p(frame->prototype)))
    prototype=probe_annotation(frame->prototype,name);
  else prototype=NULL;
  if (local) clear_frame_deleted_p(local);
  else local=add_annotation(frame,name,prototype);
  /* Set the frame type appropriately. */
  set_frame_type(local,ephemeral_frame);
  /* Ephemeral frames are always current. */
  set_frame_current_p(local);
  return local;
}

/* Makes a frame which is an alias pointer to another frame. */
Frame make_alias(Frame frame,char *name,Frame for_frame)
{
  Frame alias; alias=use_annotation(frame,name);
  /* Aliased frames are never current in order to force dereferencing. */
  set_frame_type(alias,alias_frame); clear_frame_current_p(alias);
  alias->prototype=for_frame;
  add_spinoff(alias,for_frame);
  return alias;
}

/* An alias for use_annotation */
Frame make_annotation (Frame frame,char *name)
{
  return use_annotation(frame,name);
}

/* make_unique_annotation generates a new annotation with a counter
   stored on the frame. */
Frame make_unique_annotation(Frame frame,char *name)
{
  Frame counter, result; Grounding counter_ground; int counter_value; char namebuf[100];
  if (NULLP(probe_annotation(frame,name))) return use_annotation(frame,name);
  strcpy(namebuf,"+"); strcat(namebuf,name); strcat(namebuf,".count");
  counter=use_annotation(frame,namebuf); counter_ground=frame_ground(counter);
  if (counter_ground) counter_value=GINTEGER(counter_ground); else counter_value=0;
  set_ground(counter,integer_to_ground(counter_value+1));
  sprintf(namebuf,"%s.%d",name,counter_value+1);
  result=use_annotation(frame,namebuf);
  set_ground(use_annotation(result,"+root"),string_to_ground(name));
  return result;
}

/* `inherits_frame' returns the annotation named `name' of `frame' or
   the nearest `name' annotation of one of `frame's prototypes.
   It conses no structure and returns NULL if there is no
   inherited annotation named `name'.
*/
Frame inherits_frame (Frame frame,char *name)
{
  Frame local=NULL;
  while ((frame) && (NOT(local)))
    {local=local_probe_annotation(frame,name);
     if ((local) && (frame_deleted_p(local))) local=NULL;
     frame=frame->prototype;}
  return local;
}


/* Computing virtual spinoffs */

/* The spinoff algorithm is somewhat complicated; we only store spinoffs
   if they violate the default prototype rule and to get all the spinoffs
   we need to find those descriptions whose prototypes do obey the default
   rule.  We do this by ascending the annotation lattice and then looking
   down for like named annotations of the spinoffs on the way up.  But
   we have to do the same thing to find those spinoffs.

   For instance, suppose `Ken' has a prototype of `John'; since the names
   are different, it cannot obey the default prototype rule.  Now consider
   the frame `Ken/birth/location'; its prototype should be `John/birth/location'
   but  since that obeys the default prototype rule (given that Ken's prototype is
   John), we do not store `Ken/birth/location' as a spinoff of `John/birth/location'.
   We must regenerate this fact by looking up the hierarchy from `Ken/birth/location'
   and for each stored spinoff, look back down for paths to frames whose
   dependence on `John/birth/location' obeys the default prototype rule.

   In the algorithm we use, a `ladder' is maintained describing the list
   of names leading to some home or grand-home; for each stored spinoff of
   this home frame, we send down a `probe' which calls local_check to determine
   if an appropriate path descends to an unrecorded prototype. */

/* Returns a computed array of the spinoffs of FRAME */
Frame_Array *frame_spinoffs(Frame frame)
{
  /* We assume that annotation trees will never be more than 100 deep.
     (reasonable but kludgy assumption, but we don't want to check for
     overflows). */
  int i, j, height; char *ladder[100]; Frame iladder[100];
  struct FRAME_APPENDIX *appendix;
  Frame_Array *array; Frame probe, *spinoffs;
  array=make_frame_array(0);
  if (array==NULL) return NULL;
  for (height=0;frame != NULL;
       ladder[height]=frame->aname, iladder[height++]=frame, frame=frame->home)
    if ((appendix=frame->appendix) && ((appendix->spinoffs)))
      {spinoffs=appendix->spinoffs->elements;
       for (i=appendix->spinoffs->size-1;i>=0;i--)
	 {probe=spinoffs[i];
	  if ((probe) && (NOT(frame_deleted_p(probe))))
	    for (j=height;((j>=0) && (probe != NULL));j--)
	      {if (j == 0) add_frame(probe,array);
	      else {probe=local_probe_annotation(probe,ladder[j-1]);
		    if (probe && (probe->prototype != iladder[j-1]))
		      probe = NULL;}}}}
  return array;
}


/* Ensuring that frames are current */

/* This function makes sure that a frame is `up to date'; the macro
   frame_current_p returns true if a frame has been `initialized' either
   by reading from a file, creation by a program, or retrieval over
   the network. This function is usually called only when a frame
   *isn't* current, by the macro ensure_current(x).
   If the frame is an alias, get_current dereferences the alias;
   aliased frames are always not current, so this will be called
   to dereference aliases.
*/

Frame get_current(Frame frame)
{
  Frame update_remote_frame AP((Frame f));
  /* Avoids some nasty errors in places. */
  if (frame == NULL) return frame;
  /* If it's not a frame, we're way off base! */
  else if (NOT(FRAMEP(frame))) 
    {ground_error(Type_Error,"not a frame",frame_to_ground(frame));
     return NULL;}
  /* If it's current already, you don't have to do a thing. */
  else if (frame_current_p(frame)) return frame;
  /* If it's an alias, dereference the alias. */
  else if ((frame_type(frame)) == alias_frame)
    return frame->prototype;
  /* If it's deleted, undelete it. */
  else if ((frame_type(frame)) == deleted_frame)
    {ensure_current(frame_home(frame));
     set_frame_type(frame,frame_type(frame_home(frame)));
     return frame;}
  /* If it's remote, update the remote frame. */
  else if (remote_frame_p(frame))
    return update_remote_frame(frame);
  /* The root is always up to date. */
  else if (frame==root_frame) return root_frame;
  /* If it's a `backup root' (i.e. it has a +filename annotation)
     restore the corresponding file. */
  else if ((active_image_p) && (restore_from_image(frame)))
    return get_current(frame);
  else if (backup_root_p(frame))
    {restore_frame_from_file(frame); return frame;}
  /* If it's home is current, but it isn't, it must be a dangling
     reference from some other file.  In this case, we need to
     fix its prototype (since it was created without one). */
  else if (frame_current_p(frame->home))
    {Frame home_proto, new_proto;
     set_frame_current_p(frame); home_proto=frame_prototype(frame_home(frame));
     if (home_proto) new_proto=probe_annotation(home_proto,frame_name(frame));
     else new_proto=NULL;
#if ALPHA_RELEASE
     if (new_proto)
       {Frame p; p=new_proto; while ((p) && (p != frame)) p=p->prototype;
	if (p) raise_crisis(Recursive_Prototype);}
#endif /* ALPHA_RELEASE */
     if (NULLP(new_proto))
       if (frame_read_only_p(frame_home(frame)))
	 {fprintf(stderr,"Warning: "); fprint_frame(stderr,frame); 
	  fprintf(stderr," is a dangling reference which can't be saved out.");}
       else set_frame_modified_p(frame_home(frame));
     frame->prototype=new_proto; return frame;}
  /* Otherwise, make the home current and try again. */
  else {get_current(frame->home); return get_current(frame);}
}


/* Detecting the presence of spinoffs */

/* Returns some spinoff of `frame' if one exists or NULL otherwise.
   This is used to determine whether or not it is safe to delete
   a frame (if it has spinoffs, it isn't).  It uses the same control structure
   as frame_spinoffs, but returns as soon as it finds any spinoff.  It
   also excludes frames with the FRAMER_SYSTEM_BIT set.  This bit is set
   by delete_annotation in preparation for deleting annotations; this makes
   it possible to delete an annotation with spinoffs providing that you are
   deleting the spinoffs too! */
static Frame any_spinoffs_p(Frame frame)
{
  /* We assume that annotation trees will never be more than 100 deep.
     (reasonable but kludgy assumption, but we don't want to check for
     overflows). */
  int i, j, height; char *ladder[100]; Frame iladder[100];
  struct FRAME_APPENDIX *appendix;
  Frame probe, *spinoffs;
  for (height=0;frame != NULL;
       ladder[height]=frame->aname, iladder[height++]=frame, frame=frame->home)
    if ((appendix=frame->appendix) && (appendix->spinoffs))
      {spinoffs=appendix->spinoffs->elements;
       for (i=appendix->spinoffs->size-1;i>=0;i--)
	 {probe=spinoffs[i];
	  if ((probe) && (NOT(frame_deleted_p(probe))))
	    for (j=height;((j>=0) && (probe != NULL));j--)
	      {if (j == 0)
		 {if (NOT(search_bit_p(probe,FRAMER_SYSTEM_BIT)))
		    /* Return right away */
		    return probe;}
	      else {probe=local_probe_annotation(probe,ladder[j-1]);
		    if (probe && (probe->prototype != iladder[j-1]))
		      probe = NULL;}}}}
  return NULL;
}

static Frame real_spinoffs(Frame_Array *array)
{
  Frame *head, *tail; head=array->elements; tail=head+array->size;
  while (head < tail) 
    if (((*head) && (NOT(frame_deleted_p(*head))))) return *head; else head++;
  return NULL;
}

#define any_real_spinoffs_p(frame) \
      ((frame->appendix) && (frame->appendix->spinoffs) && \
       (frame->appendix->spinoffs->size > 0) && \
       (real_spinoffs(frame->appendix->spinoffs)))

/* This returns the spinoff of some annotation of FRAME, if such a spinoff
   exists.  This is also used to check if it is okay to delete a frame.
   Note that it only applies to loaded frames, causing some inconsistencies
   when deleting a frame which unloaded frames point to.  Tough.  */
static Frame any_sub_spinoffs_p(Frame frame)
{
  if (search_bit_p(frame,FRAMER_SYSTEM_BIT)) return NULL;
  else if (any_real_spinoffs_p(frame)) return frame;
  else if (frame->appendix == NULL) return NULL;
  else {{DO_SOFT_ANNOTATIONS(x,frame)
	   {Frame so; so=any_sub_spinoffs_p(x);
	    if (so != NULL) return so;}}
	return NULL;}
}


/* Copying frames */

/* Copies a frame reference relative to `source' to a new frame
   reference relative to `dest'.  This creates a frame with a corresponding
   path. */
static Frame copy_frame_ground(Frame frame,Frame source,Frame dest)
{
  Frame home;
  if (frame == NULL) return NULL;
  else if (frame == source) return dest;
  else for (home=frame;(home != NULL) && (home != source);home=frame_home(home));
  if (home)
    return use_annotation(copy_frame_ground(frame_home(frame),source,dest),
			  frame_name(frame));
  else return frame;
}

/* Copies a pair relative to `source' to a pair relative to `dest' */
static Grounding copy_pair_ground(Grounding ground,Frame source,Frame dest)
{
  Grounding copy_ground AP((Grounding ground,Frame source,Frame dest));
  if (ground == empty_list) return empty_list;
  else return cons_pair(copy_ground(GCAR(ground),source,dest),
			copy_ground(GCDR(ground),source,dest));
}

/* Copies a result set relative to `source' to a set relative to `dest' */
static Grounding copy_results_ground(Grounding ground,Frame source,Frame dest)
{
  Grounding copy_ground AP((Grounding ground,Frame source,Frame dest));
  Grounding result, *ptr; 
  NDMAKE(result,ND_SIZE(ground)); ptr=ND_ELTS(result);
  {DO_RESULTS(r,ground) {*ptr++=copy_ground(r,source,dest);}}
  return result;
}

/* Copies a ground relative to `source' to a ground relative to `dest' */
Grounding copy_ground(Grounding ground,Frame source,Frame dest)
{
  if (FRAMEP(ground))
    return frame_to_ground(copy_frame_ground(GFRAME(ground),source,dest));
  else if (TYPEP(ground,pair_ground))
    return copy_pair_ground(ground,source,dest);
  else if (TYPEP(ground,vector_ground))
    {Grounding copy; GVMAKE(copy,GVSIZE(ground));
     {DO_TIMES(i,(GVSIZE(ground)))
	{GVSET(copy,i,copy_ground(GVREF(ground,i),source,dest));}}
     return copy;}
  else if (TYPEP(ground,nd_ground))
    return copy_results_ground(ground,source,dest);
  /* We could do fancy things with procedures, but we don't (right now). */
  else return ground;
}

/* Copies the annotation and ground structures for s and d underneath
   source and dest */
static void frame_copier(Frame s,Frame d,Frame source,Frame dest)
{
  set_ground(d,copy_ground(frame_ground(s),source,dest));
  set_prototype(d,copy_frame_ground(frame_prototype(s),source,dest));
  {DO_SOFT_ANNOTATIONS(a,s)
     if (NOT(dest == a))
       frame_copier(a,use_annotation(d,frame_name(a)),source,dest);}
}

/* Copies all the structure of source into dest.  The FRAXL
   analog of this function is COPY-FRAME. */
Frame copy_frame(Frame source,Frame dest)
{
  frame_copier(source,dest,source,dest);
  return dest;
}


/* Deleting frames */

static void delete_annotation_internal AP((Frame frame));
static void mark_children_with_system_bit AP((Frame frame));
static void clear_children_of_system_bit(Frame frame);
void reject_deletion AP((Frame frame,Frame so));

/* This removes <frame> from the tree of frames, providing that no
   one else depends on it.  Actually, it just checks that no other
   frames refer to this one as a prototype, without checking for grounds.
*/
boolean delete_annotation(Frame frame)
{
  Frame so; boolean result=False;
  TOUCH(frame_home(frame)); TOUCH(frame); 
  {WITH_HANDLING
     {mark_children_with_system_bit(frame);
      so=any_spinoffs_p(frame);
      if (so != NULL) {reject_deletion(frame,so); return False;}
      else {so=any_sub_spinoffs_p(frame);
	    if (so != NULL) 
	      {reject_deletion(so,real_spinoffs(so->appendix->spinoffs)); 
	       result=False;}
	    else {TOUCH(frame->home); delete_annotation_internal(frame);
		  result=True;}}}
   ON_EXCEPTION
    {clear_children_of_system_bit(frame); reraise();}
   END_HANDLING}
  return result;
}

/* This removes a frame and its subframes by reclaiming
   their grounds and marking them as deleted. It doesn't
   reclaim the frames themselves, because other frames may
   have dangling pointers to them.  If we ever give frames
   reference counters, we may be able to clean that up though.
*/
static void delete_annotation_internal(Frame frame)
{
  /* These do the appropriate GCing and spinoff removal */
  if (frame_current_p(frame))
    if (frame_type(frame) != alias_frame)
      {set_ground(frame,(Grounding ) NULL); frame->prototype=NULL;}
  set_frame_type(frame,deleted_frame); 
  /* These clean up the annotations */
  if (frame->appendix != NULL)
    {DO_SOFT_ANNOTATIONS(a,frame) delete_annotation_internal(a);}
  frame->bits=0;
  if ((frame->appendix) && (frame->appendix->spinoffs))
    {Frame_Array *spinoffs; spinoffs=frame->appendix->spinoffs;
     free(spinoffs->elements); spinoffs->elements=NULL; 
     free(spinoffs); frame->appendix->spinoffs=NULL;}
  /* We don't free the frame because some grounds may still
     foolishly have pointers to it.  This is a kludge, but
     it avoids the overhead of real garbage collection. */
}

/* Marks the children of a frame; used in large scale deletions
   where frames have spinoffs but the spinoffs are also being deleted. */
static void mark_children_with_system_bit(Frame frame)
{
  set_search_bit(frame,FRAMER_SYSTEM_BIT);
  {DO_SOFT_ANNOTATIONS(a,frame) mark_children_with_system_bit(a);}
}

static void clear_children_of_system_bit(Frame frame)
{
  clear_search_bit(frame,FRAMER_SYSTEM_BIT);
  {DO_SOFT_ANNOTATIONS(a,frame) clear_children_of_system_bit(a);}
}


/* Moving Frames */

Frame move_frame(Frame frame,Frame into,char *new_name)
{
  char *old_name; Frame old_home; INTERNED_FRAME_NAME(new_name);
  if (local_probe_annotation(into,new_name))
    if (frame_deleted_p(local_probe_annotation(into,new_name)))
      remove_annotation(local_probe_annotation(into,new_name));
    else frame_error(Frame_Collision,"Cannot move to an existing frame",
		     local_probe_annotation(into,new_name));
  old_name=frame->aname; old_home=frame->home;
  frame->aname=new_name; frame->home=into; remove_annotation(frame);
  update_annotations(into,frame,new_name);
  make_alias(old_home,old_name,frame);
  return frame;
}


/* Zip coding */

/* Zip coding associates integers with frames for use in fast printing
  and reading of frames.  The idea is that within a given context, each
  time a frame is printed or parsed for the first time, it is given a
  number and that subsequently the number can just be used to refer to it.
  Two different processes, providing they look at the same interaction
  corpus, can then use this to communicate much more compactly. */

/* Returns the zip code of frame from either its local cache or by
   looking it up.  This assumes the global variable zip_codes establishes
   a context for storing this sort of information. */
int zip_code(Frame frame)
{
  struct FRAME_APPENDIX *appendix;
  appendix=frame->appendix; if (NULLP(appendix)) return -1;
  else return appendix->zip_code;
}

/* Assigns a new zip_code to frame */
void new_zip_code(Frame frame)
{
  struct FRAME_APPENDIX *appendix;
  appendix=frame->appendix; if (NULLP(appendix)) appendix=add_appendix(frame);
  appendix->zip_code  = zip_codes->size; caching_zip_codes=True;
  add_frame_at_end(frame,zip_codes);
}

void free_zip_codes(Frame_Array *codes)
{
  Frame *ptr, *tail; ptr=codes->elements; tail=ptr+codes->size;
  if (caching_zip_codes)
    while (ptr < tail) (*ptr++)->appendix->zip_code=-1;
  caching_zip_codes=False;
  free(codes->elements); free(codes);
}


/* Printing and Reading frames */

/* The function print_frame_name prints a slashified version of a frame name.
   Frames print out with slashification of all non-graphic
   characters as well as '/', '\', '>', and ')'; this allows
   them to be included in pairs and vectors with impunity. */
void print_frame_name(char *string,generic_stream *stream)
{
  while (*string != '\0')
    {if ((IS_TERMINATOR_P(*string)) || (*string == '\\'))
       gsputc('\\',stream);
    gsputc(*string++,stream);}
}

/* Frames read in with the same slashification; indeed,
   anything following a '\' is stripped of any syntactic character
   and simply included in the frame name. */
char *read_frame_name(generic_stream *stream,char *into_string)
{
  char input, *ptr=into_string;
  input=gsgetc(stream);
  for (;;)
    {if (input=='\\')
       {*ptr++=gsgetc(stream);input=gsgetc(stream);}
     else
       {if (!(IS_TERMINATOR_P(input)))
	  {*ptr++=input;input=gsgetc(stream);}
	else {gsungetc(input,stream); break;};};}
  *ptr='\0';
  return into_string;
}

/* This prints a frame path relative to some other frame.
   It allocates new zip codes when zip_codes is non NULL,
   but it doesn't pay attention to read_root or to printing
   out zip codes; those are the responsibility of whoever calls
   print_frame_path.  It also only does two levels of upward
   ^ indirection (this is a cognitive assumption of how much
   context people really keep around). */
void print_frame_path(generic_stream *stream,Frame frame,Frame relative_to)
{
  if (frame == relative_to) return;
  else if (frame_deleted_p(frame))
    fprintf(stderr,"Printing dangling reference to deleted frame\n");
  else {print_frame_path(stream,frame->home,relative_to);
	gsputc('/',stream);
	if (zip_codes != NULL) new_zip_code(frame);
	print_frame_name(frame->aname,stream);}}

/* Prints a frame to a stream, taking care of read_root, zip_code
   and other specialized output conventions. */
void print_frame(generic_stream *stream,Frame frame)
{
  Frame root; root=frame;
  if (frame == root_frame) {gsputs("/",stream); return;}
  if (zip_codes != NULL)
    {while ((root) && (zip_code(root) < 0)) root=frame_home(root);
     if (root) gsprintf(stream,"%d",zip_code(root));
     else root=root_frame;}
  else if (read_root)
    {int depth, bit; bit=grab_search_bit();
     {{DO_HOMES(rr,read_root) set_search_bit(rr,bit);}
      root=frame; while ((root) && (NOT(search_bit_p(root,bit)))) root=root->home;
      {DO_HOMES(rr,read_root) clear_search_bit(rr,bit); release_search_bit(bit);}
      if (root == root_frame) {}
      else {Frame temp; temp=read_root; depth=1;
	    while ((temp) && (temp != root)) {temp=frame_home(temp); depth++;}
	    if ((NULLP(temp)) || (depth > root_print_depth)) root=root_frame;
	    else {DO_TIMES(i,depth) gsputc('^',stream);}}}}
  else root=root_frame; 
  if (frame == root) gsputc('/',stream);
  else print_frame_path(stream,frame,root);
}

/* Prints a frame to a stream, taking care of read_root, zip_code
   and other specialized output conventions. */
void print_frame_under(generic_stream *stream,Frame frame,Frame under)
{
  Frame root; root=frame;
  if (frame == root_frame) {gsputs("/",stream); return;}
  if (zip_codes != NULL)
    {while ((root) && (zip_code(root) < 0)) root=frame_home(root);
     if (root) gsprintf(stream,"%d",zip_code(root));
     else root=root_frame;}
  else if (under)
    {int depth, bit; bit=grab_search_bit();
     {{DO_HOMES(rr,under) set_search_bit(rr,bit);}
      root=frame; while ((root) && (NOT(search_bit_p(root,bit)))) root=root->home;
      {DO_HOMES(rr,under) clear_search_bit(rr,bit); release_search_bit(bit);}
      if (root == root_frame) {}
      else {Frame temp; temp=under; depth=1;
	    while ((temp) && (temp != root)) {temp=frame_home(temp); depth++;}
	    if ((NULLP(temp)) || (depth > root_print_depth)) root=root_frame;
	    else {DO_TIMES(i,depth) gsputc('^',stream);}}}}
  else root=root_frame; 
  if (frame == root_frame) gsputc('/',stream);
  else print_frame_path(stream,frame,root);
}

/* Parses a frame path from a stream, handling the recording
   but not the interpretation of zip codes and not handling read_root
   stuff.  That's the reponsibility of whoever calls the procedure.
   When trap_unknown_frames is True, it calls reader_make_annotation
   to create new frames. */
Frame parse_frame_path(generic_stream *stream,Frame root)
{
  char buffer[1024], input;
  Frame reader_make_annotation AP((Frame root,char *buffer));
  if (stream->stream_type == file_io) fread_frame_name(stream->ptr.file,buffer);
  else read_frame_name(stream,buffer);
  input=gsgetc(stream);
  if (input == '/')
    {Frame new_root;
     if ((trap_unknown_frames) &&
         ((probe_annotation(root,buffer)) == NULL))
       new_root=reader_make_annotation(root,buffer);
     else new_root=use_annotation(root,buffer);
     if (zip_codes != NULL) new_zip_code(new_root);
     return parse_frame_path(stream,new_root);}
  else if (*buffer == '\0')
    {gsungetc(input,stream); return root;}
  else {Frame new_root; gsungetc(input,stream);
        if ((trap_unknown_frames) &&
            ((probe_annotation(root,buffer)) == NULL))
          new_root=reader_make_annotation(root,buffer);
        else new_root=use_annotation(root,buffer);
        if (zip_codes != NULL) new_zip_code(new_root);
	return new_root;}
}

/* Utility function for parsing zip coded frames. */
static Frame parse_zip_coded_frame(generic_stream *stream)
{
  Frame root; int code; char input;
  code=gsgetint(stream);
  if (code >= zip_codes->size) raise_crisis(Unknown_Zip_Code);
  root=zip_codes->elements[code]; input=gsgetc(stream);
  if (input != '/') {gsungetc(input,stream); return root;}
  else return parse_frame_path(stream,root);
}

/* Parses a frame reference, handling relative paths and zip codes.
   If you change this, make sure to change fparse_frame also.  */
Frame parse_frame(generic_stream *stream)
{
  char input;
  input=gsgetc(stream);
  if (input == '#') input=gsgetc(stream);
  if (input == '/')
    return parse_frame_path(stream,root_frame);
  else if (input == '^')
    {Frame read_from; read_from=read_root;
     while (True)
       {input=gsgetc(stream);
	if (input == '^') 
	  if (read_from) read_from=read_from->home;
	  else raise_crisis(Read_Error);
	else if (input == '/') return parse_frame_path(stream,read_from);
	else raise_crisis(Read_Error);}}
  else if (zip_codes == NULL)
    {raise_crisis(Read_Error); return NULL;}
  else if (isdigit(input))
    {gsungetc(input,stream);
     return parse_zip_coded_frame(stream);}
  else if (read_root == NULL)
    {raise_crisis(Rootless_Reference); return NULL;}
  else return parse_frame_path(stream,read_root);
}


/* Functions just for file streams */

/* Reads a frame name from a file stream.  Used in speeding up
  reading from archive files. */
char *fread_frame_name(FILE *stream,char *into_string)
{
  char input, *ptr=into_string;
  input=getc(stream);
  for (;;)
    {if (input=='\\')
       {*ptr++=getc(stream);input=getc(stream);}
     else
       {if (!(IS_TERMINATOR_P(input)))
	  {*ptr++=input;input=getc(stream);}
	else {ungetc(input,stream); break;};};}
  *ptr='\0';
  return into_string;
}

/* Turns a string with slashification characters into a
   frame name. */
char *get_frame_name_from_string(char *input,char *output)
{
  generic_stream stream;
  stream.stream_type = string_input; stream.ptr.string_in = &input;
  read_frame_name(&stream,output);
  return input;
}

/* Parses a frame from a file stream */
Frame fparse_frame(FILE *file_stream)
{
  generic_stream stream;
  stream.stream_type = file_io; stream.ptr.file = file_stream;
  return parse_frame(&stream);
}


/* Prints a frame name (slashified) to a file stream.
   Used in speeding up archiving. */
void fprint_frame_name(char *string,FILE *stream)
{
  while (*string != '\0')
    {if ((IS_TERMINATOR_P(*string)) || (*string == '\\'))
       putc('\\',stream);
    putc(*string++,stream);}
}

/* This prints a frame path relative to some other frame.
   It allocates new zip codes when zip_codes is non NULL,
   but it doesn't pay attention to read_root or to printing
   out zip codes; those are the responsibility of whoever calls
   print_frame_path.  It also only does two levels of upward
   ^ indirection (this is a cognitive assumption of how much
   context people really keep around). */
void fprint_frame_path(FILE *stream,Frame frame,Frame relative_to)
{
  if (frame == relative_to) return;
  else if (frame_deleted_p(frame))
    fprintf(stderr,"Printing dangling reference to deleted frame\n");
  else {fprint_frame_path(stream,frame->home,relative_to);
	putc('/',stream);
	if (zip_codes != NULL) new_zip_code(frame);
	fprint_frame_name(frame->aname,stream);}}


/* Prints a frame to a stream, taking care of read_root, zip_code
   and other specialized output conventions. */
void fprint_frame(FILE *stream,Frame frame)
{
  Frame root; root=frame; 
  if (frame == root_frame) {fputs("/",stream); return;}
  if (zip_codes != NULL)
    {root=frame; while ((root) && (zip_code(root) < 0)) root=frame_home(root);
     if (root) fprintf(stream,"%d",zip_code(root));
     else root=root_frame;}
  else if (read_root)
    {int depth, bit; bit=grab_search_bit();
     {{DO_HOMES(rr,read_root) set_search_bit(rr,bit);}
      root=frame; while ((root) && (NOT(search_bit_p(root,bit)))) root=root->home;
      {DO_HOMES(rr,read_root) clear_search_bit(rr,bit); release_search_bit(bit);}
      if (root == root_frame) {}
      else {Frame temp; temp=read_root; depth=1;
	    while ((temp) && (temp != root)) {temp=frame_home(temp); depth++;}
	    if ((NULLP(temp)) || (depth > root_print_depth)) root=root_frame;
	    else {DO_TIMES(i,depth) fputc('^',stream);}}}}
  else root=root_frame;
  if (frame == root) fputc('/',stream);
  else fprint_frame_path(stream,frame,root);
}
 
/* Parses a frame from a string. */
Frame parse_frame_from_string(char *string)
{
  generic_stream stream;
  stream.stream_type = string_input; stream.ptr.string_in = &string;
  return parse_frame(&stream);
}

/* Prints a frame into a string.
   (mallocs the string, so be sure to free() it.) */
char *print_frame_to_string(Frame frame)
{
  generic_stream stream;
  INITIALIZE_STRING_STREAM(ss,sbuf,1);
  stream.stream_type = string_output; stream.ptr.string_out = &ss;
  FLET(Frame,read_root,root_frame)
    print_frame(&stream,frame);
  END_FLET(read_root);
  return ss.head;
}


/* This is used to establish frame references from C code */

Frame frame_ref(char *pathname)
{
  Frame result;
  {FLET(boolean,suppress_autoload,True)
     {FLET(frame_bits,frame_creation_mask,(FRAME_P_MASK | FRAME_TYPE_MASK))
	result=parse_frame_from_string(pathname);
      END_FLET(frame_creation_mask);}
   END_FLET(suppress_autoload);}
  return result;
}

/*
  Local variables: ***
  compile-command: "../etc/compile-framer" ***
  End: **
*/

