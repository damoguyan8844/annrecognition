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

   This file implements the dynamic allocation kernel and attempts
   to save space and time by malloc'ing arrays of structures instead
   of mallocing the structures one at a time.
 
  Modification history:
      13 June 1992 - Began tracking modification history (Haase)
*****************************************************************/
 
#include "framer.h"
#include "internal/private.h"

/* Macrology */

#define ALLOCATE_NEW_ARRAY(start,end,type,size) \
   start=fra_allocate(size,sizeof(type)); end=start+size
#define ALLOCATOR(first,last,overflow) \
   ((first == last) ? (overflow) : (first++))
#ifdef SUN
#define reallocate (ptr,new_size) ((ptr == NULL) ? malloc(new_size) : realloc(ptr,new_size))
#endif
#ifndef SUN
#define reallocate(ptr,new_size) realloc(ptr,new_size)
#endif

#if __MSDOS__
#include <alloc.h>
#else
#include <limits.h>
#define coreleft() ULONG_MAX
#endif 

/* For debugging */

boolean trace_allocation=False;
long ground_memory_in_use=0;

void announce_watched_ground_handout();
void announce_watched_ground_reclaim();

extern exception Out_Of_Memory;
Grounding watch_for_ground=NULL;
struct PAIR *watch_for_pair=NULL;
Grounding watch_for_frame=NULL;
boolean watch_memory=False;
struct MEMORY_RECORD {Grounding record; struct MEMORY_RECORD *more;} *memory_record;


/* Declarations */

extern Grounding t_symbol;

long grounds_requested=0, grounds_allocated=0, grounds_reclaimed=0;

struct FRAME initial_frames[FRAME_BLOCK_SIZE];
Frame frame_block, end_of_frame_block;
struct FRAME_APPENDIX initial_frame_appendices[FRAME_BLOCK_SIZE/3],
            *frame_appendix_block, *end_of_frame_appendix_block;
struct GROUNDING initial_grounds[GROUND_BLOCK_SIZE],
  *ground_block, *end_of_ground_block, *free_grounds;
char initial_permanent_strings[FRAME_BLOCK_SIZE*10], 
     *permanent_string_space, *end_of_permanent_string_space;


/* Traced allocation and reallocation functions */

void *careful_allocate(size_t number,size_t size)
{
  void *tmp;
  if (trace_allocation)
    {tmp=calloc(number,size);
     fprintf(stderr,
	     "%%%% Allocating %ld objects of size %ld\n",
	     (long) number, (long) size);}
  else tmp=calloc(number,size);
  if (tmp == NULL)
   raise_crisis(Out_Of_Memory);
  return tmp;
}

void *careful_reallocate(void *ptr,size_t new_size)
{
  void *tmp;
  if (trace_allocation)
    {if (ptr == NULL) tmp=malloc(new_size);
     else tmp=realloc(ptr,new_size);
     fprintf(stderr,"%%%% Growing pointer to %ld elements\n",(long) new_size);}
  else if (ptr == NULL) tmp=malloc(new_size);
     else tmp=realloc(ptr,new_size);
  if (tmp == NULL)
    raise_crisis(Out_Of_Memory);
  return tmp;
}

Grounding toggle_allocation_tracing()
{
  trace_allocation=(NOT(trace_allocation));
  if (trace_allocation) return t_symbol; 
  else return NULL;
}


Frame frame_underflow()
{
   if (trace_allocation)
     fprintf(stderr,
      "%%%% Allocating a block of %d new frames (%d bytes)\n",
	FRAME_BLOCK_SIZE,FRAME_BLOCK_SIZE*sizeof(Frame));
   ALLOCATE_NEW_ARRAY(frame_block,end_of_frame_block,struct FRAME,FRAME_BLOCK_SIZE);
   if (frame_block == NULL) 
     {raise_crisis(Out_Of_Memory); return NULL;}
   else	return frame_block++;
}

Frame_Appendix *frame_appendix_underflow()
{
   if (trace_allocation)
     fprintf(stderr,
      "%%%% Allocating a block of %d frame arrays (%d bytes)\n",
      FRAME_BLOCK_SIZE/3,(FRAME_BLOCK_SIZE/3)*sizeof(Frame_Appendix));
   ALLOCATE_NEW_ARRAY(frame_appendix_block,end_of_frame_appendix_block,
                      Frame_Appendix,FRAME_BLOCK_SIZE/3);
   if (frame_appendix_block == NULL) 
     {raise_crisis(Out_Of_Memory); return NULL;}
   else	return frame_appendix_block++;
}

struct GROUNDING *ground_underflow()
{
   if (trace_allocation)
    {fprintf(stderr,"%%%% Allocating %d new grounds (%d bytes)\n",
	     GROUND_BLOCK_SIZE,GROUND_BLOCK_SIZE*sizeof(struct GROUNDING));
     fprintf(stderr,
       "%%%% after %ld requests, %ld allocations, and %ld reclamations\n",
       grounds_requested,grounds_allocated,grounds_reclaimed);}
   ALLOCATE_NEW_ARRAY(ground_block,end_of_ground_block,
	              struct GROUNDING,GROUND_BLOCK_SIZE);
   if (ground_block == NULL) 
     {raise_crisis(Out_Of_Memory); return NULL;}
   else	return ground_block++;
}

Grounding new_ground()
{
  struct GROUNDING *result;
  grounds_requested++;
  if (free_grounds == NULL)
    {grounds_allocated++;
     result= ALLOCATOR(ground_block,end_of_ground_block,ground_underflow());}
  else {result=free_grounds; free_grounds=result->contents.next_free;}
#if ALPHA_RELEASE
  if (watch_for_ground == (Grounding) result) 
    announce_watched_ground_handout();
  if (watch_memory)
    {struct MEMORY_RECORD *new; ALLOCATE(new,struct MEMORY_RECORD,1);
     new->record=(Grounding) result; new->more=memory_record; memory_record=new;}
#endif /* ALPHA_RELEASE */
  ground_memory_in_use=ground_memory_in_use+(sizeof(struct GROUNDING));
  return (Grounding) result;
}

void free_up_ground(struct GROUNDING *ground)
{
#if ALPHA_RELEASE
  if (watch_for_ground == (Grounding) ground) 
    announce_watched_ground_reclaim(ground);
#endif /* ALPHA_RELEASE */
  /* Don't garbage collect twice. */
  if (ground_type(ground) == any_ground) 
    raise_crisis("Internal: Double garbage collection!");
  else
    {grounds_reclaimed++;
     set_ground_type(ground,any_ground);
     ground->contents.next_free=free_grounds;
     ground_memory_in_use=ground_memory_in_use-(sizeof(struct GROUNDING));
#if ALPHA_RELEASE
     if (watch_memory)
       {struct MEMORY_RECORD *last, *ptr; ptr=memory_record;
	if (ptr == NULL) {}
	else if (ptr->record == ((Grounding) ground))
	  {memory_record=ptr->more; free(ptr);}
	else {last=ptr; ptr=ptr->more;
	      while ((ptr != NULL) && (ptr->record != ((Grounding) ground))) 
		{last=ptr; ptr=ptr->more;}
	      if (ptr) {last->more=ptr->more; free(ptr);}}}
#endif /* ALPHA_RELEASE */
     free_grounds=ground;}
}

char *allocate_permanent_string(char *string)
{
   char *result; int size; size=strlen(string);
   if ((end_of_permanent_string_space - permanent_string_space) < size+1)
     {permanent_string_space=fra_allocate(FRAME_BLOCK_SIZE*10,sizeof(char));
      end_of_permanent_string_space=permanent_string_space+FRAME_BLOCK_SIZE*10;}
   result=permanent_string_space; 
   permanent_string_space=permanent_string_space+size+1;
   return strcpy(result,string);
}


void init_framer_memory()
{
   frame_block= initial_frames;
   end_of_frame_block=frame_block+FRAME_BLOCK_SIZE;
   frame_appendix_block= initial_frame_appendices;
   end_of_frame_appendix_block=frame_appendix_block+FRAME_BLOCK_SIZE/3;
   ground_block= initial_grounds;
   end_of_ground_block= initial_grounds+GROUND_BLOCK_SIZE;
   permanent_string_space= initial_permanent_strings;
   end_of_permanent_string_space=permanent_string_space+FRAME_BLOCK_SIZE*10;
}


/* For Debgging purposes */

Frame find_frame(Frame frame,Frame beneath)
{
  Frame tmp;
  if (frame == beneath) return beneath;
  else {DO_ANNOTATIONS(a,beneath)
	  {tmp=find_frame(frame,a);
	   if (tmp != NULL) return tmp;}}
  return NULL;
}

Grounding find_free_ground(ground,among)
     Grounding ground, among;
{
  if (ground == among) return ground;
  else if (among == NULL) return NULL;
  else return find_free_ground(ground,the(next_free,among));
}

Frame frame_checker(frame)
     Frame frame;
{
  if (frame == (Frame ) watch_for_frame) raise_crisis("Danger danger");
  return frame;
}

void print_memory_leaks()
{
  struct MEMORY_RECORD *ptr=memory_record;
  while (ptr != NULL)
    {fprint_ground(stderr,ptr->record); fputc('\n',stderr); ptr=ptr->more;}
}

void announce_watched_ground_handout()
{
  fprintf(stderr,"Handing out watched for ground");
}

void announce_watched_ground_reclaim(Grounding gnd)
{
  fprintf(stderr,"Reclaiming watched for ground");
  fprint_ground(stderr,gnd);
}

