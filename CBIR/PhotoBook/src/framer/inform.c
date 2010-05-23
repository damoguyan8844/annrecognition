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
  This file contains functions for the messages with which
  FRAMER informs the user of events and problems.  Different
  implementations may wish to define their own versions of
  these functions which (for instance) go to a window system,
  speech synthesizer, or flashing lights to communicate
  important messages.
*************************************************************************/
 
#include <stdio.h>
#include <signal.h>
#include "framer.h"

extern boolean trap_unknown_frames;
extern Frame_Array *zip_codes;

/* Setting this to False shuts up most of the messages in this file. */
boolean announce_file_ops=True;
/* Setting this to False causes the query function to be called when
   a new annotation is created by the reader. */
boolean reject_unknown_frames=False;
/* This is signalled when a new annotation, created by the reader, is rejected. */
exception Unknown_Frame_Reference="Tried to read a non-existent frame",
  Abort_to_Shell="Aborted to top level";

/* Announcing and reporting normal restores and backups */

/* This announces the beginning of a restore. */
void announce_restore(char *file,Frame home,FILE *stream)
{
  /* Currently we don't use `stream' but some interfaces
     may wish to dynamically display loading information and
     a pointer to `stream' might help that process. */
  if (announce_file_ops)
    if (home==NULL)
      fprintf(stderr,";; Loading an overlay from `%s' onto root\n",file);
    else
      {fprintf(stderr,";; Loading a frame from `%s' under #",file);
       {FLET(Frame,read_root,NULL) fprint_frame(stderr,home); END_FLET(read_root);}
       fprintf(stderr,"\n");};
}
 
/* This announces a successful restore. */
void announce_successful_restore(char *file,Frame frame)
{
  if (announce_file_ops)
    if (frame == root_frame)
      fprintf(stderr,";; Successful overlay of `%s' onto root\n",file);
    else
      {fprintf(stderr,";; Successfully restored `%s' as #",file);
       {FLET(Frame,read_root,NULL) fprint_frame(stderr,frame); END_FLET(read_root);}
       fprintf(stderr,"\n");};
}
 
/* This announces an aborted restore. */
void announce_aborted_restore(char *file,Frame top,Frame f)
{
  if (f)
    {fprintf(stderr,";; Aborted restore of #");
     {FLET(Frame,read_root,NULL) fprint_frame(stderr,top); END_FLET(read_root);}
     fprintf(stderr," from `%s' while reading #",file);
     {FLET(Frame,read_root,NULL) fprint_frame(stderr,f); END_FLET(read_root);}
     fprintf(stderr,"\n");}
  else fprintf(stderr,";; Aborted restore from `%s'\n",file);
}
 
/* This announces the start of a backup. */
void announce_backup(char *file,Frame frame,FILE *stream)
{
  /* Currently we don't use `stream' but some future interfaces
     may wish to dynamically display loading information and
     a pointer to `stream' might help that process. */
  if (announce_file_ops)
    {fprintf(stderr,";; Backing up the frame #");
     {FLET(Frame,read_root,NULL) fprint_frame(stderr,frame); END_FLET(read_root);}
     fprintf(stderr," into the file `%s'\n",file);}
}
 
/* This reports a successful backup. */
void announce_successful_backup(char *file,Frame frame)
{
  if (announce_file_ops)
    if (frame == root_frame)
      fprintf(stderr,";; Successfuly backed up the root frame into `%s'\n",file);
    else
      {fprintf(stderr,";; Successfully updated `%s' with #",file);
       {FLET(Frame,read_root,NULL) fprint_frame(stderr,frame); END_FLET(read_root);}
       fprintf(stderr,"\n");};
}

/* This anounces a backup which has been aborted. */
void announce_aborted_backup(char *file)
{
  fprintf(stderr,";; Aborted backup to `%s'\n",file);
}


/* Signalling errors */

/* frame_error signals an error having to do with a frame, printing
   the frame pathname (with a herald prefix) into exception_details */
void frame_error(exception ex,char *herald,Frame frame)
{
  WITH_OUTPUT_TO_EXISTING_STRING(gstream,exception_details,EXCEPTION_DETAILS_SIZE);
  FLET(Frame,read_root,NULL) 
    {gsputs(herald,gstream); gsputs(" #",gstream); print_frame(gstream,frame);} 
  END_FLET(read_root);
  raise_crisis(ex);
}

/* frame_error2 signals an error having to do with two frames, printing
   the frame pathnames (with a herald prefix) into exception_details */
void frame_error2(exception ex,Frame frame1,char *relation,Frame frame2)
{
  WITH_OUTPUT_TO_EXISTING_STRING(gstream,exception_details,EXCEPTION_DETAILS_SIZE);
  FLET(Frame,read_root,NULL) 
    {print_frame(gstream,frame1); gsputc(' ',gstream);
     gsputs(relation,gstream); gsputs(" #",gstream);
     print_frame(gstream,frame2);} 
  END_FLET(read_root);
  raise_crisis(ex);
}

/* This signals an error involving the printing of a ground. */
void ground_error(exception ex,char *herald,Grounding ground)
{
  WITH_OUTPUT_TO_EXISTING_STRING(gstream,exception_details,EXCEPTION_DETAILS_SIZE);
  {FLET(Frame,read_root,NULL) 
     {gsputs(herald,gstream); print_ground(gstream,ground);}
   END_FLET(read_root);}
  raise_crisis(ex);
}


/* Rejecting deletion */

/* This tells the user that a frame cannot be deleted because of some spinoff. */
void reject_deletion(Frame frame,Frame spinoff)
{
  /* This reports when a frame cannot be razed because it
     has spinoffs. */
  frame_error2("Can't delete an annotation with spinoffs",frame,
	       " has the spinoff ",spinoff);
}


/* Letting us know about new frame creation */ 

void error_print_frame(Frame f)
{
  FLET(Frame_Array *,zip_codes,NULL)
    {
      FLET(Frame,read_root,NULL)
	fprint_frame(stderr,f);
      END_FLET(read_root);
    }
  END_FLET(zip_codes);
}

/* This lets us know (if trap_unknown_frames is true) if a new
   (non inherited) annotation is being created.  It catches a lot
   of typos. */
Frame reader_make_annotation(Frame root,char *buffer)
{
  char herald[100]; 
  fprintf(stderr,";;; Warning: creating new annotation \"%s\" under #",buffer);
  error_print_frame(root); fprintf(stderr,"\n");
  if (reject_unknown_frames)
    {strcpy(herald,"no annotation \""); strcat(herald,buffer); strcat(herald,"\" of ");
     frame_error(Unknown_Frame_Reference,herald,root); return NULL;}
  else return use_annotation(root,buffer);
}



/* Handling signals */

void handle_abort_signal(int signal)
{ raise_crisis(Abort_to_Shell); }

void setup_abort_handling()
{ signal(SIGINT,handle_abort_signal); }

