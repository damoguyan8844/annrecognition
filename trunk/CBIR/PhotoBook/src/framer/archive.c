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

*/

/*****************************************************************
  This file implements the dumping of frames to files
  for later reloading.
 
*****************************************************************/
 
#include "framer.h"
#include "internal/private.h"
#include "internal/fileinfo.h"

extern int root_print_depth;
 
void free_zip_codes(Frame_Array *array);

/* Whether to load on reference, write on with zip codes, or
   whether you're reading a read only frame (which will only
   be read only once its read, for obvious reasons). */
boolean load_on_reference=False, use_zip_codes=True;
static boolean reading_read_only=False;
/* Where the root image is stored (first argument to executable) */ 
char *root_filename="radix"; 
void write_image_modifications(void);

/* A variety of archival errors. */
exception Read_Error="Error in reading";
exception Write_Error="Error in writing";
exception Unexpected_EOF="Unexpected end of file marker";
exception Dislocated_Frame="Trying to dislocate file frame!";
exception Not_Backup_File="Trying to restore a file which isn't a backup file!";
extern exception Recursive_Prototype;

/* For storing the backup search path */
char *backup_search_path[64];
 
char *frame_name_buffer;
/* A state variable for debugging the undumper.
   This is bound to whatever frame is currently being `undumped'. */
Frame undumping_frame;


/* Relevant prototypes for later definitions */
 
static Frame load_frame_from_stream(Frame root,FILE *stream);
static Frame write_backup_root(Frame frame,char *file);
static void dump_frame(FILE *stream,Frame frame,int depth);
static Frame undump_frame(FILE *stream,Frame under,int depth);
Frame raw_local_probe_annotation(Frame frame,char *name);
boolean backup_root_p(Frame f);
char *frame_filename(Frame f);
char *frame_server_name(Frame f);

/* These are defined in inform.c or its analog */
void announce_backup(char *name,Frame frame,FILE *stream);
void announce_successful_backup(char *name,Frame frame);
void announce_aborted_backup(char *file);
void announce_restore(char *name,Frame frame,FILE *stream);
void announce_successful_restore(char *name,Frame frame);
void announce_aborted_restore(char *file,Frame top,Frame troublemaker);

 
/* User functions for backup and restore*/
 
void backup_frame(Frame frame)
{
  if (remote_frame_p(frame)) return;
  else if (active_image_p) return;
  else while ((frame) && (NOT(backup_root_p(frame)))) frame=frame_home(frame);
  if (frame) write_backup_root(frame,frame_filename(frame));
  else write_backup_root(root_frame,root_filename);
}
 
static void write_root(Frame frame)
{
  if (frame == root_frame) write_backup_root(root_frame,root_filename);
  else write_backup_root(frame,frame_filename(frame));
}

/* backup_tree iterates down the FRAMER structure, saving every modified
   frame it finds, skipping ephemeral and remote subtrees. */
void backup_tree(Frame frame,Frame root)
{
  if (frame_ephemeral_p(frame)) return;
  else if (remote_frame_p(frame)) return;
  else if (frame_modified_p(frame))
    if (backup_root_p(frame)) 
      {if (frame_modified_p(probe_annotation(frame,"+filename"))) write_root(root);
       write_backup_root(frame,frame_filename(frame));}
    else write_root(root);
  if (NOT(frame_current_p(frame))) {}
  else if (backup_root_p(frame))
    {DO_ANNOTATIONS(a,frame) backup_tree(a,frame);}
  else {DO_ANNOTATIONS(a,frame) backup_tree(a,root);}
}

void backup_everything()
{
  if (active_image_p) write_image_modifications();
  else backup_tree(root_frame,root_frame);
}

void backup_root_frame(char *filename)
{
  write_backup_root(root_frame,filename);
  backup_tree(root_frame,root_frame);
}

/* Loads the filename where <frame> is stored. */
void restore_frame_from_file(Frame frame)
{
  load_frame_from_file(frame_filename(frame));
}

/* Determining modification information */

Frame find_modified_annotation(Frame frame)
{
  if (frame_modified_p(frame)) return frame;
  else {DO_SOFT_ANNOTATIONS(a,frame)
	 if (frame_ephemeral_p(a)) {}
	 else if (remote_frame_p(a)) {}
	 else if (backup_root_p(a))
	   /* Here's a special case; a backup root merits saving if
	      we've just learned that it is a backup root. */
	   {if (frame_modified_p(raw_local_probe_annotation(a,"+filename")))
	      return raw_local_probe_annotation(a,"+filename");}
	 else {Frame mod; mod=find_modified_annotation(a);
	       if (mod) return mod;}}
  return NULL;
}

/* Marks a frame and its provided annotations as not current.
   This is used when a load aborts for some reason. */
static void unload_frame(Frame frame)
{
  if (backup_root_p(frame)) return;
  clear_frame_modified_p(frame); clear_frame_current_p(frame);
  {DO_SOFT_ANNOTATIONS(a,frame) unload_frame(a);}
}
 
/* Writing frames to files */
 
/* Writes a backup of <frame> onto the file <name>. */
static Frame write_backup_root(Frame frame,char *name)
{
  char *physical_name=NULL; FILE *stream=NULL; boolean success=False;
  Frame old_rr; boolean old_syntax; Frame_Array *old_zip_codes;
  /* We dynamically bind a numbber of global variables to ensure that anybody
     can read the file back in without problems. */
  physical_name=backup_file_name(name);
  if (physical_name) stream=open_safe_stream(physical_name);
  else {raise_crisis_with_details(File_Unwritable,physical_name); return NULL;}
  if (NULLP(stream))
    {raise_crisis_with_details(File_Unwritable,physical_name); return NULL;}
  else {UNWIND_PROTECT
	  {old_rr=read_root; old_syntax=strict_syntax; old_zip_codes=zip_codes;
	   if (zip_codes) raise_crisis("Two zip code tables at once!");
	   zip_codes=NULL; read_root=NULL; strict_syntax=True;
	   announce_backup(name,frame,stream);
	   /* Output the header */
	   if (frame->home) fprint_frame(stream,frame->home); putc('\n',stream); 
	   read_root=frame;	/* Move the read root */
	   if (use_zip_codes) zip_codes=make_frame_array(INITIAL_ZIP_CODES);
	   dump_frame(stream,frame,0); 
	   clear_frame_modified_p(frame); success=True;}
	  ON_UNWIND
	    /* In clean up, we first handle memory and dynamic bindings. */
	    {if (zip_codes) free_zip_codes(zip_codes); 
	     zip_codes=NULL; read_root=NULL; strict_syntax=old_syntax;
	     if (NOT(success)) 
	       {announce_aborted_backup(name); abort_safe_stream(stream);
		unload_frame(frame); zip_codes=old_zip_codes; read_root=old_rr;}
	     else {announce_successful_backup(name,frame);
		   close_safe_stream(stream);
		   zip_codes=old_zip_codes; read_root=old_rr;}}
	  END_UNWIND}
  return frame;
}


#define INDENT(stream,indent) {int p; for (p=indent;p>0;p--) putc(' ',stream);}
#define FRAME_STUB(stream,name,code,depth,string_arg)              \
   char *arg; int p; for (p=depth;p>0;p--) putc(' ',stream);          \
   fprint_frame_name(name,stream); fprintf(stream," 0 %s\n",code); \
   arg=string_arg; if (arg)                                        \
     {INDENT(stream,depth); putc('"',stream);                      \
      while (*arg != '\0')                                         \
	{if ((*arg == '"') || (*arg == '\\')) putc('\\',stream);   \
	 putc(*arg++,stream);}                                     \
      fprintf(stream,"\"\n");}

/* This function determines whether or not a frame is worth dumping. 
   A frame is worth dumping if it is not ephemeral and either has a
   ground, a variant prototype, or some annotations that are worth dumping.
*/
boolean dump_p(Frame frame)
{
  if (frame_ephemeral_p(frame)) return False;
  else if (NOT(frame_current_p(frame)))
    if ((frame_type(frame) == alias_frame) || (backup_root_p(frame)) )
      return True;
    else return False;
  else if (frame->ground) return True;
  else if (NOT(default_prototype_p(frame,frame->prototype))) return True;
  else {DO_SOFT_ANNOTATIONS(a,frame) if (dump_p(a)) return True;}
  return False;
}

static void make_tree_unmodified(Frame f)
{
  clear_frame_modified_p(f);
  {DO_SOFT_ANNOTATIONS(a,f) make_tree_unmodified(a);}
}

static void dump_frame(FILE *stream,Frame frame,int depth)
{
  int how_many_annotations=0; boolean obeys_default_rule;
  /* This catches frames which have been given filenames 
     but haven't had the backup_root bit set or which have had
     their filenames removed. */
  if ((frame_type(frame)) == alias_frame)
    {FRAME_STUB(stream,frame->aname,"AIN",depth,NULL);
     INDENT(stream,depth); fprint_frame(stream,frame->prototype);
     fputc('\n',stream); clear_frame_modified_p(frame); return;}
  else if (remote_frame_p(frame))
    {FRAME_STUB(stream,frame->aname,"RNN",depth,frame_server_name(frame));
     return;}
  else if (backup_root_p(frame) && (depth>0))
    {FRAME_STUB(stream,frame->aname,"FNN",depth,frame_filename(frame));
     return;}
  /* Finally, this is the standard case, dumping a local frame.  We need to know
     whether it obeys the default prototype rule and how many annotations
     of it we will dump. */
  obeys_default_rule=default_prototype_p(frame,frame->prototype);
  {DO_SOFT_ANNOTATIONS(a,frame) 
     if (dump_p(a)) 
       {set_search_bit(a,FRAMER_SYSTEM_BIT); /* Save the work... */
	how_many_annotations++;}}
  INDENT(stream,depth); fprint_frame_name(frame->aname,stream);
  fprintf(stream," %d L%c%c\n",
	  how_many_annotations,
	  ((obeys_default_rule) ? 'D' : ((frame->prototype) ? 'V' : 'N')),
	  ((frame->ground) ? 'G' : 'N'));
  if (NOT((obeys_default_rule==True) || (frame->prototype == NULL)))
    {INDENT(stream,depth); putc('#',stream); 
     fprint_frame(stream,frame->prototype); putc('\n',stream); }
  if (NOT(NULLP(frame->ground)))
    {INDENT(stream,depth); fprint_ground(stream,frame->ground); putc('\n',stream);}
  /* Just in case */
  if (ferror(stream)) raise_crisis(Write_Error);
  {DO_SOFT_ANNOTATIONS(a,frame)
     if (search_bit_p(a,FRAMER_SYSTEM_BIT))  /* Set above. */
       {clear_search_bit(a,FRAMER_SYSTEM_BIT); dump_frame(stream,a,depth+1);}
     else make_tree_unmodified(a);}
  clear_frame_modified_p(frame);
}
 

/* Restoring frames from files */
 
/* Loads the file <filename>, returning the frame it found there. */
Frame load_frame_from_file(char *filename)
{
  Frame root, frame=NULL; boolean old_ro, trapping; Frame_Array *old_zip_codes;
  char first, *physical_name; FILE *stream=NULL; 
  old_zip_codes=zip_codes; trapping=trap_unknown_frames; old_ro=reading_read_only;
  physical_name=backup_file_name(filename);
  if (physical_name == NULL) 
    {raise_crisis_with_details(Nonexistent_File,filename); return NULL;}
  stream=fopen(physical_name,"r"); 
  if (stream == NULL)
    if (FILE_EXISTS_P(physical_name))
      raise_crisis_with_details(File_Unreadable,filename); 
    else raise_crisis_with_details(Nonexistent_File,filename); 
  WITH_HANDLING
    {/* Bind some globals */
      if (zip_codes) raise_crisis("Two zip code tables at once!");
      zip_codes=NULL; trap_unknown_frames=False; 
      reading_read_only=read_only_p(physical_name);
      first=getc(stream);
      if (first == '\n') root=NULL; /* It's a root image. */
      else if ((first == '#') || (first == '/')) /* It's a valid backup file. */
	{FLET(Frame,read_root,NULL)
	   {ungetc(first,stream); root=fparse_frame(stream);}
	   END_FLET(read_root);}
      else raise_crisis_with_details(Not_Backup_File,physical_name);
      announce_restore(physical_name,root,stream);
      frame=load_frame_from_stream(root,stream);}
  ON_EXCEPTION
    {/* The frame things broke on. */
      Frame troublemaker=undumping_frame, top_frame; undumping_frame=NULL;
      if (zip_codes) free_zip_codes(zip_codes); 
      zip_codes=old_zip_codes;  /* Reset bindings */
      trap_unknown_frames=trapping; reading_read_only=old_ro;
      /* Find the highest frame between troublemaker and the root; this is
	 what load_frame_from_stream should have returned. */
      top_frame=troublemaker; while ((top_frame != NULL) && (top_frame->home != root))
	top_frame=top_frame->home;
      /* `Dirty' the troublemaker and its annotations. */ 
      if (top_frame) 
	{clear_frame_current_p(top_frame); clear_frame_modified_p(top_frame);
	 {DO_SOFT_ANNOTATIONS(a,top_frame) unload_frame(a);}}
      zip_codes=NULL; read_root=NULL;
      announce_aborted_restore(physical_name,top_frame,troublemaker); fclose(stream);
      reraise();}
  END_HANDLING;
  /* Clean up.  We set zip_codes to NULL, so the announcements come out
     readably (not zip coded). */
  if (zip_codes) free_zip_codes(zip_codes); zip_codes=NULL; 
  announce_successful_restore(physical_name,frame); fclose(stream);
  zip_codes=old_zip_codes;  /* Reset bindings */
  trap_unknown_frames=trapping; reading_read_only=old_ro;
  free(physical_name);
  return frame;
}

/* Loads a frame from a stream beneath another frame. */
static Frame load_frame_from_stream(Frame root,FILE *stream)
{Frame result_frame=NULL, old_rr; char *old_frame_name_buffer; boolean old_al, old_syn; 
 UNWIND_PROTECT
   /* Temporarily bind various global variables to ensure re-readability */
   {old_rr=read_root; old_frame_name_buffer=frame_name_buffer; 
    old_al=suppress_autoload; old_syn=strict_syntax; 
    frame_name_buffer=fra_allocate(512,sizeof(char));
    suppress_autoload=(NOT(load_on_reference)); strict_syntax=True;
    zip_codes=make_frame_array(INITIAL_ZIP_CODES);
    result_frame = undump_frame(stream,root,0);}
 ON_UNWIND
   /* Reset what you bound above. */
   {free(frame_name_buffer); 
    read_root=old_rr; frame_name_buffer=old_frame_name_buffer; 
    strict_syntax=old_syn; suppress_autoload=old_al;}
 END_UNWIND
 return result_frame;
}

static Frame undump_frame(FILE *stream,Frame under,int depth)
{
  Frame new_frame; Grounding filename;
  int i, size; char file_code, default_code, ground_code, peek, *name;
  name=frame_name_buffer;
  /* Skip whitespace (indentation) */
  for (peek=getc(stream);;) 
    if (isgraph(peek)) {ungetc(peek,stream); break;}
    else if (peek == EOF) 
      {frame_error(Unexpected_EOF,"while reading under ",under);}
    else peek=getc(stream);
  fread_frame_name(stream,name);
  if (fscanf(stream," %d %c%c%c", &size, &file_code, &default_code, &ground_code) < 4)
    {frame_error(Unexpected_EOF,"while reading under ",under);}
  if (ferror(stream)) {frame_error(Unexpected_EOF,"while reading under ",under);}
  else if (under == NULL) undumping_frame=new_frame=root_frame; /* Reading root */
  else undumping_frame=new_frame=use_annotation(under,name);
  /* This is now the default read root */
  if ((under != NULL) && (depth == 0)) 
    {read_root = new_frame; 
     if (reading_read_only) set_frame_type(new_frame,read_only_frame);}
  set_frame_current_p(new_frame);
  switch ((int) file_code)
    {
    case 'F': 
      filename = fparse_ground(stream);
      {Frame backup_annotation;
       set_frame_type(new_frame,local_frame);
       backup_annotation = use_annotation(new_frame,"+filename");
       if ((backup_annotation->ground) && 
	   (TYPEP(backup_annotation->ground,string_ground)) &&
	   (string_compare(GSTRING(backup_annotation->ground),GSTRING(filename))))
	 {FREE_GROUND(filename); filename=NULL;}
       else set_ground(backup_annotation,filename);
       clear_frame_modified_p(backup_annotation); clear_frame_modified_p(new_frame);}
      /* If the file frame was dumped normally, then either load it or
	 mark it as uncurrent. */
      if (filename != NULL)
	if (depth == 0) 
	  {char *physical; physical=backup_file_name(GSTRING(filename));
	   if (read_only_p(physical)) set_frame_type(new_frame,read_only_frame);
	   free(physical);}
	else if (load_on_reference) ENSURE_CURRENT(new_frame);
	else clear_frame_current_p(new_frame); 
      break;
    case 'A':
      peek=getc(stream); while (isspace(peek)) peek=getc(stream);
      ungetc(peek,stream); new_frame->prototype=fparse_frame(stream);
      set_frame_type(new_frame,alias_frame); clear_frame_current_p(new_frame);
      break;
    case 'R':
      filename = fparse_ground(stream);
      {Frame server_annotation;
       set_frame_type(new_frame,remote_frame);
       server_annotation = use_annotation(new_frame,"+server");
       set_ground(server_annotation,filename);}
      clear_frame_current_p(new_frame);
      break;
    };
  switch ((int) default_code)
    {
    case 'D':
      new_frame->prototype=use_annotation(under->prototype,new_frame->aname);
      break;
    case 'V':
      {Frame proto; peek=getc(stream);
       while (isspace(peek)) peek=getc(stream);
       ungetc(peek,stream); proto=fparse_frame(stream);
       if (frame_type(proto) == alias_frame) proto=proto->prototype;
       add_spinoff(new_frame,proto);
       new_frame->prototype=proto;
       break;}
    case 'N':
      new_frame->prototype= (Frame ) NULL;
      break;
    case 'I':
      break;
    };
  if (ground_code=='G') 
    {FREE_GROUND(new_frame->ground);
     new_frame->ground=fparse_ground(stream);
     USE_GROUND(new_frame->ground);}
  /* Make just enough space for its annotations, *IF*
     they aren't any already there. */
  if ((size > 0) && (new_frame->appendix == NULL))
    {struct FRAME_APPENDIX *appendix; 
     appendix=add_appendix(new_frame);
     appendix->annotations=fra_allocate(size,sizeof(struct ANNOTATION));
     appendix->limit=size; appendix->size=0;}
  /* Load the definitions of all the annotations */
  for (i=size-1;i>=0;i--) undump_frame(stream,new_frame,depth+1);
  clear_frame_modified_p(new_frame);
  return new_frame;
}


/* Providers: Getting and Setting */

/* This macro returns true if a frame is read only, loading the
   frame if neccessary to discover this. */
#define read_only_frame_p(frame) \
  ((frame_current_p(frame)) ? (frame_read_only_p(frame)) \
   : (frame_read_only_p(get_current(frame))))

/* backup_root_p returns true if a frame has a filename annotation with
   a non-null value.  Note that this used to be a bit on the frame, though
   most (if not all) vestiges of that are gone.  
   The only tricky thing in this is looking for the filename without
   loading the frame; thus the call to raw_local_probe_annotation rather
   than probe_annotation. 
   (Would it work to just bind suppress_autloading?)
*/
boolean backup_root_p(Frame frame)
{
  Frame backup_annotation;
  backup_annotation = raw_local_probe_annotation(frame,"+filename");
  if ((backup_annotation != NULL) &&
      (backup_annotation->ground != NULL) &&
      (ground_type(backup_annotation->ground) == string_ground))
    return True;
  else return False;
}

/* frame_filename gets the filename in which this frame is stored as a root.
   This returns a filename only for the top of the subtree stored in the file.
   The only tricky thing in this is looking for the filename without
   loading the frame; thus the call to raw_local_probe_annotation. */
char *frame_filename(Frame frame)
{
  Frame backup_annotation;
  backup_annotation = raw_local_probe_annotation(frame,"+filename");
  if ((backup_annotation != NULL) &&
      (backup_annotation->ground != NULL) &&
      (ground_type(backup_annotation->ground) == string_ground))
    return GSTRING(backup_annotation->ground);
  else return NULL;
}

/* Gets the server on which this frame is stored as a root.
   This returns a server name only for the top of the subtree stored in the file.
   The only tricky thing in this is looking for the filename without
   loading the frame; thus the call to raw_local_probe_annotation. */
char *frame_server_name(Frame frame)
{
  Frame server_annotation;
  server_annotation = raw_local_probe_annotation(frame,"+server");
  if ((server_annotation != NULL) &&
      (server_annotation->ground != NULL) &&
      (ground_type(server_annotation->ground) == string_ground))
    return GSTRING(server_annotation->ground);
  else return NULL;
}

/* set_frame_filename sets the filename of a frame, loading the 
   corresponding file if it exists and signalling an error if a 
   different frame is already stored in that file. */
void set_frame_filename(Frame frame,char *name)
{
  Frame backup_annotation, loads_as;
  Grounding string_to_ground(char *string);
  TOUCH(frame_home(frame)); 
  /* We try to load the specified file first; if it doesn't exist,
     we just go ahead; if it does, we make sure that the frame stored
     there is the frame we're talking about. */
  WITH_HANDLING
    loads_as = load_frame_from_file(name);
  ON_EXCEPTION
    if (theException == Nonexistent_File)
       {loads_as = frame; 
	set_frame_current_p(frame);
	set_frame_modified_p(frame);
	CLEAR_EXCEPTION();}
    else reraise();
  END_HANDLING;
  if (loads_as != frame) 
    frame_error2(Dislocated_Frame,frame," loaded as ",loads_as);
  else {backup_annotation=use_annotation(frame,"+filename");
	set_ground(backup_annotation,string_to_ground(name));
	TOUCH(frame);}
}

void set_backup_path(char *paths)
{
  set_search_path(paths,backup_search_path);
}

/* This returns a `backup file name' for name, which ideally
   searches some path and returns the first matching pathname it
   finds. */
char *backup_file_name(char *name)
{
  char *file_with_suffix, *result; int size; size=strlen(name);
  ALLOCATE(file_with_suffix,char,size+5); strcpy(file_with_suffix,name);
  /* If the user already has specified .fra, don't add it. */
  if ((strlen(name) <= 4) || (strcmp(name+strlen(name)-4,".fra") != 0))
    strcat(file_with_suffix,".fra");   /* Add the .fra suffix */
  result=search_path_for_name(file_with_suffix,backup_search_path);
  free(file_with_suffix); return result;
}

#if (FOR_UNIX || FOR_MSDOS)
void get_backup_paths_from_env()
{
  char *search_path_name="FRAMER_SEARCH_PATH", *result;
  result=getenv(search_path_name);
  if (result == NULL) set_backup_path(""); 
  else set_backup_path(result);
}
#endif
#if FOR_MAC
void get_backup_paths_from_env()
{
  set_backup_path("Framer:local-data,Framer:data");
}
#endif

/*
  Local variables: ***
  compile-command: "../etc/compile-framer" ***
  tags-file-name: "../sources/TAGS" ***
  End: **
*/

