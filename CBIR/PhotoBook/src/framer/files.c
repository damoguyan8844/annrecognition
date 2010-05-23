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
  This file implements functions of probing, opening and closing files for
  FRAMER.  It handles the translation between FRAMER pathnames (with directories
  separated by semicolons) into the host file system.  It also provides the
  search path functions for overlaying several physical directory trees into
  one logical tree.

  Modification history:
    13 June 1992 - Began tracking modification history (Haase)
*************************************************************************/
 
/* File functions for FRAMER */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "internal/common.h"
#include "internal/fileinfo.h"

/* TPM */
#include "sys/stat.h"

/* Checking writability */

boolean read_only_p(char *file)
{
  return !(FILE_WRITABLE_P(file));
}

struct stat tmp_buf;

#if FOR_UNIX
boolean directory_writable_p(char *path)
{
  char copy[200], *dir_end;
  strcpy(copy,path);
  dir_end=strrchr(copy+1,'/');
  if (dir_end == NULL) dir_end=copy+1;
  *dir_end='\0';
  return (boolean) (access(copy,W_OK) != -1);
}
#endif /* FOR_UNIX */

#if FOR_MSDOS
boolean directory_writable_p(char *path)
{
  char copy[200], *dir_end;
  strcpy(copy,path);
  dir_end=strrchr(copy+1,'\\');
  if (dir_end == NULL) return True;
  *dir_end='\0';
  return (boolean) (access(copy,W_OK) != -1);
}
#endif /* FOR_MSDOS */

exception File_Unwritable="Cannot write frame file";
exception File_Unreadable="Cannot read frame file";
exception Nonexistent_File="Non-existent file";
exception Out_Of_Space="Out of disk space";


/* Dealing with search paths */

/* This stores the comma separated path list <paths> into 
   a null-terminated array of paths in path_ptr. */
void set_search_path(char *paths,char **path_ptr)
{
  char *local_paths;
  ALLOCATE(local_paths,char,strlen(paths)+1);
  strcpy(local_paths,paths);
  for (;;)
    {*path_ptr++=local_paths;
     while ((*local_paths != ',') && (*local_paths != '\0')) local_paths++;
     if (*local_paths == '\0') {*path_ptr=NULL; return;}
     *local_paths++ = '\0';};
}

/* Searches the null-terminated array of prefixes <path> for one containing
   a file <name> which passes <access_test>. */
char *search_path_for_name(char *name,char **path)
{
  char *physical_name, *result, **prefix, sep[2]; int size;
  size=strlen(name); ALLOCATE(physical_name,char,size+1); 
  /* If you're being given a physical pathname, just return it. */
  if (NOT(NULLP(strchr(name,DIRECTORY_SEPARATOR))))
    {strcpy(physical_name,name); return physical_name;}
  /* Otherwise, search the path. */
  sep[0]=DIRECTORY_SEPARATOR; sep[1]='\0';
  /* First, replace physical name */
  ALLOCATE(physical_name,char,size+1); 
  {char *read, *write; read=name; write=physical_name;
   while (*read != '\0')
     {if ((*read) == ';') *write=DIRECTORY_SEPARATOR;
      else (*write)=tolower((*read)); read++; write++;}
   *write='\0';}
  /* Try prepending each string in backup_search_path to the name and
     doing an access check, returning if it returns true */
  prefix=path; while (*prefix != NULL)
    {result=fra_allocate(size+strlen(*prefix)+2,sizeof(char));
     strcpy(result,*prefix); strcat(result,sep); strcat(result,physical_name);
     if (FILE_EXISTS_P(result))
       {free(physical_name); return result;}
     else {free(result); prefix++;}}
  prefix=path; while (*prefix != NULL)
    {result=fra_allocate(size+strlen(*prefix)+2,sizeof(char));
     strcpy(result,*prefix); strcat(result,sep); strcat(result,physical_name);
     if (directory_writable_p(result))
       {free(physical_name); return result;}
     else {free(result); prefix++;}}
  free(physical_name);
  return NULL;
}

char *search_for_name(char *name,char *suffix,char **path)
{
  char *basename, *full_name;
  ALLOCATE(basename,char,strlen(name)+1+strlen(suffix)+1);
  sprintf(basename,"%s.%s",name,suffix);
  full_name=search_path_for_name(basename,path); free(basename);
  return full_name;
}


/* File utilities */

/* Moves the file `from' into `to'; this tries to do a rename, but
   if it fails, does a copy and delete. */
static boolean move_file(char *from,char *to)
{
  if ((rename(from,to) != 0) && (from) && (to))
    {
      FILE *in, *out; int ch;
      in=fopen(from,"r"); out=fopen(to,"w");
      if ((in) && (out))
	{
	  while ((ch=getc(in)) != EOF) putc(ch,out);
	  fclose(in); fclose(out); 
	  remove(from);
	  return True;
	}			/* if ((in) && (out)) */
      else
	{
	  if (in) fclose(in);
	  if (out) fclose(out);
	  return False;
	}			/* else */
    }
    else return True;
}


/* Dealing with temporary files. */

struct TEMP_FILE_MAP { FILE *stream; char *real, *temp; struct TEMP_FILE_MAP *previous;} 
  *temp_map=NULL;

FILE *open_safe_stream(char *realname)
{
  char *temp_file; struct TEMP_FILE_MAP *ptr; 
  ALLOCATE(ptr,struct TEMP_FILE_MAP,1); ALLOCATE(temp_file,char,L_tmpnam+1);
  if (NOT(FILE_WRITABLE_P(realname)))
    {raise_crisis_with_details(File_Unwritable,realname); return NULL;}
  else tmpnam(temp_file); 
  if ((ptr->stream=fopen(temp_file,"w")) == NULL)
    raise_crisis_with_details(File_Unwritable,temp_file);
  else {ptr->real=realname; ptr->temp=temp_file; ptr->previous=temp_map; temp_map=ptr;}
  return ptr->stream;
}

void close_safe_stream(FILE *stream)
{
  struct TEMP_FILE_MAP **last, *map; last=(&temp_map);
  while (NOT(NULLP((*last))))
    if (((*last)->stream) == stream)
      {fclose(stream); move_file((*last)->temp,(*last)->real);
       map=(*last); (*last)=(*last)->previous; free(map); return;}
    else last=(&(*last)->previous);
  raise_crisis("Trying to safely close an unsafe stream"); 
  return;
}

void abort_safe_stream(FILE *stream)
{
  struct TEMP_FILE_MAP **last, *map; last=(&temp_map);
  while (NOT(NULLP(*last)))
    if (((*last)->stream) == stream)
      {fprintf(stderr,";;; Aborted writing to %s, remains are in %s\n",
	       (*last)->real,(*last)->temp);
       fclose(stream); map=(*last); (*last)=(*last)->previous; 
       free(map); return;}
    else last=(&(*last)->previous);
  raise_crisis("Trying to safely close an unsafe stream"); 
  return;
}

