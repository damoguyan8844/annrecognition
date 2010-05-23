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
  This file implements functions for generating formatted tables.  It is used
  by FRAXL for formatted text output.

  Modification history:
    13 June 1992 - Began tracking modification history (Haase)
*************************************************************************/
 
#include "framer.h"
#include "fraxl.h"

Grounding table_being_made=NULL;
Grounding apply0(Grounding thunk);

#define ROW_MAX_COL(row) (GINTEGER(GVREF(row,0)))
#define ROW_REF(row,i) (GVREF(row,i+1))

Grounding make_row()
{
  Grounding vector; GVMAKE(vector,21);
  {DO_TIMES(i,21) GVSET(vector,i,NULL);}
  {GVSET(vector,0,integer_to_ground(0));}
  return vector;
}

void add_cell(int column,char *string)
{
  if ((table_being_made == empty_list) ||
      (NOT(TYPEP(GCAR(table_being_made),vector_ground))) || 
      (column < (GINTEGER(GVREF((GCAR(table_being_made)),0)))))
    table_being_made=cons_pair(make_row(),table_being_made);
  {GVSET(GCAR(table_being_made),column+1,string_to_ground(string));}
  {GVSET(GCAR(table_being_made),0,integer_to_ground(column+1));}
}

static int string_width(char *ptr)
{
  int counter=0, width=0; 
  while (*ptr != '\0')
    if (*ptr++ != '\n') counter++;
    else if (counter > width) 
      {width=counter; counter=0;}
    else counter=0;
  if (counter > width) width=counter;
  return width;
}



void print_row(Grounding row,generic_stream *stream,int *widths)
{
  boolean do_more=True; int xpos=0, goal=0;
  char *ptrs[MAX_TABLE_WIDTH];
  {DO_TIMES(i,GINTEGER(GVREF(row,0))+1) 
    {Grounding v; v=ROW_REF(row,i);
     if (v) ptrs[i]=(GSTRING(v)); else ptrs[i]=NULL;}}
  while (do_more)
    {do_more=False;
     {DO_TIMES(i,ROW_MAX_COL(row)+1)
       if (ptrs[i] != NULL)
	 {char *ptr; ptr=ptrs[i];
	  while ((*ptr != '\n') && (*ptr != '\0'))
	    {xpos++; gsputc(*ptr++,stream);}
	  goal=goal+widths[i];
	  while (xpos < goal) {gsputc(' ',stream); xpos++;}
	  goal=xpos;
	  if (*ptr == '\n') 
	    {ptrs[i]=ptr+1; do_more=True;}
	  else ptrs[i]=NULL;}
       else {goal=xpos+widths[i]; 
	     while (xpos < goal) {gsputc(' ',stream); xpos++;}}}
     gsputc('\n',stream);}
}

void print_table(Grounding table,generic_stream *stream)
{
  int widths[MAX_TABLE_WIDTH];
  {DO_TIMES(i,MAX_TABLE_WIDTH) widths[i]=0;}
  {DO_LIST(row,table)
     if (TYPEP(row,vector_ground))
       {DO_TIMES(i,ROW_MAX_COL(row)+1)
	  {int width; 
	   if (ROW_REF(row,i) == NULL) width=0; 
	   else width=string_width(GSTRING(ROW_REF(row,i)));
	   if (width > widths[i]) widths[i]=width;}}}
  {DO_LIST(row,table)
     if (TYPEP(row,vector_ground))
       print_row(row,stream,widths);
     else print_table(row,stream);}
}


/* Table functions */

Grounding make_table(Grounding thunk)
{
  Grounding result;
  WITH_TABLE()
    result=apply0(thunk);
  END_WITH_TABLE();
  FREE_GROUND(result);
  return NULL;
}

Grounding string_cell(Grounding column,Grounding string)
{
  add_cell(the(integer,column),the(string,string));
  return string;
}

Grounding ground_cell(Grounding column,Grounding ground)
{
  add_cell(the(integer,column),print_ground_to_string(ground));
  return ground;
}

Grounding compound_cell(Grounding column,Grounding thunk)
{
  Grounding result;
  {WITH_OUTPUT_TO_STRING(gs,1024);
   {FLET(generic_stream *,standard_output,gs)
      {result=apply0(thunk); FREE_GROUND(result);
       add_cell(GINTEGER(column),string_so_far(gs));}
    END_FLET(standard_output);}
   END_WITH_OUTPUT_TO_STRING(gs);}
  return NULL;
}


#if 0
(make-table (lambda () 
               (string-cell 0 "   ")
               (string-cell 1 "Testing ")
               (ground-cell 2 3)
               (ground-cell 2 4)))
#endif


