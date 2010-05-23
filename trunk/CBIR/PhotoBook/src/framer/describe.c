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
  This file implements functions for describing frames customizable by their
  annotations.  In particular the +summarizer annotations and +viewer annotations
  describe functions for viewing whole frames or individual components thereof.

  Modification history:
    13 June 1992 - Began tracking modification history (Haase)
*************************************************************************/
 
#include "framer.h"
#include "internal/private.h"
#include "fraxl.h"

Frame notables_frame; Grounding geq_symbol, peq_symbol;
boolean default_hide_internals=True;

void ground_in_column(int column,Grounding ground)
{
  WITH_OUTPUT_TO_STRING(stream,1024);
    {pprint_ground(stream,ground,0,40);
     add_cell(column,string_so_far(stream));}
  END_WITH_OUTPUT_TO_STRING(stream);
}

void labelled_ground_in_column(int column,char *label,Grounding ground)
{
  WITH_OUTPUT_TO_STRING(stream,1024);
    {gsputs(label,stream); 
     pprint_ground(stream,ground,strlen(label),40);
     add_cell(column,string_so_far(stream));}
  END_WITH_OUTPUT_TO_STRING(stream);
}

void frame_in_column(int column,Frame frame,Frame under)
{
  if (frame_home(frame) == under)
    add_cell(column,frame_name(frame));
  else {WITH_OUTPUT_TO_STRING(stream,1024);
	{gsputc('#',stream); print_frame_under(stream,frame,under);
	 add_cell(column,string_so_far(stream));}
	END_WITH_OUTPUT_TO_STRING(stream);}
}

void labelled_frame_in_column(int column,char *label,Frame frame,Frame under)
{
  {WITH_OUTPUT_TO_STRING(stream,1024);
   {gsputs(label,stream); 
    if (frame_home(frame) == under)
      {gsputs(frame_name(frame),stream);}
    else {gsputc('#',stream); print_frame_under(stream,frame,under);}
    add_cell(column,string_so_far(stream));}
   END_WITH_OUTPUT_TO_STRING(stream);}
}


/* Describing frames */

void describe_annotation(Frame annotation,Frame under);
void describe_annotations
  (Frame frame,Frame top,generic_stream *stream,
   int column,int depth,boolean hide_internals);

void describe_frame_to_stream(frame,stream)
     Frame frame; generic_stream *stream;
{
  int annotations_count=0,depth=1;
  ENSURE_CURRENT(frame);
  if (NOT(NULLP(probe_annotation(frame,"+depth"))))
    {Grounding gdepth; gdepth=get_inherited_ground(probe_annotation(frame,"+depth"));
     if (TYPEP(gdepth,integer_ground)) depth=GINTEGER(gdepth);}
  {FLET(Frame,read_root,NULL)
     {gsprintf(stream,"> The frame %s ( #",(frame)->aname);
      print_frame(stream,frame); 
      gsputs(" )\n",stream);
      if ((frame)->home)
	{gsputs(">   is in #",stream); 
	 print_frame(stream,(frame)->home);
	 gsputc('\n',stream);}
      if ((frame)->prototype)
	{gsputs(">  with prototype #",stream);
	 print_frame(stream,(frame)->prototype);
	 gsputc('\n',stream);}
      else gsputs(">  has no prototype\n",stream);}
   END_FLET(read_root);}
  if (!((frame)->ground)) {gsputs(">  and has no grounding\n",stream);}
  else {gsputs(">  has a ground of: ",stream);
	pprint_ground(stream,((frame)->ground),strlen(">  has a ground of: "),60);
	gsputc('\n',stream);}
  {DO_ANNOTATIONS(a,frame) annotations_count++;}
  if (annotations_count == 0)
    {gsputs(">  and has no annotations\n",stream);return;}
  else {gsprintf(stream,">  and has %d annotations\n",annotations_count);
	describe_annotations(frame,frame,stream,1,depth,False);}
}

void describe_annotations(frame,top,stream,level,depth,hide_internals)
     Frame frame, top; generic_stream *stream; int level, depth; 
     boolean hide_internals;
{
  WITH_TABLE()
   {DO_ANNOTATIONS(a,frame)
      {if ((*(a->aname) != '+') || NOT(hide_internals))
	 {Frame method; 
	  {WITH_OUTPUT_TO_STRING(stream,1024);
	     {int space; space=level*3;
	      {DO_TIMES(i,space) gsputc(' ',stream);}
	      {DO_TIMES(i,level) gsputc('>',stream);}
	      gsputc(' ',stream);
	      add_cell(0,string_so_far(stream));}
	   END_WITH_OUTPUT_TO_STRING(stream);}
	  frame_in_column(1,a,top); add_cell(2," ..... ");
	  if ((frame_type(a) == alias_frame))
	    {labelled_frame_in_column(3,"aliased to ",a->prototype,top);
	     ENSURE_CURRENT(a);}
	  if ((backup_root_p(a) == True) && (frame_current_p(a) == False))
	    add_cell(3,"(not loaded)");
	  else {method = probe_annotation(a,"+summarizer");
		if ((method == NULL) || (get_inherited_ground(method) == NULL))
		  {Frame prototype; Grounding ground; 
		   prototype=frame_prototype(a); ground=frame_ground(a);
		   if (NULLP(prototype)) add_cell(3,"(no prototype)");
		   else if (NOT(prototype_is_default(a,prototype)))
		     labelled_frame_in_column(3,"is like ",prototype,top);
		   if (NOT(NULLP(ground)))
		     labelled_ground_in_column(3,"ground= ",ground);}
		else {Grounding method_proc=NULL, temp=NULL; 
		      WITH_HANDLING
			{method_proc=fraxl_eval(get_inherited_ground(method));
			 temp=apply2(method_proc,frame_to_ground(a),
				     frame_to_ground(top));}
		      ON_EXCEPTION
			{add_cell(3,"(error in describing)");
			 CLEAR_EXCEPTION();}
		      END_HANDLING;
		      FREE_GROUND(method_proc); FREE_GROUND(temp);}
		if ((depth > level) ||
		    ((local_probe_annotation(a,"+open") != NULL) &&
		     (frame_ground(local_probe_annotation(a,"+open")) != NULL)))
		  {WITH_TABLE()
		     describe_annotations(a,top,stream,level+1,depth,
					  default_hide_internals);
		   END_WITH_TABLE();}}}}}
  END_WITH_TABLE();
}

void describe_frame(frame)
     Frame frame;
{
  describe_frame_to_stream(frame,standard_output);
}

Grounding value_column(ground)
     Grounding ground;
{
  ground_in_column(3,ground);
  return NULL;
}

Grounding labelled_value_column(label,ground)
     Grounding label, ground;
{
  labelled_ground_in_column(3,GSTRING(label),ground);
  return NULL;
}

char* spinoffs_string(here)
     Frame here;
{
  generic_stream stream; Frame_Array *spinoffs;
  INITIALIZE_STRING_STREAM(sstream,buffer,10000);
  stream.stream_type = string_output; stream.ptr.string_out = &sstream;
  spinoffs=frame_spinoffs(here);
  {DO_FRAMES(s,spinoffs) describe_frame_to_stream(s,&stream);}
  free(spinoffs->elements);
  return(sstream.head);
}


Grounding describe_frame_primitive(x)
     Frame x;
{
  describe_frame(x);
  return NULL;
}


/* Viewing frames */

void view_frame(Frame frame)
{
  if (NOT(NULLP(get_handler(frame,"view"))))
    {Grounding temp; temp=send0(frame,"view"); FREE_GROUND(temp);}
  else if ((probe_annotation(frame,"+viewer") == NULL) ||
	   (NULLP(get_inherited_ground
		  (probe_annotation(frame,"+viewer")))))
    describe_frame(frame);
  else {Frame viewer; 
	viewer=probe_annotation(frame,"+viewer");
	{Grounding temp=NULL;
	 WITH_HANDLING
	   {DO_LIST(view_method,get_inherited_ground(viewer)) 
	      {temp=apply1(view_method,frame_to_ground(frame));}}
	 ON_EXCEPTION
	   {gsputs("<<error in describing>>",standard_output);
	    CLEAR_EXCEPTION();}
	 END_HANDLING;
	 FREE_GROUND(temp);}}
}

char* description_string(Frame here)
{
  generic_stream stream;
  INITIALIZE_STRING_STREAM(sstream,buffer,1);
  stream.stream_type = string_output; stream.ptr.string_out = &sstream;
  {FLET(generic_stream *,standard_output,&stream)
     {FLET(Frame,read_root,here)
	if (probe_annotation(here,"+viewer") == NULL)
	  describe_frame(here);
	else view_frame(here);
      END_FLET(read_root);}
   END_FLET(standard_output);}
  return(sstream.head);
}

Grounding gdescription_string(gframe)
     Grounding gframe;
{
  Grounding string_to_ground(char *string);
  char *description_string(Frame frame);
  return string_to_ground(description_string(GFRAME(gframe)));
}



/* Generating WITHIN-FRAME descriptions */

#define WITH_SEARCH_BIT(bit) int bit; UNWIND_PROTECT bit=grab_search_bit();
#define END_WITH_SEARCH_BIT(bit) ON_UNWIND release_search_bit(bit); END_UNWIND

void output_description(Frame f,generic_stream *stream);
void output_annotations
  (Frame f,generic_stream *stream,int b,Grounding notables,int indent);
void output_subdescription
  (Frame f,int notable_bit,Grounding notables,int indent,generic_stream *stream);
void fill_in_fields(Frame f,generic_stream *stream,Grounding fields,int indent);
Grounding get_notables(Frame f);
Grounding get_inherited_names(Frame f);
Grounding quotify(Grounding x);

Grounding output_description_primitive(Grounding frame)
{
  output_description(GFRAME(frame),standard_output);
  return NULL;
}

void output_description(Frame f,generic_stream *stream)
{
  Grounding fields, notables;
  ENSURE_CURRENT(f); notables=get_notables(f);
  {DO_RESULTS(r,notables) if (r) ensure_current(GFRAME(r));}
  {FLET(Frame,read_root,NULL)
     {gsputs("(WITHIN-FRAME #",stream); print_frame(stream,f);
      if (frame_home(f))
	{gsputs("\n  ;; Home is #",stream); print_frame(stream,frame_home(f));}
      fields=send0(f,"get-fields"); read_root=f;
      {DO_RESULTS(field,fields)
	 {gsputs("\n  ",stream); pprint_elt(stream,field,2,80,2);}
       fill_in_fields(f,stream,fields,2);
       FREE_GROUND(fields);}
      {WITH_SEARCH_BIT(b)
	 {{DO_RESULTS(notable,notables)
	     if (notable)
	       {DO_HOMES(h,GFRAME(notable)) set_search_bit(h,b);}}
	  output_annotations(f,stream,b,notables,2);
	  {DO_RESULTS(notable,notables)
	     if (notable)
	       {DO_HOMES(h,GFRAME(notable)) clear_search_bit(h,b);}}}
       END_WITH_SEARCH_BIT(b);}
      {Grounding inherited; inherited=get_inherited_names(f);
       if (inherited) gsputs("\n  ;; Inheritable annotations",stream);
       {DO_RESULTS(name,inherited)
	  {gsputs("\n  (WITHIN-FRAME ",stream);
	   print_frame_name(GSTRING(name),stream); gsputc(')',stream);}}
       FREE_GROUND(inherited);}}
   END_FLET(read_root);}
  gsputs(")\n",stream);
  FREE_GROUND(notables);
}

void fill_in_fields(Frame f,generic_stream *stream,Grounding fields,int indent)
{
  boolean fill_in_ground=True, fill_in_prototype=True;
  {DO_RESULTS(field,fields) 
     if ((TYPEP(field,pair_ground)) && (field != empty_list))
       if ((GCAR(field)) == geq_symbol) fill_in_ground=False;
       else if ((GCAR(field)) == peq_symbol) fill_in_prototype=False;}
  if (fill_in_ground)
    {Grounding field=empty_list; USE_GROUND(field);
     {DO_RESULTS(r,frame_ground(f)) field=cons_pair(quotify(r),field);
      field=cons_pair(geq_symbol,field);
      gsputc('\n',stream); {DO_TIMES(i,indent) gsputc(' ',stream);}
      pprint_elt(stream,field,indent,80,indent);
      FREE_GROUND(field);}}
  if (fill_in_prototype)
    {Grounding field; 
     if (frame_prototype(f))
       field=cons_pair(peq_symbol,cons_pair(frame_to_ground(frame_prototype(f)),
					     empty_list));
     else field=cons_pair(peq_symbol,empty_list);
     gsputc('\n',stream); {DO_TIMES(i,indent) gsputc(' ',stream);}
     pprint_elt(stream,field,indent,80,indent);
     FREE_GROUND(field);}
}

void output_annotations
  (Frame f,generic_stream *stream,int b,Grounding notables,int indent)
{
  {DO_ANNOTATIONS(a,f)
     if (search_bit_p(a,b)) output_subdescription(a,b,notables,indent,stream);}
  {DO_ANNOTATIONS(a,f)
     if (NOT(search_bit_p(a,b)))
       output_subdescription(a,b,notables,indent,stream);}
}

void output_subdescription
  (Frame f,int notable_bit,Grounding notables,int indent,generic_stream *stream)
{
  Grounding fields;
  gsputc('\n',stream);
  {DO_TIMES(i,indent) gsputc(' ',stream);} indent=indent+2;
  gsputs("(WITHIN-FRAME ",stream); print_frame_name(frame_name(f),stream);
  if (find_result(frame_to_ground(f),notables))
    {WITH_HANDLING
       {fields=send0(f,"get-fields");}
     ON_EXCEPTION
       {fields=NULL; 
	gsputc('\n',stream); {DO_TIMES(i,indent) gsputc(' ',stream);}
	gsprintf(stream,"#;\"Error getting fields (%s)\"",theException);
	fill_in_fields(f,stream,NULL,indent); CLEAR_EXCEPTION();}
     END_HANDLING
       {DO_RESULTS(field,fields)
	  {gsputc('\n',stream); {DO_TIMES(i,indent) gsputc(' ',stream);}
	   pprint_elt(stream,field,indent,80,indent);}
	FREE_GROUND(fields);}
     output_annotations(f,stream,notable_bit,notables,indent);}
  else {{DO_SOFT_ANNOTATIONS(a,f)
	   if (search_bit_p(a,notable_bit))
	     output_subdescription(a,notable_bit,notables,indent,stream);}}
  gsputc(')',stream);
}

Grounding get_inherited_names(Frame f)
{
  Grounding list_to_result_set(Grounding list);
  Grounding names=empty_list;
  {DO_PROTOTYPES(p,frame_prototype(f))
     {DO_SOFT_ANNOTATIONS(ap,p)
	if (NOT(raw_local_probe_annotation(f,frame_name(ap))))
	  names=cons_pair(string_to_ground(frame_name(ap)),names);}}
  if (names == empty_list) return NULL;
  else {Grounding results; results=list_to_result_set(names); 
	USE_GROUND(results); FREE_GROUND(names);
	return results;}
}

Grounding get_notables(Frame f)
{
  {DO_RESULTS(r,frame_ground(notables_frame))
     if (EQ(GCAR(r),frame_to_ground(f)))
       {USE_GROUND(GCDR(r)); return GCDR(r);}}
  return send0(f,"default-notables");
}

Grounding add_notable(Grounding f,Grounding notable)
{
  {DO_RESULTS(r,frame_ground(notables_frame))
     if (EQ(GCAR(r),f))
       if (find_result(notable,GCDR(r))) return NULL;
       else {Grounding new_entry; 
	     new_entry=cons_pair(f,merge_results(notable,GCDR(r)));
	     add_to_ground(notables_frame,new_entry);
	     remove_from_ground(notables_frame,r);
	     return NULL;}}
  {Grounding defaults; defaults=send0(GFRAME(f),"default-notables");
   add_to_ground(notables_frame,cons_pair(f,merge_results(notable,defaults)));
   FREE_GROUND(defaults);}
  return NULL;
}

Grounding remove_notable(Grounding f,Grounding notable)
{
  Grounding zap_result(Grounding res,Grounding from);
  {DO_RESULTS(r,frame_ground(notables_frame))
     if (EQ(GCAR(r),frame_to_ground(f)))
       if (find_result(notable,GCDR(r)))
	 {Grounding new_entry; 
	  new_entry=cons_pair(f,zap_result(notable,GCDR(r)));
	  add_to_ground(notables_frame,new_entry);
	  remove_from_ground(notables_frame,r);
	  return NULL;}
       else return NULL;}
  {Grounding defaults; defaults=send0(GFRAME(f),"default-notables");
   if (find_result(notable,defaults))
     add_to_ground(notables_frame,cons_pair(f,zap_result(notable,defaults)));
   FREE_GROUND(defaults);}
  return NULL;
}

void expand_to_depth(Frame root,Frame f,int depth)
{
  if (depth == 1) 
    {add_notable(frame_to_ground(root),frame_to_ground(f));}
  else {DO_ANNOTATIONS(a,f) expand_to_depth(root,a,depth-1);
	add_notable(frame_to_ground(root),frame_to_ground(f));}
}

Grounding add_notables(Grounding root,Grounding from,Grounding depth)
{
  expand_to_depth(GFRAME(root),GFRAME(from),GINTEGER(depth));
  return NULL;
}

Grounding remove_all_notables(Frame root,Frame f)
{
  {DO_RESULTS(r,frame_ground(notables_frame))
     if (EQ(GCAR(r),frame_to_ground(root)))
       {Grounding new_notables=NULL, new_entry;
	{DO_RESULTS(n,GCDR(r))
	   if ((NOT((GFRAME(n)) == f)) && (NOT(has_home(GFRAME(n),f)))) 
	     new_notables=merge_results(n,new_notables);
	 new_entry=cons_pair(frame_to_ground(root),new_notables);
	 add_to_ground(notables_frame,new_entry);
	 remove_from_ground(notables_frame,r);
	 return NULL;}}}
  return NULL;
}

void init_description_functions()
{
  geq_symbol=intern("ground="); peq_symbol=intern("prototype=");
  notables_frame=parse_frame_from_string("/system/editor/notables");
  declare_unary_function(describe_frame_primitive,"describe-frame",
			 frame_ground_type);
  declare_unary_function(describe_frame_primitive,"df",frame_ground_type);
  declare_unary_function(output_description_primitive,"output-description",
			 frame_ground_type);
  declare_binary_function(add_notable,"add-notable",
			  frame_ground_type,frame_ground_type);
  declare_binary_function(remove_notable,"remove-notable",
			  frame_ground_type,frame_ground_type);
  declare_binary_function(remove_all_notables,"remove-notables",
			  frame_ground_type,frame_ground_type);
  declare_function(add_notables,"add-notables",3,
		   frame_ground_type,frame_ground_type,
		   integer_ground,any_ground);
  declare_function(gdescription_string,"description-string",
		   1,frame_ground_type,any_ground,any_ground,any_ground);
  declare_binary_function
    (labelled_value_column,"labelled-value-column",string_ground,any_ground);
  declare_unary_function(value_column,"value-column",any_ground);
  declare_unary_function(make_table,"make-table",any_ground);
  declare_binary_function(string_cell,"string-cell",integer_ground,string_ground);
  declare_binary_function(ground_cell,"ground-cell",integer_ground,any_ground);
  declare_binary_function(compound_cell,"compound-cell",integer_ground,any_ground);
}
