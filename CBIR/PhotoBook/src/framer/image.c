/* $Header: /mas/framer/sources/RCS/image.c,v 1.6 1994/01/14 00:58:59 rnewman Exp $ */

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

char *search_for_name(char *name,char *suffix,char **path);

typedef int four_bytes;

int zip_code(Frame frame);
void new_zip_code(Frame frame);
boolean dump_p(Frame f);
void free_zip_codes(Frame_Array *array);
Grounding make_bignum(char *digits);

static char *image_name;
boolean active_image_p=False;


/* Writing and reading summaries (which are grounds) */

extern Grounding file_tag, read_only_tag, alias_tag; 

Grounding generate_frame_summary(Frame frame)
{
  Grounding result=empty_list; 
  if (NOT(frame_current_p(frame)))
    if (frame_type(frame) == alias_frame)
      return cons_pair(alias_tag,cons_pair(frame_to_ground(frame->prototype),result));
    else if (backup_root_p(frame))
      return cons_pair(file_tag,cons_pair(string_to_ground(frame_filename(frame)),
					  result));
  {DO_ANNOTATIONS(a,frame)
     if (dump_p(a))
       result=cons_pair(string_to_ground(frame_name(a)),result);}
  result=cons_pair(frame->ground,result);
  if (default_prototype_p(frame,frame->prototype))
    result=cons_pair(cons_pair(frame_to_ground(frame->prototype),empty_list),result);
  else result=cons_pair(frame_to_ground(frame->prototype),result);
  if (frame_read_only_p(frame))
    return cons_pair(read_only_tag,result);
  else return result;
}

four_bytes write_summary_to_image(Frame frame,FILE *image_file)
{
  Grounding summary; four_bytes position; Frame_Array *ozc;
  summary=generate_frame_summary(frame); position=ftell(image_file);
  {boolean old_ss; Frame old_rr; UNWIND_PROTECT
     {old_ss=strict_syntax; strict_syntax=True; 
      old_rr=read_root; read_root=frame_home(frame);
      ozc=zip_codes; zip_codes=make_frame_array(INITIAL_ZIP_CODES);
      fprint_ground(image_file,summary);}
     ON_UNWIND
       {strict_syntax=old_ss; read_root=old_rr; 
	free_zip_codes(zip_codes); FREE_GROUND(summary);}
   END_UNWIND;}
  return position;
}

void interpret_summary(Frame frame,Grounding summary)
{
  Grounding prototype, ground; boolean old_sa, early_exit=False;
  UNWIND_PROTECT
    {old_sa=suppress_autoload; suppress_autoload=True;
     if ((TYPEP(GCAR(summary),symbol_ground)))
       {Grounding type_symbol; type_symbol=GCAR(summary); summary=GCDR(summary);
	if (type_symbol == alias_tag) 
	  {set_frame_type(frame,alias_frame); 
	   frame->prototype=GFRAME(GCAR(summary));
	   early_exit=True;}
	else if (type_symbol == read_only_tag) set_frame_type(frame,read_only_frame);
	else if (type_symbol == file_tag)
	  {load_frame_from_file(GSTRING(GCAR(summary))); early_exit=True;}}
     if (NOT(early_exit))
       {set_frame_current_p(frame);
	prototype=GCAR(summary); summary=GCDR(summary);
	ground=GCAR(summary); summary=GCDR(summary);
	if (NULLP(prototype)) frame->prototype=NULL;
	if (prototype == NULL) frame->prototype=NULL;
	else if (prototype == empty_list)
	  frame->prototype=
	    use_annotation(frame_prototype(frame_home(frame)),frame_name(frame));
	else if (TYPEP(prototype,pair_ground))
	  frame->prototype=GFRAME(GCAR(prototype));
	else {frame->prototype=GFRAME(prototype);
	      add_spinoff(frame,GFRAME(prototype));}
	FREE_GROUND(frame->ground); frame->ground=ground; USE_GROUND(ground);
	{DO_LIST(a_name,summary) use_annotation(frame,GSTRING(a_name));}}}
  ON_UNWIND
    suppress_autoload=old_sa;
  END_UNWIND;
}


/* Resolves all the dangling references */

long resolve_dangling_references(Frame beneath)
{
  long count=1;
  if (frame_current_p(beneath))
    {DO_ANNOTATIONS(a,beneath) count=count+resolve_dangling_references(a);}
  else if ((frame_type(beneath) == alias_frame) || (backup_root_p(beneath))) {}
  else {DO_ANNOTATIONS(a,beneath) 
	  count=count+resolve_dangling_references(a);}
  return count;
}

char *bignum_to_string(Grounding big);


/* Binary I/0 */

void write_four_bytes(FILE *stream,four_bytes code)
{
  code=normalize_word(code);
  fwrite(&code,sizeof(four_bytes),1,stream);
}

void write_array(four_bytes *array,FILE *stream,int length)
{
#if strange_byte_order
  {DO_TIMES(i,length) write_four_bytes(stream,array[i]);}
#else
  fwrite(array,sizeof(four_bytes),length,stream);
#endif
}

four_bytes read_four_bytes(FILE *stream)
{
  four_bytes temp;
  fread(&temp,sizeof(four_bytes),1,stream);
  return normalize_word(temp);
}

void read_array(four_bytes *array,FILE *stream,int length)
{
  fread(array,sizeof(four_bytes),length,stream);
#if strange_byte_order
  {DO_TIMES(i,length) array[i]=normalize_word(array[i]);}
#endif
}


/* Name tables */

static char **image_name_table;
static int image_name_table_size;
struct CHAR_BUCKET image_name_lookup = {'\0',{NULL,NULL}}; 

void construct_image_name_lookup()
{
  char **ptr, **end; int size; size=1;
  ptr=frame_name_table; end=ptr+number_of_frame_names;
  while (ptr < end) 
    {struct CHAR_BUCKET *b; b=find_bucket(*ptr++,&image_name_lookup);
     b->data.code=size++;}
}

void update_image_name_lookup()
{
  char **ptr, **end; int size; size=1;
  image_name_table=
    fra_reallocate(image_name_table,sizeof(char *)*number_of_frame_names);
  ptr=frame_name_table; end=ptr+number_of_frame_names;
  while (ptr < end) 
    {struct CHAR_BUCKET *b; b=find_bucket(*ptr,&image_name_lookup);
     if (b->data.code == 0)
       {image_name_table[image_name_table_size++]=*ptr++;
	b->data.code=image_name_table_size;}
     else ptr++;}
}

void write_name_table(char *filename)
{
  char **ptr, **end; FILE *out; out=fopen(filename,"wb"); 
  write_four_bytes(out,number_of_frame_names);
  ptr=frame_name_table; end=ptr+number_of_frame_names;
  while (ptr < end) 
    {int len; len=strlen(*ptr)+1; write_four_bytes(out,len);
     fwrite(*ptr++,sizeof(char),len,out);}
  fclose(out);
}

void write_image_name_table(char *filename)
{
  char **ptr, **end; FILE *out; out=fopen(filename,"wb"); 
  write_four_bytes(out,number_of_frame_names);
  ptr=image_name_table; end=ptr+image_name_table_size;
  while (ptr < end) 
    {int len; len=strlen(*ptr)+1; write_four_bytes(out,len);
     fwrite(*ptr++,sizeof(char),len,out);}
  fclose(out);
}

void read_name_table(char *filename)
{
  char buf[1000], **ptr; FILE *in;
  /* TPM: filename may be NULL! */
  if(filename == NULL) filename = "";
  in=fopen(filename,"rb"); 
  if (NULLP(in)) raise_crisis_with_details(File_Unreadable,filename);
  image_name_table_size=read_four_bytes(in);
  ptr=image_name_table=fra_allocate(image_name_table_size,sizeof(char *));
  {DO_TIMES(i,image_name_table_size)
     {int size; struct CHAR_BUCKET *b;
      size=read_four_bytes(in); fread(buf,sizeof(char),size,in);
      *ptr++=intern_frame_name(buf); b=find_bucket(buf,&image_name_lookup);
      b->data.code=i+1;}}
  fclose(in);
}

four_bytes name_to_code(char *name)
{
  struct CHAR_BUCKET *result; result=find_bucket(name,&image_name_lookup);
  return result->data.code-1; 
}

/* Writing for an image */

void fasdump_frame_path(FILE *stream,Frame frame,Frame root)
{
  if (frame == root) return;
  else {fasdump_frame_path(stream,frame_home(frame),root); new_zip_code(frame);
	write_four_bytes(stream,name_to_code(frame_name(frame)));}
}

void fasdump_frame_reference(FILE *stream,Frame frame)
{
  int code, depth=0; Frame root; 
  while ((frame_type(frame) == alias_frame)) frame=frame->prototype;
  putc(frame_ground_type,stream);
  root=frame; while ((root) && ((code=zip_code(root)) < 0))
    {root=frame_home(root); depth++;}
  if (code < 0) {root=root_frame; depth--;}
  write_four_bytes(stream,depth);
  if (code > 0) write_four_bytes(stream,code); else write_four_bytes(stream,0);
  fasdump_frame_path(stream,frame,root);
}

void fasdump_ground(FILE *stream,Grounding ground)
{
  if (ground == NULL) putc('\0',stream);
  else if (FRAMEP(ground)) fasdump_frame_reference(stream,GFRAME(ground));
  else switch (ground_type(ground))
    {
    case string_ground:
      {int size; size=strlen(GSTRING(ground)); 
       putc(string_ground,stream); write_four_bytes(stream,size); 
       fwrite(unground(ground,string),sizeof(char),size,stream);
       break;}
    case integer_ground:
      putc(integer_ground,stream); 
      write_four_bytes(stream,unground(ground,integer)); break;
    case float_ground:
      putc(float_ground,stream);
      fwrite(&(ground->ground.contents.flonum),sizeof(float),1,stream); break;
    case vector_ground: case nd_ground:
      putc(ground_type(ground),stream); write_four_bytes(stream,GVSIZE(ground));
      {DO_VECTOR(elt,ground) fasdump_ground(stream,elt);}
      break;
    case pair_ground: case rational_ground:
      putc(ground_type(ground),stream); 
      fasdump_ground(stream,GCAR(ground));
      fasdump_ground(stream,GCDR(ground));
      break;
    case symbol_ground:
      {int size; size=strlen(SYMBOL_NAME(ground));
       putc(symbol_ground,stream); write_four_bytes(stream,size);
       fwrite(SYMBOL_NAME(ground),sizeof(char),size,stream);
       break;}
    case bignum_ground:
      {char *string; int size; string=bignum_to_string(ground);
       size=strlen(string); 
       putc(bignum_ground,stream); write_four_bytes(stream,size); 
       fwrite(GSTRING(ground),sizeof(char),size,stream);
       free(string); break;}
    case comment_ground:
      putc(comment_ground,stream); 
      fasdump_ground(stream,the(comment,ground)); break;
    case procedure_ground:
      putc(procedure_ground,stream);
      fasdump_ground(stream,proc_env(ground));
      fasdump_ground(stream,proc_text(ground));
      break;
    default:
      {Grounding canonical_form; putc(ground_type(ground),stream);
       canonical_form=canonicalize_native_ground(ground);
       USE_GROUND(canonical_form);
       fasdump_ground(stream,canonical_form);
       FREE_GROUND(canonical_form);
       break;}
    }
}

Grounding fasload_ground(FILE *stream)
{
  Ground_Type type; type = (Ground_Type) getc(stream);
  switch (type)
    {
    case any_ground: return NULL;
    case frame_ground_type:
      {Frame root; four_bytes size, code; 
       size=read_four_bytes(stream); code=read_four_bytes(stream);
       if (code) root=zip_codes->elements[code]; else root=root_frame; 
       {DO_TIMES(i,size) 
	  {_interned=image_name_table[read_four_bytes(stream)];
	   root=make_annotation(root,_interned);
	   new_zip_code(root);}}
       return frame_to_ground(root);}
    case string_ground:
      {four_bytes size; char *string; Grounding gnd; 
       size=read_four_bytes(stream); ALLOCATE(string,char,size+1);
       fread(string,sizeof(char),size,stream); string[size]='\0';
       INITIALIZE_NEW_GROUND(gnd,string_ground); unground(gnd,string)=string;
       return gnd;}
    case integer_ground:
      return integer_to_ground(read_four_bytes(stream));
    case float_ground:
      {Grounding ground; INITIALIZE_NEW_GROUND(ground,float_ground);
       fread(&(ground->ground.contents.flonum),sizeof(float),1,stream); 
       return ground;}
    case vector_ground: case nd_ground:
      {Grounding ground; four_bytes size; size=read_four_bytes(stream);
       if (type == vector_ground) {GVMAKE(ground,size);} else {NDMAKE(ground,size);}
       {DO_TIMES(i,size) {GVSET(ground,i,fasload_ground(stream));}}
       return ground;}
    case pair_ground: case rational_ground:
      {Grounding car, cdr, ground; 
       car=fasload_ground(stream); cdr=fasload_ground(stream);
       if ((NULLP(car)) && (NULLP(cdr))) return empty_list;
       INITIALIZE_NEW_GROUND(ground,type);
       GCAR(ground)=car; USE_GROUND(car); GCDR(ground)=cdr; USE_GROUND(cdr);
       return ground;}
    case symbol_ground:
      {int size; char *string; Grounding gnd; 
       size=read_four_bytes(stream); ALLOCATE(string,char,size+1);
       fread(string,sizeof(char),size,stream); string[size]='\0';
       gnd=intern(string); free(string);
       return gnd;}
    case bignum_ground:
      {int size; char *string; Grounding gnd; 
       size=read_four_bytes(stream); ALLOCATE(string,char,size+1);
       fread(string,sizeof(char),size,stream); string[size]='\0';
       gnd=make_bignum(string); free(string);
       return gnd;}
    case comment_ground:
      return make_comment(fasload_ground(stream));
    case procedure_ground:
      {Grounding proc, env, text; 
       env=fasload_ground(stream); text=fasload_ground(stream);
       proc=close_procedure(text,env);
       return proc;}
    default:
      {Grounding canonical_form; canonical_form=fasload_ground(stream);
       return interpret_native_ground(type,canonical_form);}
    }
}


four_bytes write_frame_summary_to_image(Frame frame,FILE *stream)
{
  {four_bytes file_pos; int n_annotations=0; Frame_Array *ozc; unsigned char ft; 
   UNWIND_PROTECT
     {ft=frame->type; file_pos=ftell(stream); 
      ozc=zip_codes; zip_codes=make_frame_array(INITIAL_ZIP_CODES); 
      new_zip_code(root_frame);
      if ((frame_current_p(frame)) && 
	  (default_prototype_p(frame,frame->prototype)))
	ft=ft-128;
      putc(ft,stream);
      fasdump_ground(stream,frame_to_ground(frame->prototype));
      fasdump_ground(stream,frame->ground);
      {DO_SOFT_ANNOTATIONS(a,frame) if (dump_p(a)) n_annotations++;}
      write_four_bytes(stream,n_annotations);
      {DO_SOFT_ANNOTATIONS(a,frame) 
	 if (dump_p(a))
	   write_four_bytes(stream,name_to_code(frame_name(a)));}}
   ON_UNWIND
     {free_zip_codes(zip_codes); zip_codes=ozc;}
   END_UNWIND
     return file_pos;}
}

void read_frame_summary_from_image(Frame frame,FILE *stream)
{
  unsigned char ft; boolean old_sa, dflt=True; Frame_Array *ozc;
  {UNWIND_PROTECT
     {old_sa=suppress_autoload; suppress_autoload=True;
      ozc=zip_codes; zip_codes=make_frame_array(INITIAL_ZIP_CODES); 
      new_zip_code(root_frame); ft=getc(stream);
      if (ft < 128) {ft=ft+128; dflt=False;} frame->type=ft;
      {Grounding proto; proto=fasload_ground(stream);
       if (proto) frame->prototype=GFRAME(proto); else frame->prototype=NULL;
       if ((proto) && (NOT(dflt))) add_spinoff(frame,GFRAME(proto));}
      {Grounding ground; ground=fasload_ground(stream); 
       frame->ground=ground; USE_GROUND(ground);}
      {four_bytes size; size=read_four_bytes(stream);
       {DO_TIMES(i,size)
	  {char *name; _interned=name=image_name_table[read_four_bytes(stream)];
	   use_annotation(frame,name);}}}}
  ON_UNWIND
    {suppress_autoload=old_sa; free_zip_codes(zip_codes); zip_codes=ozc;}
  END_UNWIND;}
}


/* Writing image and index files */

static FILE *image_file, *index_file; 
void load_modifications(char *filename);
boolean debugging_images=False;

struct FRAME_FILE_BLOCK {Frame frame; four_bytes summary, start, end;} root_block;

/* For qsorting */
int compare_frame_names(struct ANNOTATION *f1,struct ANNOTATION *f2)
{
  int k1, k2; k1=name_to_code(f1->key); k2=name_to_code(f2->key);
  if (k1 == k2) return 0; else if (k1 < k2) return -1; else return 1;
}

void write_frame_to_image(Frame frame,FILE *image_file,FILE *index_file,
			  struct FRAME_FILE_BLOCK *block)
{
  int i=0, size;
  struct FRAME_FILE_BLOCK *blocks;
  block->frame=frame; block->summary=write_frame_summary_to_image(frame,image_file);
  if (frame->appendix) size=frame->appendix->size; else size=0;
  /* Sort the annotations into alphabetical order before dumping them. 
     (We didn't have to do this before...) */
  if (size)
    qsort(frame->appendix->annotations,size,sizeof(struct ANNOTATION),
	  compare_frame_names);
  ALLOCATE(blocks,struct FRAME_FILE_BLOCK,size);
  {DO_ANNOTATIONS(a,frame)
     if (frame_ephemeral_p(a)) {}
     else if (NOT(frame_current_p(a)))
       {blocks[i].frame=a; blocks[i].summary=write_frame_summary_to_image(a,image_file);
	blocks[i].start=blocks[i].end=ftell(index_file)/(sizeof(four_bytes)*4); 
	i++;}
     else write_frame_to_image(a,image_file,index_file,&blocks[i++]);}
  block->start=ftell(index_file)/(sizeof(four_bytes)*4);
  {DO_TIMES(j,i)
     {four_bytes block[4]; 
      block[0]=name_to_code(frame_name(blocks[j].frame)); 
      block[1]=blocks[j].summary; block[2]=blocks[j].start; block[3]=blocks[j].end;
      write_array(block,index_file,4);}}
  block->end=ftell(index_file)/(sizeof(four_bytes)*4);
  free(blocks);
}

void write_framer_image(char *name)
{
  char *imgfile, *idxfile, *namfile; FILE *img, *idx; long frame_count;
  struct FRAME_FILE_BLOCK block; four_bytes final_block[4]; int name_count;
  imgfile=search_for_name(name,"img",backup_search_path); img=fopen(imgfile,"wb"); 
  idxfile=search_for_name(name,"idx",backup_search_path); idx=fopen(idxfile,"wb");
  fprintf(stderr,";; Resolving dangling references....\n");
  frame_count=resolve_dangling_references(root_frame); name_count=number_of_frame_names;
  fprintf(stderr,";; Dumping %ld frames....\n",frame_count);
  construct_image_name_lookup();
  write_frame_to_image(root_frame,img,idx,&block);
  final_block[1]=block.summary; final_block[2]=block.start; final_block[3]=block.end;
  write_array(final_block,idx,4);
  fclose(img); fclose(idx); free(imgfile); free(idxfile); 
  if (number_of_frame_names == name_count)
    {namfile=search_for_name(name,"nam",backup_search_path);
     fprintf(stderr,";; Writing name table (%d names in all)....\n",
	     number_of_frame_names);
     write_name_table(namfile); free(namfile);}
  else raise_crisis("Image corrupted (new names)");
}


/* Restoring from indexed image files */

void open_framer_image_file(char *name)
{
  {char *namfile; namfile=search_for_name(name,"nam",backup_search_path);
   read_name_table(namfile); free(namfile);}
  {char *imgfile; imgfile=search_for_name(name,"img",backup_search_path);
   image_file=fopen(imgfile,"rb"); free(imgfile);}
  {char *idxfile; four_bytes buf[4]; 
   idxfile=search_for_name(name,"idx",backup_search_path);
   index_file=fopen(idxfile,"rb");
   fseek(index_file,0,SEEK_END); 
   fseek(index_file,ftell(index_file)-sizeof(four_bytes)*4,SEEK_SET);
   read_array(buf,index_file,4);
   root_block.summary=buf[1]; root_block.start=buf[2]; root_block.end=buf[3];
   root_block.frame=root_frame;}
  {char *modfile; modfile=search_for_name(name,"mod",backup_search_path);
   active_image_p=True; load_modifications(modfile);}
  /* Set up global name */
  ALLOCATE(image_name,char,strlen(name)+1); strcpy(image_name,name);
}

four_bytes binary_search
  (four_bytes key,FILE *index,four_bytes from,four_bytes to,four_bytes block[4])
{
  four_bytes midpoint; midpoint=from+((to-from)/2); 
  while (midpoint > from)
    {fseek(index,midpoint*sizeof(four_bytes)*4,SEEK_SET); 
     read_array(block,index,4);
     if (key == block[0]) return midpoint;
     else if (key < block[0]) to=midpoint; else from=midpoint;
     midpoint=from+((to-from)/2);}
  fseek(index,midpoint*sizeof(four_bytes)*4,SEEK_SET); read_array(block,index,4);
  if (key == block[0]) return midpoint;
  else return -1;
}


/* Searching the file */

boolean find_indexed_frame(Frame frame,struct FRAME_FILE_BLOCK *block)
{
  if (frame == root_frame)
    {block->summary=root_block.summary; block->frame=root_block.frame;
     block->start=root_block.start; block->end=root_block.end;
     return True;}
  else {struct FRAME_FILE_BLOCK home_block; four_bytes entry[4]; four_bytes name_code;
	if (NOT(find_indexed_frame(frame_home(frame),&home_block))) return False;
	if (home_block.start == home_block.end) return False;
	name_code=name_to_code(frame_name(frame));
	binary_search(name_code,index_file,home_block.start,home_block.end,entry);
	block->summary=entry[1]; block->start=entry[2]; block->end=entry[3];
	if (entry[0] == name_code) return True;
	else return False;}
}

four_bytes get_image_index(Frame frame)
{
  struct FRAME_FILE_BLOCK block; 
  if (find_indexed_frame(frame,&block)) return block.summary;
  else return -1;
}

Grounding get_summary_from_image(Frame frame)
{
  Grounding result=NULL; four_bytes where; 
  Frame_Array *old_zc; boolean old_sa, old_tuf; Frame old_rr;
  where=get_image_index(frame);
  if (where < 0) return NULL;
  fseek(image_file,where,SEEK_SET);
  {UNWIND_PROTECT
     {old_rr=read_root; read_root=frame_home(frame);
      old_sa=suppress_autoload; suppress_autoload=True;
      old_tuf=trap_unknown_frames; trap_unknown_frames=False;
      old_zc=zip_codes; zip_codes=make_frame_array(INITIAL_ZIP_CODES);
      result=fparse_ground(image_file);}
   ON_UNWIND
     {read_root=old_rr; suppress_autoload=old_sa; trap_unknown_frames=old_tuf; 
      free_zip_codes(zip_codes); zip_codes=old_zc;}
   END_UNWIND}
  return result;
}

Grounding fraxl_get_image_index(Grounding gframe)
{
  four_bytes result; result=get_image_index(GFRAME(gframe));
  if (result == -1) return NULL;
  else return integer_to_ground((int) result);
}


/* Restoring from images */

boolean restore_from_image(Frame frame)
{
  four_bytes position; position=get_image_index(frame);
  if (position > 0)
    {fseek(image_file,position,SEEK_SET);
     read_frame_summary_from_image(frame,image_file);
     return frame_current_p(frame);}
  else return False;
}


static int modifications_count, frame_total_count;

void load_modifications(char *modfile)
{
  FILE *mods; Frame_Array *ozc; mods=fopen(modfile,"rb");
  if (mods)
    {ozc=zip_codes; zip_codes=make_frame_array(INITIAL_ZIP_CODES);
     UNWIND_PROTECT
       while (mods) 
	 {Grounding frame; int probe; probe=fgetc(mods); 
	  if (probe == EOF) {fclose(mods); mods=NULL;}
	  else {ungetc(probe,mods); frame=fasload_ground(mods);
		read_frame_summary_from_image(GFRAME(frame),mods);}}
     ON_UNWIND
       {free_zip_codes(zip_codes); zip_codes=ozc;}
     END_UNWIND}
}

void write_modifications_to_stream(Frame frame,FILE *stream)
{
  if ((frame_ephemeral_p(frame)) || (remote_frame_p(frame))) return;
  else if (frame_modified_p(frame))
    {fasdump_frame_reference(stream,frame);
     write_frame_summary_to_image(frame,stream);
     modifications_count++;}
  frame_total_count++;
  {DO_SOFT_ANNOTATIONS(a,frame) write_modifications_to_stream(a,stream);}
}

void write_image_modifications()
{
  {int old; old=image_name_table_size; update_image_name_lookup();
   if (old != image_name_table_size) 
     fprintf(stderr,"%d new names (%d in all)\n",
	     image_name_table_size-old,image_name_table_size);}
  {char *modfile; FILE *modstream; 
   modfile=search_for_name(image_name,"mod",backup_search_path);
   modstream=fopen(modfile,"wb"); 
   modifications_count=0; frame_total_count=0;
   write_modifications_to_stream(root_frame,modstream); 
   fprintf(stderr,"%ld changed frames (out of %ld in all)\n",
	   modifications_count,frame_total_count);
   fclose(modstream); free(modfile);}
  {char *namfile; namfile=search_for_name(image_name,"nam",backup_search_path);
   write_image_name_table(namfile); free(namfile);}
}

/*
  Local variables: ***
  compile-command: "../etc/compile-framer" ***
  End: **
*/

