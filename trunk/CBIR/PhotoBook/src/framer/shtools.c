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
  This file implements an interactive shell for browsing FRAMER structures.

  Modification history:
    13 June 1992 - Began tracking modification history (Haase)
*************************************************************************/
 
/* This file defines the basic datatyeps for the C implementation of FRAMER */
 
#include "framer.h"
#include "fraxl.h"

extern exception Unexpected_EOF;
exception Not_A_Command="Input is not a command";
extern boolean trap_unknown_frames;

#if __TURBOC__
#include <alloc.h>
#endif

Frame parse_frame_path AP((generic_stream *stream,Frame relative_to));
void view_frame(Frame frame);
Frame tty_shell(Frame frame);
extern Frame do_command(Frame here,char *input,Grounding (*get_arg)());
extern Grounding parse_arg(generic_stream *stream);
extern boolean trap_unknown_frames;
extern long ground_memory_in_use;
extern exception Not_A_Function;
extern Frame read_root;


/* Executing commands */

Grounding parse_arg(generic_stream *stream)
{
 Grounding result;
   {FLET(boolean,trap_unknown_frames,True)
     result=parse_ground(stream);
    END_FLET(trap_unknown_frames);}
 return result;
}

#define POPPER(arg) if (arg != empty_list) arg=GCDR(arg);
Grounding interactive_send
  (Frame target,char *message,char *command_line,Grounding (*get_arg)())
{
  Grounding nd_applier(Grounding nd_rail,Grounding d_rail);
  Frame handler; 
  Grounding procedure, arg_names, arg_ptr, default_generators; 
  boolean parsing_command_line=True;
  handler=get_handler_frame(target,message); procedure=get_inherited_ground(handler);
  if (NOT((TYPEP(procedure,framer_function_ground)) || (TYPEP(procedure,procedure_ground))))
    procedure=fraxl_eval(procedure);
  else USE_GROUND(procedure);
  {Frame aframe; aframe=probe_annotation(handler,"arguments");
   if (aframe) arg_names=get_inherited_ground(aframe); 
   else if (TYPEP(procedure,procedure_ground))
     arg_names=proc_args(procedure);
   else if (TYPEP(procedure,framer_function_ground))
     {Grounding fake_args; fake_args=empty_list;
      {DO_TIMES(i,(the(primitive,procedure)->arity)+1) 
	 {GCONS(t_symbol,fake_args,fake_args);}
       arg_names=fake_args;}}
   else ground_error(Not_A_Function,"can't use to invoke a command",procedure);
   USE_GROUND(arg_names); arg_ptr=arg_names;}
  {Frame dframe; dframe=probe_annotation(handler,"defaults");
   if (dframe) default_generators=get_inherited_ground(dframe); 
   else default_generators=empty_list;}
  {Grounding rail, result; rail=empty_list;
   {GCONS(procedure,rail,rail);} POPPER(arg_ptr); POPPER(default_generators);
   {GCONS(frame_to_ground(target),rail,rail);} POPPER(arg_ptr); 
   POPPER(default_generators);
   while (parsing_command_line && (arg_ptr != empty_list))
     {WITH_INPUT_FROM_STRING(stream,command_line)
	{WITH_HANDLING
	   {Grounding arg; arg=parse_arg(stream); GCONS(arg,rail,rail);
	    POPPER(arg_ptr); POPPER(default_generators);}
	 ON_EXCEPTION
	   if (theException == Unexpected_EOF) 
	     {theException=NULL; parsing_command_line=False;}
	   else reraise();
	 END_HANDLING;}}
   while (arg_ptr != empty_list)
     {Grounding dflt, arg; 
      if (default_generators == empty_list) dflt=NULL;
      else {dflt=apply1(GCAR(default_generators),frame_to_ground(target));
	    USE_GROUND(dflt);}
      arg=get_arg(frame_name(handler),SYMBOL_NAME(GCAR(arg_ptr)),dflt);
      GCONS(arg,rail,rail); FREE_GROUND(dflt);
      POPPER(arg_ptr); POPPER(default_generators);}
   result=nd_applier(rail,empty_list);
   FREE_GROUND(procedure); FREE_GROUND(rail); FREE_GROUND(arg_names);
   return result;}
}

Frame do_command(Frame here,char *command_line,Grounding (*get_arg)())
{
  char *command, *args;
  command=args=command_line; 
  /* Skip over the command */
  while ((*args != ' ') && (*args != '\n') && (*args != '\0'))
    if ((*args == '\\') && (*(args+1) != '\0')) args=args+2;
    else args++;
  /* Terminate the command (with '\0') separating it from its arguments
     and skip any whitespace leading up to its args. */
  if ((*args) != '\0') {*(args++)='\0'; while (*args == ' ')  args++;}
  if (get_handler(here,command))
    {Grounding cresult; cresult=interactive_send(here,command,args,get_arg);
     if (FRAMEP(cresult)) return GFRAME(cresult);
     else {FREE_GROUND(cresult); return NULL;}}
  else {Frame parse_root; if (*command == '#') command++;
        if (*command == '/') {parse_root=root_frame;command++;}
        else if ((*command == '^')  && (command[1] == '/'))
         {parse_root=read_root;command=command+2;}
        else if ((*command == '^')  && (command[1] == '^') 
                 && (command[2] == '/'))
         {parse_root=frame_home(read_root);command=command+3;}
        else parse_root=here;
        {WITH_INPUT_FROM_STRING(gstream,command)
	  {Frame relative_to;
           relative_to=parse_frame_path(gstream,parse_root);
           if (*args == '\0') return relative_to;
	   else do_command(relative_to,args,get_arg);
           if (has_home(relative_to,here)) return here;
	   else return NULL;}}}
}

Grounding never_get_arg(char *command,char *parameter,Grounding dflt)
{
  raise_crisis_with_details("Command needs parameter",parameter);
  return NULL;
}

Grounding do_command_primitive(Grounding frame,Grounding command_string)
{
  return frame_to_ground(do_command(GFRAME(frame),GSTRING(command_string),
				    never_get_arg));
}

void declare_function_as_command(command_name,function)
     char *command_name, *function;
{
  Frame table;
  table=use_annotation(use_annotation(use_annotation(root_frame,"system"),
				      "defaults"),
		       "commands");
  set_ground(use_annotation(table,command_name),find_function(function));
}

void declare_command(command_name,func,string,arity,type0,type1,type2,type3)
     char *command_name;
     Grounding (*func)();
     char *string;
     int arity;
     Ground_Type type0, type1, type2, type3;
{
  declare_function(func,string,arity,type0,type1,type2,type3);
  declare_function_as_command(command_name,string);
}


/* Useful commands */

Grounding push_shell(here)
     Grounding here;
{
  tty_shell(GFRAME(here));
  return here;
}

Grounding describe_commands_for_frame(Grounding frame)
{
  Frame handlers; handlers=get_handlers(GFRAME(frame));
  /* Fill in all the commands */
  {DO_PROTOTYPES(p,handlers)
     {DO_ANNOTATIONS(a,p)
	{Grounding g; g=frame_ground(a);
	 if (g && ((TYPEP(g,pair_ground)) || (TYPEP(g,procedure_ground)) || (TYPEP(g,framer_function_ground))))
	  probe_annotation(handlers,frame_name(a));}}}
  {DO_ANNOTATIONS(a,handlers)
     {Grounding g; g=get_inherited_ground(a);
      if (g && ((TYPEP(g,pair_ground)) || (TYPEP(g,procedure_ground)) || (TYPEP(g,framer_function_ground))))
	if (probe_annotation(a,"documentation"))
	  printf("%s      %s\n",frame_name(a),GSTRING(get_inherited_ground(probe_annotation(a,"documentation"))));
	else printf("%s\n",frame_name(a));}}
  return NULL;
}


/* The TTY shell */


/* The tty shell */

Grounding tty_get_arg(char *name,char *prompt,Grounding dflt)
{
  Grounding result;
  if (dflt != NULL) 
    {printf("%s/%s...> ",name,prompt); fprint_ground(stdout,dflt); printf(" < ");}
  else printf("%s/%s...< ",name,prompt); fflush(stdout);
  {FLET(boolean,trap_unknown_frames,True)
     result=fparse_ground(stdin);
   END_FLET(trap_unknown_frames);}
  return result;
}

#define description_herald \
   "%%%%================================================================\n"
void setup_abort_handling(void);

Frame tty_shell(Frame here)
{
  char buffer[1024], *input;
  Frame command_result; boolean verbose=True, redisplay=True, exiting=False;
  input = buffer;
  {FLET(Frame,read_root,here)
     while (NOT(exiting))
       {if ((redisplay) && (verbose)) 
	  {WITH_HANDLING
	     {setup_abort_handling(); printf(description_herald); view_frame(here);}
	   ON_EXCEPTION
	     {fprintf(stderr,"Error while describing frame %s: %s",
		      theException,exception_details);
	      printf("%%%% In ");
	      {FLET(Frame,read_root,NULL) fprint_frame(stdout,here); END_FLET(read_root);}
	      printf("\n");
	      setup_abort_handling(); /* Reset, just in case */
	      CLEAR_EXCEPTION();}
	   END_HANDLING;}
       else {printf("%%%% In ");
	     {FLET(Frame,read_root,NULL) fprint_frame(stdout,here); END_FLET(read_root);}
	     printf("\n");}
	  printf("%%%% (%lu used) Command: ",ground_memory_in_use);
	  buffer[0]='\0';
	  fflush(stdout);
	  /* Read a line */
	  input=gets(buffer);
	  /* Ignore blank lines */
	  while (strlen(input)==0) input=gets(buffer);
	  /* Skip whitespace */
	  while (*input == ' ') input++;
	  /* These are all commands that change shell state */
	  if (strncmp(input,"pop",3)==0) exiting=True;
	  else if (strncmp(input,"exit",4)==0) exiting=True;
	  else if (strncmp(input,"verbose",7)==0) verbose=True;
	  else if (strncmp(input,"taciturn",8)==0) verbose=False;
	  else if (strncmp(input,"quit",4)==0) exiting=True;
	  else if (strncmp(input,"abort",5)==0) exit(0);
	  else if (*input == '(') 
	    /* Input is an expression */
	    {WITH_HANDLING
	       {FLET(boolean,trap_unknown_frames,True)
		  {Grounding form, result, print_result;
		   /* Read form */
		   form   = parse_ground_from_string(input); USE_GROUND(form);
		   /* Eval form */
		   result = fraxl_eval(form); USE_GROUND(result);
		   /* Print result(s) */
		   print_result=apply1(find_function("PRINT-RESULT"),result);
		   FREE_GROUND(print_result);
		   /* GC form and result */
		   FREE_GROUND(form); FREE_GROUND(result);}
		END_FLET(trap_unknown_frames);}
	     ON_EXCEPTION
	       {fprintf(stdout,"Hey! %s: %s\n",theException,exception_details);
		CLEAR_EXCEPTION();}
	     END_HANDLING;}
	  else {WITH_HANDLING
		  {FLET(boolean,trap_unknown_frames,True)
		     {Frame result; result=do_command(here,input,tty_get_arg);
		      if (result) {redisplay=True; read_root=here=result;}
		      else redisplay=False;}
		     END_FLET(trap_unknown_frames);}
		  ON_EXCEPTION
		    {fprintf(stdout,"Hey! %s: %s\n",theException,exception_details);
		     CLEAR_EXCEPTION();}
		END_HANDLING;}}
  END_FLET(read_root);}
  if (exiting) return here;
  else return tty_shell(here);
}
 

/* Initializing shell commands */

void define_command_in_scheme(command,defn_string)
     char *command, *defn_string;
{
  Frame table;
  table=use_annotation(use_annotation(use_annotation(root_frame,"system"),
				      "defaults"),
		       "commands");
  set_ground(use_annotation(table,command),
	     parse_ground_from_string(defn_string));
}

Grounding setup_default_shell_commands()
{
  Grounding parse_ground_from_string(char *string);
  declare_function_as_command("up","FRAME-HOME");
  declare_function_as_command("prototype","FRAME-PROTOTYPE");
  declare_function_as_command("use","USE-FRAME");
  declare_function_as_command("find","FIND-FRAME");
  declare_function_as_command("set-ground","SET-FRAME-GROUND");
  declare_function_as_command("set-prototype","SET-FRAME-PROTOTYPE");
  declare_function_as_command("delete","DELETE-ANNOTATION");
  declare_function_as_command("describe","DESCRIBE-FRAME");
  declare_function_as_command("help","HELP-COMMANDS");
  declare_function_as_command("push","PUSH-SHELL");
  declare_function_as_command("spinoffs","FRAME-SPINOFFS");
  declare_function_as_command("help","HELP-COMMANDS");
  declare_function_as_command("push","PUSH-SHELL");
  define_command_in_scheme
    ("eval","(:DEFINE ({} :UNIT :EXPRESSION)\
                ((:DEFINE ({} :V)\
                     (:PRINT-STRING \"Value: \") (:PRINT-GROUND :V) (:NEWLINE)) \
                 (:EVAL :EXPRESSION)))");
  define_command_in_scheme
    ("backup","(:DEFINE ({} :IGNORE) (:BACKUP-ROOT-FRAME \"radix.framer\"))");
  define_command_in_scheme
    ("load","(:DEFINE ({} :IGNORE :FILE) (:LOAD-FRAME-FROM-FILE :FILE))");
  return NULL;
}

void init_shell_functions()
{
  declare_function(describe_commands_for_frame,"describe-commands",
		   1,frame_ground_type,any_ground,any_ground,any_ground);
  declare_function(push_shell,"push-shell",1,frame_ground_type,
		   any_ground,any_ground,any_ground);
  declare_function(do_command_primitive,"do-command",
		   2,frame_ground_type,string_ground,any_ground,any_ground);
  declare_function(frame_fcn(frame_spinoffs),"spinoffs-command",
		   1,frame_ground_type,any_ground,any_ground,any_ground);
  declare_function(push_shell,"push-shell",1,frame_ground_type,
		   any_ground,any_ground,any_ground);
  declare_function(setup_default_shell_commands,
		   "setup-default-shell-commands",0,0,0,0,0);
}

