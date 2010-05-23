/* C Mode */

/* A general-purpose exception-handling system for C
   by Jonathan Amsterdam, 1991 
   modified by Ken Haase for inclusion in FRAMER, 1991, 1992
*/

#include <stdio.h>
#include <setjmp.h>
#include <string.h>
#include "internal/common.h"

/* We make a distinction between events and crises, where
   crises need to be aborted from and events can be safely ignored */

/* Called when no handlers are established */
extern void (*crisis_abort_func)(char *);
/* Called for events when no handler exists */
extern void (*event_func)(char *);
/* Extra information for more useful error messages */
char exception_details[EXCEPTION_DETAILS_SIZE];
/* Called on all events (a hook for debuggers) */
void (*crisis_handler)()=NULL;

#if (!(CODE_RESOURCE))
#define unhandled_event(ex) fprintf(stderr,"Unhandled exception: %s",ex)
#define unhandled_crisis(ex) report_crisis(ex)
void report_crisis(ex)
   char *ex;
   {
   void exit(int condition);
   fprintf(stderr,"Unhandled exception: %s (%s)",ex,exception_details); 
   exit(-1);
   }
#else
#define unhandled_event(ex) (*event_func)(ex)
#define unhandled_crisis(ex) (*crisis_abort_func)(ex)
#endif
 
/* This is bound when exceptions are signaled. */
exception theException = NULL;
/* This is the current exception context */
jmp_buf_rec *cur_rec = NULL;
 
/* Pushes an exception context */
void push_jbr(jbr)
     jmp_buf_rec *jbr;
{
  jbr->next = cur_rec;
  jbr->self = jbr;
  cur_rec   = jbr;
}
 
/* Pops an exception context */
void pop_jbr()
{
  char *error = "Attempt to pop empty exception stack!";
  if (cur_rec == NULL) 
    unhandled_crisis(error);
  else cur_rec=cur_rec->next;
}
 
/* Raises an `event' --- a non fatal problem --- */
int raise_event(ex)
     exception ex;
{
  jmp_buf_rec *jbr;
  if (cur_rec == NULL) unhandled_event(ex);
  else {theException=ex; jbr=cur_rec;
	if (jbr->self != jbr)
	  {unhandled_crisis("Corrupted exception stack!");
	   cur_rec=NULL;}
	else {pop_jbr(); longjmp(jbr->jb,1);};
      };
  return 1;
}

/* Raises a crisis.  Unhandled crises cause an exit. */
int raise_crisis(ex)
     exception ex;
{
  jmp_buf_rec *jbr;
  if (crisis_handler) crisis_handler(ex);
  if (cur_rec == NULL) unhandled_crisis(ex);
  else {theException=ex; jbr=cur_rec;
	if (jbr->self != jbr)
	  {unhandled_crisis("Corrupted exception stack!");
	   cur_rec=NULL;}
	else {pop_jbr(); longjmp(jbr->jb,1);};
      };
  /* Never reached */
  return 1;
}
 
/* Raises a crisis and copies a string into the `exception details'
   used in error reports and such. */
int raise_crisis_with_details(ex,details)
     exception ex; char *details;
{
  jmp_buf_rec *jbr;
  strcpy(exception_details,details);
  if (cur_rec == NULL) unhandled_crisis(ex);
  else {theException=ex; jbr=cur_rec;
	if (jbr->self != jbr)
	  {unhandled_crisis("Corrupted exception stack!");
	   cur_rec=NULL;}
	else {pop_jbr(); longjmp(jbr->jb,1);};
      };
  /* Never reached */
  return 1;
}
 
/* Reraises the current exception.  Called in handlers */
int reraise()
{
  return raise_event(theException);
}
