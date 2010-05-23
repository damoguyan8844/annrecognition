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
  This file is the top level for the FRAMER shell program.

  Modification history:
    13 June 1992 - Began tracking modification history (Haase)
*************************************************************************/

#include "framer.h"
#include "fraxl.h"
#if FOR_UNIX
#include "sys/ioctl.h"
long ignored_ioctl;
#endif
#ifdef TIOCFLUSH
#define flush_standard_input() ioctl(0,TIOCFLUSH,&ignored_ioctl)
#else
#define flush_standard_input() 
#endif

void flush_input_on_error(void);
void backup_everything();
 
#if __TURBOC__
      extern unsigned _stklen = 20000;
#endif


int main(int argc,char *argv[])
{
  /* Get root file from comand line, defaulting to "radix"*/
  if (argc>1) root_filename=argv[1];
  INITIALIZE_FRAMER();
  printf("\n%%%%%%%% Welcome to the FRAMER shell under %s %%%%%%%%\n",OPSYS);
  read_eval_print(stdin,stdout,NULL,flush_input_on_error);
  backup_everything();
  printf("Exited FRAMER -- see you later");
  return 0;
}

void flush_input_on_error()
{
  flush_standard_input();
  fprintf(stderr,"#%%!& Error: %s: %s\n",theException,exception_details);
  fflush(stderr); 
  CLEAR_EXCEPTION();
}
