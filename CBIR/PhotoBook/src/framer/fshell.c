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
 
extern void init_shell_functions(void);
extern Frame tty_shell(Frame here);
 
#if __TURBOC__
      #include <alloc.h>
      extern unsigned _stklen = 20000;
#endif

int main(argc,argv)
     int argc;
     char *argv[];
{
  /* Get root file from comand line, defaulting to "radix"*/
  if (argc>1) root_filename=argv[1];
  INITIALIZE_FRAMER();
  init_shell_functions();
  printf("\n%%%%%%%% Welcome to the FRAMER shell under %s %%%%%%%%\n",OPSYS);
  tty_shell(root_frame);
  backup_everything();
  return 0;
}

