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
  This file contains non-functioning stubs for the FRAMER client/server functions.

  Modification history:
    13 June 1992 - Began tracking modification history (Haase)
*************************************************************************/

/* This file defines the basic operations on frames. */

#include "framer.h"

Frame update_remote_frame(Frame f)
{
  fprintf(stderr,"update_remote_frame: Network notwork!\n");
  return f;
}

Frame set_remote_ground(Frame f,Grounding g)
{
  fprintf(stderr,"set_remote_ground: Network notwork!\n");
  return f;
}

Frame set_remote_prototype(Frame f,Frame p)
{
  fprintf(stderr,"set_remote_prototype: Network notwork!\n");
  return f;
}

void add_remote_annotation(Frame f,char *name)
{
  fprintf(stderr,"add_remote_annotation: Network notwork!\n");
}

void init_network()
{
}
