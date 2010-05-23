#ifndef MULTILIST_H_INCLUDED
#define MULTILIST_H_INCLUDED

#include <xtpm/xtpm.h>
#include <tpm/list.h>

/* This is a transparent structure */
typedef struct XtpmMLRec {
  int num_lists;
  XtCallbackProc callback;
  XtPointer userData;
  Widget pane, *lists;
  List *words;
  int *selected;
} *XtpmML;

/* callback type */
typedef List XtpmMLCallback(XtpmML ml, int pos);

/* accessor functions */
XtpmML XtpmCreateML(Widget parent, String name, 
		    int num_lists, XtpmMLCallback cb, XtPointer userData);
void XtpmFreeML(XtpmML ml);
#define XtpmMLSelection(ml, pos) \
(String)ListValueAtIndex(ml->words[pos], ml->selected[pos]-1)

#endif
