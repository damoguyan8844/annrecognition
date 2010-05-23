#include "photobook.h"
#include "ui.h"
#include <xtpm/multilist.h>

/* Globals *******************************************************************/

static XtpmML symbolML = NULL;

/* Prototypes ****************************************************************/

void SymbolsCallback(Widget shell, void *userData);

/* Functions *****************************************************************/

/* Appends str to the command text */
static void AppendComText(String str)
{
  XmTextPosition old_pos;

  old_pos = XmTextGetInsertionPosition(appData->com_text);
  XmTextInsert(appData->com_text, old_pos, str);
  XmTextSetInsertionPosition(appData->com_text, old_pos + strlen(str));
}

static List SymbolCallback(XtpmML ml, int pos)
{
  List result;

  /* Case 1: item selected */
  if(pos == ml->num_lists-1) {
    AppendComText("[");
    AppendComText(XtpmMLSelection(ml, 0));
    AppendComText("/");
    AppendComText(XtpmMLSelection(ml, 1));
    AppendComText("] ");
    return NULL;
  }

  /* Case 2: initialize first list */
  if(ml->selected[pos] == 0) {
    result = Ph_GetSymbols(phandle);
    ml->selected[pos] = 1;
  }
  /* Case 3: initialize second list */
  else {
    result = Ph_GetSubSymbols(phandle, 
			      ListValueAtIndex(ml->words[0],
					       ml->selected[0]-1));
    ml->selected[pos+1] = 1;
  }
  return result;
}

static void SpitText(Widget w, char *str,
		     XmAnyCallbackStruct *callData)
{
  AppendComText(str);
}

static void CloseAll(Widget w, void *userData,
		     XmAnyCallbackStruct *callData)
{
  char *p, *str;
  int paren_count = 0;
  str = XmTextGetString(appData->com_text);
  MEM_ALLOC_NOTIFY(str);
  for(p=str;*p;p++) {
    if(*p == '(') paren_count++;
    else if(*p == ')') paren_count--;
  }
  if(paren_count < 0) ErrorPopup("Missing left parenthesis");
  else while(paren_count) {
    AppendComText(")"); paren_count--;
  }
  free(str);
}

static void ClearText(Widget w, void *userData,
		      XmAnyCallbackStruct *callData)
{
  XmTextSetString(appData->com_text, "");
}

static void Apply(Widget w, void *userData,
		  XmAnyCallbackStruct *callData)
{
  DoSearch();
}

void SymbolsCallback(Widget shell, void *userData)
{
  int n;
  Arg arg[10];
  Widget pane, w, subpane;
  char *textspit[]={
    "AND", "(AND ",
    "OR", "(OR ",
    "NOT", "(NOT ",
    "EXCEPT", "(EXCEPT ",
    ")", ")",
    NULL
  };
  char **p;

  pane = XmCreateForm(shell, "symbolsPane", NULL, 0);
  
  n = 0;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  w = XmCreatePushButton(pane, "applyButton", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)Apply, NULL);
  XtManageChild(w);

  n = 0;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNbottomWidget, w); n++;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  w = XmCreatePushButton(pane, "clearButton", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)ClearText, NULL);
  XtManageChild(w);

  n = 0;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNbottomWidget, w); n++;
  XtSetArg(arg[n], XmNorientation, XmHORIZONTAL); n++;
  subpane = XmCreateRowColumn(pane, "symbolsRowCol", arg, n);

  /* text spitters */
  n = 0;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  for(p=textspit;*p;p+=2) {
    w = XmCreatePushButton(subpane, p[0], arg, n);
    XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)SpitText, p[1]);
    XtManageChild(w);
  }
  w = XmCreatePushButton(subpane, ")*", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)CloseAll, NULL);
  XtManageChild(w);
  XtManageChild(subpane);

  if(symbolML) XtpmFreeML(symbolML);
  symbolML = XtpmCreateML(pane, "symbolML", 2, SymbolCallback, NULL);
  XtVaSetValues(symbolML->pane,
		XmNleftAttachment, XmATTACH_FORM,
		XmNrightAttachment, XmATTACH_FORM,
		XmNtopAttachment, XmATTACH_FORM,
		XmNbottomAttachment, XmATTACH_WIDGET,
		XmNbottomWidget, subpane,
		NULL);

  XtManageChild(pane);
}

void SymbolsFree(void)
{
  if(!symbolML) return;
  XtpmFreeML(symbolML); 
  symbolML = NULL;
}
