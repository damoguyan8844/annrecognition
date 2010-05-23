#include "photobook.h"
#include "ui.h"

/* Prototypes ****************************************************************/

TShell TShellCreate(Widget parent, char *name, 
		    TShellCB *callback, void *userData);
void TShellReset(TShell ts);
void TShellDestroy(TShell ts);

/* Functions *****************************************************************/

static void TShellCallback(Widget w, TShell ts,
			   XmToggleButtonCallbackStruct *toggleData)
{
  if(!toggleData->set) {
    if(ts->shell) XtPopdown(ts->shell);
    return;
  }
  if(!ts->shell) {
    Position x,y;
    Dimension height;
    Arg arg[10];
    int n;
    char str[100];

    /* Create the shell */
    /* place just under the left pane */
    XtVaGetValues(appData->main_window, XmNx, &x, XmNy, &y, NULL);
    XtVaGetValues(appData->left_pane, XmNheight, &height, NULL);
    n=0;
    XtSetArg(arg[n], XmNvisual, appData->visual); n++;
    XtSetArg(arg[n], XmNdepth, appData->depth); n++;
    XtSetArg(arg[n], XmNcolormap, appData->colormap); n++;
/*
    XtSetArg(arg[n], XmNx, x); n++;
    XtSetArg(arg[n], XmNy, y+height); n++;
*/
    /* use prefix of the button name to get name of shell */
    strcpy(str, XtName(ts->button));
    str[strlen(str)-6] = '\0';
    strcat(str, "Shell");
    ts->shell = 
      XtCreatePopupShell(str, applicationShellWidgetClass,
			 appData->main_window, arg, n);

    /* Create the widgets under the shell */
    ts->callback(ts->shell, ts->userData);
  }
  /* Pop up the shell */
  XtPopup(ts->shell, XtGrabNone);
}

TShell TShellCreate(Widget parent, char *name, 
		    TShellCB *callback, void *userData)
{
  int n;
  Arg arg[10];
  Widget w;
  char str[100];
  TShell ts = Allocate(1, struct TShell);
  ts->callback = callback;
  ts->userData = userData;
  ts->shell = NULL;

  n=0;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  sprintf(str, "%sButton", name);
  w = XmCreateToggleButton(parent, str, arg, n);
  XtManageChild(w);
  XtAddCallback(w, XmNvalueChangedCallback,
		(XtCallbackProc)TShellCallback, ts);
  ts->button = w;
  return ts;
}

void TShellReset(TShell ts)
{
  if(!ts) return;
  XmToggleButtonSetState(ts->button, 0, 0);
}

void TShellDestroy(TShell ts)
{
  if(!ts) return;
  if(ts->shell) XtDestroyWidget(ts->shell);
  ts->shell = NULL;
}
