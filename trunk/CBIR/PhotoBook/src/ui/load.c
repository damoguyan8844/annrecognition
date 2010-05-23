#include "photobook.h"
#include "ui.h"
#include <Xm/DialogS.h>
#include <Xm/FileSB.h>

/* Prototypes ****************************************************************/

void LoadDialog(Widget w, void *userData,
		XmAnyCallbackStruct *callData);
void SaveDialog(Widget w, void *userData,
		XmAnyCallbackStruct *callData);

/* Functions *****************************************************************/

static void cancelCallback(Widget w, Widget box, 
			   XmAnyCallbackStruct *callData)
{
  XtUnmanageChild(XtParent(box));
}

static void MakeFSDialog(char *name, XtCallbackProc callback)
{
  int n;
  Arg arg[10];
  Widget shell, box;
  char str[100];

  n = 0;
  XtSetArg(arg[n], XmNvisual, appData->visual); n++;
  XtSetArg(arg[n], XmNdepth, appData->depth); n++;
  XtSetArg(arg[n], XmNcolormap, appData->colormap); n++;

  XtSetArg(arg[n], XmNdialogStyle, XmDIALOG_FULL_APPLICATION_MODAL); n++;
  sprintf(str, "%sDialogShell", name);
  shell = XmCreateDialogShell(appData->main_window, str,
			      arg, n);
  sprintf(str, "%sDialogBox", name);
  box = XmCreateFileSelectionBox(shell, str, NULL, 0);
  XtUnmanageChild(XmFileSelectionBoxGetChild(box, XmDIALOG_HELP_BUTTON));
  XtUnmanageChild(XmFileSelectionBoxGetChild(box, XmDIALOG_DIR_LIST));
  XtUnmanageChild(XmFileSelectionBoxGetChild(box, XmDIALOG_DIR_LIST_LABEL));
  XtAddCallback(box, XmNokCallback, callback, box);
  XtAddCallback(box, XmNcancelCallback, (XtCallbackProc)cancelCallback, box);
  XtManageChild(box);
  XtManageChild(shell);
}

static void loadCallback(Widget w, Widget box,
			 XmAnyCallbackStruct *callData)
{
  XmString xs;
  char *fname;
  XtVaGetValues(box, XmNtextString, &xs, NULL);
  XtUnmanageChild(XtParent(box));

  fname = XmStringToCString(xs);
  printf("loading %s\n", fname);
  if(Ph_LoadWS(phandle, fname) == PH_ERROR) {
    ErrorPopup("Cannot open query file `%s'", fname);
    return;
  }

  /* invalidate the search filter */
  if(appData->filter) {
    free(appData->filter);
    appData->filter = NULL;
  }
  /* read new member set */
  SetMembers();
  /* let distance text be displayed */
  appData->mask_dist = 0;
  /* reset to page zero */
  XtpmIPJumpTo(appData->imp, 0);
  UpdateDisplay();
}

void LoadDialog(Widget w, void *userData,
		XmAnyCallbackStruct *callData)
{
  MakeFSDialog("load", (XtCallbackProc)loadCallback);
}

static void saveCallback(Widget w, Widget box,
			 XmAnyCallbackStruct *callData)
{
  XmString xs;
  char *fname;
  XtVaGetValues(box, XmNtextString, &xs, NULL);
  XtUnmanageChild(XtParent(box));

  fname = XmStringToCString(xs);
  printf("saving %s\n", fname);
  if(Ph_SaveWS(phandle, fname) == PH_ERROR) {
    ErrorPopup("Cannot save to query file `%s'", fname);
    return;
  }
  InfoPopup("Query saved to `%s'\n", fname);
}

void SaveDialog(Widget w, void *userData,
		XmAnyCallbackStruct *callData)
{
  MakeFSDialog("save", (XtCallbackProc)saveCallback);
}
