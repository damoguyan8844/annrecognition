#include "photobook.h"
#include "ui.h"
#include <math.h> /* for ceil() */

/* Prototypes ****************************************************************/

Widget MakeRightPane(Widget parent);
void CloseDB(void);

/* Functions *****************************************************************/

void CloseDB(void)
{
  XtpmIPHalt(appData->imp);
  XtpmITFree(appData->imp->imt);
  TShellDestroy(appData->view_shell);
  TShellDestroy(appData->metric_shell);
  TShellDestroy(appData->labeling_shell);
  TShellDestroy(appData->symbol_shell);
#if HOOKS
  TShellDestroy(appData->hooks_shell);
#endif
#if GLABEL
  TShellDestroy(appData->glabel_shell);
#endif
  XFreePixmap(appData->display, appData->no_image_pixmap);
  PixCacheFree();
  if(debug) XC_BLOCKS();

  free(appData->selected);
  if(appData->filter) free(appData->filter);
  LabelFree();
  SymbolsFree();
#if GLABEL
  GLabelFree();
#endif
  if(appData->members) free(appData->members);
  ListFree(appData->garbage);
}

static void Quit(Widget w, void *userData, XmAnyCallbackStruct *callData)
{
  CloseDB();
  XtpmSetCursor(appData->main_window, XC_pirate);
  XtpmIPFree(appData->imp);
  TShellFree(appData->view_shell);
  TShellFree(appData->metric_shell);
  TShellFree(appData->labeling_shell);
  TShellFree(appData->symbol_shell);
#if HOOKS
  TShellFree(appData->hooks_shell);
#endif
#if GLABEL
  TShellFree(appData->glabel_shell);
#endif
  if(appData->gamma_table) free(appData->gamma_table);
  if(appData->color_table) free(appData->color_table);
  ListFree(appData->colorList);
  Ph_Shutdown(phandle);
  XC_BLOCKS();
  exit(0);
}

/* Callback for initialize button.
 * Clears the filter and the query set. 
 */
void Initialize(Widget w, void *userData,
		XmAnyCallbackStruct *callData)
{
  /* clear the text widget */
  XtVaSetValues(appData->com_text, XmNvalue, "", NULL);
  Ph_SetFilter(phandle, "");

  SetMembers();
  /* disable distance text */
  appData->mask_dist = 1;
  XtpmIPJumpTo(appData->imp, 0);
  UpdateDisplay();
}

/* Callback for shuffle button */
void Shuffle(Widget w, void *userData, 
	     XmAnyCallbackStruct *callData)
{
  XtpmIPHalt(appData->imp);
  Ph_Shuffle(phandle);
  SetMembers();
  /* disable distance text */
  appData->mask_dist = 1;
  XtpmIPJumpTo(appData->imp, 0);
  /* don't need to UpdateDisplay() since the number of members 
     hasn't changed */
}

/* Callback for resize button */
void Resize(Widget w, void *userData, 
	    XmAnyCallbackStruct *callData)
{
  int pos;
  XtpmSetCursor(appData->main_window, XC_gumby);
  XtpmIPHalt(appData->imp);
  pos = appData->imp->page * appData->imp->page_size;
  XtpmITFree(appData->imt);
  XFreePixmap(appData->display, appData->no_image_pixmap);
  ConfigurePhotos();
  /* jump back to the page we were looking at */
  XtpmIPJumpTo(appData->imp, pos / appData->imp->page_size);
}

/* Callback for refresh button */
static void Refresh(Widget w, void *userData, 
		    XmAnyCallbackStruct *callData)
{
  XtpmSetCursor(appData->main_window, XC_gumby);
  XtpmIPHalt(appData->imp);
  PixCacheFree();
  PixCacheCreate();
  XtpmIPUpdate(appData->imp);
}

/* Callback for page text entry */
static void SetPage(Widget w, void *userData, 
		    XmAnyCallbackStruct *callData)
{
  int page;
  char *str = XmTextGetString(w);
  MEM_ALLOC_NOTIFY(str);
  page = atoi(str);
  free(str);
  if(XtpmIPJumpTo(appData->imp, page-1) == 1) {
    ErrorPopup("No page %d", page);
  }
}

/* Callback for member text entry */
static void SetMember(Widget w, void *userData, 
		      XmAnyCallbackStruct *callData)
{
  int i;
  Ph_Member member;
  char *str = XmTextGetString(w);
  MEM_ALLOC_NOTIFY(str);
  member = Ph_MemberWithName(phandle, str);
  for(i=0;i<appData->num_members;i++) {
    if(appData->members[i] == member) break;
  }
  if(i == appData->num_members) {
    ErrorPopup("No member named \"%s\"\nCheck your search filter", str);
  }
  else {
    appData->selected[i] = 1;
    XtpmIPJumpTo(appData->imp, i/appData->imp->page_size);
  }
  free(str);
}

/* Callback for the photo table (XtpmIP).
 * Called just before images are refreshed.
 */
static void UpdateStart(XtpmIP imp, void *userData)
{
  char str[100];
  int pages = ceil((double)appData->imp->num_images/appData->imp->page_size);
  /* Update the page label */
  sprintf(str, "Page %d of %d", appData->imp->page+1, pages);
  XtVaSetValues(appData->page_label,
                XmNlabelString, MakeXmString(str),
                NULL);
  ShowStatus("loading images");
}

/* Callback for the photo table (XtpmIP).
 * Called after images are refreshed.
 */
static void UpdateStop(XtpmIP imp, void *userData)
{
  BusyOff();
}

Widget MakeRightPane(Widget parent)
{
  Arg arg[10];
  int n;
  Widget pane, w;

  n=0;
  XtSetArg(arg[n], XmNtopAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_FORM); n++;
  pane = XmCreateRowColumn(parent, "rightPane", arg, n);

  n=0;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;

  /* Initialize */
  w=XmCreatePushButton(pane,"initializeButton",arg,n);
  XtAddCallback(w, XmNactivateCallback,
		(XtCallbackProc)Initialize, NULL);
  XtManageChild(w);

  /* Shuffle */
  w=XmCreatePushButton(pane,"shuffleButton",arg,n);
  XtManageChild(w);
  XtAddCallback(w, XmNactivateCallback,
		(XtCallbackProc)Shuffle, NULL);

  /* Load */
  w=XmCreatePushButton(pane,"loadButton",arg,n);
  XtAddCallback(w, XmNactivateCallback,
		(XtCallbackProc)LoadDialog, NULL);
  XtManageChild(w);

  /* Save */
  w=XmCreatePushButton(pane,"saveButton",arg,n);
  XtAddCallback(w, XmNactivateCallback,
		(XtCallbackProc)SaveDialog, NULL);
  XtManageChild(w);

#if 0
  /* Probe */
  w=XmCreatePushButton(pane,"probeButton",arg,n);
  XtAddCallback(w, XmNactivateCallback,
		(XtCallbackProc)Probe, NULL);
  XtManageChild(w);
#endif

  /* Configure Text */
  w = XmCreatePushButton(pane, "configText", arg, n);
  XtAddCallback(w, XmNactivateCallback,
		(XtCallbackProc)ConfigureText, NULL);
  XtManageChild(w);

#if TCL
  /* Tcl */
  w = XmCreatePushButton(pane, "tclButton", arg, n);
  XtAddCallback(w, XmNactivateCallback,
		(XtCallbackProc)DoTcl, NULL);
  XtManageChild(w);
#endif

  Separator(pane);

  /* Symbols toggle */
  appData->symbol_shell = 
    TShellCreate(pane, "symbol", 
		 (TShellCB*)SymbolsCallback, NULL);

  /* Labeling toggle */
  appData->labeling_shell = 
    TShellCreate(pane, "labeling", 
		 (TShellCB*)LabelingCallback, NULL);
  
#if HOOKS
  /* Hooks toggle */
  appData->hooks_shell = 
    TShellCreate(pane, "hooks",
		 (TShellCB*)HooksCallback, NULL);
#else
  appData->hooks_shell = NULL;
#endif

  /* GLabel toggle */
  appData->glabels = NULL;
#if GLABEL
  appData->glabel_shell = 
    TShellCreate(pane, "glabel",
		 (TShellCB*)GLabelingCallback, NULL);
#else
  appData->glabel_shell = NULL;
#endif

  Separator(pane);

  /* Resize */
  w=XmCreatePushButton(pane,"resizeButton",arg,n);
  XtAddCallback(w, XmNactivateCallback,
		(XtCallbackProc)Resize, NULL);
  XtManageChild(w);

  /* Refresh */
  w=XmCreatePushButton(pane,"refreshButton",arg,n);
  XtAddCallback(w, XmNactivateCallback,
		(XtCallbackProc)Refresh, NULL);
  XtManageChild(w);
  
  /* Arrow label */
  n=0;
  XtSetArg(arg[n], XmNrecomputeSize, FALSE); n++;
  w=XmCreateLabel(pane, "arrowLabel", arg, n);
  XtManageChild(w);

  /* Photo table */
  appData->imp = XtpmIPCreate(pane, "photoTable");
  XtpmIPStartCallback(appData->imp, (XtpmIPCallback*)UpdateStart, NULL);
  XtpmIPStopCallback(appData->imp, (XtpmIPCallback*)UpdateStop, NULL);

  /* Page label */
  n=0;
  XtSetArg(arg[n], XmNrecomputeSize, FALSE); n++;
  /* the init string needs to be big enough to establish the widget size */
  XtSetArg(arg[n], XmNlabelString, MakeXmString("Page 100 of 100")); n++;
  w=XmCreateLabel(pane, "pageLabel", arg, n);
  XtManageChild(w);
  appData->page_label = w;

  Separator(pane);

  /* Page text entry */
  w = XmCreateLabel(pane, "pageTextLabel", NULL, 0);
  XtManageChild(w);
  n=0;
  w = XmCreateText(pane, "pageText", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)SetPage, NULL);
  XtManageChild(w);

  /* Member text entry */
  w = XmCreateLabel(pane, "memberTextLabel", NULL, 0);
  XtManageChild(w);
  n=0;
  w = XmCreateText(pane, "memberText", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)SetMember, NULL);
  XtManageChild(w);

  Separator(pane);

  /* Quit */
  n=0;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  w = XmCreatePushButton(pane, "quitButton", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)Quit, NULL);
  XtManageChild(w);

  XtManageChild(pane);
  return pane;
}

