#include "photobook.h"
#include "ui.h"
#include "config_shell.h"

/* Prototypes ****************************************************************/

Widget MakeLeftPane(Widget parent);
void ConfigureMenus(void);

/* private */
static void ConfigureMenu(List names, Ph_Object obj, XtCallbackProc callback,
			  Widget menu_parent, TShell conf);

/* Functions *****************************************************************/

static void ChangeDatabase(Widget w, void *userData, 
			   XmAnyCallbackStruct *callData)
{
  char *name;

  XtVaGetValues(w, XmNuserData, &name, NULL);
  if(!strcmp(name, Ph_GetDatabase(phandle))) return;
  if(debug) printf("change database: %s\n", name);

  XtpmSetCursor(appData->main_window, XC_watch);
  CloseDB();
  Ph_SetDatabase(phandle, name);
  SetupDB();
  XtpmSetCursor(appData->main_window, NoCursor);
}

/* TShellCB for ViewConfig */
static void ViewConfigCallback(Widget shell, void *userData)
{
  Ph_Object obj;
  ConfigShellFunc *shellFunc;

  obj = Ph_GetView(phandle);
  shellFunc = GetConfigShell(obj);
  if(shellFunc) shellFunc(shell, obj);
}

/* TShellCB for MetricConfig */
static void MetricConfigCallback(Widget shell, void *userData)
{
  Ph_Object obj;
  ConfigShellFunc *shellFunc;

  obj = Ph_GetMetric(phandle);
  shellFunc = GetConfigShell(obj);
  if(shellFunc) shellFunc(shell, obj);
}

Widget MakeLeftPane(Widget parent)
{
  Arg arg[10];
  int n;
  Widget pane, w;
  Widget menu, def;
  List databases;

  n=0;
  XtSetArg(arg[n], XmNtopAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_FORM); n++;
  pane = XmCreateRowColumn(parent, "leftPane", arg, n);

  /* database option menu */
  databases = Ph_GetDatabases();
  menu = MakePulldownMenu(pane, databases,
			  (XtCallbackProc)ChangeDatabase, NULL,
			  Ph_GetDatabase(phandle), &def);
  ListFree(databases);
  n=0;
  XtSetArg(arg[n], XmNsubMenuId, menu); n++;
  XtSetArg(arg[n], XmNmenuHistory, def); n++;
  w = XmCreateOptionMenu(pane, "dbOpMenu", arg, n);
  n=0;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  XtSetValues(XmOptionButtonGadget(w), arg, n);
  XtManageChild(w);

  Separator(pane);

  /* View option menu */
  n=0;
  w = XmCreateOptionMenu(pane, "viewOpMenu", arg, n);
  n=0;
  XtSetArg(arg[n], XmNrecomputeSize, TRUE); n++;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  XtSetValues(XmOptionButtonGadget(w), arg, n);
  XtManageChild(w);
  appData->view_menu = w;

  /* View configure toggle */
  appData->view_shell =
    TShellCreate(pane, "viewConfig", 
		 (TShellCB*)ViewConfigCallback, NULL);

  Separator(pane);

  /* Metric option menu */
  n=0;
  w = XmCreateOptionMenu(pane, "metricOpMenu", arg, n);
  n=0;
  XtSetArg(arg[n], XmNrecomputeSize, TRUE); n++;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  XtSetValues(XmOptionButtonGadget(w), arg, n);
  XtManageChild(w);
  appData->metric_menu = w;

  /* Metric configure toggle */
  appData->metric_shell =
    TShellCreate(pane, "metricConfig", 
		 (TShellCB*)MetricConfigCallback, NULL);
  
  Separator(pane);

  /* Count label */
  n=0;
  XtSetArg(arg[n], XmNrecomputeSize, FALSE); n++;
  XtSetArg(arg[n], XmNlabelString, MakeXmString("Working Set: ")); n++;
  w = XmCreateLabel(pane, "countLabel", arg, n);
  XtManageChild(w);
  appData->count_label = w;

  Separator(pane);

  /* Help text */
  n=0;
  w = XmCreateLabel(pane, "helpLabel", arg, n);
  XtManageChild(w);

  XtManageChild(pane);
  return pane;
}

static void ChangeMetric(Widget w, void *userData, 
			 XmAnyCallbackStruct *callData)
{
  char *name;

  XtVaGetValues(w, XmNuserData, &name, NULL);
  if(!strcmp(name, Ph_ObjName(Ph_GetMetric(phandle)))) return;
  if(debug) printf("change metric: %s\n", name);
  if(!Ph_SetMetric(phandle, name)) return;
  TShellDestroy(appData->metric_shell);
  TShellReset(appData->metric_shell);
  TShellSetSensitive(appData->metric_shell, 
		     !!GetConfigShell(Ph_GetMetric(phandle)));
}

static void ChangeView(Widget w, void *userData, 
		       XmAnyCallbackStruct *callData)
{
  char *name;
  int old_width, old_height, new_width, new_height;
  Ph_Object obj;

  XtVaGetValues(w, XmNuserData, &name, NULL);
  obj = Ph_GetView(phandle);
  if(!strcmp(name, Ph_ObjName(obj))) return;
  if(debug) printf("change view: %s\n", name);
  /* remember current size */
  Ph_ObjGet(obj, "width", &old_width);
  Ph_ObjGet(obj, "height", &old_height);
  obj = Ph_SetView(phandle, name);
  if(!obj) return;
  TShellDestroy(appData->view_shell);
  TShellReset(appData->view_shell);
  TShellSetSensitive(appData->view_shell, 
		     !!GetConfigShell(Ph_GetView(phandle)));
  /* has the size changed? */
  Ph_ObjGet(obj, "width", &new_width);
  Ph_ObjGet(obj, "height", &new_height);
  if(new_width != old_width || new_height != old_height) {
    Resize(NULL, NULL, NULL);
  }
  else {
    XtpmIPUpdate(appData->imp);
  }
}

void ConfigureMenus(void)
{
  ConfigureMenu(Ph_GetDBMetrics(phandle), Ph_GetMetric(phandle),
		(XtCallbackProc)ChangeMetric,
		appData->metric_menu, appData->metric_shell);
  ConfigureMenu(Ph_GetDBViews(phandle), Ph_GetView(phandle),
		(XtCallbackProc)ChangeView,
		appData->view_menu, appData->view_shell);
}

static void ConfigureMenu(List names, Ph_Object obj, XtCallbackProc callback,
			  Widget menu_parent, TShell conf)
{
  Arg arg[10];
  int n;
  Widget menu, def;

  if(names && !ListEmpty(names)) {
    char *current;
    if(obj) current = Ph_ObjName(obj);
    else current = NULL;
    menu = MakePulldownMenu(appData->left_pane, names, 
			    callback, NULL,
			    current, &def);
    n=0;
    XtSetArg(arg[n], XmNsubMenuId, menu); n++;
    XtSetArg(arg[n], XmNmenuHistory, def); n++;
    XtSetValues(menu_parent, arg, n);
    XtSetSensitive(menu_parent, 1);
  }
  else 
    XtSetSensitive(menu_parent, 0);
/*
  TShellSetSensitive(conf,
                     CanSearch && (appData->metric->shellFunc != NULL));
*/
  if(names) ListFree(names);
}
