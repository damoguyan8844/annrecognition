#include "photobook.h"
#include "ui.h"

/* Globals *******************************************************************/

struct AppDataStruct theAppData;
AppData appData = &theAppData;
Ph_Handle phandle;

/* Prototypes ****************************************************************/

void SetMembers(void);
void UpdateDisplay(void);
void SetupDB(void);

/* Functions *****************************************************************/

void main(int argc, char *argv[])
{
  MEM_USE_PICKETS(1);
  if(ProcessArgs(&argc, argv) == 0) exit(0);

  phandle = Ph_Startup();
  if(Ph_SetDatabase(phandle, argv[1]) == PH_ERROR) {
    List db_names;
    char *s;
    fprintf(stderr, "Unknown database `%s'\n", argv[1]);
    fprintf(stderr, "\nAvailable databases:\n");
    db_names = Ph_GetDatabases();
    if(ListSize(db_names)) {
      ListIter(p, s, db_names) {
	fprintf(stderr, "  %s\n", s);
      }
    }
    else {
      fprintf(stderr, "  none. Go make some.\n");
    }
    ListFree(db_names);
    exit(1);
  }

  InitDisplay();
  SetupDB();
  if(debug) printf("->XtAppMainLoop\n");
  XtAppMainLoop(appData->app_context);
}

void SetupDB(void)
{
  appData->garbage = ListCreate(GenericFree);
  appData->members = NULL;
  appData->selected = NULL; /* initialization for SetMembers() */
  appData->filter = NULL;   /* init for DoSearch() */
  appData->glabels = NULL;  /* not pretty, but needed */
  SetMembers();
  Ph_SetView(phandle, NULL);
  Ph_SetMetric(phandle, NULL);
  appData->show_name = 1;
  appData->show_annotation = NULL;
  appData->show_label = 0;
  appData->show_tree = 0;
  appData->show_dist = 0;
  appData->mask_dist = 1;
  ConfigureDisplay();
  UpdateDisplay();
}

void SetMembers(void)
{
  if(appData->members) free(appData->members);
  appData->members = Ph_GetWorkingSet(phandle, &appData->num_members);
  appData->members = ArrayCopy(appData->members, appData->num_members, 
			       sizeof(Ph_Member));
  /* must update num_images whenever num_members changes.
   * the IP must be halted and jump to page 0 for this to work.
   */
  appData->imp->num_images = appData->num_members;

  /* refresh the selection flags */
  if(appData->selected) free(appData->selected);
  appData->selected = (char*)calloc(appData->num_members, 1);
}

void UpdateDisplay(void)
{
  char str[100];

  /* Update the count */
  sprintf(str, "Working Set: %d", appData->num_members);
  XtVaSetValues(appData->count_label,
		XmNlabelString, MakeXmString(str),
		NULL);

  XtpmIPUpdate(appData->imp);
}

void BusyOn(char *s)
{
  ShowStatus(s);
  /* put a watch cursor in all windows */
  XtpmSetCursor(appData->main_window, XC_watch);
  XtpmSetCursor(appData->labeling_shell->shell, XC_watch);
  XtpmSetCursor(appData->symbol_shell->shell, XC_watch);
  /* Have to get a clean way to get these events processed */
  if(XtAppPending(appData->app_context))
    XtpmAppForceEvent(appData->app_context);
}

void BusyOff(void)
{
  /* remove watch cursors */
  XtpmSetCursor(appData->main_window, NoCursor);
  XtpmSetCursor(appData->labeling_shell->shell, NoCursor);
  XtpmSetCursor(appData->symbol_shell->shell, NoCursor);
  ShowStatus(NULL);
}

void ShowStatus(char *s)
{
  char str[100];
  if(!s) strcpy(str, "Photobook");
  else sprintf(str, "Photobook : %s", s);
  XtVaSetValues(appData->main_window, 
		XmNtitle, str,
		NULL);
}

/* Pop up an error notification window */
void ErrorPopup(char *format, ...)
{
  va_list args;
  Widget mbox;
  Arg arg[20];
  int n;
  char str[100];

  va_start(args, format);
  vsprintf(str, format, args);
  va_end(args);

  n=0;
  XtSetArg(arg[n], XmNvisual, appData->visual); n++;
  XtSetArg(arg[n], XmNdepth, appData->depth); n++;
  XtSetArg(arg[n], XmNcolormap, appData->colormap); n++;

  XtSetArg(arg[n], XmNnoResize, TRUE); n++;
  XtSetArg(arg[n], XmNdialogStyle, XmDIALOG_FULL_APPLICATION_MODAL); n++;
  XtSetArg(arg[n], XmNokLabelString, MakeXmString("Dismiss")); n++;
  XtSetArg(arg[n], XmNmessageString, MakeXmString(str)); n++;
  mbox = XmCreateErrorDialog(appData->shell, "Error", arg, n);
  XtUnmanageChild(XmMessageBoxGetChild(mbox, XmDIALOG_CANCEL_BUTTON));
  XtUnmanageChild(XmMessageBoxGetChild(mbox, XmDIALOG_HELP_BUTTON));
  XtManageChild(mbox);
}

/* Pop up an info window */
void InfoPopup(char *format, ...)
{
  va_list args;
  Widget mbox;
  Arg arg[20];
  int n;
  char str[100];

  va_start(args, format);
  vsprintf(str, format, args);
  va_end(args);

  n=0;
  XtSetArg(arg[n], XmNvisual, appData->visual); n++;
  XtSetArg(arg[n], XmNdepth, appData->depth); n++;
  XtSetArg(arg[n], XmNcolormap, appData->colormap); n++;

  XtSetArg(arg[n], XmNnoResize, TRUE); n++;
  XtSetArg(arg[n], XmNdialogStyle, XmDIALOG_FULL_APPLICATION_MODAL); n++;
  XtSetArg(arg[n], XmNokLabelString, MakeXmString("Dismiss")); n++;
  XtSetArg(arg[n], XmNmessageString, MakeXmString(str)); n++;
  mbox = XmCreateInformationDialog(appData->shell, "Information", arg, n);
  XtUnmanageChild(XmMessageBoxGetChild(mbox, XmDIALOG_CANCEL_BUTTON));
  XtUnmanageChild(XmMessageBoxGetChild(mbox, XmDIALOG_HELP_BUTTON));
  XtManageChild(mbox);
}
