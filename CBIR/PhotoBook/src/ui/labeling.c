#include "photobook.h"
#include "ui.h"
#include <xtpm/multilist.h>
#include <tpm/stream.h>

#define EXCLUSIVE 1

typedef struct {
  char pos, label;
  int index;
} Ex;

/* Globals *******************************************************************/

static XtpmML labelML = NULL;
static List Examples;

/* Prototypes ****************************************************************/

void LabelingCallback(Widget shell, void *userData);
void LabelFree(void);

/* Private */
static void LabelInit(int loadlast);
static void LabelClear(void);

/* Functions *****************************************************************/

static void ShowCovers(Widget w, void *userData, 
		       XmAnyCallbackStruct *callData)
{
  int label;
  printf("----------------------------------\n");
  for(label=0;label<ListSize(appData->labels);label++) {
    printf("%10s has %d covers\n", 
	   ListValueAtIndex(appData->labels, label), LearnNumCovers(label));
  }
}

static void WhichImages(Widget w, void *userData, 
			XmAnyCallbackStruct *callData)
{
  int i, page, count, label;
  char *labels;

  label = labelML->selected[0]-1;
  printf("Page   Patches labeled %s\n", 
	 ListValueAtIndex(appData->labels, label));
  printf("----------------------------------\n");
  page = 0;
  count = 0;
  for(i=0;i<appData->num_members;i++) {
    if(i && i % appData->imp->page_size == 0) {
      page++;
      if(count) printf("%4d   %d\n", page, count);
      count = 0;
    }
    labels = Ph_MemLabels(appData->members[i], NULL);
    if(labels[label]) count++;
    free(labels);
  }
  if(count) printf("%4d   %d\n", page+1, count);
}

static void AddExample(int pos, int label, int index)
{
  Ex *ex = Allocate(1, Ex);
  ex->pos = pos;
  ex->label = label;
  ex->index = index;
  ListAddRear(Examples, ex);
}

static void PosAction(Widget w, int exclusive, XmAnyCallbackStruct *callData)
{
  int label, i, j, found;
  Ph_Member member;

  BusyOn("relabeling");
  label = labelML->selected[0]-1;
  found = 0;
  for(i=0;i<appData->num_members;i++) {
    if(!appData->selected[i]) continue;
    found = 1;
    appData->selected[i] = 0;
    member = appData->members[i];
    AddExample(1, label, Ph_MemIndex(member));
    LearnPosEx(member, label);
    if(exclusive) {
      /* neg ex for all other labels */
      for(j=0;j<NumLabels;j++)
	if(j != label) {
	  LearnNegEx(member, j);
	AddExample(0, j, Ph_MemIndex(member));
	}
    }
  }
  if(!found) {
    ErrorPopup("No images selected to label");
  }
  else {
    LearnUpdate();
    UpdateDisplay();
  }
  BusyOff();
}

static void NegAllAction(Widget w, void *userData, 
			 XmAnyCallbackStruct *callData)
{
  int label, i, found;
  Ph_Member member;

  BusyOn("relabeling");
  found = 0;
  for(i=0;i<appData->num_members;i++) {
    if(!appData->selected[i]) continue;
    found = 1;
    appData->selected[i] = 0;
    member = appData->members[i];
    for(label=0;label<ListSize(appData->labels);label++) {
      AddExample(0, label, Ph_MemIndex(member));
      LearnNegEx(member, label);
    }
  }
  if(!found) {
    ErrorPopup("No images selected");
  }
  else {
    LearnUpdate();
    UpdateDisplay();
  }
  BusyOff();
}

static void NegAction(Widget w, void *userData, XmAnyCallbackStruct *callData)
{
  int label, i, found;
  Ph_Member member;

  BusyOn("relabeling");
  label = labelML->selected[0]-1;
  found = 0;
  for(i=0;i<appData->num_members;i++) {
    if(!appData->selected[i]) continue;
    found = 1;
    appData->selected[i] = 0;
    member = appData->members[i];
    AddExample(0, label, Ph_MemIndex(member));
    LearnNegEx(member, label);
  }
  if(!found) {
    ErrorPopup("No images selected");
  }
  else {
    LearnUpdate();
    UpdateDisplay();
  }
  BusyOff();
}

static List LabelCallback(XtpmML ml, int pos)
{
  List result;

  /* Case 1: double click action */
  if((pos == ml->num_lists-1) && ml->selected[pos]) {
    PosAction(NULL, 1, NULL);
    return NULL;
  }

/*
  result = ListCreate(NULL);
*/
  /* Case 2: build first list */
  if(ml->selected[pos] == 0) {
    ml->selected[pos] = 1;
    return ListCopy(appData->labels, NULL);
  }
  /* Case 3: selection */
  else {
    /* never happens */
    assert(0);
    ml->selected[pos+1] = 1;
  }
  return result;
}

static void RevertAction(Widget w, int loadlast,
			 XmAnyCallbackStruct *callData)
{
  BusyOn("reverting labels");
  LabelClear();
  LabelInit(loadlast);
  UpdateDisplay();
}

static void SaveExamples(Widget w, void *userData, 
			 XmAnyCallbackStruct *callData)
{
  FileHandle fp;
  /* dump the example list */
  fp = FileOpen("examples", "w");
  {Ex *ex;ListIter(p, ex, Examples) {
    fprintf(fp, "%d %d %d\n", ex->pos, ex->label, ex->index);
  }}
  FileClose(fp);
  InfoPopup("Wrote labeling state to `%s'", "examples");
  if(debug) printf("examples saved.\n");
}

/* Callback for the popup Ok button */
static void OkCallback(Widget w, void *userData, 
		       XmAnyCallbackStruct *callData)
{
  LearnUpdate();
  /* Refresh this page to show new labels */
  XtpmIPUpdate(appData->imp);
}

/* Toggle callback for tree buttons */
static void toggleCB(Widget w, int tree, 
		     XmToggleButtonCallbackStruct *toggleData)
{
  LearnEnable(tree, toggleData->set);
}

/* Pop up a toggle button dialog for enabling/disabling trees. */
static void EnableTrees(Widget w, void *userData, 
			XmAnyCallbackStruct *callData)
{
  Widget shell, pane;
  Arg arg[20];
  int n, tree;
  char *name;

  n=0;
  XtSetArg(arg[n], XmNvisual, appData->visual); n++;
  XtSetArg(arg[n], XmNdepth, appData->depth); n++;
  XtSetArg(arg[n], XmNcolormap, appData->colormap); n++;

  XtSetArg(arg[n], XmNdialogStyle, XmDIALOG_FULL_APPLICATION_MODAL); n++;
  shell = XmCreateMessageDialog(appData->shell, "Enable Trees", arg, n);
  XtUnmanageChild(XmMessageBoxGetChild(shell, XmDIALOG_HELP_BUTTON));
  XtUnmanageChild(XmMessageBoxGetChild(shell, XmDIALOG_CANCEL_BUTTON));
  XtAddCallback(shell, XmNokCallback, (XtCallbackProc)OkCallback, NULL);
  
  n=0;
  pane = XmCreateRowColumn(shell, "enableTreePane", arg, n);

  tree = 0;
  {ListIter(p, name, appData->trees) {
    n = 0;
    XtSetArg(arg[n], XmNset, LearnEnabled(tree)); n++;
    w = XmCreateToggleButton(pane, name, arg, n);
    XtAddCallback(w, XmNvalueChangedCallback,
		  (XtCallbackProc)toggleCB, (void*)tree);
    XtManageChild(w);
    tree++;
  }}
  XtManageChild(pane);
  XtManageChild(shell);
}

static void LabelWidgets(Widget parent)
{
  int n;
  Arg arg[10];
  Widget w, which;

  n=0;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  w = XmCreatePushButton(parent, "saveButton", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)SaveExamples, NULL);
  XtManageChild(w);
  
  n=0;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNbottomWidget, w); n++;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  w = XmCreatePushButton(parent, "revertButton", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)RevertAction, 
		(void*)1);
  XtManageChild(w);

  n=0;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNbottomWidget, w); n++;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  w = XmCreatePushButton(parent, "clearLabelButton", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)RevertAction, 
		(void*)0);
  XtManageChild(w);

  n=0;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNbottomWidget, w); n++;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  w = XmCreatePushButton(parent, "coversButton", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)ShowCovers, NULL);
  XtManageChild(w);

  n=0;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNbottomWidget, w); n++;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  w = XmCreatePushButton(parent, "whichButton", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)WhichImages, NULL);
  XtManageChild(w);

  n=0;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNbottomWidget, w); n++;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  w = XmCreatePushButton(parent, "enableButton", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)EnableTrees, NULL);
  XtManageChild(w);

  n=0;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNbottomWidget, w); n++;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  w = XmCreatePushButton(parent, "negAllButton", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)NegAllAction, NULL);
  XtManageChild(w);
  
  n=0;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNbottomWidget, w); n++;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  w = XmCreatePushButton(parent, "negButton", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)NegAction, NULL);
  XtManageChild(w);
  
  n=0;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNbottomWidget, w); n++;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  w = XmCreatePushButton(parent, "addPosButton", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)PosAction, (void*)0);
  XtManageChild(w);

  n=0;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNbottomWidget, w); n++;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  w = XmCreatePushButton(parent, "posButton", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)PosAction, (void*)1);
  XtManageChild(w);

  if(labelML) XtpmFreeML(labelML);
  labelML = XtpmCreateML(parent, "labelML", 1, LabelCallback, NULL);
  XtVaSetValues(labelML->pane,
		XmNleftAttachment, XmATTACH_FORM,
		XmNrightAttachment, XmATTACH_FORM,
		XmNtopAttachment, XmATTACH_FORM,
		XmNbottomAttachment, XmATTACH_WIDGET,
		XmNbottomWidget, w,
		NULL);

}

static void InitExamples(void)
{
  int count = 0;
  Ex ex;
  char str[100];
  FileHandle fp;
  /* read in the example file */
  fp = fopen("examples", "r");
  if(fp) {
    ShowStatus("loading examples");
    XtpmAppForceEvent(appData->app_context);
    for(;;) {
      getline(str, 100, fp);
      if(feof(fp)) break;
      ex.pos = atoi(strtok(str, " "));
      ex.label = atoi(strtok(NULL, " "));
      ex.index = atoi(strtok(NULL, " "));
      if(debug) printf("%d\n", count++); 
      AddExample(ex.pos, ex.label, ex.index);
      /* use of total_set is an abstraction violation */
      if(ex.pos)
	LearnPosEx(phandle->total_set[ex.index], ex.label);
      else
	LearnNegEx(phandle->total_set[ex.index], ex.label);
    }
    fclose(fp);
    LearnUpdate();
  }
}

static void LabelInit(int loadlast)
{
  List trees;
  int i;

  BusyOn("loading trees");

  trees = Ph_GetDBTrees(phandle);
  LearnInit(phandle, trees, NumLabels);

  /* set appData->trees to trees with file extensions removed */
  appData->trees = ListCreate(GenericFree);
  {char *fname;ListIter(p, fname, trees) {
    char *tree = strdup(fname);
    for(i=0;tree[i] && tree[i] != '.';i++);
    if(tree[i]) tree[i] = 0;
    ListAddRear(appData->trees, tree);
  }}
  ListFree(trees);

  for(i=0;i<NumLabels;i++) {
    AddLabel();
  }
  Examples = ListCreate(GenericFree);
  if(loadlast) InitExamples();
}

/* TShellCB for labeling toggle */
void LabelingCallback(Widget shell, void *userData)
{
  Widget pane;

  LabelInit(0);
#if GLABEL
  GLabelInit();
#endif
  /* enable hooks */
  if(appData->hooks_shell)
    TShellSetSensitive(appData->hooks_shell, 1);

  pane = XmCreateForm(shell, "labelingPane", NULL, 0);
  LabelWidgets(pane);
  XtManageChild(pane);
  UpdateDisplay(); /* because of LabelInit() */
}

static void LabelClear(void)
{
  ListFree(Examples);
  ListFree(appData->trees);
  LearnFree();
}

void LabelFree(void)
{
  /* if no shell was created, then learning was never initialized */
  if(!labelML) return;
  XtpmFreeML(labelML); 
  labelML = NULL;
  ListFree(appData->labels);
  LabelClear();
  /* disable hooks */
  if(appData->hooks_shell)
    TShellSetSensitive(appData->hooks_shell, 0);
}
