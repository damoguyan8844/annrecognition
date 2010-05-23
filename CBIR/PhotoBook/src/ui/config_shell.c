#include "photobook.h"
#include "ui.h"
#include "config_shell.h"

/* Prototypes ****************************************************************/

ConfigShellFunc *GetConfigShell(Ph_Object obj);

/* Private */
static ConfigShellFunc 
  EuclideanShell, 
  ViewImageShell, 
  ViewBarShell,
  ViewZoomShell,
  ViewChannelShell,
  ViewTswShell,
  TswShell,
  WoldShell, 
  CombinationShell,
  LabelProbShell,
  PeaksShell,
  RankComboShell;

/* Globals *******************************************************************/

typedef struct {
  char *object;
  ConfigShellFunc *shellFunc;
} ShellMap;
static ShellMap Shells[]={
  { "euclidean", EuclideanShell },
  { "image", ViewImageShell },
  { "bar_graph", ViewBarShell },
  { "zoom", ViewZoomShell },
  { "channel", ViewChannelShell },
  { "tsw_tree", ViewTswShell },
  { "tsw", TswShell },
  { "wold", WoldShell },
  { "combination", CombinationShell },
  { "label_prob", LabelProbShell },
  { "peaks", PeaksShell },
  { "rank_combo", RankComboShell },
  { NULL, NULL }
};

/* Functions *****************************************************************/

ConfigShellFunc *GetConfigShell(Ph_Object obj)
{
  ShellMap *p;

  for(;;) {
    /* search on name */
    for(p=Shells;p->object;p++) {
      if(!strcmp(p->object, Ph_ObjName(obj))) break;
    }
    if(p->object) break;
    /* search on class */
    for(p=Shells;p->object;p++) {
      if(!strcmp(p->object, Ph_ObjClass(obj))) break;
    }
    if(p->object || !obj->super) break;
    /* try the superclass */
    obj = obj->super;
  }
  return p->shellFunc;
}

static void EuclideanShell(Widget shell, Ph_Object obj)
{
  Widget pane;
  int n;

  /* RowColumn */
  pane = XmCreateRowColumn(shell, "euclideanPane", NULL, 0);

  if(Ph_ObjGet(obj, "vector-size", &n) == PH_ERROR) {
    fprintf(stderr, "Error getting vector-size of %s\n", Ph_ObjName(obj));
  }
  else {
    ScaleVar(pane, obj, "from", 0, n-1, 0);
    ScaleVar(pane, obj, "to", 0, n-1, 0);
  }

  XtManageChild(pane);
}

static void ViewImageShell(Widget shell, Ph_Object obj)
{
  TextVar(shell, obj, "field");
}

static void ViewBarShell(Widget shell, Ph_Object obj)
{
  Widget pane;

  /* RowColumn */
  pane = XmCreateRowColumn(shell, "woldPane", NULL, 0);

  TextVar(pane, obj, "minimum");
  TextVar(pane, obj, "maximum");
  TextVar(pane, obj, "color");

  XtManageChild(pane);
}

static void ViewZoomShell(Widget shell, Ph_Object obj)
{
  TextVar(shell, obj, "zfact");
}

static void ViewChannelShell(Widget shell, Ph_Object obj)
{
  TextVar(shell, obj, "channel");
}

static void ViewTswShell(Widget shell, Ph_Object obj)
{
  TextVar(shell, obj, "maximum");
}

static void TswShell(Widget shell, Ph_Object obj)
{
  Widget pane;

  /* RowColumn */
  pane = XmCreateRowColumn(shell, "woldPane", NULL, 0);

  TextVar(pane, obj, "cutoff");
  TextVar(pane, obj, "keep");

  XtManageChild(pane);
}

static void WoldShell(Widget shell, Ph_Object obj)
{
  Widget pane;
  List items;

  /* RowColumn */
  pane = XmCreateRowColumn(shell, "woldPane", NULL, 0);

  TextVar(pane, obj, "nbr-size");
  TextVar(pane, obj, "alt-metric");
  TextVar(pane, obj, "orien-label");

  items = ListCreate(NULL);
  ListAddRear(items, "none");
  ListAddRear(items, "gorkani");
  ListAddRear(items, "tamura");
  MenuVar(pane, obj, "orien-type", items, NULL);
  ListFree(items);

  XtManageChild(pane);
}

static void PeaksShell(Widget shell, Ph_Object obj)
{
  Widget pane;
  int n;

  /* RowColumn */
  pane = XmCreateRowColumn(shell, "peaksPane", NULL, 0);

  TextVar(pane, obj, "nbr-size");
  TextVar(pane, obj, "peaks");

  XtManageChild(pane);
}

static void CombinationShell(Widget shell, Ph_Object obj)
{
  Widget pane;
  int n;

  /* RowColumn */
  pane = XmCreateRowColumn(shell, "woldPane", NULL, 0);

  if(Ph_ObjGet(obj, "num-metrics", &n) == PH_ERROR) {
    fprintf(stderr, "Error getting num-metrics of %s\n", Ph_ObjName(obj));
  }
  else {
    TextVar(pane, obj, "metrics");
    TextVar(pane, obj, "factors");
    TextVar(pane, obj, "weights");
  }

  XtManageChild(pane);
}

static void RankComboShell(Widget shell, Ph_Object obj)
{
  Widget pane;
  int n;

  /* RowColumn */
  pane = XmCreateRowColumn(shell, "woldPane", NULL, 0);

  if(Ph_ObjGet(obj, "num-metrics", &n) == PH_ERROR) {
    fprintf(stderr, "Error getting num-metrics of %s\n", Ph_ObjName(obj));
  }
  else {
    TextVar(pane, obj, "num-metrics");
    TextVar(pane, obj, "metrics");
    TextVar(pane, obj, "weights");
  }

  XtManageChild(pane);
}

static void LabelProbShell(Widget shell, Ph_Object obj)
{
  ScaleVar(shell, obj, "label", 0, NumLabels, 0);
}
