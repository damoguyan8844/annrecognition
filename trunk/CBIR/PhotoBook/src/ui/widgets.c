#include "photobook.h"
#include "ui.h"
#include <math.h> /* for pow() */

typedef struct {
  Ph_Object obj;
  char *field;
  Widget w;
  int i;
} CBInfo;

/* Prototypes ****************************************************************/

Widget ScaleVar(Widget parent, Ph_Object obj, char *field, 
		double from, double to, int points);
Widget TextVar(Widget parent, Ph_Object obj, char *field);
Widget MenuVar(Widget parent, Ph_Object obj, char *field,
	       List items, List values);

/* Functions *****************************************************************/

static CBInfo *NewCBInfo(Ph_Object obj, char *field, Widget w, int i)
{
  CBInfo *data = Allocate(1, CBInfo);
  data->obj = obj;
  data->field = field;
  data->w = w;
  data->i = i;
  return data;
}

/* Drag callback for scale */
static void ScaleDrag(Widget w, CBInfo *data,
		      XmScaleCallbackStruct *scaleData)
{
  double value;
  char str[20];

  value = (double)scaleData->value;
  if(data->i) value /= pow(10.0, (double)data->i);
  sprintf(str, "%lg", value);
  if(Ph_ObjSetString(data->obj, data->field, str) == PH_ERROR) {
    fprintf(stderr, "Error setting field %s of %s\n", 
	    data->field, Ph_ObjName(data->obj));
  }
}

/* Watch callback for scale */
static void ScaleWatch(Ph_Object obj, char *field, CBInfo *data)
{
  int value;
  char *str;
  if(Ph_ObjGetString(obj, field, &str) == PH_ERROR) {
    fprintf(stderr, "Error getting field %s of %s\n", 
	    field, Ph_ObjName(obj));
    return;
  }
  /* must cast exponent to double for alphas */
  value = (int)(strtod(str, NULL)*pow(10.0, (double)data->i) + 0.5);
  free(str);
  XmScaleSetValue(data->w, value);
}

/* Create a Scale widget which is tethered to field of obj 
 * (which must parse as a double).
 * from and to are the minimum and maximum values of the field.
 * points is the number of digits after the decimal point to be displayed.
 */
Widget ScaleVar(Widget parent, Ph_Object obj, char *field, 
		double from, double to, int points)
{
  int n;
  Arg arg[10];
  Widget w;
  char name[100];
  CBInfo *info;

  n=0;
  XtSetArg(arg[n], XmNminimum, from*pow(10,(double)points)); n++;
  XtSetArg(arg[n], XmNmaximum, to*pow(10,(double)points)); n++;
  XtSetArg(arg[n], XmNdecimalPoints, points); n++;
  XtSetArg(arg[n], XmNtitleString, MakeXmString(field)); n++;
  sprintf(name, "%sScale", field);
  w = XmCreateScale(parent, name, arg, n);
  XtManageChild(w);

  info = NewCBInfo(obj, field, w, points);
  ScaleWatch(obj, field, info);
  Ph_ObjWatch(obj, field, (ObjFieldCallback*)ScaleWatch, info);
  XtAddCallback(w, XmNdragCallback, 
		(XtCallbackProc)ScaleDrag, info);
  XtAddCallback(w, XmNvalueChangedCallback, 
		(XtCallbackProc)ScaleDrag, info);
  /* let the garbage collector clean it up */
  ListAddRear(appData->garbage, info);
  return w;
}

/* Activation callback for Text */
static void TextActivate(Widget w, CBInfo *data,
			 XmAnyCallbackStruct *callData)
{
  char *str;
  str = XmTextGetString(w);
  MEM_ALLOC_NOTIFY(str);
  if(Ph_ObjSetString(data->obj, data->field, str) == PH_ERROR) {
    fprintf(stderr, "Error setting field %s of %s\n", 
	    data->field, Ph_ObjName(data->obj));
  }
  free(str);
}

/* Watch callback for Text */
static void TextWatch(Ph_Object obj, char *field, CBInfo *data)
{
  char *value;
  if(Ph_ObjGetString(obj, field, &value) == PH_ERROR) {
    fprintf(stderr, "Error getting field %s of %s\n",
	    field, Ph_ObjName(obj));
    return;
  }
  XmTextSetString(data->w, value);
  free(value);
}

/* Create a Text widget which is tethered to field of obj */
Widget TextVar(Widget parent, Ph_Object obj, char *field)
{
  int n;
  Arg arg[10];
  Widget w, pane;
  char name[100];
  CBInfo *info;

  /* create a rowcol to hold the text entry and its label */
  pane = XmCreateRowColumn(parent, "TextRowCol", NULL, 0);

  w = XmCreateLabel(pane, field, NULL, 0);
  XtManageChild(w);
  n=0;
  sprintf(name, "%sText", field);
  w = XmCreateText(pane, name, arg, n);
  XtManageChild(w);
  XtManageChild(pane);

  info = NewCBInfo(obj, field, w, 0);
  TextWatch(obj, field, info);
  Ph_ObjWatch(obj, field, (ObjFieldCallback*)TextWatch, info);
  XtAddCallback(w, XmNactivateCallback, 
		(XtCallbackProc)TextActivate, info);
  /* let the garbage collector clean it up */
  ListAddRear(appData->garbage, info);
  return w;
}

/* Activation callback for menu options */
static void MenuActivate(Widget w, CBInfo *data,
			 XmAnyCallbackStruct *callData)
{
  char *str;
  XtVaGetValues(w, XmNuserData, &str, NULL);
  if(Ph_ObjSetString(data->obj, data->field, str) == PH_ERROR) {
    fprintf(stderr, "Error setting field %s of %s\n", 
	    data->field, Ph_ObjName(data->obj));
  }
}

/* Watch callback for Menu */
static void MenuWatch(Ph_Object obj, char *field, CBInfo *data)
{
  int i,n;
  Widget w, *children;
  char *value, *v;
  if(Ph_ObjGetString(obj, field, &value) == PH_ERROR) {
    fprintf(stderr, "Error getting field %s of %s\n",
	    field, Ph_ObjName(obj));
    return;
  }
  /* find which widget has that value */
  XtVaGetValues(data->w,
		XmNchildren, &children,
		XmNnumChildren, &n, NULL);
  for(i=0;i<n;i++) {
    XtVaGetValues(children[i], XmNuserData, &v, NULL);
    if(!strcmp(v, value)) break;
  }
  if(i == n) {
    fprintf(stderr, "MenuVar: value `%s' is not a menu item\n", value);
    return;
  }
  free(value);
  /* the PulldownMenu has the OptionMenu in its userData */
  XtVaGetValues(data->w, XmNuserData, &w, NULL);
  XtVaSetValues(w, XmNmenuHistory, children[i], NULL);
}

/* Create an OptionMenu which is tethered to field of obj.
 * The menu options are specified in items (strings).
 * The menu values are specified in values (strings).
 * If values is NULL, items is used instead.
 */
Widget MenuVar(Widget parent, Ph_Object obj, char *field,
	       List items, List values)
{
  int n;
  Arg arg[10];
  char menu_name[100], op_name[100];
  Widget menu, w;
  ListNode p1,p2;
  CBInfo *info;

  /* Create a pulldown menu widget, with children corresponding
   * to the contents of the items and values Lists.
   */
  n=0;
  /* XmCreatePulldownMenu requires the proper visual settings, because
   * it creates a new window.
   */
  XtSetArg(arg[n], XmNvisual, appData->visual); n++;
  XtSetArg(arg[n], XmNdepth, appData->depth); n++;
  XtSetArg(arg[n], XmNcolormap, appData->colormap); n++;
  sprintf(menu_name, "%sMenu", field);
  sprintf(op_name, "%sOption", menu_name);
  menu = XmCreatePulldownMenu(parent, menu_name, arg, n);
  info = NewCBInfo(obj, field, menu, 0);
  ListAddRear(appData->garbage, info);
  if(!values) values = items;
  for(p1=ListFront(items),p2=ListFront(values);
      p1 && p2;
      p1=p1->next,p2=p2->next) {
    char str[100], *item, *value;
    item = (char*)p1->data;
    value = strdup((char*)p2->data);
    ListAddFront(appData->garbage, value);
    n=0;
    XtSetArg(arg[n], XmNlabelString, MakeXmString(item)); n++;
    XtSetArg(arg[n], XmNuserData, value); n++;
    w = XmCreatePushButtonGadget(menu, op_name, arg, n);
    XtManageChild(w);
    XtAddCallback(w, XmNactivateCallback, 
		  (XtCallbackProc)MenuActivate, info);
  }
  /* create the OptionMenu parent of the PulldownMenu */
  n=0;
  XtSetArg(arg[n], XmNsubMenuId, menu); n++;
  XtSetArg(arg[n], XmNlabelString, MakeXmString(field)); n++;
  sprintf(menu_name, "%sOpMenu", field);
  w = XmCreateOptionMenu(parent, menu_name, arg, n);
  XtVaSetValues(XmOptionButtonGadget(w), XmNtraversalOn, FALSE, NULL);
  XtManageChild(w);
  /* store the OptionMenu widget in the PulldownMenu's userData */
  XtVaSetValues(menu, XmNuserData, w, NULL);
 
  MenuWatch(obj, field, info);
  Ph_ObjWatch(obj, field, (ObjFieldCallback*)MenuWatch, info);
  return menu;
}
