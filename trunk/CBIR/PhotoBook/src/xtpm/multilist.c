#include <xtpm/multilist.h>

#if 0
XtpmMLCallback mlcb;

List mlcb(XtpmML ml, int pos)
{
  List result;

  printf("->mlcb\n");
  /* Case 1: rightmost list; user made a selection */
  if((pos == ml->num_lists-1) && ml->selected[pos]) {
    printf("selected: %s\n", XtpmMLSelection(ml, pos));
    return NULL;
  }

  result = ListCreate(NULL);
  /* Case 2: first-time initialization */
  if(!ml->selected[pos]) {
    {ListNode ptr;ListIterate(ptr, (List)ml->userData) {
      ListAddRear(result, ((Association)ptr->data)->key);
    }}
    ml->selected[pos] = 1;
  }
  /* Case 3: user selection; define the next list */
  else {
    {ListNode ptr;
     ListIterate(ptr, 
		 (List)((Association)ListValueAtIndex(ml->userData, 
						ml->selected[pos]-1))->value) {
       ListAddRear(result, ptr->data);
     }}
    ml->selected[pos+1] = 1;
  }
  printf("<-mlcb\n");
  return result;
}
#endif

#define CString(text) XmStringLtoRCreate(text, XmSTRING_DEFAULT_CHARSET)
static void installWords(XtpmML ml, int pos, List list)
{
  XmString *xmstrings;
  int i;

  xmstrings = Allocate(ListSize(list), XmString);
  for(i=0;i<ListSize(list);i++) {
    xmstrings[i] = CString(ListValueAtIndex(list, i));
  }
  XtVaSetValues(ml->lists[pos],
		XmNitems, xmstrings,
		XmNitemCount, ListSize(list),
		NULL);
  free(xmstrings);
  if(ml->selected[pos] == 0) {
    fprintf(stderr, "Warning: ML callback didn't set the selection\n");
    ml->selected[pos] = 1;
  }
  XmListSelectPos(ml->lists[pos], ml->selected[pos], FALSE);
  ml->words[pos] = list;
}

static void XtpmMLHandler(Widget w, XtpmML ml, XmListCallbackStruct *callData)
{
  int pos;
  XmString *xmstrings;

  /* figure out which list this is */
  for(pos = 0;ml->lists[pos] != w; pos++);
  ml->selected[pos] = callData->item_position;

  if(!ml->selected[pos]) {
    /* no selection, so we must refresh ourselves */
    ListFree(ml->words[pos]);
    installWords(ml, pos, ((XtpmMLCallback*)ml->callback)(ml, pos));
  }
  else if(pos == ml->num_lists-1) {
    /* rightmost list */
    ((XtpmMLCallback*)ml->callback)(ml, pos);
    return;
  }
  /* tell all lists to the right to update */
  for(;pos < ml->num_lists-1;pos++) {
    ListFree(ml->words[pos+1]);
    installWords(ml, pos+1, ((XtpmMLCallback*)ml->callback)(ml, pos));
  }
}

static void browseHandler(Widget w, XtpmML ml, 
			  XmListCallbackStruct *callData)
{
  ml->selected[ml->num_lists-1] = callData->item_position;
}

XtpmML XtpmCreateML(Widget parent, String name, 
		    int num_lists, XtpmMLCallback cb, XtPointer userData)
{
  XtpmML ml = Allocate(1, struct XtpmMLRec);
  String str = Allocate(strlen(name)+10, char);
  int i,n;
  Arg arg[10];
  XmListCallbackStruct listData;

  ml->num_lists = num_lists;
  ml->callback = (XtCallbackProc)cb;
  ml->userData = userData;

  /* create the pane */
  sprintf(str, "%sPane", name);
  n = 0;
  ml->pane = XmCreateForm(parent, str, arg, n);

  /* create the lists */
  ml->words = Allocate(num_lists, List);
  ml->selected = Allocate(num_lists, int);
  ml->lists = Allocate(num_lists, Widget);
  for(i=0;i<num_lists;i++) {
    ml->selected[i] = 0;
    ml->words[i] = ListCreate(NULL);

    sprintf(str, "%sList%d", name, i+1);
    n = 0;
    XtSetArg(arg[n], XmNtopAttachment, XmATTACH_FORM); n++;
    XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_FORM); n++;
    if(i == 0) {
      XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
    }
    else {
      XtSetArg(arg[n], XmNleftAttachment, XmATTACH_WIDGET); n++;
      XtSetArg(arg[n], XmNleftWidget, ml->lists[i-1]); n++;
    }
    if(i == num_lists-1) {
      XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
    }
    ml->lists[i] = XmCreateScrolledList(ml->pane, str, arg, n);
    XtManageChild(ml->lists[i]);
    if(i == num_lists-1) {
      XtAddCallback(ml->lists[i], 
		    XmNbrowseSelectionCallback,
		    (XtCallbackProc)browseHandler, ml);
      XtAddCallback(ml->lists[i], 
		    XmNdefaultActionCallback,
		    (XtCallbackProc)XtpmMLHandler, ml);
    }
    else {
      XtAddCallback(ml->lists[i], 
		    XmNbrowseSelectionCallback,
		    (XtCallbackProc)XtpmMLHandler, ml);
    }
  }
  free(str);
  XtManageChild(ml->pane);

  /* call the handler to initialize */
  listData.item_position = 0;
  XtpmMLHandler(ml->lists[0], ml, &listData);

  return ml;
}

void XtpmFreeML(XtpmML ml)
{
  int i;

  XtDestroyWidget(ml->pane);
  free(ml->selected);
  for(i=0;i<ml->num_lists;i++) {
    ListFree(ml->words[i]);
  }
  free(ml->words);
  free(ml->lists);
  free(ml);
}
