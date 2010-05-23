#include <xtpm/photo_table.h>

/* Prototypes ****************************************************************/

XtpmIP XtpmIPCreate(Widget parent, char *name);
void XtpmIPFree(XtpmIP imp);
void XtpmIPUpdate(XtpmIP imp);
int XtpmIPJumpTo(XtpmIP imp, int page);
void XtpmIPSetIT(XtpmIP imp, Widget imt, int num_images, 
		 XtpmIPLoad *loadFunc, void *userData);
void XtpmIPStartCallback(XtpmIP imp, XtpmIPCallback *callback, void *userData);
void XtpmIPStopCallback(XtpmIP imp, XtpmIPCallback *callback, void *userData);

/* Functions *****************************************************************/

/* Maps and sensitizes w according to flag */
static void SetArrow(Widget w, int flag)
{
  if(flag) {
    XMapWindow(XtDisplay(w), XtWindow(w));
    XtSetSensitive(w, TRUE);
  }
  else {
    XtSetSensitive(w, FALSE);
    XUnmapWindow(XtDisplay(w), XtWindow(w));
  }
}

/* WorkProc used by XtpmIPUpdate to refresh the images */
static int UpdateIter(XtpmIP imp)
{
  Widget w;
  Pixmap pixmap;

  for(;;) {
    w = XtpmITWidget(imp->imt, imp->row, imp->col);
    if(w) break;
    else if(!imp->col) goto endwork;
    else { imp->row++; imp->col=0; }
  }
  pixmap = imp->loadFunc(imp->page, imp->row, imp->col, imp->loadData);
  if(!pixmap) {
    SetArrow(imp->downArrow, 0);
    goto endwork;
  }
  XtSetSensitive(w, TRUE);
  XtpmITSetWPixmap(imp->imt, w, pixmap);
  imp->col++;
  return 0;

 endwork:
  imp->work_proc = 0;
  XtpmSetCursor(imp->imt, NoCursor);
  /* call the stop callback, if any */
  if(imp->stopCB) imp->stopCB(imp, imp->stop_data);
  return 1;
}

/* Refresh the IP */
void XtpmIPUpdate(XtpmIP imp)
{
  Widget w;
  int row, col;
  Pixmap pix;

  XtpmIPHalt(imp);
  /* call the start callback, if any */
  if(imp->startCB) imp->startCB(imp, imp->start_data);

  XtpmSetCursor(imp->imt, XC_gumby);
  SetArrow(imp->upArrow, imp->page > 0);
  SetArrow(imp->downArrow, (imp->page+1)*imp->page_size < imp->num_images);

  /* clear all pixmaps and make the photos insensitive */
  for(row=0;;row++) {
    for(col=0;;col++) {
      w = XtpmITWidget(imp->imt, row, col);
      if(!w) break;
      XtSetSensitive(w, FALSE);
      pix = XtpmITGetWPixmap(imp->imt, w);
      if(pix) XFreePixmap(XtDisplay(imp->imt), pix);
      XtpmITSetWPixmap(imp->imt, w, (Pixmap)NULL);
    }
    if(!col) break;
  }

  /* start the workproc */
  imp->row = imp->col = 0;
  imp->work_proc = 
    XtAppAddWorkProc(XtWidgetToApplicationContext(imp->imt), 
		     (XtWorkProc)UpdateIter, imp);
}

/* Callback for up arrow */
static void PageUp(Widget w, XtpmIP imp, XmAnyCallbackStruct *callData)
{
  imp->page--;
  XtpmIPUpdate(imp);
}

/* Callback for down arrow */
static void PageDown(Widget w, XtpmIP imp, XmAnyCallbackStruct *callData)
{
  imp->page++;
  XtpmIPUpdate(imp);
}

/* Sets the displayed page to "page".
 * Returns 1 if the page is out of bounds.
 */
int XtpmIPJumpTo(XtpmIP imp, int page)
{
  if((page < 0) || (page*imp->page_size >= imp->num_images)) {
/*
    fprintf(stderr, "XtpmIPJumpTo: bad page number (%d)\n", page);
*/
    return 1;
  }
  imp->page = page;
  XtpmIPUpdate(imp);
  return 0;
}

/* Sets the image table for imp to use */
void XtpmIPSetIT(XtpmIP imp, Widget imt, int num_images, 
		 XtpmIPLoad *loadFunc, void *loadData)
{
  int row,col;
  XtpmIPHalt(imp);
  imp->imt = imt;
  imp->loadFunc = loadFunc;
  imp->loadData = loadData;
  imp->num_images = num_images;
  /* compute size of imt */
  imp->page_size = 0;
  for(row=0;;row++) {
    for(col=0;;col++) {
      if(XtpmITWidget(imt, row, col)) imp->page_size++;
      else break;
    }
    if(!col) break;
  }
  imp->page = 0;
  imp->work_proc = 0;
}

/* Sets the start callback, which will be called just before page refreshes.
 */
void XtpmIPStartCallback(XtpmIP imp, XtpmIPCallback *callback, void *userData)
{
  imp->startCB = callback;
  imp->start_data = userData;
}

/* Sets the stop callback, which will be called after page refreshes.
 */
void XtpmIPStopCallback(XtpmIP imp, XtpmIPCallback *callback, void *userData)
{
  imp->stopCB = callback;
  imp->stop_data = userData;
}

/* Creates a new XtpmIP. Places arrow buttons under parent. */
XtpmIP XtpmIPCreate(Widget parent, char *name)
{
  XtpmIP imp;
  int n;
  Arg arg[10];
  Widget w;
  char str[100];
  
  imp = (XtpmIP)malloc(sizeof(*imp));
  imp->startCB = imp->stopCB = NULL;
  imp->work_proc = 0;
  imp->imt = NULL;

  n=0;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  sprintf(str, "%sUpArrow", name);
  w = XmCreateArrowButton(parent, str, arg, n);
  XtManageChild(w);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)PageUp, imp);
  imp->upArrow = w;

  n=0;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  XtSetArg(arg[n], XmNarrowDirection, XmARROW_DOWN); n++;
  sprintf(str, "%sDownArrow", name);
  w = XmCreateArrowButton(parent, str, arg, n);
  XtManageChild(w);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)PageDown, imp);
  imp->downArrow = w;

  return imp;
}

/* Destroys an IP, including its arrow buttons, but not the IT. */
void XtpmIPFree(XtpmIP imp)
{
  XtpmIPHalt(imp);
  XtDestroyWidget(imp->upArrow);
  XtDestroyWidget(imp->downArrow);
  free(imp);
}
