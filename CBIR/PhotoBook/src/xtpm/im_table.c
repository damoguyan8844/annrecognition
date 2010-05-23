#include <xtpm/im_table.h>

/* Functions *****************************************************************/

static void ReBlt(Widget w, void *userData, XmAnyCallbackStruct *callData)
{
  Pixmap pixmap;

  XtVaGetValues(w, XmNuserData, &pixmap, NULL);
  if(!pixmap) {
    XClearWindow(XtDisplay(w), XtWindow(w));
  }
  else {
    Dimension width, height;
    Widget pane = (Widget)userData;
    GC gc;
    XtVaGetValues(w, XmNwidth, &width, XmNheight, &height, NULL);
    XtVaGetValues(pane, XmNuserData, &gc, NULL);
    if(!gc) {
      fprintf(stderr, "XtpmIT: you forgot to set the GC\n");
      exit(1);
    }
    XCopyArea(XtDisplay(w), pixmap, XtWindow(w),
	      gc,0,0,width,height,0,0);
  }
}

Widget XtpmITCreate(Widget parent, char *name, List im_rows,
		    XtCallbackProc cb, void *userData)
{
  Arg arg[10];
  int n, index;
  Widget pane, rows, row, w;
  List im_row;
  XtpmITRec *im;
  char str[100];

  /* ScrolledWindow */
  n=0;
  XtSetArg(arg[n], XmNscrollingPolicy, XmAUTOMATIC); n++;
  XtSetArg(arg[n], XmNuserData, NULL); n++;
  sprintf(str, "%sScrollW", name);
  pane = XmCreateScrolledWindow(parent, str, arg, n);

  /* Row manager */
  n=0;
  XtSetArg(arg[n], XmNmarginHeight, 0); n++;
  XtSetArg(arg[n], XmNmarginWidth, 0); n++;
  XtSetArg(arg[n], XmNspacing, 0); n++;
  XtSetArg(arg[n], XmNadjustLast, 0); n++;
  sprintf(str, "%sPane", name);
  rows = XmCreateRowColumn(pane, str, arg, n);
  index = 0;
  {ListIter(p, im_row, im_rows) {
    /* One row */
    n=0;
    XtSetArg(arg[n], XmNmarginHeight, 0); n++;
    XtSetArg(arg[n], XmNmarginWidth, 0); n++;
    XtSetArg(arg[n], XmNspacing, 0); n++;
    XtSetArg(arg[n], XmNadjustLast, 0); n++;
    XtSetArg(arg[n], XmNorientation, XmHORIZONTAL); n++;
    sprintf(str, "%sRowPane", name);
    row = XmCreateRowColumn(rows, str, arg, n);
    {ListIter(p2, im, im_row) {
      n=0;
      XtSetArg(arg[n], XmNwidth, im->width);  n++;
      XtSetArg(arg[n], XmNheight, im->height);  n++;
      XtSetArg(arg[n], XmNuserData, NULL); n++;
      XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
      XtSetArg(arg[n], XmNresizePolicy, XmRESIZE_NONE); n++;
      sprintf(str, "%sDrawArea", name);
      w = XmCreateDrawingArea(row, str, arg, n);  
      if(cb) XtAddCallback(w, XmNinputCallback, cb, (void*)(index++));
      XtAddCallback(w, XmNexposeCallback, (XtCallbackProc)ReBlt, pane);
      XtManageChild(w);
    }}
    XtManageChild(row);
  }}
  XtManageChild(rows);
  XtManageChild(pane);

  return pane;
}

void XtpmITSetGC(Widget imt, GC gc)
{
  XtVaSetValues(imt, XmNuserData, gc, NULL);
}

#if 0
Widget XtpmITWidget(Widget imt, int n)
{
  Widget rows;
  WidgetList rowList, colList;
  int num_rows, num_cols, i;

  XtVaGetValues(imt, XmNworkWindow, &rows, NULL);
  XtVaGetValues(rows, XmNnumChildren, &num_rows, XmNchildren, &rowList, NULL);
  for(i=0;i<num_rows;i++) {
    XtVaGetValues(rowList[i], XmNnumChildren, &num_cols, 
		  XmNchildren, &colList, NULL);
    if(n >= num_cols) {
      n -= num_cols;
      continue;
    }
    return colList[n];
  }
  return NULL;
}
#endif

Widget XtpmITWidget(Widget imt, int row, int col)
{
  Widget rows;
  WidgetList rowList, colList;
  int num_rows, num_cols;

  XtVaGetValues(imt, XmNworkWindow, &rows, NULL);
  XtVaGetValues(rows, XmNnumChildren, &num_rows, XmNchildren, &rowList, NULL);
  if(row >= num_rows) {
    /* printf("XtpmITWidget: bad row index (%d)\n", row); */
    return NULL;
  }
  XtVaGetValues(rowList[row], XmNnumChildren, &num_cols, 
		XmNchildren, &colList, NULL);
  if(col >= num_cols) {
    /* printf("XtpmITWidget: bad column index (%d)\n", col); */
    return NULL;
  }
  return colList[col];
}

void XtpmITSetWPixmap(Widget imt, Widget w, Pixmap pix)
{
  XtVaSetValues(w, XmNuserData, pix, NULL);
  if(XtIsRealized(w)) ReBlt(w, imt, NULL);
}

Pixmap XtpmITGetWPixmap(Widget imt, Widget w)
{
  Pixmap pix;
  XtVaGetValues(w, XmNuserData, &pix, NULL);
  return pix;
}

void XtpmITFree(Widget imt)
{
  int row, col;
  Widget w;
  Pixmap pix;

  /* flush events before we start removing things which may be involved
   * in said events.
   */
  XFlush(XtDisplay(imt));
/*
  XtpmAppForceEvent(XtWidgetToApplicationContext(imt));
*/
  /* Free all pixmaps */
  for(row=0;;row++) {
    for(col=0;;col++) {
      w = XtpmITWidget(imt, row, col);
      if(!w) break;
      pix = XtpmITGetWPixmap(imt, w);
      if(pix) XFreePixmap(XtDisplay(imt), pix);
    }
    if(!col) break;
  }
  XtDestroyWidget(imt);
}
