#include <math.h>
#include "photobook.h"
#include "ui.h"
#include <xtpm/colors.h>
#include <xtpm/converters.h>

/* Globals *******************************************************************/

struct AppDataStruct theAppData;
AppData appData = &theAppData;
Ph_Handle phandle;

#define RColorList "ColorList"

/* X Resources */
XtResource resources[]={
  { "cacheSize", "CacheSize", XtRInt, sizeof(int),
      XtOffset(AppData, cache_size), XtRString, "128" },
  { "gamma", "Gamma", RDouble, sizeof(double),
      XtOffset(AppData, gamma), XtRString, "1.0" },
  { "pickedColor", "PickedColor", XtRColor, sizeof(XColor),
      XtOffset(AppData, color[COLOR_PICKED]), XtRString, "red" },
  { "unpickedColor", "UnpickedColor", XtRColor, sizeof(XColor),
      XtOffset(AppData, color[COLOR_UNPICKED]), XtRString, "black" },
  { "textColor", "TextColor", XtRColor, sizeof(XColor), 
      XtOffset(AppData, color[COLOR_TEXT]), XtRString, "white" },
  { "photoBgColor", "PhotoBgColor", XtRColor, sizeof(XColor),
      XtOffset(AppData, color[COLOR_PHOTOBG]), XtRString, "black" },
  { "catColors", "CatColors", RColorList, sizeof(List),
      XtOffset(AppData, colorList), XtRString, "" },
};

/* Prototypes ****************************************************************/

void ProcessArgs(int *argc_p, char **argv);
void InitDisplay(void);
void SetupX(void), SetupWidgets(void);
Boolean CvtStringToColorList(Display *dpy,
                             XrmValue *args, Cardinal *nargs, 
                             XrmValue *fromVal, XrmValue *toVal,
                             XtPointer *data);

/* Functions *****************************************************************/

void main(int argc, char *argv[])
{
  char *db_name = "pieces";
  char *view_name = "image";
  List members;
  int i;
  Ph_Object obj;
  Ph_Image image;

  ProcessArgs(&argc, argv);
  phandle = Ph_Startup();
  InitDisplay();

  Ph_SetDatabase(phandle, db_name);
  obj = Ph_SetView(phandle, view_name);
  Ph_ObjSetString(obj, "height", "64");
  Ph_ObjSetString(obj, "width", "64");
  Ph_ObjSetString(obj, "channels", "3");
  Ph_ObjSetString(obj, "field", "image");

  appData->num_rows = 8;
  appData->num_cols = 8;
  SetupWidgets();

  /* Racing... we must establish the GC after the shell has a window
   * but before the ImTable gets drawn.
   */
  XtRealizeWidget(appData->shell);
  appData->gc = XCreateGC(appData->display,
			  XtWindow(appData->shell), 0, NULL);
  XSetBackground(appData->display, appData->gc, 
                 appData->color[COLOR_PHOTOBG].pixel);
  XtpmITSetGC(appData->imt, appData->gc);

  if(appData->depth == 8) {
    appData->color_table = 
      GetNamedColorTable(appData->display, XtWindow(appData->shell),
                         "grey", 16, 256, appData->gamma, 
                         CT_BEST, &appData->colormap);
    if(!appData->color_table) {
      printf("No color table\n");
      exit(1);
    }
    MEM_ALLOC_NOTIFY(appData->color_table);
    appData->color[COLOR_PHOTOBG].pixel = appData->color_table[0];
    appData->color[COLOR_TEXT].pixel = appData->color_table[255];
    appData->color[COLOR_UNPICKED].pixel = appData->color[COLOR_PHOTOBG].pixel;
    appData->color[COLOR_PICKED].pixel = appData->color_table[255];
  }

  appData->start_index = 0;
  members = Ph_GetMembers(phandle);
  appData->members = ListToPtrArray(members, &appData->num_members);
  ListFree(members);

  appData->selected = Allocate(appData->num_members, int);
  for(i=0;i<appData->num_members;i++) appData->selected[i] = 0;
  appData->imp->num_pages = (int)((float)appData->num_members/
				  appData->page_size+0.5);
  UpdateDisplay();

  XtAppMainLoop(appData->app_context);
}

void ProcessArgs(int *argc_p, char **argv)
{
  /* Open X connection */
  appData->shell = XtAppInitialize(&appData->app_context, "Photobook",
				   NULL, 0, argc_p, argv, NULL, NULL, 0);
  XtAppSetTypeConverter(appData->app_context,
                        XtRString, RDouble, CvtStringToDouble, 
                        NULL, 0, XtCacheNone, NULL);
  XtAppSetTypeConverter(appData->app_context,
                        XtRString, XtRColor, CvtStringToColor, 
                        NULL, 0, XtCacheNone, NULL);
  XtAppSetTypeConverter(appData->app_context,
                        XtRString, RColorList, CvtStringToColorList, 
                        NULL, 0, XtCacheNone, NULL);
  /* Read in the X resources */
  XtGetApplicationResources(appData->shell, appData, 
			    resources, XtNumber(resources), NULL, 0);
}

void InitDisplay(void)
{
  SetupX();
  PixCacheCreate();
  PixCacheSize(appData->cache_size);
}

void UpdateDisplay(void)
{
  XtpmIPUpdate(appData->imp);
}

static void Quit(Widget w, void *userData, XmAnyCallbackStruct *callData)
{
  free(appData->selected);
  LabelFree();
  XtpmIPFree(appData->imp);
  free(appData->members);
  PixCacheFree();
  free(appData->color_table);
  ListFree(appData->colorList);
  Ph_Shutdown(phandle);
  exit(0);
}

static Pixmap PhotoLoad(int page, int row, int col, void *userData)
{
  int i, width, height, label;
  Ph_Member member;
  Pixmap pixmap;
  Ph_Object view;
  XColor *color;
  int *labels;
  Widget w;

  view = Ph_GetView(phandle);
  Ph_ObjGet(view, "width", &width);
  Ph_ObjGet(view, "height", &height);

  i = page*appData->page_size + row*appData->num_rows + col;
  w = XtpmITWidget(appData->imt, row, col);
  XtVaSetValues(w, XmNborderColor,
		appData->selected[i] ?
		appData->color[COLOR_PICKED].pixel :
		appData->color[COLOR_UNPICKED].pixel,
		NULL);
  member = appData->members[i];
  pixmap = GetMemPixmap(member);

  /* draw the label under the image */
  labels = Ph_MemLabels(member);
  for(label=0;label<ListSize(appData->labels);label++) {
    if(!labels[label]) continue;
    color = ListValueAtIndex(appData->colorList, label);
    assert(color);
    XSetForeground(appData->display, appData->gc, color->pixel);
    XFillRectangle(appData->display, pixmap, appData->gc,
		   0, height, width, TEXTHEIGHT);
    XSetForeground(appData->display, appData->gc, 
		   appData->color[COLOR_TEXT].pixel);
    XtpmDrawCenteredString(appData->display, appData->gc, pixmap, 
			   width, height + TEXTHEIGHT,
			   ListValueAtIndex(appData->labels, label));
  }
  free(labels);
  
  return pixmap;
}

/* callback for image widgets */
#define MemIndex(i) (appData->imp->page*appData->page_size+(i))
static void PhotoCB(Widget w, int i, XmAnyCallbackStruct *callData)
{
  if(callData->event->type != ButtonPress) return;
  if(callData->event->xbutton.button == Button1) {
    appData->selected[MemIndex(i)] = !appData->selected[MemIndex(i)];
    XtVaSetValues(w, XmNborderColor,
		  appData->selected[MemIndex(i)] ?
		  appData->color[COLOR_PICKED].pixel :
		  appData->color[COLOR_UNPICKED].pixel,
		  NULL);
  }
}

static void PhotoTable(Widget parent)
{
  Widget w;
  List image_rows, image_col;
  XtpmITRec *p;
  int i,j;
  int width, height;
  Ph_Object view;

  view = Ph_GetView(phandle);
  Ph_ObjGet(view, "width", &width);
  Ph_ObjGet(view, "height", &height);

  appData->page_size = appData->num_rows * appData->num_cols;
  image_rows = ListCreate((FreeFunc*)ListFree);
  for(i=0;i<appData->num_rows;i++) {
    image_col = ListCreate(GenericFree);
    for(j=0;j<appData->num_cols;j++) {
      p = Allocate(1, XtpmITRec);
      p->width = width;
      p->height = height+TEXTHEIGHT;
      ListAddRear(image_col, p);
    }
    ListAddRear(image_rows, image_col);
  }
  appData->imt = XtpmITCreate(parent, "imTable", image_rows, 
			      (XtCallbackProc)PhotoCB, NULL);
  ListFree(image_rows);
}

void SetupWidgets(void)
{
  Arg arg[10];
  int n;
  Widget w, pane, subpane;

  n=0;
  XtSetArg(arg[n], XmNvisual, appData->visual); n++;
  XtSetArg(arg[n], XmNdepth, appData->depth); n++;
  XtSetArg(arg[n], XmNcolormap, appData->colormap); n++;
  XtSetValues(appData->shell, arg, n);

  n=0;
  pane = XmCreateForm(appData->shell, "mainPane", arg, n);

  PhotoTable(pane);

  /* button pane */
  n=0;
  XtSetArg(arg[n], XmNtopAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_FORM); n++;
  subpane = XmCreateRowColumn(pane, "buttonPane", arg, n);

  LabelMenu(subpane);
  LabelButtons(subpane);
  appData->imp = XtpmIPCreate(subpane, "photoTable", appData->imt, 0,
			      (XtpmIPLoad*)PhotoLoad, NULL);

  n=0;
  XtSetArg(arg[n], XmNtraversalOn, FALSE); n++;
  w = XmCreatePushButton(subpane, "quitButton", arg, n);
  XtAddCallback(w, XmNactivateCallback, (XtCallbackProc)Quit, NULL);
  XtManageChild(w);
  XtManageChild(subpane);

  n=0;
  XtSetArg(arg[n], XmNtopAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNrightWidget, subpane); n++;
  XtSetValues(appData->imt, arg, n);

  XtManageChild(pane);
}

static void Alloc24BitColor(XColor *color)
{
/*
  color->pixel = (color->red << 16) + (color->green << 8) + color->blue;
*/
/*
  XImage *ximage;

  ximage = XCreateImage(appData->display, appData->visual, appData->depth,
                        ZPixmap, 0, NULL, 1, 1, 32, 0);
  ximage->data=(char *)malloc(ximage->bytes_per_line);
  XPutPixel(ximage, 0, 0, 
            (color->red << 16) + (color->green << 8) + (color->blue));
  color->pixel = *(Pixel*)ximage->data;
  XDestroyImage(ximage);
*/
  if(XAllocColor(appData->display, appData->colormap, color)) return;
  if(debug) printf("R/O allocate failed\n");
  if(XAllocColorCells(appData->display, appData->colormap, 0,
                      NULL, 0, &color->pixel, 1)) {
    XStoreColor(appData->display, appData->colormap, color);
    return;
  }
  if(debug) printf("Alloc24BitColor failed for (%d,%d,%d)\n",
                   color->red,color->green,color->blue);
  color->pixel = 0;
}

void SetupX(void)
{
  int scr;
  XVisualInfo vlist;
  int i;

  appData->display = XtDisplay(appData->shell);
  appData->screen = XtScreen(appData->shell);
  appData->root = RootWindowOfScreen(appData->screen);

  /* Find best visual */
  scr = DefaultScreen(appData->display);
  if (!(XMatchVisualInfo(appData->display,
			 scr,
			 24, TrueColor, &vlist)))  {
    if (!(XMatchVisualInfo(appData->display,
			   scr,
			   24, DirectColor, &vlist)))  {
/*
      if (!(XMatchVisualInfo(appData->display,
			     scr,
			     8, PseudoColor, &vlist)))  {
*/
	vlist.visual = DefaultVisualOfScreen(appData->screen);
	vlist.depth = DefaultDepthOfScreen(appData->screen);
/*
      }
*/
    }
  }
  
  appData->visual = vlist.visual; 
  appData->depth = vlist.depth;
  printf("%d bits per pixel\n", appData->depth);
  if(appData->depth < 8) {
    printf("This screen type (depth %d) is not supported.\n");
    exit(1);
  }

  appData->colormap = 
    XCreateColormap(appData->display, appData->root,
		    appData->visual, AllocNone);

  if(appData->depth == 24) {
    /* gamma correction table */
    appData->color_table = Allocate(256, int);
    for(i=0;i<256;i++) {
      appData->color_table[i] = 
	(int)(pow((double)i/255, 1/appData->gamma)*255+0.5);
    }

    for(i=0;i<NUM_COLORS;i++) {
      Alloc24BitColor(&appData->color[i]); 
    }
    {ListNode ptr;ListIterate(ptr, appData->colorList) {
      Alloc24BitColor((XColor*)ptr->data);
    }}
  }
}

Boolean CvtStringToColorList(Display *dpy,
                             XrmValue *args, Cardinal *nargs, 
                             XrmValue *fromVal, XrmValue *toVal,
                             XtPointer *data)
{
  String str = (String)fromVal->addr, word;
  XrmValue from, to;
  XColor *color;
  
  toVal->size = sizeof(List);
  *(List*)toVal->addr = ListCreate(GenericFree);
  for(word = strtok(str, " ");word;word = strtok(NULL, " ")) {
    from.addr = word;
    from.size = strlen(word);
    to.addr = (XtPointer)Allocate(1,XColor);
    to.size = sizeof(XColor);
    if(XtConvertAndStore(appData->shell, XtRString, &from, XtRColor, &to)
       == FALSE) {
      XtAppError(appData->app_context, 
                 "CvtStringToColorList: String to Color conversion failed.\n");
      return FALSE;
    }
    ListAddRear(*(List*)toVal->addr, to.addr);
  }
  return TRUE;
}
