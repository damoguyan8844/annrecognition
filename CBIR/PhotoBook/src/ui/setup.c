#include <math.h>
#include "photobook.h"
#include "ui.h"
#include <xtpm/converters.h>
#include <xtpm/colors.h>
#include <X11/Xmu/StdCmap.h>

/* Globals *******************************************************************/

#define RColorList "ColorList"

/* X Resources */
XtResource resources[]={
  { "cacheSize", "CacheSize", XtRInt, sizeof(int),
      XtOffset(AppData, cache_size), XtRString, "1048576" },
  { "gamma", "Gamma", RDouble, sizeof(double),
      XtOffset(AppData, gamma), XtRString, "1.0" },
  { "pickedColor", "PickedColor", XtRColor, sizeof(XColor),
      XtOffset(AppData, color[COLOR_PICKED]), XtRString, "red" },
  { "unpickedColor", "UnpickedColor", XtRColor, sizeof(XColor),
      XtOffset(AppData, color[COLOR_UNPICKED]), XtRString, "black" },
  { "textColor", "TextColor", XtRColor, sizeof(XColor), 
      XtOffset(AppData, color[COLOR_TEXT]), XtRString, "white" },
  { "textFont", "TextFont", XtRFont, sizeof(Font),
      XtOffset(AppData, text_font), XtRString, "6x13" },
  { "photoBgColor", "PhotoBgColor", XtRColor, sizeof(XColor),
      XtOffset(AppData, color[COLOR_PHOTOBG]), XtRString, "black" },
  { "catColors", "CatColors", RColorList, sizeof(List),
      XtOffset(AppData, colorList), XtRString, "" },
};

/* Prototypes ****************************************************************/

int ProcessArgs(int *argc_p, char **argv);
void InitDisplay(void);
Widget MakePopShell(void);
Widget MakePulldownMenu(Widget parent, List menu_list, 
			XtCallbackProc callback, void *userData,
			char *menu_default, Widget *def_widget);

/* private */
static void SetupX(void);
static void SetupWidgets(void);
static Widget MakeTextEntry(Widget pane);
static void Alloc24BitColor(XColor *color);
static Boolean CvtStringToColorList(Display *dpy,
				    XrmValue *args, Cardinal *nargs, 
				    XrmValue *fromVal, XrmValue *toVal,
				    XtPointer *data);

/* Functions *****************************************************************/

int ProcessArgs(int *argc_p, char **argv)
{
  /* Init the debugging flag */
  /* Debugging is on iff "-debug" is the last arg on the command line */
  if(!strcmp(argv[(*argc_p)-1], "-debug")) {
    debug = 1;
    (*argc_p)--;
  }
  else {
    debug = 0;
  }

  /* If no args, display usage information */
  if(*argc_p < 2) {
    printf("Usage:\n\nphotobook [options] <database>\n\n");
    printf("<database> is the name of the database under #/dbs in the FRAMER file (e.g. faces).\n");
    return 0;
  }

  /* Open X connection */
  appData->shell = XtAppInitialize(&appData->app_context, "Photobook",
				   NULL, 0, argc_p, argv, NULL, NULL, 0);

  /* Initial database should be the last argument in argv now */
  if(*argc_p < 2) {
    printf("Missing database name. Type `photobook' with no arguments for usage info.\n");
    return 0;
  }

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
  return 1;
}

void InitDisplay(void)
{
  if(debug) printf("->InitDisplay\n");
  SetupX();
  SetupWidgets();
  if(debug) printf("<-InitDisplay\n");
}

static void SetupWidgets(void)
{
  Arg arg[10];
  int n;
  Widget pane, w;
  XFontStruct *font_struct;

  /* Create the widgets */
  n=0;
  XtSetArg(arg[n], XmNvisual, appData->visual); n++;
  XtSetArg(arg[n], XmNdepth, appData->depth); n++;
  XtSetArg(arg[n], XmNcolormap, appData->colormap); n++;
  XtSetValues(appData->shell, arg, n);

  n=0;
  XtSetArg(arg[n], XmNvisual, appData->visual); n++;
  XtSetArg(arg[n], XmNdepth, appData->depth); n++;
  XtSetArg(arg[n], XmNcolormap, appData->colormap); n++;
  XtSetArg(arg[n], XmNtitle, "Photobook"); n++;
  appData->main_window = 
    XtCreatePopupShell("mainWindow", topLevelShellWidgetClass,
		       appData->shell, arg, n);
  
  n=0;
  pane = XmCreateForm(appData->main_window, "mainPane", arg, n);
  appData->form = pane;

  if(debug) printf("left pane\n");
  appData->left_pane = MakeLeftPane(pane);

  if(debug) printf("right pane\n");
  appData->right_pane = MakeRightPane(pane);

  appData->com_text = MakeTextEntry(pane);
  appData->pop_shell = NULL;

  XtManageChild(pane);

  /* Realize the widgets and establish a GC and colormap */
  XtPopup(appData->main_window, XtGrabNone);

  /* Create the GC */
  appData->gc = XCreateGC(appData->display,
			  XtWindow(appData->main_window), 0, NULL);
  XSetFont(appData->display, appData->gc, appData->text_font);
  font_struct = XQueryFont(appData->display, XGContextFromGC(appData->gc));
  if(font_struct == NULL) {
    fprintf(stderr, "XQueryFont returned NULL\n");
    exit(1);
  }
  appData->text_ascent = font_struct->ascent + 1;
  appData->text_height = font_struct->ascent + font_struct->descent + 2;
/*
  XFreeFont(appData->display, font_struct);
*/

  /* If the depth == 24, then SetupX() has already established the colormap.
   * Otherwise, we have to create one.
   */
  appData->color_table = NULL;
  if(appData->depth == 8) {
    /* Make a greyscale color table */
    appData->color_table = 
      GetNamedColorTable(appData->display, XtWindow(appData->main_window),
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
  XSetBackground(appData->display, appData->gc, 
                 appData->color[COLOR_PHOTOBG].pixel);
}

static void TextActivate(Widget w, void *userData,
			 XmAnyCallbackStruct *callData)
{
  DoSearch();
}

static Widget MakeTextEntry(Widget pane)
{
  int n;
  Arg arg[10];
  Widget w;

  /* The text entry widget has left and right attachments,
     so it will be resized with the main window */
  n=0;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_WIDGET);  n++; 
  XtSetArg(arg[n], XmNleftWidget, appData->left_pane);  n++; 
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNrightWidget, appData->right_pane); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_FORM);  n++; 
  w = XmCreateText(pane, "comText", arg, n);
  XtManageChild(w);
  /* This callback is activated when enter is pressed in the text widget */
  XtAddCallback(w, XmNactivateCallback,
		(XtCallbackProc)TextActivate, NULL);
  return w;
}

Widget MakePopShell(void)
{
  int n;
  Arg arg[10];
  Widget shell, w;

  /* Shell */
  n=0;
  XtSetArg(arg[n], XmNvisual, appData->visual); n++;
  XtSetArg(arg[n], XmNdepth, appData->depth); n++;
  XtSetArg(arg[n], XmNcolormap, appData->colormap); n++;
  shell = XtCreatePopupShell("popShell", overrideShellWidgetClass,
			     appData->main_window, arg, n);

  /* Label */
  n=0;
  w=XmCreateLabel(shell, "popLabel", arg, n);
  XtManageChild(w);
  appData->pop_label = w;

  /* Add translations for popping down, if not specified by user */
  XtAugmentTranslations(shell,
    XtParseTranslationTable("<Btn3Up>: XtMenuPopdown(popShell)"));

  return shell;
}

static void SetupX(void)
{
  int scr;
  XVisualInfo vlist;
  int i;
  int gotcmap;
  XStandardColormap *stdcmap;

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
  if(appData->depth == 8) 
    printf("Color images will be displayed in grayscale\n");

  /* Create colormap; try to get a standard map */
#if 0
  gotcmap = XmuLookupStandardColormap(appData->display, scr, 
				      appData->visual->visualid,
				      appData->depth,
				      XA_RGB_BEST_MAP,
				      1, 1);
  if(gotcmap) {
    gotcmap = XGetRGBColormaps(appData->display, appData->root,
			       &stdcmap, &i, XA_RGB_BEST_MAP);
    if(gotcmap) {
      if(debug) printf("using standard colormap\n");
      appData->colormap = stdcmap->colormap;
    }
  }
#else
  gotcmap = 0;
/* XmuStandardColormap is broken on our suns... */
#ifndef sun
  if((appData->depth == 24) && (appData->visual->class != TrueColor)) {
    if(debug) printf("trying to get standard colormap\n");
    stdcmap = XmuStandardColormap(appData->display, scr,
				  appData->visual->visualid,
				  appData->depth,
				  XA_RGB_BEST_MAP,
				  None, 255, 255, 255);
    if(stdcmap) {
      if(debug) printf("using standard colormap\n");
      appData->colormap = stdcmap->colormap;
      gotcmap = 1;
    }
  }
#endif
#endif
  if(!gotcmap) {
    if(debug) printf("not using standard colormap\n");
    if(appData->visual->class != TrueColor) {
      fprintf(stderr, "Warning: colors may not be accurate\n");
    }
    appData->colormap = 
      XCreateColormap(appData->display, appData->root,
		      appData->visual, AllocNone);
  }

  appData->gamma_table = NULL;
  if(appData->gamma != 1.0) {
    /* gamma correction table */
    appData->gamma_table = Allocate(256, int);
    for(i=0;i<256;i++) {
      appData->gamma_table[i] = 
	(int)(pow((double)i/255, 1/appData->gamma)*255+0.5);
    }
  }

  for(i=0;i<NUM_COLORS;i++) {
    Alloc24BitColor(&appData->color[i]); 
  }
  {ListNode ptr;ListIterate(ptr, appData->colorList) {
    Alloc24BitColor((XColor*)ptr->data);
  }}
}

static void Alloc24BitColor(XColor *color)
{
/*
  color->pixel = (color->red << 16) + (color->green << 8) + color->blue;
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

static Boolean CvtStringToColorList(Display *dpy,
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

/* Create a pulldown menu whose labels and userData fields are the
 * data values in the menu_list.
 * def_widget = returned widget whose name matches menu_default.
 * callback = callback proc for each menu item.
 */
Widget MakePulldownMenu(Widget parent, List menu_list, 
			XtCallbackProc callback, void *userData,
			char *menu_default, Widget *def_widget)
{
  Arg arg[10];
  int n;
  Widget menu, w;
  ListNode listPtr;

  *def_widget = NULL;
  n=0;
  /* XmCreatePulldownMenu requires the proper visual settings. (why??) */
  XtSetArg(arg[n], XmNvisual, appData->visual); n++;
  XtSetArg(arg[n], XmNdepth, appData->depth); n++;
  XtSetArg(arg[n], XmNcolormap, appData->colormap); n++;
  menu = XmCreatePulldownMenu(parent, "opMenu", arg, n);
  ListIterate(listPtr, menu_list) {
    n=0;
    XtSetArg(arg[n], XmNlabelString, MakeXmString((String)listPtr->data)); n++;
    XtSetArg(arg[n], XmNuserData, listPtr->data); n++;
    w = XmCreatePushButtonGadget(menu, "opMenuOption", arg, n);
    XtManageChild(w);
    XtAddCallback(w, XmNactivateCallback, callback, userData);
    if(menu_default) {
      if(!strcmp(menu_default, (String)listPtr->data)) *def_widget = w;
    }
  }
  return menu;
}

