#include <xtpm/xtpm.h>
#include <stdlib.h>

/* Prototypes ****************************************************************/

XmString MakeXmString(String s);
void XtpmSetCursor(Widget w, int c);
void XtpmAppForceEvent(XtAppContext ac);
void XtpmDrawCenteredString(Display *dpy, GC gc, Pixmap pixmap, 
			    int width, int y, char *str);
char *XmStringToCString(XmString xs);

/* Functions *****************************************************************/

/* Convert a string with newlines into an XmString.
 * Returns the XmString.
 */
XmString MakeXmString(String s)
{
  return XmStringCreateLtoR(s, XmSTRING_DEFAULT_CHARSET);
}

/* Change the widget's cursor to c */
void XtpmSetCursor(Widget w, int c)
{
  if(!w || !XtIsRealized(w)) return;
  XDefineCursor(XtDisplay(w), XtWindow(w),
                (c == NoCursor) ? None :
                XCreateFontCursor(XtDisplay(w), c));
}

/* Dispatch all pending events; blocks until at least one event is processed */
void XtpmAppForceEvent(XtAppContext ac)
{
  do {
    XtAppProcessEvent(ac, XtIMAll);
  } while(XtAppPending(ac));
}

void XtpmDrawCenteredString(Display *dpy, GC gc, Pixmap pixmap, 
			    int width, int y, char *str)
{
  int strsize,x;
  XFontStruct *font_struct;

  strsize = strlen(str);
  font_struct = XQueryFont(dpy,XGContextFromGC(gc));
  if(font_struct == NULL) {
    fprintf(stderr, "XQueryFont returned NULL\n");
    return;
  }
  x = (width - XTextWidth(font_struct,str,strsize))/2;
  /* XFreeFont doesn't work when passed a FontStruct created
   * from a GContext, but we've got to free the memory somehow.
   * These lines were taken from the X11R5 source for XFreeFont.
   * The calls to free are designed to escape a memcheck warning.
   */
  if(font_struct->per_char) (free)(font_struct->per_char);
  if(font_struct->properties) (free)(font_struct->properties);
  (free)(font_struct);
/*
  XFreeFont(dpy, font_struct);
*/

  XDrawImageString(dpy, pixmap, gc,
		   x, y,
		   str,
		   strsize);
}

char *XmStringToCString(XmString xs)
{
  XmStringContext context;
  char *text = "(XmStringToCString failed)";
  XmStringCharSet charset;
  XmStringDirection direction;
  Boolean separator;
  int status;

  status = XmStringInitContext(&context, xs);
  while(status) {
    status = XmStringGetNextSegment(context, &text, 
                                    &charset, &direction, &separator);
    break;
  }
  return text;
}
