/* Include all of the X headers you'll probably ever need */
#ifndef XTPM_H_INCLUDED
#define XTPM_H_INCLUDED

#include <xtpm/xbasic.h>
#include <Xm/Xm.h>

/* Widgets */
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>
#include <Xm/ToggleB.h>
#include <Xm/ArrowB.h>
#include <Xm/MessageB.h>
#include <Xm/Text.h>
#include <Xm/BulletinB.h>
#include <Xm/List.h>
#include <Xm/Separator.h>
#include <Xm/Scale.h>
#include <Xm/RowColumn.h>
#include <Xm/DrawingA.h>
#include <Xm/ScrollBar.h>
#include <Xm/ScrolledW.h>
#include <Xm/Form.h>

#define XmSDC XmSTRING_DEFAULT_CHARSET
#define DoAllColors DoRed|DoGreen|DoBlue
#define NoCursor 1000
#define DestroyXImage(xi) { if(xi->data) { free(xi->data); xi->data=NULL; } XDestroyImage(xi); }

/* xtpm.c ********************************************************************/
XmString MakeXmString(String s);
void XtpmSetCursor(Widget w, int c);
void XtpmAppForceEvent(XtAppContext ac);
void XtpmDrawCenteredString(Display *dpy, GC gc, Pixmap pixmap, 
			    int width, int y, char *str);
char *XmStringToCString(XmString xs);

#endif
