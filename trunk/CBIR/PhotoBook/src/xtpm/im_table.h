#ifndef IMTABLE_H_INCLUDED
#define IMTABLE_H_INCLUDED

#include <xtpm/xtpm.h>
#include <tpm/list.h>

typedef struct {
  int width, height;
} XtpmITRec;

Widget XtpmITCreate(Widget parent, char *name, List im_rows,
		    XtCallbackProc cb, void *userData);
void XtpmITSetGC(Widget imt, GC gc);
void XtpmITSetWPixmap(Widget imt, Widget w, Pixmap pix);
Pixmap XtpmITGetWPixmap(Widget imt, Widget w);
Widget XtpmITWidget(Widget imt, int row, int col);
void XtpmITFree(Widget imt);

#endif
