/* Definitions for standard converters */
#ifndef CONVERTERS_H_INCLUDED
#define CONVERTERS_H_INCLUDED

#include <xtpm/xbasic.h>

#define RDouble "Double"

/* Converters */
Boolean CvtStringToColor(Display *dpy,
			 XrmValue *args, Cardinal *nargs, 
			 XrmValue *fromVal, XrmValue *toVal,
			 XtPointer *data);
Boolean CvtStringToDouble(Display *dpy,
			  XrmValue *args, Cardinal *nargs, 
			  XrmValue *fromVal, XrmValue *toVal,
			  XtPointer *data);

#endif
