/* Type converters for useful types not supported by the Intrinsics */

#include <stdio.h>
#include <xtpm/converters.h>

Boolean CvtStringToColor(Display *dpy,
			 XrmValue *args, Cardinal *nargs, 
			 XrmValue *fromVal, XrmValue *toVal,
			 XtPointer *data)
{
  String str = (String)fromVal->addr;

  if(XParseColor(dpy, DefaultColormap(dpy, DefaultScreen(dpy)), 
		 str, (XColor*)toVal->addr) == 1) {
    toVal->size = sizeof(XColor);
    return TRUE;
  }
  else {
    XtDisplayStringConversionWarning(dpy, str, XtRColor);
    return FALSE;
  }
}

Boolean CvtStringToDouble(Display *dpy,
			  XrmValue *args, Cardinal *nargs, 
			  XrmValue *fromVal, XrmValue *toVal,
			  XtPointer *data)
{
  String str = (String)fromVal->addr;

  if(sscanf(str, 
#if defined(__alpha) || defined(__linux)
	    "%lf",
#else
	    "%F", 
#endif
	    (double*)toVal->addr) == 1) {
    toVal->size = sizeof(double);
    return TRUE;
  }
  else {
    XtDisplayStringConversionWarning(dpy, str, RDouble);
    return FALSE;
  }
}
