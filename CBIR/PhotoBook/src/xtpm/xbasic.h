/* Fundemental X library headers, with workarounds for known compiler 
 * difficulties. 
 */
#ifndef XBASIC_H_INCLUDED
#define XBASIC_H_INCLUDED

/* C* has trouble with Xlib.h because of it's use of "current". 
 * This kludges around the problem. 
 */
#ifdef cstar
#  define current __CURRENT
#endif

/* X includes */
#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#include <X11/cursorfont.h>
#include <X11/Shell.h>

#ifdef cstar
#  undef current
#endif

/* C* also barfs on uses of the standard X macro XtOffset */
#ifdef cstar
#  undef XtOffset
#  define XtOffset(v,f) (Cardinal)((String)&v.f - (String)&v)
#endif

#include <xtpm/xcheck.h>

#endif
