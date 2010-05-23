#ifndef XBASICP_H_INCLUDED
#define XBASICP_H_INCLUDED

/* C* has trouble with Xlib.h because of it's use of "current". */
#ifdef cstar
#  define current __CURRENT
#endif

/* X includes */
#include <X11/IntrinsicP.h>

#ifdef cstar
#  undef current
#endif

#endif
