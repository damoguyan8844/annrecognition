#ifndef XCHECK_H_INCLUDED
#define XCHECK_H_INCLUDED

#if MEMCHECK

#ifdef AP
#undef AP
#endif

#if defined(__STDC__) || defined(__cplusplus)
#define AP(x) x
#else
#define AP(x) ()
#endif

Pixmap _xc_XCreatePixmap AP((Display *dpy, Drawable d, 
			     unsigned width, unsigned height,
			     unsigned depth,
			     char *file, unsigned line));
void _xc_XFreePixmap AP((Display *dpy, Pixmap p, char *file, unsigned line));

#define XCreatePixmap(dpy, d, w, h, depth) \
       _xc_XCreatePixmap(dpy, d, w, h, depth, __FILE__, __LINE__)
#define XFreePixmap(dpy, p) _xc_XFreePixmap(dpy, p, __FILE__, __LINE__)

#define XC_ABORT_ON_ERROR(flag) _xc_abort_on_error(flag)
#define XC_ALLOC_NOTIFY(dpy, p) _xc_alloc_notify(dpy, p, __FILE__, __LINE__)
#define XC_FREE_NOTIFY(dpy, p) _xc_free_notify(dpy, p, __FILE__, __LINE__)
#define XC_BLOCKS() _xc_x_nodes(__FILE__, __LINE__)

#else

#define XC_ABORT_ON_ERROR(flag)
#define XC_ALLOC_NOTIFY(dpy, p)
#define XC_FREE_NOTIFY(dpy, p)
#define XC_BLOCKS()

#endif
#endif
