/* Grid Widget */
/* The Grid widget is provided for easy display of a resizable block of pixels.
 * The widget will zoom each pixel to accomodate window sizes larger than
 * gridWidth by gridHeight. Window area not filled by this fixed scaling factor
 * will be black, so resize increments of gridWidth and gridHeight are
 * recommended.
 * The output is cached in a pixmap, for fast redraw after an Expose event.
 * This pixmap is only regenerated after a Resize event, SetValues call,
 * or GridUpdate call (utility function).
 *
 * Resources:
 *   Ngrid       - int vector (default NULL)
 *   NgridWidth  - horizontal dimension (default 0)
 *   NgridHeight - vertical dimension (default 0)
 *   Nprogress   - TRUE to display the regeneration process (default FALSE)
 *   NcolorTable - ColorTable for mapping pixel values to colors (default NULL)
 *
 * Callbacks:
 *   None
 *
 * Utility functions:
 *   GridUpdate(Widget) - Refreshes the grid display.
 */

#include <stdio.h>
#include <malloc.h>
#include <xtpm/xbasicP.h>
#include <xtpm/xbasic.h>
#include <xtpm/GridP.h>
#include <xtpm/Grid.h>

static void xi_draw_grid(GridWidget w);
static void nb_draw_grid(GridWidget w);
static void cb_draw_grid(GridWidget w);
static void Initialize(GridWidget request, GridWidget new);
static void Redisplay(GridWidget w, XExposeEvent *event, Region region);
static void Resize(GridWidget w);
static void Destroy(GridWidget w);
static Boolean SetValues(GridWidget cur, GridWidget request,
			 GridWidget new);	

#ifdef cstar
static struct _GridRec base;
#else
typedef struct _GridRec *base;
#endif

static XtResource resources[]={
  { Ngrid, CGrid, XtRString, sizeof(String),
      XtOffset(base, grid.grid_data), XtRString, NULL },
  { NgridWidth, CGridWidth, XtRInt, sizeof(int),
      XtOffset(base, grid.grid_width), XtRImmediate, (caddr_t)0 },
  { NgridHeight, CGridHeight, XtRInt, sizeof(int),
      XtOffset(base, grid.grid_height), XtRImmediate, (caddr_t)0 },
  { NcolorTable, CColorTable, XtRString, sizeof(int*),
      XtOffset(base, grid.color_table), XtRString, NULL },
  { Nprogress, CProgress, XtRBoolean, sizeof(Boolean),
      XtOffset(base, grid.progress), XtRString, "FALSE" },
  { NminValue, CMinValue, XtRInt, sizeof(int),
      XtOffset(base, grid.min_value), XtRString, "0" },
  { NmaxValue, CMaxValue, XtRInt, sizeof(int),
      XtOffset(base, grid.max_value), XtRString, "255" },
};

GridClassRec gridClassRec = {
  /* CoreClassPart */
  {
    (WidgetClass)&widgetClassRec,         /* superclass */
    "Grid",                               /* class_name */
    sizeof(GridRec),                      /* widget_size */
    NULL,                                 /* class_initialize */
    NULL,                                 /* class_part_initialize */
    FALSE,                                /* class_inited */
    (XtInitProc)Initialize,               /* initialize */
    NULL,                                 /* initialize_hook */
    XtInheritRealize,                     /* realize */
    NULL,                                 /* actions */
    0,                                    /* num_actions */
    resources,                            /* resources */
    XtNumber(resources),                  /* num_resources */
    NULLQUARK,                            /* xrm_class */
    TRUE,                                 /* compress_motion */
    TRUE,                                 /* compress_exposure */
    TRUE,                                 /* compress_enterleave */
    TRUE,                                 /* visible_interest */
    (XtWidgetProc)Destroy,                /* destroy */
    (XtWidgetProc)Resize,                 /* resize */
    (XtExposeProc)Redisplay,              /* expose */
    (XtSetValuesFunc)SetValues,           /* set_values */
    NULL,                                 /* set_values_hook */
    XtInheritSetValuesAlmost,             /* set_values_almost */
    NULL,                                 /* get_values_hook */
    NULL,                                 /* accept_focus */
    XtVersion,                            /* version */
    NULL,                                 /* callback private */
    NULL,                                 /* tm_table */
    NULL,                                 /* query_geometry */
    NULL,                                 /* display_accelerator */
    NULL,                                 /* extension */
  },
  /* GridClassPart */
  {
    0,
  },
};

WidgetClass gridWidgetClass = (WidgetClass)&gridClassRec;

void GridUpdate(Widget w)
{
  GridWidget g = (GridWidget)w;

  if(g->grid.grid_width <= 0) {
    XtWarning("Invalid grid width");
    return;
  }
  if(g->grid.grid_height <= 0) {
    XtWarning("Invalid grid height");
    return;
  }
  Resize(g);
}

static void Initialize(GridWidget request, GridWidget new)
{
  Display *dpy = XtDisplay(new);

  if(request->core.width == 0)
    new->core.width = request->grid.grid_width;
  if(request->core.height == 0)
    new->core.height = request->grid.grid_height;

  /* Initialize private resources */
  new->grid.gc = XCreateGC(dpy, DefaultRootWindow(dpy), 0L, NULL);
  new->grid.pix = (Pixmap)NULL;
}

static void Redisplay(GridWidget w, XExposeEvent *event, Region region)
{
  if(w->grid.pix) {
    XCopyArea(XtDisplay(w), w->grid.pix, XtWindow(w), w->grid.gc,
	      event->x, event->y, event->width, event->height,
	      event->x, event->y);
  }
}

static void Resize(GridWidget w)
{
  Display *dpy;
  int width, height;

  width = w->core.width;
  height = w->core.height;
  dpy = XtDisplay(w);
  if(w->grid.pix) {
    XFreePixmap(dpy, w->grid.pix);
  }
  w->grid.pix = XCreatePixmap(dpy,
			    DefaultRootWindow(dpy),
			    width, height,
			    DefaultDepthOfScreen(XtScreen(w)));
  XSetForeground(dpy, w->grid.gc,
		 BlackPixelOfScreen(XtScreen(w)));
  XFillRectangle(dpy, w->grid.pix, w->grid.gc, 0,0, 
		 width, height);
  cb_draw_grid(w);
  if(XtIsRealized((Widget)w)) {
    XCopyArea(XtDisplay(w), w->grid.pix, XtWindow(w), w->grid.gc,
	      0,0, width, height, 0,0);
  }
}

/* XImage version */
static void xi_draw_grid(GridWidget w)
{
  int x, y, zoom_x, zoom_y, i, j, *p;
  XImage *xi;
  Display *dpy;

  dpy = XtDisplay(w);

  /* Calculate zooming factors */
  zoom_x = w->core.width / w->grid.grid_width;
  if(!zoom_x) zoom_x = 1;
  zoom_y = w->core.height / w->grid.grid_height;
  if(!zoom_y) zoom_y = 1;

  /* Allocate XImage */
  xi = XCreateImage(dpy, DefaultVisualOfScreen(XtScreen(w)),
		    DefaultDepthOfScreen(XtScreen(w)),
		    ZPixmap, 0, NULL, w->core.width, w->core.height, 32, 0);
  xi->data = (char *)XtMalloc(xi->bytes_per_line*w->core.height);

  /* Fill the XImage */
  p = w->grid.grid_data;
  for(j=0;j < w->grid.grid_height;j++) {
    for(y=0;y < zoom_y;y++) {
      for(i=0;i < w->grid.grid_width;i++) {
	for(x=0;x < zoom_x;x++) {
	  if(w->grid.color_table)
	    XPutPixel(xi, x+i*zoom_x, y+j*zoom_y, w->grid.color_table[*p]);
	  else
	    XPutPixel(xi, x+i*zoom_x, y+j*zoom_y, *p);
	}
	p++;
      }
      p-=w->grid.grid_width;
    }
    p+=w->grid.grid_width;
  }

  /* Copy XImage to Pixmap */
  XPutImage(dpy, w->grid.pix, w->grid.gc, xi, 
	    0,0,0,0, w->core.width, w->core.height);

  XtFree(xi->data);
  XFree((caddr_t)xi);
}

/* Non-buffered version */
static void nb_draw_grid(GridWidget w)
{
  int x, y, zoom_x, zoom_y, i, j, v;
  int *p;
  Display *dpy;
  Window win;
  int ncolors, buf_size, *count;
  XRectangle **rects, *rect;

  dpy = XtDisplay(w);
  win = XtWindow(w);

  zoom_x = w->core.width / w->grid.grid_width;
  if(!zoom_x) zoom_x = 1;
  zoom_y = w->core.height / w->grid.grid_height;
  if(!zoom_y) zoom_y = 1;

  p=w->grid.grid_data;
  for(j=y=0;j < w->grid.grid_height;j++,y+=zoom_y) {
    for(i=x=0;i < w->grid.grid_width;i++,x+=zoom_x) {
      v = *p++;
      if(w->grid.color_table) {
	v = w->grid.color_table[v];
      }
      XSetForeground(dpy, w->grid.gc, v);
      XFillRectangle(dpy, w->grid.pix, w->grid.gc, x, y, zoom_x, zoom_y);
      if(w->grid.progress && XtIsRealized((Widget)w)) {
	XFillRectangle(dpy, win, w->grid.gc, x, y, zoom_x, zoom_y);
      }
    }
  }
}

#define set_color \
if(w->grid.color_table) \
  XSetForeground(dpy, w->grid.gc, w->grid.color_table[v]); \
else \
  XSetForeground(dpy, w->grid.gc, v)

#define draw_rects \
XFillRectangles(dpy, w->grid.pix, w->grid.gc, rects[v], count[v]); \
if(w->grid.progress && XtIsRealized((Widget)w)) \
  XFillRectangles(dpy, win, w->grid.gc, rects[v], count[v])


/* Color-buffered version */
static void cb_draw_grid(GridWidget w)
{
  int x, y, zoom_x, zoom_y, i, j, v;
  int *p;
  Display *dpy;
  Window win;
  int ncolors, buf_size, *count;
  XRectangle **rects, *rect;

  dpy = XtDisplay(w);
  win = XtWindow(w);

  /* Calc zoom factors */
  zoom_x = w->core.width / w->grid.grid_width;
  if(!zoom_x) zoom_x = 1;
  zoom_y = w->core.height / w->grid.grid_height;
  if(!zoom_y) zoom_y = 1;

  /* Calc bin size */
  buf_size = (XMaxRequestSize(dpy)-3)/2;
  i = w->grid.grid_width*w->grid.grid_height;
  if(buf_size > i) buf_size = i;
  ncolors = w->grid.max_value - w->grid.min_value + 1;
  buf_size /= ncolors/16;
  if(!buf_size) buf_size = 1;
  for(;(buf_size>0) && 
      !(rects = (XRectangle**)malloc(ncolors*(sizeof(XRectangle*)+
					      buf_size*sizeof(XRectangle))))
      ;buf_size-=1024);
  if(buf_size <= 0) {
    fprintf(stderr, "cb_draw_grid: Out of memory\n");
    fflush(stderr);
    exit(2);
  }
  free(rects);

  /* Allocate the color bins */
  rects = (XRectangle**)XtMalloc(ncolors*sizeof(XRectangle*));
  count = (int*)XtMalloc(ncolors*sizeof(int));
  for(i=0;i<ncolors;i++) {
    count[i] = 0;
    rects[i] = (XRectangle*)XtMalloc(buf_size*sizeof(XRectangle));
  }

  /* Loop the image */
  p=w->grid.grid_data;
  for(j=y=0;j < w->grid.grid_height;j++,y+=zoom_y) {
    for(i=x=0;i < w->grid.grid_width;i++,x+=zoom_x) {
      v = (*p++) - w->grid.min_value;
      /* Add a new rectangle to the appropriate bin */
      rect = &rects[v][count[v]];
      rect->x = x;
      rect->y = y;
      rect->width = zoom_x;
      rect->height = zoom_y;
      count[v]++;
      /* If we've filled up the bin, empty it 
	 by drawing all rectangles inside */
      if(count[v] == buf_size) {
	set_color;
	draw_rects;
	count[v]=0;
      }
    }
  }
  /* Loop all bins and empty them */
  for(v=0;v<ncolors;v++) {
    set_color;
    draw_rects;
    XtFree((caddr_t)rects[v]);
  }
  XtFree((caddr_t)rects);
  XtFree((caddr_t)count);
}

static void Destroy(GridWidget w)
{
  XFreePixmap(XtDisplay(w), w->grid.pix);
  XtReleaseGC((Widget)w, w->grid.gc);
}

#define CHANGED(v) (new->v != cur->v)

static Boolean SetValues(GridWidget cur, GridWidget request,
			 GridWidget new)
{
  Boolean redraw = FALSE;

  if(CHANGED(grid.grid_data) || 
     CHANGED(grid.grid_width) || 
     CHANGED(grid.grid_height) ||
     CHANGED(grid.color_table))
    redraw = TRUE;
  return redraw;
}
