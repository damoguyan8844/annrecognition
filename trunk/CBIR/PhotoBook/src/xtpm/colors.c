/***************************************************************************
 * File: colors.c
 * Author: Thomas Minka
 * Date: 12/1/93
 *
 *   Copyright 1993 by Massachusetts Institute of Technology.
 *   All rights reserved.
 *
 * Description: ColorTable/colormap allocation routines
 *              Requires the X11 and math libraries (-lX11 -lm)
 * Compilation: cc myXcode.c colors.c -lX11 -lm
 *
 ***************************************************************************/

/* A ColorTable is a mapping from integers to pixel values, designed
 * to allow hardware-independent easily indexable smooth palettes.
 * This way you can bind index 0 to black, index 255 to white,
 * and everything in between to shades of grey, on any display.
 *
 * Example:
 *   ColorTable ct = GetNamedColorTable(dpy, win, "grey", 0, 256, 
 *                                      1.0, 0);
 * makes
 *   ct[0] = black, ct[255] = white, ct[128] = mid-grey
 * so that
 *   XSetForeground(dpy, gc, ct[0]);
 * will set the drawing color to black.
 *   
 * This module supports ColorTables up to 256 entries (ncolors <= 256).
 * The "keep" parameter is used to preserve colors in the low end 
 * of the palette (which are used heavily by window managers). It does not
 * affect the resulting ColorTable, but it restricts the number of color cells
 * available. keep = -1 preserves everything and is useful if you don't want
 * other windows to change color.
 */

/* GetNamedColorTable returns a built-in ColorTable selected by name.
 * GetPivotColorTable returns a ColorTable which is a smooth blend 
 * between each of the given "pivot" colors, in order.
 */

/* ExpandColors enlarges an array of XColor structures using InterpolateColors.
 * InterpolateColors creates a smooth color gradient between two colors, using
 * linear interpolation.
 */

/* Globals *******************************************************************/

#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <X11/Intrinsic.h>

#define DoAllColors DoRed|DoGreen|DoBlue

/* Define DEBUG to 0 to disable debugging output */
#define DEBUG 0

#define MAXCOLORS 256
#define MAXPIVOTS 16
#define PTABLES 2

typedef int *ColorTable;
typedef enum { CT_BEST, CT_NICE, CT_WINDOW } CtMode;

/* Prototypes ****************************************************************/
ColorTable GetNamedColorTable(Display *dpy, Window w, String mapname, 
			      int keep, int ncolors, float gamma,
			      CtMode mode, Colormap *cmap_return);
ColorTable GetPivotColorTable(Display *dpy, Window w, 
			      XColor *pivots, int npivots,
			      int keep, int ncolors, float gamma,
			      CtMode mode, Colormap *cmap_return);
void ExpandColors(XColor *dest, int to_colors,
		  XColor *src, int from_colors);
void InterpolateColors(XColor *colors, XColor from, XColor to, int steps);
void GammaCorrectColors(XColor *colors, int ncolors, float gamma);

/* Private */
void RemoveElement(XColor *array, int *length, int index);
void RemoveValue(unsigned long *array, int *length, unsigned long value);
void GammaCorrectColors(XColor *colors, int ncolors, float gamma);

/* Functions *****************************************************************/

/* Get a named color table.
 * w = window which may receive a private colormap
 * mapname = built-in ColorTable name
 * keep = number of colors at front of default palette to not destroy
 * ncolors = total number of colors desired (size of ColorTable)
 * mode = color allocation strategy (see GetPivotColorTable)
 * cmap_return = returned colormap (NULL for no return)
 */
/* Currently available ColorTables:
 *   rainbow - six-pivot ROY G BV
 *   grey    - two-pivot B to W
 */
ColorTable GetNamedColorTable(Display *dpy, Window w, String mapname, 
			      int keep, int ncolors, float gamma,
			      CtMode mode, Colormap *cmap_return)
{
  /* Type of a MAXPIVOT-entry array of RGB values */
  typedef struct {
    unsigned short red, green, blue;
  } pivotlist[MAXPIVOTS];

  /* Array of built-in pivotlists */
  static struct {
    String name;
    int npivots;
    pivotlist pivots;
  } pivot_table[PTABLES]={
    { "rainbow", 6, {
      { 65535, 0, 0 }, { 65535, 42405, 0 }, { 65535, 65535, 0 },
      { 0, 65535, 0 }, { 0, 0, 65535}, { 41120, 8224, 61680 },
    } },

    { "grey", 2, {
      { 0,0,0 }, { 65535, 65535, 65535 },
    } },
  };
  XColor colors[MAXPIVOTS];/* XColor array corresponding to chosen pivotlist */
  int table;
  int i;
  
  /* Find which pivotlist is named by mapname */
  for(table=0;table<PTABLES;table++) {
    if(!strcmp(mapname, pivot_table[table].name)) break;
  }
  if(table == PTABLES) {
    printf("GetNamedColorTable error: Unknown color table\n");
    return NULL;
  }

  /* Copy the pivotlist into XColor array */
  for(i=0;i < pivot_table[table].npivots;i++) {
    colors[i].pixel=i;
    colors[i].red=pivot_table[table].pivots[i].red;
    colors[i].green=pivot_table[table].pivots[i].green;
    colors[i].blue=pivot_table[table].pivots[i].blue;
  }

  /* Let GetPivotColorTable do the rest */
  return GetPivotColorTable(dpy, w, colors, pivot_table[table].npivots, 
			    keep, ncolors, gamma, mode, cmap_return);
}

/* Create a color table based on blending between 'pivot' colors.
 * w = window which may receive a private colormap (must be mapped)
 * pivots = colors to blend between
 * npivots = number of pivots in "pivots"
 * keep = number of colors at front of default palette to not destroy
 *        -1 means keep everything (i.e. read-only cells only)
 * ncolors = number of colors desired beyond keep
 * mode: CT_WINDOW = try to allocate colors local to the widget
 *                   if the display does not support this, acts like CT_NICE
 *       CT_NICE = try to get read-only cells; unavailable colors
 *                 are put in a local colormap, if possible
 *       CT_BEST = try to get read-only cells; then try r/w cells; then
 *                 make local colormap, if possible
 * cmap_return = returned local colormap, if any (NULL for no return)
 *
 * output = array of ncolors entries, where the first keep colors
 *          are taken from the default colormap and the remaining
 *          colors are a blending across the pivot colors, from pivot
 *          zero to pivot (npivots-1).
 * output will be NULL in the case of errors.
 *
 * The algorithm used is based on a "failure set" of colors which weren't
 * allocated color cells. The failure set starts out as all requested
 * colors. Then the algorithm negotiates with the server as much as possible
 * to empty this set.
 *
 * CT_BEST Algorithm: proceed until the failure set is empty
 * 1. Request read-only cells which match colors in the failure set.
 *    If a match is found, remove the color from the set and change the
 *    corresponding ColorTable entry to point to the read-only cell.
 * 2. Request writeable cells which match colors in the failure set.
 *    If N writeable cells are available, remove the first N colors
 *    in the failure set, change the corresponding ColorTable entry to
 *    point to the cell, and write the desired color value into the cell.
 * 3. If the display supports local colormaps, create one and fill it
 *    with as many failed colors as possible.
 * 4. Map all failed colors to an adjacent mapped color.
 */
ColorTable GetPivotColorTable(Display *dpy, Window w, 
			      XColor *pivots, int npivots,
			      int keep, int ncolors, float gamma, 
			      CtMode mode, Colormap *cmap_return)
{
#define NOCOLOR -1
  Colormap def_cmap, cmap;
  XColor temp;
  int i, got, adjacent;
  XWindowAttributes win_attr;
  XVisualInfo *visual_list, *visual_info;
  Visual *visual;       /* default visual */
  int avail_colors;     /* size of display's colormap */
  int overflow;         /* ncolors - avail_colors */
  ColorTable ctable;    /* return value */
  unsigned long pixel;
  Boolean writeable;    /* TRUE iff the display has writeable colors */
  XColor failed[MAXCOLORS]; /* failure set */
  int nfailed;              /* size of failure set */
  unsigned long pixels[MAXCOLORS]; /* available pixel set */
  int npixels;                     /* size of pixel set */

  /* Initialize some basic X parameters */
  XGetWindowAttributes(dpy, w, &win_attr);
  avail_colors = CellsOfScreen(win_attr.screen);
  visual = win_attr.visual;
  def_cmap = DefaultColormapOfScreen(win_attr.screen);

  /* Examine the visual: is it writeable? */
  visual_list = XGetVisualInfo(dpy, VisualNoMask, NULL, &i);
  for(visual_info = visual_list;
      visual_info->visual != visual;
      visual_info++);
  /* Are R/W colormaps available? */
  writeable = visual_info->class & 1;
  /* Force CT_WINDOW to act like CT_NICE */
  if(!writeable && (mode == CT_WINDOW)) mode = CT_NICE;
  XFree((caddr_t)visual_list);

  /* Create color table */
  ctable = (ColorTable)malloc(ncolors*sizeof(int));
  if(!ctable) return ctable; /* not enough memory */
  /* Initialize to bogus values */
  for(i=0;i<ncolors;i++) ctable[i] = NOCOLOR;

  /* Keep everything? */
  if((keep > avail_colors) || (keep == -1)) keep = avail_colors;

  /* Requested too many colors? */
  if(overflow = ncolors - avail_colors) {  /* assignment */
#if DEBUG
    printf("Truncating request for %d colors to %d colors\n", 
	   ncolors, avail_colors);
#endif
    ncolors = avail_colors;
  }

  /* Initialize failure set */
  nfailed = ncolors;
  for(i=0;i<ncolors;i++) {
    failed[i].pixel=i;      /* using the pixel field as index into ctable */
    failed[i].flags=DoAllColors;
  }
  /* Initialize pixel set */
  npixels = avail_colors;
  for(i=0;i<npixels;i++) {
    pixels[i] = i;
  }
  /* Remove pixels to keep from pixel set */
  if(keep) {
    XQueryColors(dpy, def_cmap, failed, keep);
    for(i=0;i<keep;i++) {
      RemoveValue(pixels, &npixels, failed[i].pixel);
    }
  }
  ExpandColors(failed, ncolors, pivots, npivots);
  GammaCorrectColors(failed, nfailed, gamma);

  /* Attempt to allocate read-only color cells 
   * and remove entries from the failure set.
   */
  if(mode != CT_WINDOW) {
    got=0;
    for(i=0;i<nfailed;i++) {
      temp = failed[i];
      if(XAllocColor(dpy, def_cmap, &temp)) {
	ctable[failed[i].pixel] = temp.pixel;
	RemoveElement(failed, &nfailed, i);
	RemoveValue(pixels, &npixels, temp.pixel);
	got++;
      }
    }
#if DEBUG
    printf("Got %d read-only cells\n", got);
#endif
    if(!nfailed) goto done;
  }

  if(writeable) {
    /* If nice mode is off, try to allocate R/W cells from the
     * default colormap 
     */
    if(mode == CT_BEST) {
      got=0;
      for(i=0;i<nfailed;i++) {
	if(!XAllocColorCells(dpy, def_cmap, 0, NULL, 0, &pixel, 1))
	  break;
	ctable[failed[i].pixel] = pixel;
	failed[i].pixel = pixel;
	XStoreColor(dpy, def_cmap, &failed[i]);
	RemoveElement(failed, &nfailed, i);
	RemoveValue(pixels, &npixels, pixel);
	got++;
      }
#if DEBUG
      printf("Got %d read-write cells\n", got);
#endif
      if(!nfailed) goto done;
    }

    /* Now we resort to a window colormap */
#if DEBUG	
    printf("Put %d colors into a window colormap\n", nfailed);
#endif
    cmap = XCreateColormap(dpy, w, visual, AllocAll);
    if(cmap_return) *cmap_return = cmap;
    /* Fill the cmap with the same stuff that is in the default cmap */
    for(i=0;i<avail_colors;i++) {
      temp.pixel = i;
      temp.flags = DoAllColors;
      XQueryColor(dpy, def_cmap, &temp);
      XStoreColor(dpy, cmap, &temp);
    }
    /* Make sure to use unused pixel values so that our other allocations
     * don't get clobbered when we flip to the window colormap
     */
    for(i=0;i<nfailed;i++) {
      if(npixels == 0) break;
      ctable[failed[i].pixel] = pixels[0];
      failed[i].pixel = pixels[0];
      RemoveValue(pixels, &npixels, pixels[0]);
    }
  }
  else {
    fprintf(stderr, "GetPivotColorTable warning: Display does not fully support desired colormap\n");
    fflush(stderr);
    i=0;
  }

  /* If we are out of pixel values, we must map remaining colors to
   * existing colors. One way to do this is to map to an adjacent pixel 
   * in ctable, since it is probably close to the color we are after.
   */
#if DEBUG
  if(i < nfailed) 
    printf("Squished %d colors\n", nfailed-i);
#endif
  got = i;
  for(;i<nfailed;i++) {
    adjacent = failed[i].pixel;
    /* Try going down */
    while((ctable[adjacent] == NOCOLOR) && adjacent) adjacent--;
    /* Try going up */
    while((ctable[adjacent] == NOCOLOR) && (adjacent < ncolors)) adjacent++;
    if(ctable[adjacent] == NOCOLOR) {
      /* No read-only cells matched. Panic! */
      free(ctable);
      return NULL;
    }
    ctable[failed[i].pixel] = ctable[adjacent];
  }

  /* Create the window colormap if we can */
  if(writeable) {
    nfailed = got;
    XStoreColors(dpy, cmap, failed, nfailed);
    XSetWindowColormap(dpy, w, cmap);
  }

 done:
  /* Set overflow colors equal to the last color */
  for(i=0;i<overflow;i++) {
    ctable[ncolors + i] = ctable[ncolors-1];
  }
  return ctable;
}
    
/* Remove the element of array at index "index" 
 * by shifting higher indices down.
 */
void RemoveElement(XColor *array, int *length, int index)
{
  int i;

  for(i=index;i<*length-1;i++) {
    array[i] = array[i+1];
  }
  *length -= 1;
}

/* Remove all elements of array whose value equals "value"
 * If there is no such element, do nothing.
 */
void RemoveValue(unsigned long *array, int *length, unsigned long value)
{
  int src,dest;

  for(src=dest=0;src<*length;src++) {
    array[dest] = array[src];
    if(array[src] != value) dest++;
  }
  *length = dest;
}

#define fix_gamma(v) (v = (int)(pow((double)(v)/65535, (double)1/gamma) * 65535 + 0.5))

/* Gamma correct a range of colors */
void GammaCorrectColors(XColor *colors, int ncolors, float gamma)
{
  int i;

  for(i=0;i<ncolors;i++) {
    fix_gamma(colors[i].red);
    fix_gamma(colors[i].green);
    fix_gamma(colors[i].blue);
  }
}
    
/* Increase the number of colors in an XColor array by interpolation.
 * src array must have at least from_colors filled entries,
 * and dest must have room for to_colors entries. 
 */
void ExpandColors(XColor *dest, int to_colors,
		  XColor *src, int from_colors)
{
  int from, to, factor;

  if(from_colors <= 1) {
    /* Fill with single color */
    InterpolateColors(dest, src[0], src[0], to_colors);
    return;
  }
  factor = (to_colors+from_colors-2) / (from_colors-1);
  to=0;
  for(from=0;from < from_colors-1;from++) {
    InterpolateColors(&dest[to], src[from], src[from+1], factor);
    to+=factor-1;
  }
  /* Fill leftovers with last color */
  InterpolateColors(&dest[to], src[from], src[from], to_colors-to-1);
}

/* Generate a range of colors between two endpoints.
 * colors = destination XColor array
 * from = first color
 * to = last color
 * steps = number of colors to generate
 */
void InterpolateColors(XColor *colors, XColor from, XColor to, int steps)
{
  int delta_red, delta_green, delta_blue;
  int i;

  delta_red=  to.red  -from.red;
  delta_green=to.green-from.green;
  delta_blue= to.blue -from.blue;
  for(i=0;i<steps;i++) {
    colors[i].red=   delta_red  *i/(steps-1) + from.red;
    colors[i].green= delta_green*i/(steps-1) + from.green;
    colors[i].blue=  delta_blue *i/(steps-1) + from.blue;
  }
}
