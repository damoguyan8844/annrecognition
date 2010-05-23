/* Defs and Protos for colors.c */
#ifndef COLORS_H_INCLUDED
#define COLORS_H_INCLUDED

typedef int *ColorTable;
typedef enum { CT_BEST, CT_NICE, CT_WINDOW } CtMode;

/*
ColorTable CreateColorTable(Display *dpy, Window w, Colormap cmap,
			    List color_list, List exclude_list, CtMode mode);
*/
ColorTable GetNamedColorTable(Display *dpy, Window w, String mapname, 
			      int startc, int ncolors, float gamma,
			      CtMode mode, Colormap *cmap_return);
ColorTable GetPivotColorTable(Display *dpy, Window w, 
			      XColor *pivots, int npivots,
			      int startc, int ncolors, float gamma,
			      CtMode mode, Colormap *cmap_return);
void ExpandColors(XColor *dest, int to_colors,
		  XColor *src, int from_colors);
void InterpolateColors(XColor *colors, XColor from, XColor to, int steps);

#endif
