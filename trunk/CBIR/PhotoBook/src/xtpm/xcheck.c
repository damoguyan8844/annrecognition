/* Code for the X checker. */

#include <stdio.h>
#include <malloc.h>
#include <X11/Xlib.h>

/* X node structure; singly linked list */
typedef struct XNodeStruct {
  Display *dpy;
  Pixmap pix;
  char *file;
  unsigned line;
  struct XNodeStruct *next;
} *XNode;

/* Globals *******************************************************************/
static XNode XNodeList = NULL;
static int abort_on_error = 1;

/* Private *******************************************************************/

/* Generic error message. Aborts execution depending on abort_on_error.
 */
#ifdef __STDC__
static void show_error(char *s, char *file, unsigned line)
#else
static void show_error(s, file, line)
     char *s;
     char *file;
     unsigned line;
#endif
{
  fprintf(stderr, 
	    "*******************************************************************************\n");
  fprintf(stderr, "%s line %d: %s", file, line, s);
  fprintf(stderr, 
	  "\n*******************************************************************************\n");
  fflush(stderr);
  if(abort_on_error) exit(1);
}

/* Lists active Pixmaps.
 */
#ifdef __STDC__
static void show_x_nodes(void)
#else
static void show_x_nodes()
#endif
{
  XNode node;

  printf("Active Pixmaps:\n");
  for(node = XNodeList; node; node = node->next) {
    printf("  %s line %d\n", node->file, node->line);
  }
}

/* Add a node to the X list.
 */
#ifdef __STDC__
static void add_x_node(Display *dpy, Pixmap pix, char *file, unsigned line)
#else
static void add_x_node(dpy, pix, file, line)
     Display *dpy;
     Pixmap pix;
     char *file;
     unsigned line;
#endif
{
  XNode node;

  node = (XNode)malloc(sizeof(struct XNodeStruct));
  if(!node) {
    show_error("INTERNAL: cannot allocate X node", file, line);
    exit(1);
  }
  node->dpy = dpy;
  node->pix = pix;
  node->file = file;
  node->line = line;
  node->next = XNodeList;
  XNodeList = node;
}

/* Remove a node from the X list.
 */
#ifdef __STDC__
static void remove_x_node(XNode *node_p)
#else
static void remove_x_node(node_p)
     XNode *node_p;
#endif
{
  XNode node;
  
  node = *node_p;
  *node_p = node->next;
  free(node);
}

/* Find a block in the x node list. Returns a modifiable pointer
 * to a node (useful for removal).
 */
#ifdef __STDC__
static XNode *find_x_node(Display *dpy, Pixmap p)
#else
static XNode *find_x_node(dpy, p)
     Display *dpy;
     Pixmap p;
#endif
{
  XNode *node_p;

  for(node_p = &XNodeList; *node_p; node_p = &(*node_p)->next) {
    if( ((*node_p)->dpy == dpy) &&
        ((*node_p)->pix == p) ) return node_p;
  }
  return node_p;
}

/* Public ********************************************************************/

/* Routine called by XC_ABORT_ON_ERROR macro.
 */
#ifdef __STDC__
void _xc_abort_on_error(int flag)
#else
void _xc_abort_on_error(flag)
     int flag;
#endif
{
  abort_on_error = flag;
}

#ifdef __STDC__
void _xc_alloc_notify(Display *dpy, Pixmap p, char *file, unsigned line)
#else
void _xc_alloc_notify(dpy, p, file, line)
     Display *dpy;
     Pixmap p;
     char *file;
     unsigned line;
#endif
{
  add_x_node(dpy, p, file, line);
}

/* Routine called by MEM_FREE_NOTIFY macro.
 */
#ifdef __STDC__
void _xc_free_notify(Display *dpy, Pixmap p, char *file, unsigned line)
#else
void _xc_free_notify(dpy, p, file, line)
     Display *dpy;
     Pixmap p;
     char *file;
     unsigned line;
#endif
{
  XNode *node;

  node = find_x_node(dpy, p);
  if(!*node) {
    show_error("attempted to free an unallocated or freed pixmap", 
	       file, line);
    return;
  }
  remove_x_node(node);
}

/* Routine which replaces XCreatePixmap()
 */
#ifdef __STDC__
Pixmap _xc_XCreatePixmap(Display *dpy,
			 Drawable d,
			 unsigned width, unsigned height,
			 unsigned depth,
			 char *file, unsigned line)
#else
Pixmap _xc_XCreatePixmap(dpy, d, width, height, depth,
			 file, line)
     Display *dpy;
     Drawable d;
     unsigned width, height, depth;
     char *file;
     unsigned line;
#endif
{
  Pixmap pix;

  pix = XCreatePixmap(dpy, d, width, height, depth);
  add_x_node(dpy, pix, file, line);
  return pix;
}

/* Routine which replaces XFreePixmap()
 */
#ifdef __STDC__
void _xc_XFreePixmap(Display *dpy, Pixmap p, char *file, unsigned line)
#else
void _xc_XFreePixmap(dpy, p, file, line)
     Display *dpy;
     Pixmap p;
     char *file;
     unsigned line;
#endif
{
  XNode *node;

  node = find_x_node(dpy, p);
  if(!*node) {
    show_error("attempted to free an unallocated or freed Pixmap",
	      file, line);
    return;
  }
  XFreePixmap(dpy, p);
  remove_x_node(node);
}

/* Routine called by X_BLOCKS macro
 */
#ifdef __STDC__
void _xc_x_nodes(char *file, unsigned line)
#else
void _xc_x_nodes(file, line)
     char *file;
     unsigned line;
#endif
{
  if(XNodeList == NULL) {
    printf("All Pixmaps have been deallocated.\n");
    return;
  }
  show_x_nodes();
}
