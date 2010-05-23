#ifndef PHOTOTABLE_H_INCLUDED
#define PHOTOTABLE_H_INCLUDED

#include <xtpm/im_table.h>

struct XtpmIP;
typedef Pixmap XtpmIPLoad(int page, int row, int col, void *userData);
typedef void XtpmIPCallback(struct XtpmIP *imp, void *userData);

typedef struct XtpmIP {
  Widget upArrow, downArrow, imt;
  XtpmIPLoad *loadFunc;
  void *loadData;
  XtpmIPCallback *startCB, *stopCB;
  void *start_data, *stop_data;
  int num_images, page_size, page, row, col;
  XtWorkProcId work_proc;
} *XtpmIP;

/* Stops the image update WorkProc. Should be executed before modifying
 * the IP in any way. (Done automatically by SetIT, JumpTo, and Free.)
 */
#define XtpmIPHalt(imp) \
  if((imp)->work_proc) XtRemoveWorkProc((imp)->work_proc);

XtpmIP XtpmIPCreate(Widget parent, char *name);
void XtpmIPFree(XtpmIP imp);
void XtpmIPUpdate(XtpmIP imp);
int XtpmIPJumpTo(XtpmIP imp, int page);
void XtpmIPSetIT(XtpmIP imp, Widget imt, int num_images, 
		 XtpmIPLoad *loadFunc, void *userData);
void XtpmIPStartCallback(XtpmIP imp, XtpmIPCallback *callback, void *userData);
void XtpmIPStopCallback(XtpmIP imp, XtpmIPCallback *callback, void *userData);

#endif
