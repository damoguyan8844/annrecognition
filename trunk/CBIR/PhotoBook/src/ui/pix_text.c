#include "photobook.h"
#include "ui.h"

/* Globals *******************************************************************/

Widget nameToggle, distToggle, labelToggle, treeToggle, annText;

/* Prototypes ****************************************************************/

void PixmapText(Pixmap pixmap, Ph_Member member);
void ConfigureText(Widget w, void *userData,
		   XmAnyCallbackStruct *callData);

/* Functions *****************************************************************/

void PixmapText(Pixmap pixmap, Ph_Member member)
{
  int ypos;

  ypos = appData->im_height + appData->text_ascent;
  if(appData->show_name) {
    /* draw the member name under the image */
    XSetForeground(appData->display, appData->gc, 
		   appData->color[COLOR_TEXT].pixel);
    XtpmDrawCenteredString(appData->display, appData->gc, pixmap, 
			   appData->im_width, ypos, 
			   Ph_MemName(member));
    ypos += appData->text_height;
  }
  if(appData->show_annotation) {
    /* draw the annotation under the image */
    char *anns, *ann, *str;
    /* must make a copy because strtok is destructive */
    anns = strdup(appData->show_annotation);
    ann = strtok(anns, "\n");
    while(ann) {
      str = Ph_MemGetAnn(member, ann);
      if(str) {
	XSetForeground(appData->display, appData->gc, 
		       appData->color[COLOR_TEXT].pixel);
	XtpmDrawCenteredString(appData->display, appData->gc, pixmap, 
			       appData->im_width, ypos, 
			       str);
      }
      ypos += appData->text_height;
      ann = strtok(NULL, "\n");
    }
    free(anns);
  }
  if(appData->show_label) {
    if(IsLabeling) {
      int i, label;
      XColor *color;
      char *labels;
      /* draw the label under the image */
#if 1
      labels = Ph_MemLabels(member, NULL);
#else
      labels = ComputeLabels(member);
#endif
      i = 0;
      for(label=0;label<NumLabels;label++) {
	if(labels[label]) i++;
      }
      if(i > 1) {
	char str[10];
	sprintf(str, "(%d)", i);
	XSetForeground(appData->display, appData->gc, 
		       appData->color[COLOR_TEXT].pixel);
	XtpmDrawCenteredString(appData->display, appData->gc, pixmap, 
			       appData->im_width, ypos,
			       str);
	ypos += appData->text_height;
	if(appData->show_tree) ypos += appData->text_height;
      }
      else {
	for(label=0;label<NumLabels;label++) {
	  if(!labels[label]) continue;
	  color = 0;
	  for(i=label;!color;i-=ListSize(appData->colorList)) {
	    color = ListValueAtIndex(appData->colorList, i);
	  }
	  XSetForeground(appData->display, appData->gc, color->pixel);
	  XFillRectangle(appData->display, pixmap, appData->gc,
			 0, ypos - appData->text_ascent, 
			 appData->im_width, appData->text_height);
	  XSetForeground(appData->display, appData->gc, 
			 appData->color[COLOR_TEXT].pixel);
	  XtpmDrawCenteredString(appData->display, appData->gc, pixmap, 
				 appData->im_width, ypos,
				 ListValueAtIndex(appData->labels, label));
	  ypos += appData->text_height;
	  if(appData->show_tree) {
	    XtpmDrawCenteredString(appData->display, appData->gc, pixmap, 
				   appData->im_width, ypos,
				   ListValueAtIndex(appData->trees, 
						    labels[label]-1));
	    ypos += appData->text_height;
	  }
	}
      }
      free(labels);
    }
    else {
      ypos += appData->text_height;
      if(appData->show_tree) ypos += appData->text_height;
    }
  }
  if(appData->show_dist) {
    if(!appData->mask_dist &&
       Ph_MemDistance(member) != NOTADISTANCE) {
      /* draw the member distance under the image */
      char str[20];
      sprintf(str, "%g", Ph_MemDistance(member));
      XSetForeground(appData->display, appData->gc, 
		     appData->color[COLOR_TEXT].pixel);
      XtpmDrawCenteredString(appData->display, appData->gc, pixmap, 
			     appData->im_width, ypos, 
			     str);
    }
    ypos += appData->text_height;
  }
  
#if 0
  if(appData->selected[Ph_MemIndex(member)]) {
    /* draw a black border around the image */
    XSetForeground(appData->display, appData->gc, 
		   appData->color[COLOR_PHOTOBG].pixel);
    XDrawRectangle(appData->display, pixmap, appData->gc,
		   0, 0, appData->pix_width-1, appData->pix_height-1);
    XDrawRectangle(appData->display, pixmap, appData->gc,
		   1, 1, appData->pix_width-3, appData->pix_height-3);
    XDrawRectangle(appData->display, pixmap, appData->gc,
		   2, 2, appData->pix_width-5, appData->pix_height-5);
  }
#endif
}

static void OkCallback(Widget w, void *userData, 
		       XmAnyCallbackStruct *callData)
{
  /* set appData to the toggle button variables */
  appData->show_name = XmToggleButtonGetState(nameToggle);
  appData->show_label = XmToggleButtonGetState(labelToggle);
  appData->show_tree = appData->show_label && 
    XmToggleButtonGetState(treeToggle);
  appData->show_dist = XmToggleButtonGetState(distToggle);
  appData->show_annotation = XmTextGetString(annText);
  MEM_ALLOC_NOTIFY(appData->show_annotation);
  if(!appData->show_annotation[0]) {
    free(appData->show_annotation);
    appData->show_annotation = NULL;
  }
  else 
    ListAddRear(appData->garbage, appData->show_annotation);

  /* similar to Resize() and CloseDB() */
  BusyOn("reconfiguring photos");
  XtpmIPHalt(appData->imp);
  XtpmITFree(appData->imt);
  XFreePixmap(appData->display, appData->no_image_pixmap);
  PixCacheFree();
  if(debug) XC_BLOCKS();
  ConfigurePhotos();
  PixCacheCreate();
  PixCacheSize(appData->cache_size);
  XtpmIPUpdate(appData->imp);
}

static void toggleCB(Widget w, void *userData, 
		     XmToggleButtonCallbackStruct *toggleData)
{
  XtSetSensitive(treeToggle, toggleData->set);
}

/* Pop up a dialog for editing the text flags.
 * If the user hits OK, clear out the pixcache and resize the phototable.
 */
void ConfigureText(Widget pb, void *userData,
		   XmAnyCallbackStruct *callData)
{
  Widget shell, pane, subpane, w;
  Arg arg[20];
  int n;
  char str[100];

  n=0;
  XtSetArg(arg[n], XmNvisual, appData->visual); n++;
  XtSetArg(arg[n], XmNdepth, appData->depth); n++;
  XtSetArg(arg[n], XmNcolormap, appData->colormap); n++;

  XtSetArg(arg[n], XmNdialogStyle, XmDIALOG_FULL_APPLICATION_MODAL); n++;
  shell = XmCreateMessageDialog(appData->shell, "Configure Text", arg, n);
  XtUnmanageChild(XmMessageBoxGetChild(shell, XmDIALOG_HELP_BUTTON));
  XtAddCallback(shell, XmNokCallback, (XtCallbackProc)OkCallback, NULL);

  n=0;
  pane = XmCreateRowColumn(shell, "configTextPane", arg, n);

  n = 0;
  XtSetArg(arg[n], XmNset, appData->show_name); n++;
  w = XmCreateToggleButton(pane, "nameToggle", arg, n);
  XtManageChild(w);
  nameToggle = w;

  n = 0;
  XtSetArg(arg[n], XmNorientation, XmHORIZONTAL); n++;
  subpane = XmCreateRowColumn(pane, "annSubpane", arg, n);
  n = 0;
  if(appData->show_annotation)
    XtSetArg(arg[n], XmNvalue, appData->show_annotation); n++;
  XtSetArg(arg[n], XmNeditMode, XmMULTI_LINE_EDIT); n++;
  w = XmCreateText(subpane, "annText", arg, n);
  XtManageChild(w);
  annText = w;
  w = XmCreateLabel(subpane, "annLabel", NULL, 0);
  XtManageChild(w);
  XtManageChild(subpane);

  n = 0;
  XtSetArg(arg[n], XmNset, appData->show_label); n++;
  w = XmCreateToggleButton(pane, "labelToggle", arg, n);
  XtAddCallback(w, XmNvalueChangedCallback,
		(XtCallbackProc)toggleCB, NULL);
  XtManageChild(w);
  labelToggle = w;

  n = 0;
  XtSetArg(arg[n], XmNset, appData->show_tree); n++;
  XtSetArg(arg[n], XmNsensitive, appData->show_label); n++;
  w = XmCreateToggleButton(pane, "treeToggle", arg, n);
  XtManageChild(w);
  treeToggle = w;

  n = 0;
  XtSetArg(arg[n], XmNset, appData->show_dist); n++;
  w = XmCreateToggleButton(pane, "distToggle", arg, n);
  XtManageChild(w);
  distToggle = w;

  XtManageChild(pane);
  XtManageChild(shell);
}
