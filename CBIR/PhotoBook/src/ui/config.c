#include "photobook.h"
#include "ui.h"

/* Prototypes ****************************************************************/

void ConfigureDisplay(void);
void ConfigurePhotos(void);
void DoSearch(void);

/* private */
static void PhotoTable(Widget parent);

/* Functions *****************************************************************/

void ConfigureDisplay(void)
{
  List lst;

  if(debug) printf("->ConfigureDisplay\n");

  ConfigureMenus();
  ConfigurePhotos();

  /* pop down all shells */
  /* view shell */
  TShellReset(appData->view_shell);
  TShellSetSensitive(appData->view_shell, 
		     !!GetConfigShell(Ph_GetView(phandle)));

  /* metric shell */
  TShellReset(appData->metric_shell);
  TShellSetSensitive(appData->metric_shell, 
		     !!Ph_GetMetric(phandle) &&
		     !!GetConfigShell(Ph_GetMetric(phandle)));

  /* labeling shell */
  TShellReset(appData->labeling_shell);
  appData->labels = Ph_GetDBLabels(phandle);
  if(appData->labels && !NumLabels) {
    ListFree(appData->labels);
    appData->labels = NULL;
  }
  TShellSetSensitive(appData->labeling_shell, !!appData->labels);

  /* hooks shell */
  if(appData->hooks_shell) {
    TShellReset(appData->hooks_shell);
    TShellSetSensitive(appData->hooks_shell, !!appData->labeling_shell->shell);
  }

  /* glabel shell */
  if(appData->glabel_shell) {
    TShellReset(appData->glabel_shell);
    TShellSetSensitive(appData->glabel_shell, !!appData->glabels);
  }

  /* symbol shell */
  TShellReset(appData->symbol_shell);
  lst = Ph_GetSymbols(phandle);
  TShellSetSensitive(appData->symbol_shell, lst && ListSize(lst));
  if(lst) ListFree(lst);

  /* refresh the cache */
  PixCacheCreate();
  PixCacheSize(appData->cache_size);
  if(debug) printf("<-ConfigureDisplay\n");
}

static void MakePixmaps(void)
{
  Pixmap pixmap;
  int width, height;

  width = appData->pix_width;
  height = appData->pix_height;

  /* Allocate pixmap */
  pixmap = XCreatePixmap(appData->display,
			 appData->root,
			 width,
			 height,
			 appData->depth);
  if(!pixmap) {
    XtError("MakePixmaps: XCreatePixmap failed\n");
  }
  /* Clear the pixmap */
  XSetForeground(appData->display, appData->gc, 
		 appData->color[COLOR_PHOTOBG].pixel);
  XFillRectangle(appData->display,
		 pixmap,
		 appData->gc,0,0,
		 width,
		 height);
  /* Write text in center */
  XSetForeground(appData->display, appData->gc, 
		 appData->color[COLOR_TEXT].pixel);
  XtpmDrawCenteredString(appData->display, appData->gc, pixmap, 
			 width, height/2, 
			 "NO IMAGE");
  appData->no_image_pixmap = pixmap;
}

static Pixmap PhotoLoad(int page, int row, int col, void *userData)
{
  int i;
  Ph_Member member;
  Pixmap pixmap;
  Widget w;

  i = page*appData->imp->page_size + row*appData->num_cols + col;
  if(i >= appData->num_members) return 0;
  w = XtpmITWidget(appData->imt, row, col);
  XtVaSetValues(w, XmNborderColor,
		appData->selected[i] ?
		appData->color[COLOR_PICKED].pixel :
		appData->color[COLOR_UNPICKED].pixel,
		NULL);
  member = appData->members[i];
  pixmap = GetMemPixmap(member);
  PixmapText(pixmap, member);
  return pixmap;
}

void ConfigurePhotos(void)
{
  Arg arg[10];
  int n;
  int x_border, y_border, border_width;
  Dimension width, height;
  Ph_Object view;

  view = Ph_GetView(phandle);
  if(!view) {
    fprintf(stderr, "The %s database has no display mode\n", 
	    Ph_GetDatabase(phandle));
    exit(1);
  }
  Ph_ObjGet(view, "width", &appData->im_width);
  Ph_ObjGet(view, "height", &appData->im_height);
  Ph_ObjGet(view, "channels", &appData->im_channels);
  appData->pix_width = appData->im_width;
  appData->text_lines = appData->show_name + appData->show_dist +
    appData->show_label + appData->show_tree;
  if(appData->show_annotation) {
    /* count the number of lines */
    char *lines = strdup(appData->show_annotation);
    char *line = strtok(lines, "\n");
    for(;line;line = strtok(NULL, "\n")) appData->text_lines++;
    free(lines);
  }
  appData->y_pad = appData->text_lines * appData->text_height;
  appData->pix_height = appData->im_height + appData->y_pad;

  /* calculate the number of rows and cols */
  border_width = 1;
  /* x border */
  XtVaGetValues(appData->left_pane, XmNwidth, &width, NULL);
  x_border = width;
  XtVaGetValues(appData->right_pane, XmNwidth, &width, NULL);
  x_border += width + 25;
  /* y border */
  XtVaGetValues(appData->com_text, XmNheight, &height, NULL);
  y_border = height + 25;
  /* size of form */
  XtVaGetValues(appData->form, XmNwidth, &width, XmNheight, &height, NULL);
  /* set cols */
  width -= x_border;
  appData->num_cols = width / (appData->pix_width + border_width*2);
  if(appData->num_cols == 0) appData->num_cols = 1;
  /* set rows */
  height -= y_border;
  appData->num_rows = height / (appData->pix_height + border_width*2);
  if(appData->num_rows == 0) appData->num_rows = 1;
  if(debug) printf("using %d rows and %d columns\n", 
		   appData->num_rows, appData->num_cols);

  /* create the image table */
  PhotoTable(appData->form);

  /* create the standard pixmaps */
  MakePixmaps();

  /* set the form attachments of the table 
   * so that it will resize with the main window.
   */
  n=0;
  XtSetArg(arg[n], XmNtopAttachment, XmATTACH_FORM); n++;
  XtSetArg(arg[n], XmNbottomAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNbottomWidget, appData->com_text); n++;
  XtSetArg(arg[n], XmNleftAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNleftWidget, appData->left_pane); n++;
  XtSetArg(arg[n], XmNrightAttachment, XmATTACH_WIDGET); n++;
  XtSetArg(arg[n], XmNrightWidget, appData->right_pane); n++;
  XtSetValues(appData->imt, arg, n);
  /* post-set the photo table's image table */
  XtpmIPSetIT(appData->imp, appData->imt, appData->num_members,
	      (XtpmIPLoad*)PhotoLoad, NULL);
}

void DoSearch(void)
{
  int i, status;
  List query_list; /* list of strings */
  char *str;

  BusyOn("searching");
  /* set the filter */
  str = XmTextGetString(appData->com_text);
  MEM_ALLOC_NOTIFY(str);
  if(!appData->filter || strcmp(str, appData->filter)) {
    if(Ph_SetFilter(phandle, str) == PH_ERROR) {
      BusyOff();
      ErrorPopup(phandle->error_string);
      free(str);
      return;
    }
    if(appData->filter) free(appData->filter);
    appData->filter = str;
  }
  else free(str);

  /* collect the selected members into query_list */
  query_list = ListCreate(NULL);
  for(i=0;i<appData->num_members;i++) {
    if(appData->selected[i]) 
      ListAddRear(query_list, Ph_MemName(appData->members[i]));
  }
  
  /* make the query */
  status = Ph_SetQuery(phandle, query_list);
  ListFree(query_list);
  if(status == PH_ERROR) {
    BusyOff();
    ErrorPopup("Query error");
    return;
  }

  /* refresh the member list */
  SetMembers();
  /* let distance text be displayed */
  appData->mask_dist = 0;
  /* reset to page zero */
  XtpmIPJumpTo(appData->imp, 0);
  UpdateDisplay();
}

static void MemInfoPopup(Ph_Member member)
{
  char *str, *name;
  List anns;

  str = Allocate(2048, char);
  sprintf(str, "FRAME: #dbs/%s/members/%s\n------",
	  Ph_GetDatabase(phandle), Ph_MemName(member));

  /* text annotations */
  anns = Ph_MemTextAnns(member);
  {ListIter(p, name, anns) {
    sprintf(str, "%s\n%s:  %s", str, name, Ph_MemGetAnn(member, name));
  }}
  ListFree(anns);

  sprintf(str, "%s\n------", str);
  /* symbol annotations */
  anns = Ph_MemSymbolAnns(member);
  {ListIter(p, name, anns) {
    sprintf(str, "%s\n%s:  %s", str, name, Ph_MemGetAnn(member, name));
  }}
  ListFree(anns);

  if(IsLabeling) {
    int i;
    float *prob;
    char *labels;
    float *prob2;
    sprintf(str, "%s\n------", str);
    labels = Ph_MemLabels(member, NULL);
    prob = Ph_MemLabelProb(member);
#if HOOKS
    prob2 = ComputeLabelProb(member);
#endif
    for(i=0;i<NumLabels;i++) {
      int j;
      /* show label probability */
      sprintf(str, "%s\n%s : %g %g", str, 
	      ListValueAtIndex(appData->labels, i),
	      prob[i], 
#if HOOKS
	      prob2[i]
#else
	      prob[i]
#endif
	      );
/*
      for(j=0;j<ListSize(appData->trees);j++) {
	sprintf(str, "%s\n%s : %g (%s)", str, 
		ListValueAtIndex(appData->labels, i),
		matrix[(j+1)*phandle->total_members+Ph_MemIndex(member)],
		ListValueAtIndex(appData->trees, j));
      }
*/
				      
      if(!labels[i]) continue;
      sprintf(str, "%s\n%s : %s", str, ListValueAtIndex(appData->labels, i),
	      ListValueAtIndex(appData->trees, labels[i]-1));
    }
#if HOOKS
    free(prob2);
#endif
    free(prob);
    free(labels);
  }

  if(!appData->mask_dist) {
    sprintf(str, "%s\n------\ndistance: %g", str, Ph_MemDistance(member));
  }

  /* set the pop-up window to contain str */
  XtVaSetValues(appData->pop_label, 
		XmNlabelString, MakeXmString(str), 
		NULL);
  free(str);

  /* Make the window appear (<BtnUp> action will pop it down) */
  XtPopupSpringLoaded(appData->pop_shell);
}

/* callback for image widgets */
#define MemIndex(i) (appData->imp->page*appData->imp->page_size+(i))
static void PhotoCB(Widget w, int i, XmAnyCallbackStruct *callData)
{
  if(callData->event->type != ButtonPress) return;
  if(callData->event->xbutton.button == Button1) {
    /* select/deselect */
    appData->selected[MemIndex(i)] = !appData->selected[MemIndex(i)];
    XtVaSetValues(w, XmNborderColor,
		  appData->selected[MemIndex(i)] ?
		  appData->color[COLOR_PICKED].pixel :
		  appData->color[COLOR_UNPICKED].pixel,
		  NULL);
  }
  else if(callData->event->xbutton.button == Button2) {
    DoSearch();
  }
  else if(callData->event->xbutton.button == Button3) {
    /* have to recreate each time to change the window size */
    if(appData->pop_shell) XtDestroyWidget(appData->pop_shell);
    appData->pop_shell = MakePopShell();
    /* Position the popup shell */
    XtVaSetValues(appData->pop_shell, 
		  XmNx, callData->event->xbutton.x_root,
		  XmNy, callData->event->xbutton.y_root,
		  NULL);

    /* pop up an info window */
    MemInfoPopup(appData->members[MemIndex(i)]);
  }
}

static void PhotoTable(Widget parent)
{
  Widget w;
  List image_rows, image_col;
  XtpmITRec *p;
  int i,j;

  image_rows = ListCreate((FreeFunc*)ListFree);
  for(i=0;i<appData->num_rows;i++) {
    image_col = ListCreate(GenericFree);
    for(j=0;j<appData->num_cols;j++) {
      p = Allocate(1, XtpmITRec);
      p->width = appData->pix_width;
      p->height = appData->pix_height;
      ListAddRear(image_col, p);
    }
    ListAddRear(image_rows, image_col);
  }
  appData->imt = XtpmITCreate(parent, "imTable", image_rows, 
			      (XtCallbackProc)PhotoCB, NULL);
  XtpmITSetGC(appData->imt, appData->gc);
  ListFree(image_rows);
}
