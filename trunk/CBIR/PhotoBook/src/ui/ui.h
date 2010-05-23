#include <assert.h>
#include <xtpm/photo_table.h>
#include "tshell.h"

#define Separator(w) XtManageChild(XmCreateSeparator(w, "sep", NULL, 0))
#define IsLabeling appData->labeling_shell->shell
#define NumLabels ListSize(appData->labels)
#define NumGLabels ListSize(appData->glabels)

#include "appdata.h"

extern AppData appData;
extern Ph_Handle phandle;

/* main.c ********************************************************************/
void SetMembers(void);
void UpdateDisplay(void);
void SetupDB(void);
void ErrorPopup(char *format, ...);
void InfoPopup(char *format, ...);
void ShowStatus(char *s);
void BusyOn(char *s);
void BusyOff(void);

/* setup.c *******************************************************************/
int ProcessArgs(int *argc_p, char **argv);
void InitDisplay(void);
Widget MakePopShell(void);
Widget MakePulldownMenu(Widget parent, List menu_list, 
			XtCallbackProc callback, void *userData,
			char *menu_default, Widget *def_widget);

/* left_pane.c ***************************************************************/
Widget MakeLeftPane(Widget parent);
void ConfigureMenus(void);

/* right_pane.c **************************************************************/
Widget MakeRightPane(Widget parent);
void CloseDB(void);

/* config.c ******************************************************************/
void ConfigureDisplay(void);
void ConfigurePhotos(void);
void DoSearch(void);

/* labeling.c ****************************************************************/
void LabelingCallback(Widget shell, void *userData);
void LabelFree(void);

/* symbols.c *****************************************************************/
void SymbolsCallback(Widget shell, void *userData);
void SymbolsFree(void);

/* cache.c *******************************************************************/
void PixCacheCreate(void);
void PixCacheFree(void);
Pixmap GetMemPixmap(Ph_Member member);
void UncacheMemPixmap(Ph_Member member, char *view);
void PixCacheSize(int size);

/* widgets.c *****************************************************************/
Widget ScaleVar(Widget parent, Ph_Object obj, char *field, 
		double from, double to, int points);
Widget TextVar(Widget parent, Ph_Object obj, char *field);
Widget MenuVar(Widget parent, Ph_Object obj, char *field,
	       List items, List values);

/* pix_text.c ****************************************************************/
void PixmapText(Pixmap pixmap, Ph_Member member);
void ConfigureText(Widget w, void *userData,
		   XmAnyCallbackStruct *callData);

/* load.c ********************************************************************/
void LoadDialog(Widget w, void *userData,
		XmAnyCallbackStruct *callData);
void SaveDialog(Widget w, void *userData,
		XmAnyCallbackStruct *callData);

/* hooks.c *******************************************************************/
float *ComputeLabelProb(Ph_Member member);
char *ComputeLabels(Ph_Member member);
void HooksOperate(int image_index);
void HooksCallback(Widget shell, void *userData);

/* glabel.c ******************************************************************/
void GLabelingCallback(Widget shell, void *userData);
void ShowGLabel(float *prob);
void GLabelInit(void);
void GLabelFree(void);

/* tcl.c *********************************************************************/
void DoTcl(Widget w, void *userData,
	   XmAnyCallbackStruct *callData);
