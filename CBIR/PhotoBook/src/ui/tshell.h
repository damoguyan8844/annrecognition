typedef void TShellCB(Widget shell, void *userData);

typedef struct TShell {
  Widget button, shell;
  TShellCB *callback;
  void *userData;
} *TShell;

#define TShellSetSensitive(ts, flag) XtSetSensitive((ts)->button, flag);
#define TShellFree(ts) free(ts)

TShell TShellCreate(Widget parent, char *name, 
		    TShellCB *callback, void *userData);
void TShellReset(TShell ts);
void TShellDestroy(TShell ts);
