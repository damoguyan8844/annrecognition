#include <math.h>
#include <framer.h>
#include <fraxl.h>

#include <tpm/list.h>
#include <tpm/hash.h>
#include "object.h"
#include "member.h"
#include "phandle.h"
#include "image.h"
#include "matrix.h"

/* globals *******************************************************************/

#define NOTADISTANCE 1e10
enum { PH_OK, PH_ERROR };
#define EVAL(x) fraxl_eval(parse_ground_from_string(x))

extern int debug;
#define debugprint(s) if(debug) fprintf(stderr, s)

/* object functions **********************************************************/

typedef void PhDistFunc(Ph_Object self, Ph_Member query, Ph_Member *test, int count);
typedef Ph_Image PhImageFunc(Ph_Object self, Ph_Member m);

/* photobook.c ***************************************************************/

Ph_Handle Ph_Startup(void);
void Ph_Shutdown(Ph_Handle phandle);
char *Ph_StringQuark(char *s);
Type Ph_TypeQuark(char *type_s);

List Ph_GetDatabases(void);
int Ph_SetDatabase(Ph_Handle phandle, char *db_name);
char *Ph_GetDatabase(Ph_Handle phandle);

List Ph_GetSymbols(Ph_Handle phandle);
List Ph_GetSubSymbols(Ph_Handle phandle, char *symbol);
List Ph_GetDBTrees(Ph_Handle phandle);
List Ph_GetDBLabels(Ph_Handle phandle);

List Ph_GetDBMetrics(Ph_Handle phandle);
List Ph_GetDBViews(Ph_Handle phandle);
Ph_Object Ph_SetMetric(Ph_Handle phandle, char *name);
Ph_Object Ph_GetMetric(Ph_Handle phandle);
Ph_Object Ph_SetView(Ph_Handle phandle, char *name);
Ph_Object Ph_GetView(Ph_Handle phandle);

Ph_Object PhLoadMetric(Ph_Handle phandle, char *name);
Ph_Object PhLoadView(Ph_Handle phandle, char *name);

int Ph_SetFilter(Ph_Handle phandle, char *filter);
int Ph_SetQuery(Ph_Handle phandle, List query_list);
void Ph_Shuffle(Ph_Handle phandle);

Ph_Member *Ph_GetMembers(Ph_Handle phandle, int *length_return);
List Ph_ListMembers(Ph_Handle phandle);
Ph_Member *Ph_GetWorkingSet(Ph_Handle phandle, int *length_return);
List Ph_ListWorkingSet(Ph_Handle phandle);
Ph_Member Ph_MemberWithName(Ph_Handle phandle, char *name);
Ph_Member Ph_AddMember(Ph_Handle phandle, char *name);
Ph_Image Ph_MemImage(Ph_Member member);

int Ph_SaveWS(Ph_Handle phandle, char *file);
int Ph_LoadWS(Ph_Handle phandle, char *file);

/* load_field.c **************************************************************/

void FreeValue(Type type, void *data);

int PhLoadField(Ph_Object obj, ObjTField *otf);

/* memann.c ******************************************************************/

char *Ph_MemGetAnn(Ph_Member m, char *field);
void Ph_MemSetAnn(Ph_Member m, char *field,
		  char *value, int symbol_f);
char *PhFrameValue(Frame frame);
List PhFrameResults(Frame frame);
List Ph_MemTextAnns(Ph_Member m);
List Ph_MemSymbolAnns(Ph_Member m);

/* parse.c *******************************************************************/

char *ParseFilter(Ph_Handle phandle, char *filter);

/* utils.c *******************************************************************/

Frame subframe(Frame root, ...);

/* class_table.c *************************************************************/

extern List MetricClasses, ViewClasses, GlobalClasses; /* List of ObjClass */
List Ph_GetMetrics(void);
List Ph_GetViews(void);
void PhClassInit(void);
Ph_Object PhLookupObject(Ph_Handle phandle, char *name);

/* learn.c *******************************************************************/

char *Ph_MemLabels(Ph_Member member, char *result);
char *Ph_MemWithLabel(int label, char *flags);
void LearnInit(Ph_Handle phandle, List trees, int labels);
void LearnFree(void);
void LearnPosEx(Ph_Member member, int label);
void LearnNegEx(Ph_Member member, int label);
void LearnUpdate(void);
void AddLabel(void);
void LearnOptimistic(char flag);
float *Ph_MemLabelProb(Ph_Member member);
int LearnEnabled(int tree);
void LearnEnable(int tree, int flag);
