/* Data types ****************************************************************/

/* "Binding" is a list of { name, type }
 * It provides assignments to the unknowns in a type.
 */
typedef List Binding;

/* Globals *******************************************************************/

#define BindingEmpty(b) (!(b) || ListEmpty(b))

/* Prototypes ****************************************************************/

/* match.c */
int TypeMatch(Type a, Type b, Binding *ba, Binding *bb);
int TypeCompare(Type a, Type b);
int TypeSimilar(Type a, Type b);
int TypeMatchPartial(Type a, Type b, Binding ba, Binding bb);

/* match.c */
Binding   BindingCreate(void);
void      BindingFree(Binding b);
int       BindingApply(Binding b, Type t);
int       BindingApplyUnique(Binding b, Type t);

int       BindingIntValue(Binding b, char *name);
Type      BindingValue(Binding b, char *name);
void      BindingSetInt(Binding b, char *name, int i);
void      BindingSetValue(Binding b, char *name, Type type);

Binding   BindingCopy(Binding b);
void      BindingMerge(Binding b1, Binding b2);
int       BindingNoVars(Binding b);
int       BindingCovers(Binding b, Type t);
Binding   BindingUnknowns(Type t);
void      BindingWrite(FILE *fp, Binding b);
int       BindingClean(Binding b);

