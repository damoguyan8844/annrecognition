#include <tpm/list.h>
#include "type.h"
#include "match.h"

/* Globals *******************************************************************/

typedef struct {
  char *name;
  Type value;
} BindingRec;

/* Prototypes ****************************************************************/

int TypeMatch(Type a, Type b, Binding *ba, Binding *bb);
int TypeCompare(Type a, Type b);
int TypeSimilar(Type a, Type b);
int TypeMatchPartial(Type a, Type b, Binding ba, Binding bb);

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

/* private */
static void BindingAdd(Binding b, char *name, Type value);
static char *getid(int star);

/* Functions *****************************************************************/

/* Performs unification on two types, returning the binding in each direction
 * s.t. ba(a) == b and bb(b) == a. 
 * Each type has a separate namespace, and no occurs check is performed.
 * Doesn't handle repeated variables properly.
 * returns 0 on failure.
 * caller must free bindings only if match was successful.
 * if unsuccessful, *ba = *bb = NULL.
 */
int TypeMatch(Type a, Type b, Binding *ba, Binding *bb)
{
  *ba = BindingCreate();
  *bb = BindingCreate();
  if(!TypeMatchPartial(a, b, *ba, *bb)) {
    BindingFree(*ba);
    BindingFree(*bb);
    *ba = NULL;
    *bb = NULL;
    return 0;
  }
  return 1;
}

/* returns 0 for failed match, frees ba and bb. */
/* requires: ba and bb already created */
int TypeMatchPartial(Type a, Type b, Binding ba, Binding bb)
{
  /* apply any bindings accumulated so far */
  if(TypeUnknown(a)) {
    BindingRec *br;
    ListNode p; 
    ListIterate(p, ba) {
      br = (BindingRec*)p->data;
      if(!strcmp(TypeUName(a), br->name)) {
	a = TypeCopy(br->value);
	break;
      }
    }
  }
  if(TypeUnknown(b)) {
    BindingRec *br;
    ListNode p;
    ListIterate(p, bb) {
      br = (BindingRec*)p->data;
      if(!strcmp(TypeUName(b), br->name)) {
	b = TypeCopy(br->value);
	break;
      }
    }
  }
  
  /* bind remaining unknowns */
  if(TypeUnknown(a)) {
    if(TypeNamed(a)) {
      BindingAdd(ba, TypeUName(a), b);
    }
    else {
      TypeError("unnamed unknown");
    }
  }

  if(TypeUnknown(b)) {
    if(TypeNamed(b)) {
      BindingAdd(bb, TypeUName(b), a);
    }
    else {
      TypeError("unnamed unknown");
    }
  }
  if(TypeUnknown(a) || TypeUnknown(b)) return 1;

  /* both known; must be equal */
  if(a->tag != b->tag) return 0;
  switch(a->tag) {
  case TYPE_INT:
    return a->value.i == b->value.i;
  case TYPE_TYPE:
    if(a->value.t->class != b->value.t->class) return 0;
    /* equal in major type; now check params */
    if(!TypeMatchPartial(&a->value.t->i_params, &b->value.t->i_params, 
			 ba, bb)) return 0;
    if(!TypeMatchPartial(&a->value.t->t_params, &b->value.t->t_params, 
			 ba, bb)) return 0;
    break;
  case TYPE_LIST:
    {ListNode p1,p2; Type type1,type2;
     for(p1=ListFront(a->value.l), p2=ListFront(b->value.l);
	 p1 && p2; p1=p1->next, p2=p2->next) {
       type1 = (Type)p1->data;
       type2 = (Type)p2->data;
       /* if type1 or type2 is a ?*, eat the rest of the list */
       if(TypeStar(type1) && !TypeStar(type2)) {
	 Type result = TypeCreate(TYPE_LIST);
	 /* nothing can follow a ?* */
	 if(p1->next) return 0;
	 for(;p2;p2=p2->next) {
	   type2 = (Type)p2->data;
	   ListAddRear(result->value.l, TypeCopy(type2));
	 }
	 BindingAdd(ba, TypeUName(type1), result);
	 TypeFree(result);
	 p1 = NULL;
	 break;
       }
       if(TypeStar(type2) && !TypeStar(type1)) {
	 Type result = TypeCreate(TYPE_LIST);
	 /* nothing can follow a ?* */
	 if(p2->next) return 0;
	 for(;p1;p1=p1->next) {
	   type1 = (Type)p1->data;
	   ListAddRear(result->value.l, TypeCopy(type1));
	 }
	 BindingAdd(bb, TypeUName(type2), result);
	 TypeFree(result);
	 p2 = NULL;
	 break;
       }
       if(!TypeMatchPartial(type1, type2, ba, bb)) return 0;
     }
     if(p1 || p2) return 0;
    }
    break;
  }
  return 1;
}

/* compare two Types without unification.
 * variable names and integers must be equal.
 * Returns 1 on failure.
 */
int TypeCompare(Type a, Type b)
{
  if(a->tag != b->tag) return 1;
  switch(a->tag) {
  case TYPE_UNKNOWN:
    return strcmp(TypeUName(a), TypeUName(b));
  case TYPE_INT:
    return a->value.i - b->value.i;
  case TYPE_TYPE:
    if(a->value.t->class != b->value.t->class) return 1;
    if(a->value.t->size != b->value.t->size) return 1;
    if(TypeCompare(&a->value.t->i_params, &b->value.t->i_params))
      return 1;
    return TypeCompare(&a->value.t->t_params, &b->value.t->t_params);
  case TYPE_LIST:
    {ListNode p1,p2; Type type1,type2;
     for(p1=ListFront(a->value.l), p2=ListFront(b->value.l);
	 p1 && p2; p1=p1->next, p2=p2->next) {
       type1 = (Type)p1->data;
       type2 = (Type)p2->data;
       if(TypeCompare(type1, type2)) return 1;
     }}
    return 0;
  }
  return 0;
}

/* compare two Types without unification.
 * integers must be equal, but not variable names.
 * Returns 1 on failure.
 */
int TypeSimilar(Type a, Type b)
{
  if(a->tag != b->tag) return 1;
  switch(a->tag) {
  case TYPE_UNKNOWN:
    return 0;
  case TYPE_INT:
    return a->value.i - b->value.i;
  case TYPE_TYPE:
    if(a->value.t->class != b->value.t->class) return 1;
    if(a->value.t->size != b->value.t->size) return 1;
    if(TypeSimilar(&a->value.t->i_params, &b->value.t->i_params))
      return 1;
    return TypeSimilar(&a->value.t->t_params, &b->value.t->t_params);
  case TYPE_LIST:
    {ListNode p1,p2; Type type1,type2;
     for(p1=ListFront(a->value.l), p2=ListFront(b->value.l);
	 p1 && p2; p1=p1->next, p2=p2->next) {
       type1 = (Type)p1->data;
       type2 = (Type)p2->data;
       if(TypeSimilar(type1, type2)) return 1;
     }}
    return 0;
  }
  return 0;
}

/* Returns the integer value bound to name in b. */
int BindingIntValue(Binding b, char *name)
{
  Type type;

  type = BindingValue(b, name);
  if(!type) {
    TypeError("BindingIntValue: no binding for %s", name);
  }
  if(type->tag != TYPE_INT) {
    TypeError("BindingIntValue used on a non-int");
  }
  return type->value.i;
}

void BindingSetInt(Binding b, char *name, int i)
{
  Type type;

  type = TypeCreate(TYPE_INT, i);
  BindingSetValue(b, name, type);
  TypeFree(type);
}

/* Returns the Type bound to name in b. */
Type BindingValue(Binding b, char *name)
{
  BindingRec *br;
  ListIter(p, br, b) {
    if(!strcmp(name, br->name)) return br->value;
  }
  return NULL;
}

void BindingSetValue(Binding b, char *name, Type type)
{
  BindingRec *br;
  ListIter(p, br, b) {
    if(!strcmp(name, br->name)) break;
  }
  if(p) {
    TypeFree(br->value);
    br->value = TypeCopy(type);
  }
  else {
    BindingAdd(b, name, type);
  }
}

/* Binds the unknowns in t with their values in b.
 * Unknowns which do not appear in b are left alone.
 * Returns the number of bindings made.
 */
int BindingApply(Binding b, Type t)
{
  Type type;
  int count;

  if(BindingEmpty(b)) return 0;
  switch(t->tag) {
  case TYPE_INT:
    return 0;
  case TYPE_UNKNOWN:
    type = BindingValue(b, TypeUName(t));
    if(type) {
      TypeReplace(t, TypeCopy(type));
      return 1;
    }
    return 0;
  case TYPE_LIST:
    {
    List result = ListCreate((FreeFunc*)TypeFree);
    count = 0;
    {ListIter(p, type, t->value.l) {
      count += BindingApply(b, type);
      if(type->tag == TYPE_LIST) {
	ListAppend(result, type->value.l);
	free(type);
      }
      else ListAddRear(result, type);
    }}
    /* free the nodes of the original list, but not the data */
    t->value.l->fp = NULL; ListFree(t->value.l);
    /* replace with the new list */
    t->value.l = result;
    }
    return count;
  case TYPE_TYPE:
    count = 0;
    count += BindingApply(b, &t->value.t->i_params);
    count += BindingApply(b, &t->value.t->t_params);
    if(count) TypeComputeSize(t);
    return count;
  }
  return 0;
}

/* Binds the unknowns in t with their values in b.
 * Unknowns which do not appear in b are mapped to unique names via getid().
 * Adds the name mappings to b.
 * Returns the number of unique names assigned.
 */
int BindingApplyUnique(Binding b, Type t)
{
  int count;
  Type type;

  switch(t->tag) {
  case TYPE_INT:
    return 0;
  case TYPE_UNKNOWN:
    type = BindingValue(b, TypeUName(t));
    if(type) {
      TypeReplace(t, TypeCopy(type));
      return 0;
    }
    else {
      char *name = TypeUName(t);
      TypeUName(t) = getid(name[0] == '*');
      BindingAdd(b, name, t);
      free(name);
      return 1;
    }
  case TYPE_LIST:
    {
    List result = ListCreate((FreeFunc*)TypeFree);
    count = 0;
    {ListIter(p, type, t->value.l) {
      count += BindingApplyUnique(b, type);
      if(type->tag == TYPE_LIST) {
	ListAppend(result, type->value.l);
	free(type);
      }
      else ListAddRear(result, type);
    }}
    /* free the nodes of the original list, but not the data */
    t->value.l->fp = NULL; ListFree(t->value.l);
    /* replace with the new list */
    t->value.l = result;
    }
    return count;
  case TYPE_TYPE:
    count = 0;
    count += BindingApplyUnique(b, &t->value.t->i_params);
    count += BindingApplyUnique(b, &t->value.t->t_params);
    if(count) TypeComputeSize(t);
    return count;
  }
  return 0;
}

/* Returns TRUE if b binds all variables in t */
int BindingCovers(Binding b, Type t)
{
  switch(t->tag) {
  case TYPE_INT:
    return 1;
  case TYPE_UNKNOWN:
    return !!BindingValue(b, TypeUName(t));
  case TYPE_LIST:
    {Type type;ListIter(p, type, t->value.l) {
      if(!BindingCovers(b, type)) return 0;
    }}
    return 1;
  case TYPE_TYPE:
    if(!BindingCovers(b, &t->value.t->i_params)) return 0;
    if(!BindingCovers(b, &t->value.t->t_params)) return 0;
    return 1;
  }
  return 1;
}

static void UnknownPartial(Type t, Binding b)
{
  switch(t->tag) {
  case TYPE_UNKNOWN:
    BindingAdd(b, TypeUName(t), t);
    break;
  case TYPE_LIST:
    {Type type;ListIter(p, type, t->value.l) {
      UnknownPartial(type, b);
    }}
    break;
  case TYPE_TYPE:
    UnknownPartial(&t->value.t->i_params, b);
    UnknownPartial(&t->value.t->t_params, b);
    break;
  }
}

/* Returns a binding of all unknowns in t to themselves. */
Binding BindingUnknowns(Type t)
{
  Binding b = BindingCreate();
  UnknownPartial(t, b);
  return b;
}

static void BindingAdd(Binding b, char *name, Type value)
{
  BindingRec *br = Allocate(1, BindingRec);
  br->name = strdup(name);
  br->value = TypeCopy(value);
  ListAddRear(b, br);
}

static BindingRec *BindingRecCopy(BindingRec *br)
{
  BindingRec *result = Allocate(1, BindingRec);
  result->name = strdup(br->name);
  result->value = TypeCopy(br->value);
  return result;
}

Binding BindingCopy(Binding b)
{
  return ListCopy(b, (void*(*)())BindingRecCopy);
}

/* Produces a single binding in b1. Frees storage used by b2.
 * b1 must not be NULL.
 */
void BindingMerge(Binding b1, Binding b2)
{
  if(b2 == NULL) return;
  ListAppend(b1, b2);
}

static void BindingRecFree(BindingRec *br)
{
  free(br->name);
  TypeFree(br->value);
  free(br);
}

/* Returns an empty binding. */
Binding BindingCreate(void)
{
  return ListCreate((FreeFunc*)BindingRecFree);
}

void BindingFree(Binding b)
{
  if(b == NULL) return;
  ListFree(b);
}

void BindingWrite(FILE *fp, Binding b)
{
  ListNode p;
  BindingRec *br;

  if(b == NULL) {
    fprintf(fp, "NULL binding\n");
    return;
  }
  if(ListEmpty(b)) {
    fprintf(fp, "<no bindings>\n");
    return;
  }
  fprintf(fp, "Binding:\n");
  ListIterate(p, b) {
    br = (BindingRec*)p->data;
    fprintf(fp, "%s ", br->name);
    TypeUnparse(fp, br->value);
    fprintf(fp, "\n");
  }
}

/* used by BindingNoVars */
static int NoVarCompare(BindingRec *r1, void *dummy)
{
  return (r1->value->tag != TYPE_UNKNOWN);
}

/* Mutates b by removing all bindings to unknowns, e.g. x -> y.
 * Returns the number of bindings removed.
 */
int BindingNoVars(Binding b)
{
  return ListRemoveAll(b, NULL, (CmpFunc*)NoVarCompare);
}

/* Used for getting unique identifiers for BindingApplyUnique().
 * Limited by the size of an integer, so should be changed in the future.
 */
static unsigned id_count = 0;

static char *getid(int star)
{
  char *s = Allocate(6, char);
  sprintf(s, "%c%d", star?'*':'_', id_count++);
  return s;
}

/* used by BindingClean */
static int NoCyclicCompare(BindingRec *r1, void *dummy)
{
  return (r1->value->tag != TYPE_UNKNOWN ||
	  strcmp(r1->value->value.s, r1->name));
}

/* used by BindingClean */
static int NoTempCompare(BindingRec *r1, void *dummy)
{
  return (r1->name[0] != '_' && r1->name[0] != '*');
}

/* Removes all bindings which refer to themselves, and
 * expands all binding values.
 * Returns the number of bindings removed.
 */
int BindingClean(Binding b)
{
  int count = ListRemoveAll(b, NULL, (CmpFunc*)NoCyclicCompare);
  BindingRec *br;
  ListIter(p, br, b) {
    BindingApply(b, br->value);
  }
/*
  return count + ListRemoveAll(b, NULL, (CmpFunc*)NoTempCompare);
*/
  return count;
}
