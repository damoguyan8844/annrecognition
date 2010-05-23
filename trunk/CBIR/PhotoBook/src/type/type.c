#include "type.h"

/* Prototypes ****************************************************************/

Type TypeCreate(TypeTag tag, ...);
void TypeFree(Type t);
void TypeReplace(Type t1, Type t2);
void TypeComputeSize(Type t);
int TypeSize(Type t);
int TypeAlign(Type t);
int TypeHasUnknowns(Type t);
Type TypeCopy(Type t);
void TypeError(char *err, ...);

/* Functions *****************************************************************/

/* Returns a new type with the given tag. 
 * The value is provided as the second argument.
 */
Type TypeCreate(TypeTag tag, ...)
{
  va_list args;
  Type result = Allocate(1, struct TypeStruct);

  result->tag = tag;
  va_start(args, tag);
  switch(tag) {
  case TYPE_INT:
    result->value.i = va_arg(args, int);
    break;
  case TYPE_TYPE:
    result->value.t = va_arg(args, TypeT);
    break;
  case TYPE_UNKNOWN:
    result->value.s = va_arg(args, char*);
    break;
  case TYPE_LIST:
    result->value.l = ListCreate((FreeFunc*)TypeFree);
    break;
  default:
    TypeError("Internal error; bad TypeThingTag");
  }
  va_end(args);
  return result;
}

static void TypeTFree(TypeT t)
{
  ListFree(t->i_params.value.l);
  ListFree(t->t_params.value.l);
  free(t);
}

static void TypeFreePartial(Type t)
{
  switch(t->tag) {
  case TYPE_TYPE:
    TypeTFree(t->value.t);
    break;
  case TYPE_UNKNOWN:
    free(t->value.s);
    break;
  case TYPE_LIST:
    ListFree(t->value.l);
    break;
  }
}

void TypeFree(Type t)
{
  TypeFreePartial(t);
  free(t);
}

/* Changes t1 to be t2, destroying t2. */
void TypeReplace(Type t1, Type t2)
{
  TypeFreePartial(t1);
  *t1 = *t2;
  free(t2);
}

/* Updates the known size of t. Should be executed whenever an unknown
 * is bound in t.
 */
void TypeComputeSize(Type t)
{
  if(t->tag == TYPE_TYPE) {
    TypeT tt = t->value.t;
    if(PValIsInt(tt->class->sizeFunc)) 
      tt->size = PValAsInt(tt->class->sizeFunc);
    else 
      tt->size = tt->class->sizeFunc(t);
    if(PValIsInt(tt->class->alignFunc)) 
      tt->align = PValAsInt(tt->class->alignFunc);
    else 
      tt->align = tt->class->alignFunc(t);
  }
}

/* Returns the size, in bytes, of an instance of t. */
int TypeSize(Type t)
{
  switch(t->tag) {
  case TYPE_TYPE:
    return t->value.t->size;
  case TYPE_INT:
    return t->value.i;
  }
  return 0;
}

/* Returns the alignment requirement of an instance of t. */
int TypeAlign(Type t)
{
  switch(t->tag) {
  case TYPE_TYPE:
    return t->value.t->align;
  }
  return 0;
}

/* Returns true if t has some unknown parameters in its tree. */
int TypeHasUnknowns(Type t)
{
  switch(t->tag) {
  case TYPE_UNKNOWN:
    return TypeNamed(t);
  case TYPE_INT:
    return 0;
  case TYPE_LIST:
    {Type tt;ListIter(p, tt, t->value.l) {
      if(TypeHasUnknowns(tt)) return 1;
    }}
    break;
  case TYPE_TYPE:
    if(TypeHasUnknowns(&t->value.t->i_params)) return 1;
    if(TypeHasUnknowns(&t->value.t->t_params)) return 1;
    break;
  }
  return 0;
}

static TypeT TypeTCopy(TypeT t)
{
  TypeT result = Allocate(1, struct TypeTStruct);
  Type type;
  *result = *t;
  type = TypeCopy(&t->i_params);
  result->i_params = *type;
  free(type); /* not TypeFree */
  type = TypeCopy(&t->t_params);
  result->t_params = *type;
  free(type);
  return result;
}

/* Returns a replica of t. */
Type TypeCopy(Type t)
{
  Type result = Allocate(1, struct TypeStruct);
  result->tag = t->tag;
  switch(t->tag) {
  case TYPE_INT:
    result->value.i = t->value.i;
    break;
  case TYPE_TYPE:
    result->value.t = TypeTCopy(t->value.t);
    break;
  case TYPE_LIST:
    result->value.l = ListCopy(t->value.l, (void*(*)())TypeCopy);
    break;
  case TYPE_UNKNOWN:
    result->value.s = strdup(t->value.s);
    break;
  }
  return result;
}

void TypeError(char *err, ...)
{
  va_list args;
  va_start(args, err);
  fprintf(stderr, "Type Error: ");
  vfprintf(stderr, err, args);
  fprintf(stderr, "\n");
  fflush(stderr);
  exit(1);
}
