/* Definitions for using Type trees.
 * A TypeT is the basic type structure, which is a named class plus
 * some number of integer or Type parameters.
 * A Type is a parameter of a TypeT and is either a string, an unknown, 
 * a list of Types, or a TypeT.
 * For example, "int" is a TypeT class with no parameters, 
 * "list" is a TypeT class with one Type parameter, and 
 * "array" is a TypeT class with one integer and one Type parameter.
 * Parameters are allowed to be unknown for the purposes of type unification;
 * see "match.h".
 */
#ifndef TYPE_H_INCLUDED
#define TYPE_H_INCLUDED

#include <tpm/list.h>

/* Data types ****************************************************************/

/* "Type" is { tag, value } */
typedef enum { TYPE_INT, TYPE_TYPE, TYPE_STRING, 
		 TYPE_UNKNOWN, TYPE_LIST } TypeTag;

struct TypeTStruct;
typedef struct TypeStruct {
  TypeTag tag;
  union {
    int i;                 /* TYPE_INT */
    struct TypeTStruct *t; /* TYPE_TYPE */
    char *s;               /* TYPE_STRING, TYPE_UNKNOWN */
    List l;                /* TYPE_LIST */
  } value;
} *Type;

/* "TypeClass" is { name, sizeFunc, alignFunc, n_i_p, n_t_p } */
typedef int TypeSizeFunc(Type type);

typedef struct TypeClassStruct {
  char *name;
  TypeSizeFunc *sizeFunc, *alignFunc;
  int num_i_params, num_t_params; /* expected number of params */
} *TypeClass;

/* "TypeT" is { class, size, align, i_p, t_p } */
typedef struct TypeTStruct { 
  TypeClass class;
  int size, align;
  struct TypeStruct i_params, t_params; /* both are TYPE_LISTs */
} *TypeT;

/* "TypeClassHook" is called on TypeDefine */
typedef void TypeClassHook(TypeClass class, char *data);

/* Globals *******************************************************************/

#define VARIABLE_PARAMS -1

#define TypeClass(type) (type)->value.t->class->name
#define TypeNInt(type) ListSize((type)->value.t->i_params.value.l)
#define TypeNType(type) ListSize((type)->value.t->t_params.value.l)
#define _TypeInt(type,n) \
  ((Type)ListValueAtIndex((type)->value.t->i_params.value.l, n))
#define TypeType(type,n) \
  ((Type)ListValueAtIndex((type)->value.t->t_params.value.l, n))
#define TypeInt(type,n) _TypeInt(type,n)->value.i

/* true if the parameter is unknown */
#define TypeIntU(t,n) TypeUnknown(_TypeInt(t,n))
#define TypeTypeU(t,n) TypeUnknown(TypeType(t,n))

/* the name of the unknown parameter */
#define TypeUInt(t,n) TypeUName(_IntParam(t,n))
#define TypeUType(t,n) TypeUName(_TypeParam(t,n))

#define TypeUnknown(x) ((x)->tag == TYPE_UNKNOWN)
#define TypeNamed(x) (*(x)->value.s != 0)
#define TypeUName(x) (x)->value.s
#define TypeStar(t) (TypeUnknown(t) && (*TypeUName(t) == '*'))

/* Prototypes ****************************************************************/

/* type.c */
Type TypeCreate(TypeTag tag, ...);
void TypeFree(Type t);
void TypeReplace(Type t1, Type t2);
void TypeComputeSize(Type t);
int TypeSize(Type t);
int TypeAlign(Type t);
int TypeHasUnknowns(Type t);
Type TypeCopy(Type t);
void TypeError(char *err, ...);

/* type_parse.c */
Type TypeParse(char *s);
void TypeUnparse(FILE *fp, Type t);

/* type_table.c */
void TypeClassDefine(char *name, 
		     int align, TypeSizeFunc *alignFunc,
		     int size, TypeSizeFunc *sizeFunc, 
		     int i_params, int t_params);
TypeClass TypeClassGet(char *s);
void TypeTableCreate(void);
void TypeTableFree(void);

/* quark.c */
void TypeQuarksCreate(void);
void TypeQuarksFree(void);
void TypeQuarksFlush(void);
Type TypeQuark(char *type_s);

#endif
