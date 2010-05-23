#include "type.h"
#include <tpm/stream.h>

/* Prototypes ****************************************************************/

void TypeDefineTpm(void);

/* Functions *****************************************************************/

static int ArrayAlign(Type t)
{
  return TypeAlign(TypeType(t,0));
}

static int ArraySize(Type t)
{
  return TypeSize(_TypeInt(t,0)) * TypeSize(TypeType(t,0));
}

static int StructAlign(Type t)
{
  if(TypeNType(t)) return TypeAlign(TypeType(t,0));
  return 1;
}

static int StructSize(Type t)
{
  int i, a, s, size;

  size = 0;
  for(i=0;i<TypeNType(t);i++) {
    s = TypeSize(TypeType(t,i));
    if(!s) return 0; /* short-circuit for unknown subtype */
    a = TypeAlign(TypeType(t,i));
    /* make space for proper alignment */
    if(size % a) size += a - (size % a);
    size += s;
  }
  return size;
}

static int DMatrixAlign(Type t)
{
  return sizeof(double);
}

static int DMatrixSize(Type t)
{
  return TypeSize(_TypeInt(t,0)) * TypeSize(_TypeInt(t,1)) * sizeof(double);
}

void TypeDefineTpm(void)
{
  TypeClassDefine("tad", sizeof(void*), NULL, sizeof(void*), NULL, 0, 0);
  TypeClassDefine("tadt", sizeof(void*), NULL, sizeof(void*), NULL, 0, 1);
  TypeClassDefine("array", 0, ArrayAlign, 0, ArraySize, 1, 1);
  TypeClassDefine("struct", 0, StructAlign, 0, StructSize, 0, VARIABLE_PARAMS);
  TypeClassDefine("stream", sizeof(void*), NULL, sizeof(Stream), NULL, 0, 1);
  TypeClassDefine("list", sizeof(void*), NULL, sizeof(List), NULL, 0, 1);
  TypeClassDefine("disk_array", sizeof(void*), NULL, sizeof(Stream), NULL, 
		  1, 1);
  TypeClassDefine("ptr", sizeof(void*), NULL, sizeof(void*), NULL, 0, 1);
  TypeClassDefine("exec", sizeof(int), NULL, sizeof(FILE), NULL, 0, 1);
  TypeClassDefine("type", sizeof(int), NULL, sizeof(Type), NULL, 0, 0);

  TypeClassDefine("char", sizeof(char), NULL, sizeof(char), NULL, 0, 0);
  TypeClassDefine("int", sizeof(int), NULL, sizeof(int), NULL, 0, 0);
  TypeClassDefine("float", sizeof(float), NULL, sizeof(float), NULL, 0, 0);
  TypeClassDefine("double", sizeof(double), NULL, sizeof(double), NULL, 0, 0);
  TypeClassDefine("string", sizeof(char*), NULL, sizeof(char*), NULL ,0, 0);
  TypeClassDefine("dmatrix", 0, DMatrixAlign, 0, DMatrixSize, 2, 0);

  TypeClassDefine("unknown", sizeof(void*), NULL, sizeof(void*), NULL, 0, 0);
}
