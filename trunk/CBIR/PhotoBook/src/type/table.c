#include "type.h"

/* Globals *******************************************************************/

typedef struct {
  int bytes;
  TypeClassHook *hook;
} DataHook;

/* Private */
static List TypeTable; /* List of TypeClass */
static List ClassData; /* List of DataHook */
static int ClassSize;

/* Prototypes ****************************************************************/

void TypeClassDefine(char *name, 
		     int align, TypeSizeFunc *alignFunc, 
		     int size, TypeSizeFunc *sizeFunc, 
		     int i_params, int t_params);
TypeClass TypeClassGet(char *s);

int TypeClassAddData(int bytes, TypeClassHook *hook);

void TypeTableCreate(void);
void TypeTableFree(void);

/* Functions *****************************************************************/

/* Get the TypeClass structure with name s */
TypeClass TypeClassGet(char *s)
{
  {ListNode ptr;ListIterate(ptr, TypeTable) {
    if(!strcmp(((TypeClass)ptr->data)->name, s)) {
      return ptr->data;
    }
  }}
  TypeError("Unknown type `%s'", s);
  return NULL;
}

/* Register a new type class.
 * If size is zero, then sizeFunc(type) will be used to get the size.
 * i_params/t_params is the number of integer/type parameters.
 * If i_p/t_p is -1, then any number of params is allowed, including zero.
 */
void TypeClassDefine(char *name, 
		     int align, TypeSizeFunc *alignFunc, 
		     int size, TypeSizeFunc *sizeFunc, 
		     int i_params, int t_params)
{
  TypeClass td;

  td = (TypeClass)malloc(ClassSize);
  td->name = strdup(name);
  if(align) td->alignFunc = (TypeSizeFunc*)PValInt(align);
  else td->alignFunc = alignFunc;
  if(size) td->sizeFunc = (TypeSizeFunc*)PValInt(size);
  else td->sizeFunc = sizeFunc;
  td->num_i_params = i_params;
  td->num_t_params = t_params;
  
  /* call the hooks */
  if(!ListEmpty(ClassData)) {
    char *q = (char*)td + sizeof(struct TypeClassStruct);
    DataHook *dh;
    ListIter(p, dh, ClassData) {
      dh->hook(td, q);
      q += dh->bytes;
    }
  }

  ListAddRear(TypeTable, td);
}

/* Define a data area of size <bytes> to be appended to each type class.
 * Call hook(tclass, p), where p is the start of the data area,
 * whenever a new type class is defined.
 */
int TypeClassAddData(int bytes, TypeClassHook *hook)
{
  DataHook *dh;

  if(!ListEmpty(TypeTable)) {
    TypeError("TypeClassAddData: TypeTable is non-empty");
  }
  /* make sure we are aligned */
  if(bytes % 4) bytes += 4 - (bytes % 4);

  dh = Allocate(1, DataHook);
  dh->bytes = bytes;
  dh->hook = hook;
  ListAddRear(ClassData, dh);
  ClassSize += bytes;
  return ClassSize - bytes;
}

static void TypeClassFree(TypeClass td)
{
  free(td->name);
  free(td);
}

/* Create the global type class table.
 * Must be called before any types can be defined.
 */
void TypeTableCreate(void)
{
  TypeTable = ListCreate((FreeFunc*)TypeClassFree);
  ClassData = ListCreate(GenericFree);
  ClassSize = sizeof(struct TypeClassStruct);
}

void TypeTableFree(void)
{
  ListFree(TypeTable);
  ListFree(ClassData);
  MEM_BLOCKS();
  MEM_STATUS();
}
