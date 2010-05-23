#include "type.h"

static int Offset;
typedef void ParseFunc(Type t, char *s, void **data);
typedef void UnparseFunc(Type t, void *data, char **s);
typedef struct {
  ParseFunc *parseFunc;
  UnparseFunc *unparseFunc;
} ParseData;
#define TParseData(type) ((ParseData*)((char*)(type->value.t->class)+Offset))

/* Parse/unparse functions ***************************************************/

static void IntParse(Type t, char *s, void **data)
{
  sscanf(s, "%d", *data);
}

static void IntUnparse(Type t, void *data, char**s)
{
  if(!*s) {
    *s = Allocate(sizeof(int)*24/10+1, char);
  }
  sprintf(*s, "%d", *(int*)data);
}

static void FloatParse(Type t, char *s, void **data)
{
  sscanf(s, "%f", *data);
}

static void FloatUnparse(Type t, void *data, char**s)
{
  if(!*s) {
    *s = Allocate(sizeof(float)*24/10+5, char);
  }
  sprintf(*s, "%g", *(float*)data);
}

/* Parse function table ******************************************************/

typedef struct {
  char *name;
  ParseData data;
} ParseTable;
ParseTable parseTable[] = {
  { "float", { FloatParse, FloatUnparse } },
  { "int",   { IntParse, IntUnparse } },
  { NULL,    { NULL, NULL } },
};

static void ParseDataHook(TypeClass class, ParseData *data)
{
  ParseTable *pt;

  /* find the ParseTable entry */
  for(pt=parseTable;pt->name;pt++) {
    if(!strcmp(pt->name, class->name)) {
      *data = pt->data;
      return;
    }
  }
  data->parseFunc = data->unparseFunc = NULL;
}

int ParseValue(Type t, char *s, void **data)
{
  if(TParseData(t)->parseFunc) {
    if(!*data) {
      *data = Allocate(TypeSize(t), char);
    }
    TParseData(t)->parseFunc(t, s, data);
    return 0;
  }
  return 1;
}

int UnparseValue(Type t, void *data, char **s)
{
  if(TParseData(t)->unparseFunc) {
    TParseData(t)->unparseFunc(t, data, s);
    return 0;
  }
  return 1;
}

main()
{
  char str[100], *s;
  Type t;
  void *data;

  TypeTableCreate();
  Offset = TypeClassAddData(sizeof(ParseData), (TypeClassHook*)ParseDataHook);
  TypeDefineTpm();

  printf("Enter a type:\n");
  fgets(str, 100, stdin);
  t = TypeParse(str);

  printf("Enter a value:\n");
  fgets(str, 100, stdin);
  data = NULL;
  if(ParseValue(t, str, &data)) {
    printf("Don't know how to parse that type\n");
    exit(1);
  }

  printf("Type was: ");
  TypeUnparse(stdout, t);
  printf("\n");
  printf("Value was: ");
  s = NULL;
  if(UnparseValue(t, data, &s)) {
    printf("Don't know how to unparse that type\n");
    exit(1);
  }
  printf("%s\n", s);
  free(s);
  free(data);

  TypeFree(t);
  TypeTableFree();
}

