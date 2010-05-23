#include <type/type.h>

typedef int ParseFunc(Type t, char *s, void **data);
typedef char *UnparseFunc(Type t, void *data);
typedef struct {
  ParseFunc *parseFunc;
  UnparseFunc *unparseFunc;
} ParseData;

typedef struct {
  char *name;
  ParseData data;
} ParseTable;

int ParseValue(Type t, char *s, void **data);
char *UnparseValue(Type t, void *data);
void TypeParseTpm(void);
void TypeParseFree(void);
void TypeParseDefine(ParseTable *pt);
ParseData *TypeParseData(TypeClass tc);

/* internal */
void TpmParseTable(List table);
