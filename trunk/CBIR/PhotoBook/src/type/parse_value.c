/* Uses a TypeClass hook to register a parse/unparse function for every
 * type. ParseValue and UnparseValue then simply dispatch to these functions.
 */

#include "parse_value.h"

/* Globals ******************************************************************/

static int Offset;
#define TCParseData(class) ((ParseData*)((char*)(class)+Offset))
#define TParseData(type) TCParseData((type)->value.t->class)

static List parseTable; /* List of ParseTable */

/* Prototypes ****************************************************************/

int ParseValue(Type t, char *s, void **data);
char *UnparseValue(Type t, void *data);
void TypeParseTpm(void);

/* Functions *****************************************************************/

static void ParseDataHook(TypeClass class, ParseData *data)
{
  ParseTable *pt;

  /* find the ParseTable entry */
  ListIter(p, pt, parseTable) {
    if(!strcmp(pt->name, class->name)) {
      *data = pt->data;
      return;
    }
  }
  data->parseFunc = NULL;
  data->unparseFunc = NULL;
}

int ParseValue(Type t, char *s, void **data)
{
  char result, allocated = 0;
  if(TParseData(t)->parseFunc) {
    if(!*data) {
      int size = TypeSize(t);
      /* avoid malloc(0) */
      if(size) {
	allocated = 1;
	*data = Allocate(size, char);
	/* initialize to zero, so that we can test for emptiness */
	memset(*data, 0, size);
      }
    }
    result = TParseData(t)->parseFunc(t, s, data);
    /* if the parsing function fails, make sure we free the data
     * we allocated.
     */
    if(!result && allocated) free(*data);
    return result;
  }
  fprintf(stderr, "ParseValue: Cannot parse type `%s'\n", TypeClass(t));
  return 0;
}

char *UnparseValue(Type t, void *data)
{
  if(TParseData(t)->unparseFunc) {
    return TParseData(t)->unparseFunc(t, data);
  }
  fprintf(stderr, "UnparseValue: Cannot unparse type `%s'\n", TypeClass(t));
  return NULL;
}

void TypeParseTpm(void)
{
  parseTable = ListCreate(NULL);
  /* add parse definitions to parseTable */
  TpmParseTable(parseTable);

  /* register the ParseDataHook */
  Offset = TypeClassAddData(sizeof(ParseData), (TypeClassHook*)ParseDataHook);
}

void TypeParseDefine(ParseTable *pt)
{
  ListAddRear(parseTable, pt);
}

ParseData *TypeParseData(TypeClass tc)
{
  return TCParseData(tc);
}

void TypeParseFree(void)
{
  ListFree(parseTable);
}
