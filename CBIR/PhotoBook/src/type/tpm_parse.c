#include "parse_value.h"
#include <math.h>  /* for strtod */

/* Parse/unparse functions ***************************************************/

#define SkipSpace(s) while(*s && isspace(*s)) s++

static int CharParse(Type t, char *s, void **data)
{
  char *p = (char*)1;
  **(char**)data = (unsigned char)strtol(s, &p, 10);
  return p-s;
}

static char *CharUnparse(Type t, void *data)
{
  char *s = Allocate(sizeof(char)*24/10+1, char);
  sprintf(s, "%d", (int)*(char*)data);
  return s;
}

static int IntParse(Type t, char *s, void **data)
{
  char *p = (char*)1;
  /* this should be strtoul, but sun4 doesn't have that */
  **(int**)data = strtol(s, &p, 10);
  return p-s;
}

static char *IntUnparse(Type t, void *data)
{
  char *s = Allocate(sizeof(int)*24/10+1, char);
  sprintf(s, "%d", *(int*)data);
  return s;
}

static int FloatParse(Type t, char *s, void **data)
{
  char *p = (char*)1;
  double d = strtod(s, &p);
  **(float**)data = (float)d;
  return p-s;
}

static char *FloatUnparse(Type t, void *data)
{
  char *s = Allocate(sizeof(float)*24/10+5, char);
  sprintf(s, "%.8g", *(float*)data);
  return s;
}

static int DoubleParse(Type t, char *s, void **data)
{
  char *p = (char*)1;
  **(double**)data = strtod(s, &p);
  return p-s;
}

static char *DoubleUnparse(Type t, void *data)
{
  char *s = Allocate(sizeof(double)*24/10+5, char);
  sprintf(s, "%.16lg", *(double*)data);
  return s;
}

/* Arrays look like "{3 4}". Nested arrays look like "{{3 4} {5 6}}".
 * requires: subtype is known
 * may bind the int param.
 */
/* Should be passing up a binding, so that e.g.
 *   struct(array[?x] int, array[?x] float)
 * will work. But right now we don't care about this case.
 */
static int ArrayParse(Type t, char *s, void **data)
{
  char ending;
  int i, j, size;
  char *p=s;
  void *d;
  Type subtype = TypeType(t,0);

  /* read the opening symbol */
  while(*p && isspace(*p)) p++;
  if((*p == '[') || (*p == '{')) { ending = *p+2; p++; }
  else ending = 0;

  size = TypeSize(subtype);
  if(_TypeInt(t,0)->tag == TYPE_UNKNOWN) {
    /* BEWARE: if the length is unknown, the data must be free-able. */
    if(*data) free(*data);
    *data = malloc(1);
    for(i=0;;i++) {
      *data = realloc(*data, (i+1)*size);
      d = (char*)*data + i*size;
      j = ParseValue(subtype, p, &d);
      if(j == 0) break;
      p += j;
    }

    /* read the ending */
    while(*p && isspace(*p) && (*p != ending)) p++;
    if(*p != ending) { free(*data); return 0; }
    p++;
  
    *data = realloc(*data, i*size);
    TypeReplace(_TypeInt(t,0), TypeCreate(TYPE_INT, i));
    TypeComputeSize(t);
  }
  else {
    int len = TypeInt(t,0);
    d = *data;
    for(i=0;i<len;i++) {
      j = ParseValue(subtype, p, &d);
      if(j == 0) return 0;
      p += j;
      d = (char*)d + size;
    }

    /* read the ending */
    while(*p && isspace(*p) && (*p != ending)) p++;
    if(*p != ending) return 0;
    p++;
  }
  return p-s;
}

/* requires: everything known */
static char *ArrayUnparse(Type t, void *data)
{
  int i, size, length;
  char *s = malloc(3), *p;

  length = 3;
  s[0] = '{'; s[1] = '\0';
  size = TypeSize(TypeType(t,0));
  for(i=0;i<TypeInt(t,0);i++) {
    p = UnparseValue(TypeType(t,0), data);
    s = realloc(s, length += strlen(p)+1);
    strcat(s, p);
    if(i < TypeInt(t,0)-1) strcat(s, " ");
    else length--;
    free(p);
    data = (char*)data + size;
  }
  s[length-2] = '}'; s[length-1] = '\0';
  return s;
}

/* Structures look like "{3 5.6}".
 */
static int StructParse(Type t, char *s, void **data)
{
  int i, j, size;
  char *p=s;
  void *d;

  /* read a { */
  while(*p && isspace(*p) && (*p != '{')) p++;
  if(*p != '{') return 0;
  p++;

  size = TypeSize(t);
  if(size == 0) {
    /* If size is zero, data must be free-able */
    if(*data) free(*data);
    *data = malloc(1);
    for(i=0;i<TypeNType(t);i++) {
      Type subtype = TypeType(t,i);
      int len;
      if((len = TypeSize(subtype)) == 0) {
	void *nd = NULL;
	j = ParseValue(subtype, p, &nd);
	if(j == 0) { free(*data); return 0; }
	size += (len = TypeSize(subtype));
	MakeAlignedInt(size, TypeSize(subtype));
	*data = realloc(*data, size);
	d = (char*)*data + size - len;
	memcpy(d, nd, TypeSize(subtype));
	free(nd);
      }
      else {
	size += len;
	MakeAlignedInt(size, TypeSize(subtype));
	*data = realloc(*data, size);
	d = (char*)*data + size - len;
	j = ParseValue(subtype, p, &d);
	if(j == 0) { free(*data); return 0; }
      }
      p += j;
    }

    /* read a } */
    while(*p && isspace(*p) && (*p != '}')) p++;
    if(*p != '}') { free(*data); return 0; }
    p++;
  }
  else {
    d = *data;
    for(i=0;i<TypeNType(t);i++) {
      Type subtype = TypeType(t,i);
      MakeAligned(d, TypeSize(subtype));
      j = ParseValue(subtype, p, &d);
      if(j == 0) return 0;
      d = (char*)d + TypeSize(subtype);
      p += j;
    }

    /* read a } */
    while(*p && isspace(*p) && (*p != '}')) p++;
    if(*p != '}') return 0;
    p++;
  }
  return p-s;
}

/* requires: everything known */
static char *StructUnparse(Type t, void *data)
{
  int i, length;
  char *s = malloc(3), *p;

  length = 3;
  s[0] = '{'; s[1] = '\0';
  for(i=0;i<TypeNType(t);i++) {
    Type subtype = TypeType(t,i);
    MakeAligned(data, TypeSize(subtype));
    p = UnparseValue(subtype, data);
    s = realloc(s, length += strlen(p)+1);
    strcat(s, p);
    if(i < TypeNType(t)-1) strcat(s, " ");
    else length--;
    free(p);
    data = (char*)data + TypeSize(subtype);
  }
  s[length-2] = '}'; s[length-1] = '\0';
  return s;
}

static int PtrParse(Type t, char *s, void **data)
{
  void **d = *(void***)data;
  Type subtype = TypeType(t,0);
  if(TypeSize(subtype) == 0) *d = NULL;
  else if(!*d) *d = Allocate(TypeSize(subtype), char);
  return ParseValue(subtype, s, d);
}

static char *PtrUnparse(Type t, void *data)
{
  return UnparseValue(TypeType(t,0), *(void**)data);
}

static int StringParse(Type t, char *s, void **data)
{
  char *p, **dest;
  int skip = 0;
  /* skip whitespace */
  for(;*s && isspace(*s);s++,skip++);
  /* quoted string? */
  if(*s == '"') {
    /* read until closing quote */
    for(p=++s;*p && *p != '"';p++);
    skip+=2; /* skip the quotes, but don't include in the string */
  }
  else {
    /* copy up to whitespace, ']', or '}' */
    for(p=s;*p && !isspace(*p) && *p != ']' && *p != '}';p++);
  }
  dest = *(char***)data;
  *dest = Allocate(p-s+1, char);
  memcpy(*dest, s, p-s+1);
  (*dest)[p-s] = 0;
  return skip+p-s;
}

static char *StringUnparse(Type t, void *data)
{
  return strdup(*(char**)data);
}

#if 0
static int NTypeParse(Type t, char *s, void **data)
{
  **(Type**)data = TypeParse(s);
}

static char *NTypeUnparse(Type t, void *data)
{
}
#endif

/* Parse function table ******************************************************/

static ParseTable tpmParseTable[] = {
  { "string", { StringParse, StringUnparse } },
  { "ptr",    { PtrParse, PtrUnparse } },
  { "struct", { StructParse, StructUnparse } },
  { "array",  { ArrayParse, ArrayUnparse } },
  { "double", { DoubleParse, DoubleUnparse } },
  { "float",  { FloatParse, FloatUnparse } },
  { "int",    { IntParse, IntUnparse } },
  { "char",   { CharParse, CharUnparse } },
  { NULL,     { NULL, NULL } },
};

void TpmParseTable(List table) 
{
  ParseTable *pt;
  
  for(pt=tpmParseTable;pt->name;pt++) {
    ListAddRear(table, pt);
  }
}
