/* Ph_MemGetData: Reading the coefficient data of a member */

#include "photobook.h"
#include <tpm/stream.h>
#include <type/match.h>

/* Globals *******************************************************************/

typedef struct {
  char *name;
  Type type;
  void *value;
} ValueCache;

/* Size of each member's value cache */
#define VALUE_MAX 5

/* Prototypes ****************************************************************/

void Ph_MemSetData(Ph_Handle phandle, Ph_Member m, char *field, 
		   Type type, void *data);
void *Ph_MemGetData(Ph_Handle phandle, Ph_Member m, char *field, 
		    Type type, int force);
void PhValueCacheFree(void *d);
void PhFFCacheFree(void);
char *PhGetQuark(Ph_Handle phandle, char *s);

/* Private */
static void *LoadData(Ph_Handle phandle, Ph_Member m, char *field, Type type);
static ValueCache *NewValue(char *name, Type type, void *value);

/* Functions *****************************************************************/

/* Adds data to m's value cache.
 * Not guaranteed to persist for very long.
 */
void Ph_MemSetData(Ph_Handle phandle, Ph_Member m, char *field, 
		   Type type, void *data)
{
  if(ListSize(m->data) == VALUE_MAX) ListRemoveRear(m->data, NULL);
  ListAddFront(m->data, NewValue(field, TypeCopy(type), data));
}

/* Returns field of m, caching it in memory for fast subsequent accesses.
 * If type is specified, coerces the data loaded from disk to that type.
 * Additionally, if force is true, coerces the data to the type
 * even if it was loaded as a different type.
 * Requires: field is a quark (see PhGetQuark)
 */
void *Ph_MemGetData(Ph_Handle phandle, Ph_Member m, char *field, 
		    Type type, int force)
{
  ValueCache *a;
  void *data;

  /* is it in the value cache? */
  {ListIter(p, a, m->data) {
    if(a->name == field) {
      Binding b1, b2;
      if(type && force) {
	if(!TypeMatch(type, a->type, &b1, &b2)) {
	  fprintf(stderr, "MemberData: type request mismatched cached type\n");
	  return NULL;
	}
	else {
	  BindingApply(b1, type);
	  BindingFree(b1);
	  BindingFree(b2);
	}
      }
      return a->value;
    }
  }}

  /* read in the data */
  data = LoadData(phandle, m, field, type);
  if(!data) return NULL;

  /* add to the cache */
  if(ListSize(m->data) == VALUE_MAX) ListRemoveRear(m->data, NULL);
  ListAddFront(m->data, NewValue(field, TypeCopy(type), data));
  return data;
}

/* Frees a node in the value cache */
void PhValueCacheFree(void *p)
{
  ValueCache *vc = p;
  TypeFree(vc->type);
  free(vc->value);
  free(vc);
}

/* Field file cache node */
typedef struct {
  char *name;
  Stream *da;
  Type type;
} FFile;

/* cache of FFiles */
static List FFiles = NULL;
#define FF_MAX 5

static void FFileFree(FFile *ff)
{
  if(debug) printf("Closing field file %s\n", ff->name);
  StreamClose(*ff->da);
  free(ff->da);
  TypeFree(ff->type);
  free(ff);
}

/* Frees the field file cache */
void PhFFCacheFree(void)
{
  if(!FFiles) return;
  ListFree(FFiles);
  FFiles = NULL;
}

/* Returns a field file node for field, caching the node for fast subsequent
 * accesses.
 * Returns NULL if no file exists for that field.
 */
static FFile *FieldFile(Ph_Handle phandle, char *field)
{
  FFile *ff;
  char str[1000], *name;
  FileHandle fp;
  int len;
  Stream *da;
 
  sprintf(str, "%s/%s/%s/%s", phandle->data_dir, phandle->db_name,
	  field, ".everything");
  name = PhGetQuark(phandle, str);

  /* it is in the cache? */
  if(FFiles == NULL) FFiles = ListCreate((FreeFunc*)FFileFree);
  else {
    ListIter(p, ff, FFiles) {
      if(ff->name == name) return ff;
    }
  }

  /* does the file exist? */
  fp = fopen(name, "r");
  if(!fp) return NULL;
  if(debug) printf("Opening field file %s\n", name);

  /* how big is each element? */
  ReadAdjust(&len, TPM_INT, 1, fp);
  
  /* create a DiskArray */
  da = Allocate(1, Stream);
  DiskArrayOpen(da, fp, phandle->total_members, len*sizeof(double));

  /* add a new cache entry */
  if(ListSize(FFiles) == FF_MAX) ListRemoveRear(FFiles, NULL);
  ff = Allocate(1, FFile);
  ff->name = name;
  ff->da = da;
  sprintf(str, "array[%d] double", len);
  ff->type = TypeParse(str);
  ListAddFront(FFiles, ff);
  return ff;
}

/* Loads field of m from disk, and returns it.
 * If type != NULL, coerces the data be that type.
 * Returns NULL if the field could not be loaded or coercion failed.
 */
static void *LoadData(Ph_Handle phandle, Ph_Member m, char *field, Type type)
{
  void *data;
  FFile *ff;
  Binding b1, b2;

  /* Try to open a field file */
  ff = FieldFile(phandle, field);
  if(ff) {
    if(debug) printf("Loading field %s for %s\n", field, Ph_MemName(m));
    data = Allocate(TypeSize(ff->type), char);
    StreamRead(*ff->da, m->index, data);
    AdjustBuffer(data, TypeInt(ff->type,0), 
		 TypeSize(TypeType(ff->type,0)));
    if(type) {
      if(!TypeMatch(type, ff->type, &b1, &b2)) {
	fprintf(stderr, "MemberData: bad type request / type mismatch\n");
	free(data);
	return NULL;
      }
      else {
	BindingApply(b1, type);
	BindingFree(b1);
	BindingFree(b2);
      }
    }
    return data;
  }
  else {
    fprintf(stderr, "Cannot find field file for `%s'\n", field);
    return NULL;
  }
}

/* Returns a quarked version of s, i.e. returns identical pointer values
 * for all equivalent strings.
 * (This naming convention is in keeping with the quantum mechanical theory 
 * that all equivalent quarks are identical.)
 */
char *PhGetQuark(Ph_Handle phandle, char *s)
{
  char *quark;
  ListIter(p, quark, phandle->quark_table) {
    int c = strcmp(quark, s);
    if(c == 0) return quark;
    if(c > 0) break;
  }
  ListAddInOrder(phandle->quark_table, quark = strdup(s), (CmpFunc*)strcmp);
  return quark;
}

static ValueCache *NewValue(char *name, Type type, void *value)
{
  ValueCache *vc = Allocate(1, ValueCache);
  vc->name = name;
  vc->type = type;
  vc->value = value;
  return vc;
}
