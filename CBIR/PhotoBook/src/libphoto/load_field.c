/* PhLoadField: compute the transient field of an object */

#include "photobook.h"
#include <tpm/stream.h>
#include <type/match.h>
#include <assert.h>

/* Globals *******************************************************************/

#define VERBOSE 0

typedef void *ProducerFunc(Ph_Object self, Ph_Object obj, void *data);

/* Prototypes ****************************************************************/

void FreeValue(Type type, void *data);

int PhLoadField(Ph_Object obj, ObjTField *otf);

/* Private */
static int LoadData(Ph_Object obj, char *field, Type type, 
		    void **value_return);

/* Functions *****************************************************************/

/* this is a kludge; fix me! */
void FreeValue(Type type, void *data)
{
  if(!strcmp(TypeClass(type), "ptr")) free(*(void**)data);
  free(data);
}

/* Computes the field of object. If otf->use_disk is true
 * and the field is found on disk, then it is read in.
 * Otherwise, the field is computed by the producer object, if any.
 * Returns an error code if the disk read failed, 
 * the producer failed or the producer wasn't specified.
 */
int PhLoadField(Ph_Object obj, ObjTField *otf)
{
  /* is the value already cached? */
  if(otf->cache_flag == 1) return PH_OK;

  /* is the transient cache full? */
  obj->cache_used += TypeSize(otf->type);
  if(obj->cache_used >= obj->cache_limit) {
    /* find the last cached field */
    ObjTField *f, *last = NULL;
    ListIter(p, f, obj->trans_fields) {
      if((f != otf) && (f->cache_flag == 1)) last = f;
    }
    if(last) {
      /* uncache it */
      if(debug) printf("uncaching %s\n", last->name);
      obj->cache_used -= TypeSize(last->type);
      FreeValue(last->type, last->data);
      last->cache_flag = 0;
    }
  }

  /* try to read from disk */
  if(otf->use_disk) {
    if(LoadData(obj, otf->name, otf->type, &otf->data) == PH_OK) goto done;
  }
  
  /* call the producer */
  if(otf->producer) {
    ProducerFunc *pf = (ProducerFunc*)PhObjFunc(otf->producer, otf->p_field);
    if(!pf) return PH_ERROR;
    /* the producer returns by value */
    otf->data = Allocate(TypeSize(otf->type), char);
    *(void**)otf->data = pf(otf->producer, obj, otf->p_data);
    if(!*(void**)otf->data) return PH_ERROR;
  }
  else return PH_ERROR;

 done:
  if(!otf->cache_flag) otf->cache_flag = 1;
  return PH_OK;
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
  /*TypeFree(ff->type);*/
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
  name = Ph_StringQuark(str);

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
  ff->type = Ph_TypeQuark(str);
  ListAddFront(FFiles, ff);
  return ff;
}

/* Loads field of obj from disk, and returns it in *value_return.
 * If type != NULL, coerces the data be that type.
 * Returns an error code if the field could not be loaded or coercion failed.
 */
static int LoadData(Ph_Object obj, char *field, Type type, void **value_return)
{
  void *data;
  FFile *ff;
  Binding b1, b2;

  /* Try to open a field file */
  ff = FieldFile(obj->phandle, field);
  if(ff) {
    int index;
#if VERBOSE
    if(debug) printf("Loading field %s for %s\n", field, Ph_ObjName(obj));
#endif
    if(Ph_ObjGet(obj, "index", &index) == PH_ERROR) {
      fprintf(stderr, "LoadData: object %s does not have an index\n",
	      Ph_ObjName(obj));
      goto plan_b;
    }
    data = Allocate(TypeSize(ff->type), char);
    StreamRead(*ff->da, index, data);
    AdjustBuffer(data, TypeInt(ff->type,0), 
		 TypeSize(TypeType(ff->type,0)));
    if(type) {
      /* TypeType(type,0) is a major kludge; fix me! */
      if(!TypeMatch(TypeType(type,0), ff->type, &b1, &b2)) {
	fprintf(stderr, "LoadData: bad type request / type mismatch\n");
	free(data);
	return PH_ERROR;
      }
      else {
	BindingApply(b1, type);
	BindingFree(b1);
	BindingFree(b2);
      }
    }
    goto success;
  }

 plan_b:
  /* Try to open an individual file */
  {
    FILE *fp;
    char str[100];
    int len;
    sprintf(str, "%s/%s/%s/%s", obj->phandle->data_dir, obj->phandle->db_name,
	    field, Ph_ObjName(obj));
    fp = fopen(str, "r");
    if(!fp) goto plan_c;
#if VERBOSE
    if(debug) printf("Loading file %s for %s\n", Ph_ObjName(obj), field);
#endif
    /* read the length */
    ReadAdjust(&len, TPM_INT, 1, fp);
    data = Allocate(len, double);
    /* read the data */
    ReadAdjust(data, TPM_DOUBLE, len, fp);
    fclose(fp);
    /* ignore typing issues */
    goto success;
  }
  
 plan_c:
  if(debug) printf("Cannot find file for field %s of %s\n", 
		   field, Ph_ObjName(obj));
  return PH_ERROR;

 success:
  /* allocate a pointer, since we are creating a "ptr array" */
  *value_return = Allocate(1, void*);
  **(void***)value_return = data;
  return PH_OK;
}
