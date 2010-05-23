/* Object manipulation 
 *   create/free objects
 *   lookup classes/objects
 *   set/get fields
 *   put watches on fields
 */

#include "photobook.h"
#include <type/parse_value.h>
#include <assert.h>

/* Globals *******************************************************************/

/* Prototypes ****************************************************************/

Ph_Object Ph_ObjCreate(Ph_Handle phandle, ObjClass class, char *name);
void Ph_ObjFree(Ph_Object obj);
ObjClass PhClassFind(List class_list, char *class_name);
Ph_Object PhObjFind(List obj_list, char *name);
ObjField *Ph_ObjField(Ph_Object obj, char *field);

int Ph_ObjSetString(Ph_Object obj, char *field, char *value_s);
int Ph_ObjSet(Ph_Object obj, char *field, void *value);
int Ph_ObjGetString(Ph_Object obj, char *field, char **value);
int Ph_ObjGet(Ph_Object obj, char *field, void *value);

int Ph_ObjWatch(Ph_Object obj, char *field, 
		ObjFieldCallback *func, void *userData);
void Ph_ObjAddField(Ph_Object obj, char *field, Type type,
		    Ph_Object producer, char *p_field, void *p_data,
		    char dont_cache, char dont_load);

ObjField *PhObjField(Ph_Object obj, char *field);
ObjTField *PhObjTField(Ph_Object obj, char *field);
GenericFunc *PhObjFunc(Ph_Object obj, char *field);

/* Private */
static GenericFunc *PhThisObjFunc(Ph_Object obj, char *field);
static void PhCallCallbacks(Ph_Object obj, ObjField *of);

/* Functions *****************************************************************/

/* Sets the field of obj to the parsed value of value_s.
 * Returns an error code if obj does not have the field
 * or if there is a parse error.
 */
int Ph_ObjSetString(Ph_Object obj, char *field, char *value_s)
{
  ObjField *of = PhObjField(obj, field);
  if(!of) {
    ObjTField *otf = PhObjTField(obj, field);
    if(!otf) return PH_ERROR;
    if(!otf->cache_flag) otf->cache_flag = 1;
    of = (ObjField*)otf;
  }

  /* the parse routines should make sure that they don't mutate on an error */
  if(!ParseValue(of->type, value_s, &of->data)) {
    return PH_ERROR;
  }
  PhCallCallbacks(obj, of);
  return PH_OK;
}

/* Sets the field of obj to *value.
 * Returns an error code if obj does not have the field.
 */
int Ph_ObjSet(Ph_Object obj, char *field, void *value)
{
  ObjField *of = PhObjField(obj, field);
  if(!of) {
    ObjTField *otf = PhObjTField(obj, field);
    if(!otf) return PH_ERROR;
    if(!otf->cache_flag) otf->cache_flag = 1;
    of = (ObjField*)otf;
  }

  /* Warning: if of->type includes quarks, the given value must already
   * be quarked by the caller.
   */
  memcpy(of->data, value, TypeSize(of->type));
  PhCallCallbacks(obj, of);
  return PH_OK;
}

/* Puts the unparsed field of obj in *value.
 * Allocates the string for *value; caller is expected to free it.
 * Returns an error code if obj does not have the field,
 * if the field has no value,
 * or if there is an unparsing error (i.e. the value is un-unparseable).
 */
int Ph_ObjGetString(Ph_Object obj, char *field, char **value)
{
  ObjTField *otf = NULL;
  ObjField *of = PhObjField(obj, field);
  if(!of) {
    otf = PhObjTField(obj, field);
    if(!otf) return PH_ERROR;
    if(PhLoadField(obj, otf) == PH_ERROR) return PH_ERROR;
    of = (ObjField*)otf;
  }
  *value = UnparseValue(of->type, of->data);
  if(otf && (otf->cache_flag != 1)) FreeValue(of->type, of->data);
  if(!*value) return PH_ERROR;
  return PH_OK;
}

/* Puts the field of obj in *value.
 * Returns an error code if obj does not have the field.
 */
int Ph_ObjGet(Ph_Object obj, char *field, void *value)
{
  ObjTField *otf = NULL;
  ObjField *of = PhObjField(obj, field);
  if(!of) {
    otf = PhObjTField(obj, field);
    if(!otf) return PH_ERROR;
    if(PhLoadField(obj, otf) == PH_ERROR) return PH_ERROR;
    of = (ObjField*)otf;
  }
  memcpy(value, of->data, TypeSize(of->type));
  if(otf && (otf->cache_flag != 1)) free(of->data);
  return PH_OK;
}

/* Establishes a watch on field of obj, so that when the field is changed
 * func(obj, field, userData) is called.
 * Returns an error code if obj does not have the field.
 */
int Ph_ObjWatch(Ph_Object obj, char *field, 
		ObjFieldCallback *func, void *userData)
{
  List *cbl;
  ObjFieldCB *cb;
  ObjField *of = PhObjField(obj, field);
  if(!of) {
    ObjTField *otf = PhObjTField(obj, field);
    if(!otf) return PH_ERROR;
    of = (ObjField*)otf;
  }
  cb = Allocate(1, ObjFieldCB);
  cb->callback = func;
  cb->userData = userData;

  cbl = &of->callbacks;
  if(!*cbl) *cbl = ListCreate(GenericFree);
  ListAddRear(*cbl, cb);

  return PH_OK;
}

/* Calls all watches on field of obj */
static void PhCallCallbacks(Ph_Object obj, ObjField *of)
{
  ObjFieldCB *cb;
  if(!of->callbacks) return;
  {ListIter(p, cb, of->callbacks) {
    cb->callback(obj, of->name, cb->userData);
  }}
}

/* Returns the ObjField structure for field of obj (perm or trans) */
ObjField *Ph_ObjField(Ph_Object obj, char *field)
{
  ObjField *of = PhObjField(obj, field);
  if(!of) {
    ObjTField *otf = PhObjTField(obj, field);
    if(!otf) return NULL;
    of = (ObjField*)otf;
  }
  return of;
}

/* Returns a (permanent) field entry of obj.
 * Returns NULL if obj does not have the field.
 */
ObjField *PhObjField(Ph_Object obj, char *field)
{
  ObjField *of;
  if(obj->fields) {
    for(of=obj->fields;of->name;of++) {
      if(!strcmp(field, of->name)) return of;
    }
  }
  if(!obj->super) return NULL;
  return PhObjField(obj->super, field);
}

/* Returns a transient field entry of obj.
 * Returns NULL if obj does not have the field.
 */
ObjTField *PhObjTField(Ph_Object obj, char *field)
{
  extern struct ObjClassStruct Member;
  ObjTField *otf;
  if(obj->trans_fields) {
    ListIter(p, otf, obj->trans_fields) {
      if(!strcmp(field, otf->name)) return otf;
    }
  }
  if(obj->class == &Member) {
    /* try to add the TField automatically */
    int i;
    for(i=0;field[i];i++) {
      if(field[i] == '/') {
	if(!strncmp(field, "view", i)) {
	  if(debug) printf("auto-loading %s for %s\n", field, Ph_ObjName(obj));
	  if(PhLoadView(obj->phandle, &field[i+1]) == NULL)
	    return NULL;
	  /* new field was added to the front */
	  return (ObjTField*)obj->trans_fields->front->data;
	}
	break;
      }
    }
  }
  if(!obj->super) return NULL;
  return PhObjTField(obj->super, field);
}

/* Returns a function field of obj, ignoring inheritance.
 * Returns NULL if obj does not have the field.
 */
static GenericFunc *PhThisObjFunc(Ph_Object obj, char *field)
{
  ObjClassFunc *cf;
  if(obj->class->funcs) {
    for(cf=obj->class->funcs;cf->name;cf++) {
      if(!strcmp(field, cf->name)) return cf->func;
    }
  }
  return NULL;
}

/* Returns a function field of obj.
 * Returns NULL if obj does not have the field.
 */
GenericFunc *PhObjFunc(Ph_Object obj, char *field)
{
  GenericFunc *f = PhThisObjFunc(obj, field);
  if(f) return f;
  if(!obj->super) return NULL;
  return PhObjFunc(obj->super, field);
}

/* Primitive object functions ************************************************/

static void FreeTField(ObjTField *otf)
{
  if(otf->cache_flag == 1) FreeValue(otf->type, otf->data);
  if(otf->callbacks) ListFree(otf->callbacks);
  free(otf);
}

void Ph_ObjAddField(Ph_Object obj, char *field, Type type,
		    Ph_Object producer, char *p_field, void *p_data,
		    char dont_cache, char dont_load)
{
  ObjTField *otf = Allocate(1, ObjTField);

  otf->name = Ph_StringQuark(field);
  otf->type = type;
  otf->callbacks = NULL;
  otf->producer = producer;
  otf->p_field = p_field;
  otf->p_data = p_data;
  otf->cache_flag = dont_cache ? 2:0;
  otf->use_disk = !dont_load;

  if(!obj->trans_fields) 
    obj->trans_fields = ListCreate((FreeFunc*)FreeTField);
  ListAddFront(obj->trans_fields, otf);
}

/* Returns the number of fields class has */
static int NumClassFields(ObjClass class)
{
  int i;
  for(i=0;;i++) if(!class->fields[i].name) break;
  return i;
}

/* Returns a new object of class oc.
 * Calls the object's constructor, if it has one.
 * Returns NULL if oc is NULL.
 */
Ph_Object Ph_ObjCreate(Ph_Handle phandle, ObjClass oc, char *name)
{
  Ph_Object obj;
  int i,n;
  PhObjConstructor *constructor;
  ObjField *of;
  char *ptr;
  ObjClassField *ft;
  
  if(!oc) return NULL;
  obj = (Ph_Object)malloc(sizeof(*obj));
  obj->name = name;
  obj->class = oc;
  obj->data = Allocate(oc->size, char);
  obj->phandle = phandle;

  /* construct the super object */
  if(!oc->super) obj->super = NULL;
  else obj->super = Ph_ObjCreate(phandle, oc->super, name);

  /* construct the object field table */
  if(!oc->fields) {
    obj->fields = NULL;
  }
  else {
    n = NumClassFields(oc);
    of = obj->fields = Allocate(n+1, ObjField);
    ptr = obj->data;
    ft = oc->fields;
    for(i=0;i<n;i++) {
      int size;
      of->name = ft->name;
      of->callbacks = NULL;
      of->type = Ph_TypeQuark(ft->type_s); /* no check */
      of->shared_type = !TypeHasUnknowns(of->type);
      /* make sure we are aligned for this type */
      size = TypeSize(of->type);
      MakeAligned(ptr, size);
      of->data = ptr;
      ptr += size;
      assert(((long)ptr - (long)obj->data) <= oc->size);
      of++;
      ft++;
    }
    of->name = NULL;
  }

  /* initialize transient fields */
  obj->trans_fields = NULL;
  obj->cache_used = 0;
  obj->cache_limit = 128*1024; /* ad hoc; change me! */

  /* call constructor */
  constructor = (PhObjConstructor*)PhThisObjFunc(obj, "constructor");
  if(constructor) constructor(obj);
  return obj;
}

/* Frees the storage used by an object.
 * Calls the object's destructor, if it has one.
 */
void Ph_ObjFree(Ph_Object obj)
{
  PhObjDestructor *destructor;
  ObjField *of;

  /* call destructor */
  destructor = (PhObjDestructor*)PhThisObjFunc(obj, "destructor");
  if(destructor) destructor(obj);

  /* free fields */
  for(of=obj->fields;of->name;of++) {
    if(!of->shared_type) TypeFree(of->type);
    if(of->callbacks) ListFree(of->callbacks);
  }
  free(obj->fields);

  /* free transient fields */
  if(obj->trans_fields) ListFree(obj->trans_fields);

  free(obj->data);
  if(obj->super) Ph_ObjFree(obj->super);
  free(obj);
}

/* Returns the class in class_list with name class_name.
 * Returns NULL if there is no such class.
 */
ObjClass PhClassFind(List class_list, char *class_name)
{
  ObjClass oc;
  ListIter(p, oc, class_list) {
    if(!strcmp(oc->name, class_name)) return oc;
  }
  return NULL;
}

/* Returns the object in obj_list whose name is <name>.
 * Returns NULL is there is no such object.
 */
Ph_Object PhObjFind(List obj_list, char *name)
{
  Ph_Object obj;
  ListIter(p, obj, obj_list) {
    if(!strcmp(Ph_ObjName(obj), name)) return obj;
  }
  return NULL;
}
